"""
Utilities for computing initial object pose fits from instance masks.
Some part of the code adapted from HOMAN: https://github.com/hassony2/homan
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from utils.constants import REND_SIZE, RENDER_H, RENDER_W, BBOX_EXPANSION_FACTOR
from utils.geometry import (
    rot6d_to_matrix,
    matrix_to_rot6d,
)
from utils.camera import get_K_crop_resize, TCO_init_from_boxes_zup_autodepth, rotation_angle_difference
from utils.losses import batch_mask_iou
from utils.bbox import  make_bbox_square, bbox_xy_to_wh, bbox_wh_to_xy, crop_and_resize
from utils.semantic_corres import min_distance_to_point_set, find_semantic_correspondences, visualize_correspondences
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    BlendParams
)
from pytorch3d.structures import Meshes
import torch.nn.functional as F
import neural_renderer as nr
from detectron2.structures import BitMasks

class ObjTracker(nn.Module):
    """
    Computes the optimal object pose from an instance mask and an exemplar mesh.
    """
    def __init__(
        self,
        ref_image,
        vertices,
        faces,
        textures,
        dino_model,
        gt_dino_feat,
        rotation_init,
        translation_init,
        num_initializations=1,
        K=None,
        rasterizer=None,
        shader=None,
        semantic_corres_infos=None,
        lw_mask=1.0,
        lw_sem=1.0
    ):
        assert ref_image.shape[0] == ref_image.shape[1], "Must be square."
        super().__init__()

        self.register_buffer("vertices", vertices)
        self.register_buffer("faces", faces.repeat(num_initializations, 1, 1))
        self.textures = textures
        # Convention for silhouette-aware loss: -1=occlusion, 0=bg, 1=fg.
        ref_mask = torch.from_numpy((ref_image > 0).astype(np.float32)) # Segmaps for objects
        keep_mask = torch.from_numpy((ref_image >= 0).astype(np.float32)) # Occlusion-aware segmaps
        self.register_buffer("ref_mask",
                             ref_mask.repeat(num_initializations, 1, 1))
        self.register_buffer("keep_mask",
                             keep_mask.repeat(num_initializations, 1, 1))
        self.register_buffer("gt_dino_feat", gt_dino_feat)

        self.dino_model = dino_model
        self.best_score = 0
        self.former_max_idx = None
        self.rotations = nn.Parameter(rotation_init.clone().float(),
                                      requires_grad=True)
        if rotation_init.shape[0] != translation_init.shape[0]:
            translation_init = translation_init.repeat(num_initializations, 1, 1)
        self.translations = nn.Parameter(translation_init.clone().float(),
                                         requires_grad=True)

        self.rasterizer = rasterizer
        self.shader = shader
        self.semantic_corres_infos = semantic_corres_infos
        origin_K = K[:, :2] * REND_SIZE
        self.pytorch3d_cameras = PerspectiveCameras(focal_length=origin_K[0:1, 0, 0].reshape(-1, 1).repeat(num_initializations, 1), principal_point=origin_K[0:1, :2, -1].repeat(num_initializations, 1), 
                                                    in_ndc=False, R=torch.eye(3).unsqueeze(0).repeat(num_initializations, 1, 1), T=torch.zeros((1, 3)).repeat(num_initializations, 1), 
                                                    device="cuda", image_size=torch.tensor([ref_mask.shape[0], ref_mask.shape[1]]).unsqueeze(0))
        # Silhouette renderer (Silhouette renderer of Pytorch3D is in [0, 1] not in {0, 1}, use the origin neural_renderer from HOMAN)
        # blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
        # sil_raster_settings = RasterizationSettings(
        #     image_size=REND_SIZE, 
        #     blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
        #     faces_per_pixel=100, 
        # )
        # self.sil_renderer = MeshRenderer(
        #     rasterizer=MeshRasterizer(
        #         cameras=self.pytorch3d_cameras, 
        #         raster_settings=sil_raster_settings
        #     ),
        #     shader=SoftSilhouetteShader(blend_params=blend_params)
        # )
        self.sil_renderer = nr.renderer.Renderer(
            image_size=ref_image.shape[0],
            K=K,
            R=torch.eye(3).unsqueeze(0).cuda(),
            t=torch.zeros(1, 3).cuda(),
            orig_size=1,
            anti_aliasing=False,
        )
        self.origin_K = origin_K
        self.lw_mask = lw_mask
        self.lw_sem = lw_sem
        self.K = K
        self.losses = None

    def apply_transformation(self):
        """
        Applies current rotation and translation to vertices.
        """
        rots = rot6d_to_matrix(self.rotations)
        return torch.matmul(self.vertices.repeat(rots.shape[0], 1, 1), rots) + self.translations

    def compute_offscreen_loss(self, verts):
        """
        Computes loss for offscreen penalty. This is used to prevent the degenerate
        solution of moving the object offscreen to minimize the chamfer loss.
        """
        # On-screen means coord_xy between [-1, 1] and far > depth > 0
        proj = nr.projection(
            verts,
            self.sil_renderer.K,
            self.sil_renderer.R,
            self.sil_renderer.t,
            self.sil_renderer.dist_coeffs,
            orig_size=1,
        )
        coord_xy, coord_z = proj[:, :, :2], proj[:, :, 2:]
        zeros = torch.zeros_like(coord_z)
        lower_right = torch.max(coord_xy - 1,
                                zeros).sum(dim=(1, 2))  # Amount greater than 1
        upper_left = torch.max(-1 - coord_xy,
                               zeros).sum(dim=(1, 2))  # Amount less than -1
        behind = torch.max(-coord_z, zeros).sum(dim=(1, 2))
        too_far = torch.max(coord_z - self.sil_renderer.far, zeros).sum(dim=(1, 2))
        return lower_right + upper_left + behind + too_far
    
    def compute_semantic_reproj_loss(self):
        pts_3d = self.semantic_corres_infos['pts_3d'].clone()
        pts_2d = self.semantic_corres_infos['pts_2d'].clone()
        proj_K = self.semantic_corres_infos['proj_K']
        # weights = self.semantic_corres_infos['cos']
        rots = rot6d_to_matrix(self.rotations)
        pts_3d = torch.matmul(pts_3d.unsqueeze(0), rots) + self.translations
        pts_3d_proj = pts_3d[0] @ proj_K.T
        pts_3d_proj = pts_3d_proj[:, :2] / (pts_3d_proj[:, 2:] + 1e-8)
        pts_2d[:, 0] = 2 * pts_2d[:, 0] / REND_SIZE - 1
        pts_2d[:, 1] = 2 * pts_2d[:, 1] / REND_SIZE - 1
        pts_3d_proj[:, 0] = 2 * pts_3d_proj[:, 0] / REND_SIZE - 1
        pts_3d_proj[:, 1] = 2 * pts_3d_proj[:, 1] / REND_SIZE - 1
        # reproj_error = weights[:, None] * (pts_3d_proj - pts_2d)
        return F.l1_loss(pts_3d_proj, pts_2d, reduction='sum')

    def corres_forward(self):
        loss_dict = {}
        verts = self.apply_transformation()
        render_sil = self.sil_renderer(verts, self.faces, mode="silhouettes")
        render_mask = self.keep_mask * render_sil
        loss_dict["iou"] = (1 - batch_mask_iou(render_mask,
                                            self.ref_mask))
        with torch.no_grad():
            iou = batch_mask_iou(render_mask.detach(),
                                            self.ref_mask.detach())
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(verts)
        if self.semantic_corres_infos is not None:
            loss_dict["semantic_reproj"] = self.compute_semantic_reproj_loss() #? 关于loss weight的设置
        return loss_dict, iou

    def coarse_forward(self):
        loss_dict = {}
        verts = self.apply_transformation()
        render_sil = self.sil_renderer(
            verts, self.faces, mode="silhouettes")
        render_mask = self.keep_mask * render_sil
        loss_dict["iou"] = (1 - batch_mask_iou(render_mask,
                                            self.ref_mask))
        with torch.no_grad():
            iou = batch_mask_iou(render_mask.detach(),
                                            self.ref_mask.detach())
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(verts)
        return loss_dict, iou

    def forward(self):
        loss_dict = {}
        verts = self.apply_transformation()
        render_sil = self.sil_renderer(verts, self.faces, mode="silhouettes")
        render_mask = self.keep_mask * render_sil
        loss_dict["iou"] = (1 - batch_mask_iou(render_mask,
                                            self.ref_mask))
        verts_pytorch3d = verts.clone()
        # transform into pytorch3d coordinate
        verts_pytorch3d[:, :, :2] *= -1
        pytorch3d_meshes = Meshes(verts=verts_pytorch3d, faces=self.faces, 
                                  textures=self.textures.join_batch([self.textures]*(verts.shape[0]-1)).to("cuda"))
        fragments = self.rasterizer(meshes_world=pytorch3d_meshes, cameras=self.pytorch3d_cameras)
        render_img = self.shader(fragments, pytorch3d_meshes, cameras=self.pytorch3d_cameras)[:, :, :, :-1].permute(0, 3, 1, 2)
        render_img = F.interpolate(render_img, self.dino_model.smaller_edge_size, mode='bicubic', align_corners=True)
        rendered_dino_feat = self.dino_model.extract_features(render_img) # bs * 1024 * 768
        with torch.no_grad():
            # resize_rend_sil = F.interpolate(render_sil.detach().unsqueeze(1), self.dino_model.feat_size, 
            #                                 mode='nearest').squeeze(1).reshape(render_sil.shape[0], -1)
            resize_image_ref = F.interpolate(self.ref_mask.detach().unsqueeze(1), self.dino_model.feat_size, 
                                             mode='nearest').squeeze(1).reshape(render_sil.shape[0], -1)
            iou = batch_mask_iou(render_mask.detach(),
                                            self.ref_mask.detach())
            gt_dino_feat = self.gt_dino_feat.repeat(rendered_dino_feat.shape[0], 1, 1) # bs * (dino_feat_h*dino_feat_w) * dino_feat_dim
            # feat_mask = torch.logical_or(resize_rend_sil, resize_image_ref)
        
        loss_dict["sem"] = self.lw_sem * ((resize_image_ref * (1 - torch.sum(gt_dino_feat * rendered_dino_feat, dim=-1) 
                                         / (torch.norm(gt_dino_feat, dim=-1) * torch.norm(rendered_dino_feat, dim=-1) + 1e-6))) / (resize_image_ref.sum(-1, keepdim=True) + 1e-6)).sum(-1)
        loss_dict["offscreen"] = 100000 * self.compute_offscreen_loss(verts)
        return loss_dict, iou

def compute_prior_features(prior_infos, dino_model):
    render_view_infos = {}
    render_crop_imgs = []
    render_crop_masks = []
    render_crop_depths = []
    render_feats = []
    render_feats_masks = []
    K_render_roi = []
    for i, (prior_rendering, prior_depth, K_) in enumerate(zip(prior_infos['prior_batched_renderings'], prior_infos['prior_depths'], prior_infos['Ks'])):
        prior_rendering = prior_rendering.to("cuda")
        prior_depth = prior_depth.to("cuda")
        render_mask = (prior_rendering[:, :, -1] == 1)
        bit_masks = BitMasks(render_mask.unsqueeze(0))
        non_zero_indices = torch.nonzero(render_mask)
        min_row = max(torch.min(non_zero_indices[:, 0]) - 5., 0)
        max_row = min(torch.max(non_zero_indices[:, 0]) + 5., RENDER_H)
        min_col = max(torch.min(non_zero_indices[:, 1]) - 5., 0)
        max_col = min(torch.max(non_zero_indices[:, 1]) + 5., RENDER_W)
        box = torch.tensor([min_col, min_row, max_col, max_row]).float()
        bbox = bbox_xy_to_wh(box)  # xy_wh
        square_bbox = make_bbox_square(bbox, BBOX_EXPANSION_FACTOR)
        square_boxes = torch.FloatTensor(
                    np.tile(bbox_wh_to_xy(square_bbox),
                            (1, 1)))
        crop_mask = bit_masks.crop_and_resize(square_boxes.to(render_mask.device),
                                                    REND_SIZE).clone()[0]
        crop_image = crop_and_resize(prior_rendering[:, :, :3].permute(2, 0, 1).unsqueeze(0),
                                            square_boxes.to(render_mask.device), REND_SIZE).clone()[0].permute(1, 2, 0)
        crop_depth = crop_and_resize(prior_depth.permute(2, 0, 1).unsqueeze(0), 
                                    square_boxes.to(render_mask.device), REND_SIZE).clone()[0]
        crop_image[crop_mask==0] = torch.ones(3).to("cuda")
        x, y, b, _ = square_bbox
        camintr_roi = get_K_crop_resize(
            torch.Tensor(K_.cpu()).unsqueeze(0), torch.tensor([[x, y, x + b, y + b]]),
            [REND_SIZE]).cuda()
        with torch.no_grad():
            dino_render_feat = dino_model.extract_features(F.interpolate(crop_image.permute(2, 0, 1).unsqueeze(0).to('cuda'), 
                                                                    dino_model.smaller_edge_size, mode='bicubic', align_corners=True))
            dino_render_feat = F.normalize(dino_render_feat, dim=-1)
            dino_render_feat_mask = F.interpolate(crop_mask[None, None].float(), dino_model.smaller_edge_size//dino_model.model.patch_size, mode='nearest')
        render_crop_imgs.append(crop_image.unsqueeze(0))
        render_crop_masks.append(crop_mask.unsqueeze(0))
        render_crop_depths.append(crop_depth.unsqueeze(0))
        render_feats.append(dino_render_feat.cpu()) # save the memory
        render_feats_masks.append(dino_render_feat_mask.squeeze(1))
        K_render_roi.append(camintr_roi)

    render_view_infos.update({
        "render_imgs": prior_infos['prior_batched_renderings'], # NUM_VIEWS * RENDER_H * RENDER_W * 4
        "render_crop_imgs": torch.cat(render_crop_imgs), # NUM_VIEWS * REND_SIZE * REND_SIZE * 3
        "render_crop_masks": torch.cat(render_crop_masks),
        "render_crop_depths": torch.cat(render_crop_depths), # NUM_VIEWS * REND_SIZE * REND_SIZE
        "render_roi_Ks": torch.cat(K_render_roi),
        "render_feats": torch.cat(render_feats), # NUM_VIEWS * patch_size**2 * 768
        "render_feats_masks": torch.cat(render_feats_masks), # NUM_VIEWS * patch_size* patch_size
        "render_rotations": prior_infos['Rs'], # NUM_VIEWS * 3 * 3 o2c transformation object-coordinate to camera coordinate
        "render_translations": prior_infos['Ts'] , # NUM_VIEWS * 3
        })
    return render_view_infos

def find_optimal_pose(
    vertices,
    faces,
    textures,
    annotation,
    dino_model,
    K,
    render_view_infos,
    view_selection_dirs,
    view_idx,
    num_iterations=50,
    num_initializations=1,
    lr=1e-3,
    best_score=0,
    sort_best=True,
    rotations_init=None,
    translations_init=None,
    lw_mask=1.0,
    lw_sem=1.0,
    rasterizer=None,
    shader=None,
    mode="fine",
    use_former=True,
    former_max_idx=None,
):
    x, y, b, _ = annotation["square_bbox"]
    bbox = annotation["bbox"]
    crop_image = annotation["crop_image"]
    mask = annotation["target_crop_mask"]
    camintr_roi = get_K_crop_resize(
        torch.Tensor(K).unsqueeze(0), torch.tensor([[x, y, x + b, y + b]]),
        [REND_SIZE]).cuda()
    # Stuff to keep around
    best_losses = np.inf
    best_rots = None
    best_trans = None
    best_loss_single = np.inf
    loop = tqdm(total=num_iterations)
    K = torch.from_numpy(K).unsqueeze(0).to(vertices.device)
    ##### View Selection #####
    dino_feat_size = dino_model.feat_size
    crop_image = F.interpolate(torch.from_numpy(crop_image.astype(np.float32)).cuda().unsqueeze(0), 
                               dino_model.smaller_edge_size, mode='bicubic', align_corners=True)
    render_mask = render_view_infos['render_feats_masks']
    crop_mask = F.interpolate(torch.from_numpy((mask > 0).astype(np.float32)).cuda()[None, None, :, :], dino_feat_size, mode='nearest').to(render_mask.device)
    with torch.no_grad():
        gt_dino_feat = dino_model.extract_features(crop_image).to(render_view_infos["render_feats"].device)
        gt_dino_feat = F.normalize(gt_dino_feat, dim=-1)
    cos_mask = crop_mask.reshape(-1, dino_feat_size**2).to(render_view_infos["render_feats"].device)
    dino_cos = (cos_mask * torch.sum(gt_dino_feat * render_view_infos["render_feats"], dim=-1) 
                                         / (torch.norm(gt_dino_feat, dim=-1) * torch.norm(render_view_infos["render_feats"], dim=-1) + 1e-6)).sum(1) / cos_mask.sum(1)
    dino_cos = dino_cos.cuda()
    if not use_former or rotations_init is None:
        max_idx = torch.argmax(dino_cos)
    else: # 约束不要与上一帧的候选和上一帧优化后的角度相差太大
        rel_angle_full = rotation_angle_difference(rotations_init.clone(), render_view_infos["render_rotations"].transpose(1, 2).clone())
        if former_max_idx != -1:
            former_rel_angle_full = rotation_angle_difference(render_view_infos["render_rotations"][former_max_idx:former_max_idx+1].transpose(1, 2).clone(),
                                                              render_view_infos["render_rotations"].transpose(1, 2).clone())
            cos_topk_num = 5
        else:
            former_rel_angle_full = torch.zeros_like(rel_angle_full)
            cos_topk_num = 10
        _, indices = torch.topk(dino_cos, cos_topk_num, largest=True)
        rel_angle = rel_angle_full[indices]
        max_idx = indices[torch.argmin(rel_angle)].item() 
        if rel_angle_full[max_idx] > 85.0 or former_rel_angle_full[max_idx] > 85.0:
            max_idx = -1
    if max_idx != -1:
        filered = False
        best_rotation_init = render_view_infos["render_rotations"][max_idx:max_idx+1].transpose(1, 2).clone()
    else:
        filered = True
        best_rotation_init = rotations_init.clone()
        if torch.min(rel_angle_full) < 15.0: # 判断是否与上一帧差异过大，或语义不够相似
            max_idx = torch.argmin(rel_angle_full) 
            if (former_max_idx != -1 and former_rel_angle_full[max_idx].item() > 30.0) or dino_cos[max_idx] < (torch.max(dino_cos) - torch.std(dino_cos)):
                semantic_corres_infos = None
                max_idx = -1
        else:
            semantic_corres_infos = None
    best_translation_init = TCO_init_from_boxes_zup_autodepth(
            bbox, torch.matmul(vertices.unsqueeze(0), best_rotation_init),
            K).unsqueeze(1)
    if max_idx != -1: # save the view selction results
        if os.path.exists(view_selection_dirs):
            def render_idx(idx):
                res_img = np.hstack([render_view_infos["render_crop_imgs"][idx].detach().cpu().numpy(),
                                    annotation["crop_image"].transpose(1, 2, 0)])
                if not use_former or rotations_init is None:
                    cv2.imwrite(os.path.join(view_selection_dirs, 
                            '{}_{}_select_view{}.jpg'.format(view_idx, dino_cos[idx], idx)), res_img*255)
                else:
                    cv2.imwrite(os.path.join(view_selection_dirs, 
                            '{}_{}_{}_{}_{}_select_view{}.jpg'.format(view_idx, dino_cos[idx], rel_angle_full[idx], former_rel_angle_full[idx], filered, idx)), res_img*255)
            render_idx(max_idx)
        if mode == 'corres' and lw_sem == 0:
            max_idx_render_feat = render_view_infos["render_feats"][max_idx:max_idx+1]
            max_idx_render_mask = render_mask[max_idx:max_idx+1]
            max_idx_render_depth = render_view_infos["render_crop_depths"][max_idx:max_idx+1]
            max_idx_render_K = render_view_infos["render_roi_Ks"][max_idx]
            max_idx_render_R = render_view_infos["render_rotations"][max_idx]
            max_idx_render_T = render_view_infos["render_translations"][max_idx]
            ##### Find semantic correspondence ##### 
            image_render_cos = torch.bmm(gt_dino_feat, max_idx_render_feat.transpose(1, 2))[0] # (dino_feat_size ** 2) * (dino_feat_size ** 2)
            selected_points_image_1, selected_points_image_2 = find_semantic_correspondences(image_render_cos, crop_mask.reshape(-1), max_idx_render_mask.reshape(-1), 
                                                                                                gt_dino_feat[0], dino_feat_size, dino_feat_size)
            uv_idxs = torch.arange(dino_feat_size * dino_feat_size)
            uv_idxs_row = (uv_idxs // dino_feat_size) * dino_model.model.patch_size + dino_model.model.patch_size / 2
            uv_idxs_col =  (uv_idxs % dino_feat_size) * dino_model.model.patch_size + dino_model.model.patch_size / 2
            uvs = torch.cat([uv_idxs_col.reshape(-1, 1), uv_idxs_row.reshape(-1, 1)], dim=1)
            sample_uvs = uvs[selected_points_image_1].to(max_idx_render_depth.device)
            render_corres_uvs = uvs[selected_points_image_2].to(max_idx_render_depth.device)
            # 缩放回原来的尺度
            scale_h = dino_model.smaller_edge_size / REND_SIZE
            scale_w = dino_model.smaller_edge_size / REND_SIZE
            sample_uvs[:, 0], render_corres_uvs[:, 0] = sample_uvs[:, 0] / scale_w, render_corres_uvs[:, 0] / scale_w
            sample_uvs[:, 1], render_corres_uvs[:, 1] = sample_uvs[:, 1] / scale_h, render_corres_uvs[:, 1] / scale_h
            ### Establish 3D-2D correspondences ###
            #? 这里取depth直接取int是否会影响到效果？ 插值的depth是否会有点问题？ 先简单的使用点距离筛选掉不合理的三维点
            render_corres_depth = max_idx_render_depth[0, render_corres_uvs[:, 1].int(), render_corres_uvs[:, 0].int()]
            fx, fy = max_idx_render_K[0, 0], max_idx_render_K[1, 1]
            cx, cy = max_idx_render_K[0, 2], max_idx_render_K[1, 2]
            X = (render_corres_uvs[:, 0] - cx) * render_corres_depth / fx
            Y = (render_corres_uvs[:, 1] - cy) * render_corres_depth / fy
            Z = render_corres_depth
            xyz = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], dim=1)
            c2w_rot = max_idx_render_R.T
            c2w_trans = -max_idx_render_R.T @ max_idx_render_T
            xyz_w = xyz @ c2w_rot.T + c2w_trans.reshape(1, 3) # object template 3D points in the object coordinate
            dis_ = min_distance_to_point_set(xyz_w, vertices)
            dis_thres = torch.mean(dis_[render_corres_depth!=-1]) + torch.std(dis_[render_corres_depth!=-1])
            cos_ = image_render_cos[selected_points_image_1, selected_points_image_2].to(dis_.device) #? 考虑semantic correspondence相似性阈值的问题
            corres_mask = (dis_ < dis_thres)
            semantic_corres_infos = {"pts_2d": sample_uvs[corres_mask], "pts_3d": xyz_w[corres_mask], "proj_K": max_idx_render_K, "cos": cos_[corres_mask]}
            ###
            visualize_correspondences(annotation["crop_image"].transpose(1, 2, 0), render_view_infos["render_crop_imgs"][max_idx].cpu().numpy(), 
                                        sample_uvs[corres_mask].cpu().numpy(), render_corres_uvs[corres_mask].cpu().numpy(), 
                                        os.path.join(view_selection_dirs, '{}_2d_corres.jpg'.format(view_idx)))
    # Bring crop K to NC rendering space
    camintr_roi[:, :2] = camintr_roi[:, :2] / REND_SIZE

    model = ObjTracker(
        ref_image=mask,
        vertices=vertices,
        faces=faces,
        textures=textures,
        dino_model=dino_model,
        gt_dino_feat=gt_dino_feat,
        rotation_init=matrix_to_rot6d(best_rotation_init),
        translation_init=best_translation_init,
        num_initializations=1,
        K=camintr_roi,
        rasterizer=rasterizer,
        shader=shader,
        semantic_corres_infos=semantic_corres_infos,
        lw_mask=lw_mask,
        lw_sem=lw_sem,
    )
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(num_iterations):
        optimizer.zero_grad()
        if mode == "fine":
            loss_dict, iou = model()
        elif mode == 'corres':
            loss_dict, iou = model.corres_forward()
        else:
            loss_dict, iou = model.coarse_forward()
        losses = sum(loss_dict.values())
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        ind = torch.argmin(losses)
        best_loss_single = losses[ind]
        loop.set_description(f"loss: {best_loss_single.item():.3g}")
        loop.update()
    if best_rots is None:
        best_rots = model.rotations
        best_trans = model.translations
        best_losses = losses
    else:
        best_rots = torch.cat((best_rots, model.rotations), 0)
        best_trans = torch.cat((best_trans, model.translations), 0)
        best_losses = torch.cat((best_losses, losses))
    if sort_best:
        inds = torch.argsort(best_losses)
        best_losses = best_losses[inds][:num_initializations].detach().clone()
        best_trans = best_trans[inds][:num_initializations].detach().clone()
        best_rots = best_rots[inds][:num_initializations].detach().clone()
    loop.close()
    model.rotations = nn.Parameter(best_rots)
    model.translations = nn.Parameter(best_trans)
    model.best_score = best_score
    model.former_max_idx = max_idx
    return model

def find_optimal_poses(dino_model,
                       faces,
                       vertices,
                       textures,
                       annotations,
                       prior_infos,
                       Ks=None,
                       num_iterations=50,
                       num_initializations=1,
                       lr=1e-2,
                       lw_sem=1.0,
                       mode='fine',
                       view_selection_dirs=None,
                       use_former=True):
    '''
    Initialize the object poses for each frame indiviually.
    '''
    # Convert inputs to tensors
    vertices = torch.from_numpy(vertices).cuda()
    faces = torch.from_numpy(faces).cuda()
    # Check input shapes
    assert faces.ndim == 2 and faces.shape[-1] == 3, "Invalid shape for faces"
    assert vertices.ndim == 2 and vertices.shape[-1] == 3, "Invalid shape for vertices"
    # Keep track of previous rotations to get temporally consistent initialization
    previous_rotations = None
    previous_translations = None
    best_score = 0
    former_max_idx = None
    all_object_parameters = []
    raster_settings = RasterizationSettings(
        image_size=(REND_SIZE, REND_SIZE), 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )
    rasterizer=MeshRasterizer(
        raster_settings=raster_settings
    )
    shader=SoftPhongShader(
        device='cuda', 
    )
    render_view_infos = compute_prior_features(prior_infos, dino_model)
    for optim_idx, (annotation, K) in enumerate(zip(annotations, Ks)):
        print("Optim_idx:{}".format(optim_idx))
        # Optimize pose to given mask and semantic evidence
        model = find_optimal_pose(
            dino_model=dino_model,
            render_view_infos=render_view_infos,
            annotation=annotation,
            vertices=vertices,
            faces=faces,
            textures=textures,
            view_idx=optim_idx,
            K=K,
            best_score=best_score,
            num_iterations=num_iterations,
            num_initializations=num_initializations,
            sort_best=True,
            rotations_init=previous_rotations,
            translations_init=previous_translations,
            lw_sem=lw_sem,
            rasterizer=rasterizer,
            shader=shader,
            mode=mode,
            lr=lr,
            use_former=use_former,
            former_max_idx=former_max_idx,
            view_selection_dirs=view_selection_dirs,
        )
        verts_trans = model.apply_transformation()
        object_parameters = {
            "rotations": rot6d_to_matrix(model.rotations).detach()[0],
            "translations": model.translations.detach()[0],
            "K_roi": model.K.detach(),
            "verts_trans": verts_trans.detach()[0],
            "semantic_corres_infos": model.semantic_corres_infos # a dic
        }
        all_object_parameters.append(object_parameters)
        previous_rotations = rot6d_to_matrix(model.rotations.detach())
        previous_translations = None
        best_score = model.best_score
        former_max_idx = model.former_max_idx
    all_final_params = []
    # Aggregate object pose candidates for all frames
    for obj_params, info in zip(all_object_parameters, annotations):
        final_params = {}
        # Get best parameters
        for key in ["rotations", "translations", "verts_trans"]:
            final_params[key] = obj_params[key].unsqueeze(0).cuda()
        # Copy useful mask information
        for key in ["K_roi"]:
            final_params[key] = obj_params[key].unsqueeze(0).cuda()
        final_params["target_masks"] = torch.from_numpy(info["target_crop_mask"]).unsqueeze(0).cuda()
        final_params["verts"] = vertices.unsqueeze(0).cuda()
        final_params["semantic_corres_infos"] = obj_params["semantic_corres_infos"]
        all_final_params.append(final_params)
    return all_final_params
