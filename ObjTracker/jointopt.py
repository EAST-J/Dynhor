"""
Joint optimiztion for all frames with smooth.
"""
from collections import defaultdict
import torch
from tqdm.auto import tqdm
from torch import nn
from utils.losses import Losses
from utils.camera import (
    compute_transformation_persp,
)
from utils.geometry import matrix_to_rot6d, rot6d_to_matrix
from utils.camera import tensorify

class Joint_Optimizer(nn.Module):
    def __init__(
        self,
        translations_object,
        rotations_object,
        verts_object_og,
        faces_object,
        camintr_rois_object,
        target_masks_object,
        int_scale_init=1.0,
        optimize_object_scale=False,
    ):
        super().__init__()
        # Initialize object pamaters
        translation_init = translations_object.detach().clone()
        self.translations_object = nn.Parameter(translation_init,
                                                requires_grad=True)
        rotations_object = rotations_object.detach().clone()
        if rotations_object.shape[-1] == 3:
            rotations_object6d = matrix_to_rot6d(rotations_object)
        else:
            rotations_object6d = rotations_object
        self.rotations_object = nn.Parameter(
            rotations_object6d.detach().clone(), requires_grad=True)
        self.register_buffer("verts_object_og", verts_object_og)
        # Inititalize person parameters
        init_scales = int_scale_init * torch.ones(1).float()
        self.optimize_object_scale = optimize_object_scale
        if optimize_object_scale:
            self.int_scales_object = nn.Parameter(
                init_scales,
                requires_grad=True,
            )
        else:
            self.register_buffer("int_scales_object", init_scales)
        self.register_buffer("int_scale_object_mean", torch.ones(1).float())
        self.register_buffer("ref_mask_object",
                             (target_masks_object > 0).float())
        self.register_buffer("keep_mask_object",
                             (target_masks_object >= 0).float())
        self.register_buffer("camintr_rois_object", camintr_rois_object)
        self.register_buffer("faces_object", faces_object)
        self.cuda()

        self.losses = Losses(
            ref_mask_object=self.ref_mask_object,
            keep_mask_object=self.keep_mask_object,
            camintr_rois_object=self.camintr_rois_object,
        )

    def get_verts_object(self):
        rotations_object = rot6d_to_matrix(self.rotations_object)
        obj_verts = compute_transformation_persp(
            meshes=self.verts_object_og,
            translations=self.translations_object,
            rotations=rotations_object,
            intrinsic_scales=self.int_scales_object.abs(),
        )
        return obj_verts

    def forward(self, loss_weights=None):
        """
        If a loss weight is zero, that loss isn't computed.
        """
        loss_dict = {}
        metric_dict = {}
        verts_object = self.get_verts_object()
        if loss_weights is None or (loss_weights["lw_smooth_obj"] > 0):
            loss_smooth = self.losses.compute_smooth_loss(
                verts_object
            )
            loss_dict.update(loss_smooth)
        if loss_weights is None or loss_weights["lw_sil_obj"] > 0:
            sil_loss_dict, sil_metric_dict = self.losses.compute_sil_loss(
                verts=verts_object, faces=self.faces_object)
            loss_dict.update(sil_loss_dict)
            metric_dict.update(sil_metric_dict)
        return loss_dict, metric_dict

def joint_optimize(
    object_parameters,
    objvertices=None,
    objfaces=None,
    loss_weights=None,
    num_iterations=400,
    lr=1e-4,
    board=None,
    optimize_object_scale=False,
):
    # Load mesh data.
    verts_object_og = tensorify(objvertices).cuda()
    faces_object = tensorify(objfaces).cuda()

    obj_trans = torch.cat([obj["translations"] for obj in object_parameters])
    obj_rots = torch.cat([obj["rotations"] for obj in object_parameters])

    obj_tar_masks = torch.cat(
        [obj["target_masks"] for obj in object_parameters])
    obj_camintr_roi = torch.cat(
        [obj["K_roi"][:, 0] for obj in object_parameters])
    model = Joint_Optimizer(
        translations_object=obj_trans,  # [B, 1, 3]
        rotations_object=obj_rots,  # [B, 3, 3]
        verts_object_og=verts_object_og,  # [B, VN, 3]
        faces_object=faces_object,  # [B, FN, 3]
        # Used for silhouette supervision
        target_masks_object=obj_tar_masks,  # [B, REND_SIZE, REND_SIZE]
        camintr_rois_object=obj_camintr_roi,
        int_scale_init=1,
        optimize_object_scale=optimize_object_scale,
    )
    rigid_parameters = [
        val for key, val in model.named_parameters()
        if "rotation" not in key
    ]

    rotation_parameters = [
        # Do not try to refactor an iter !
        val for key, val in model.named_parameters()
        if ("rotation" in key)
    ]
    optimizer = torch.optim.Adam([{
        "params": rigid_parameters,
        "lr": lr
    }, {
        "params": rotation_parameters,
        "lr": lr * 10
    }])
    loop = tqdm(range(num_iterations))
    loss_evolution = defaultdict(list)
    for step in loop:
        optimizer.zero_grad()
        loss_dict, metric_dict = model(loss_weights=loss_weights)
        loss_dict_weighted = {
            k: loss_dict[k] * loss_weights[k.replace("loss", "lw")]
            for k in loss_dict
        }
        for k, val in loss_dict.items():
            loss_evolution[k].append(val.item())
            board.add_scalar(k, val.item(), step)
        for k, val in metric_dict.items():
            loss_evolution[k].append(val)
        loss = sum(loss_dict_weighted.values())
        loss_evolution["loss"].append(loss.item())
        loop.set_description(f"Loss {loss.item():.4f}")
        loss.backward()
        optimizer.step()
    return model, dict(loss_evolution)
