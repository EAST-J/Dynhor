import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import flatten_eff_distloss

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

import models
from models.utils import cleanup
from models.ray_utils import get_rays
import utils.camera as camera
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy
import numpy as np
from systems.utils import parse_optimizer, parse_scheduler, update_module_step
from utils.camera import convert_opengl_to_opencv
# https://github.com/bennyguo/instant-nsr-pl/issues/83
# Textured mesh https://github.com/bennyguo/instant-nsr-pl/issues/60 


@systems.register('neus-barf-system')
class NeuSSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR()
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        self.sample_foreground_ratio = self.config.dataset.get('sample_foreground_ratio', 1.0)
        self.se3_refine = torch.nn.Embedding(self.config.model.n_frames, 6)
        # self.depth_factor = torch.nn.Embedding(self.config.model.n_frames, 2)
        torch.nn.init.zeros_(self.se3_refine.weight)
        # torch.nn.init.ones_(self.depth_factor.weight)
        self.automatic_optimization = False

    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        # pose_optim = parse_optimizer(self.config.system.pose_optimizer, [self.se3_refine, self.depth_factor])
        pose_optim = parse_optimizer(self.config.system.pose_optimizer, self.se3_refine)
        return (
            {
                'optimizer':optim,
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim)
            },
            {
                'optimizer':pose_optim,
                # 'lr_scheduler': parse_scheduler(self.config.system.pose_scheduler, pose_optim)
            }
        )

    def forward(self, batch):
        return self.model(batch['rays'])
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch: # validation / testing
            index = batch['index'].cpu()
        else:
            if self.config.model.batch_image_sampling:
                if self.sample_foreground_ratio < 1: # More sampling in the foreground part of the image
                    fg_ray_index = torch.randint(0, len(self.dataset.all_fg_indexs), size=(int(self.train_num_rays * self.sample_foreground_ratio),), device=self.dataset.all_fg_indexs.device)
                    bg_ray_index = torch.randint(0, len(self.dataset.all_bg_indexs), size=(self.train_num_rays - int(self.train_num_rays * self.sample_foreground_ratio),), device=self.dataset.all_bg_indexs.device)
                    
                    fg_ray_index = self.dataset.all_fg_indexs[fg_ray_index]
                    bg_ray_index = self.dataset.all_bg_indexs[bg_ray_index]
                    ray_index = torch.cat([fg_ray_index, bg_ray_index], dim=0)
                    index, y, x = ray_index[:, 0], ray_index[:, 1], ray_index[:, 2]
                else:
                    index = torch.randint(0, len(self.dataset.all_images), size=(self.train_num_rays,), device=self.dataset.all_images.device)
                    x = torch.randint(
                        0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
                    )
                    y = torch.randint(
                        0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
                    )
            else:
                index = torch.randint(0, len(self.dataset.all_images), size=(1,))
                if self.sample_foreground_ratio < 1:
                    fg_ray_index = torch.randint(0, self.dataset.fg_indexs_dic[index.item()].shape[0], size=(int(self.train_num_rays * self.sample_foreground_ratio),), device=self.dataset.all_fg_indexs.device)
                    bg_ray_index = torch.randint(0, self.dataset.bg_indexs_dic[index.item()].shape[0], size=(self.train_num_rays - int(self.train_num_rays * self.sample_foreground_ratio),), device=self.dataset.all_bg_indexs.device)
                    fg_ray_index = self.dataset.fg_indexs_dic[index.item()][fg_ray_index]
                    bg_ray_index = self.dataset.bg_indexs_dic[index.item()][bg_ray_index]
                    ray_index = torch.cat([fg_ray_index, bg_ray_index], dim=0)
                    y, x = ray_index[:, 0], ray_index[:, 1]
                else:
                    x = torch.randint(
                            0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images.device
                        )
                    y = torch.randint(
                        0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images.device
                    )
        all_c2w = self.dataset.all_c2w.clone()
        if stage in ['train', 'validation']:
            c2w_refine = camera.lie.se3_to_SE3(self.se3_refine.weight)
            all_c2w = camera.pose.compose([c2w_refine, all_c2w])
        if stage in ['train']:
            c2w = all_c2w[index]
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions[y, x]
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index, y, x]
            rays_o, rays_d = get_rays(directions, c2w)
            # _, rays_d_tmp = get_rays(directions, torch.eye(4).to(c2w.device)[None].repeat(directions.shape[0], 1, 1)[:, :3])
            # depth_scale = F.normalize(rays_d_tmp, p=2, dim=-1)[:, 2:]
            rgb = self.dataset.all_images[index, y, x].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index, y, x].view(-1).to(self.rank)
            normal = self.dataset.all_normals[index, y, x].view(-1, self.dataset.all_normals.shape[-1]).to(self.rank)
            # depth = self.dataset.all_depths[index, y, x].view(-1).to(self.rank)
            seg = self.dataset.all_segs[index, y, x].view(-1, self.dataset.all_segs.shape[-1]).to(self.rank)
            corres_index = (self.dataset.select_win - 1) + torch.randint(1, self.dataset.select_win, (self.train_num_rays,), device=index.device) * (torch.randint(0, 2, (self.train_num_rays,), device=index.device) * 2 - 1)
            corres_info = self.dataset.all_correspondences[index, corres_index, y, x].to(self.rank)
            corres_c2w = all_c2w[(corres_index-self.dataset.select_win+1+index).clamp(0, self.dataset.all_images.shape[0]-1)]
        else:
            c2w = all_c2w[index][0]
            corres_index = -1
            corres_c2w = all_c2w[0][0] 
            if self.dataset.directions.ndim == 3: # (H, W, 3)
                directions = self.dataset.directions
            elif self.dataset.directions.ndim == 4: # (N, H, W, 3)
                directions = self.dataset.directions[index][0] 
            rays_o, rays_d = get_rays(directions, c2w)
            # _, rays_d_tmp = get_rays(directions, torch.eye(4).to(c2w.device)[None].repeat(directions.shape[0], 1, 1)[:, :3])
            # depth_scale = F.normalize(rays_d_tmp, p=2, dim=-1)[:, 2:]
            rgb = self.dataset.all_images[index].view(-1, self.dataset.all_images.shape[-1]).to(self.rank)
            fg_mask = self.dataset.all_fg_masks[index].view(-1).to(self.rank)
            seg = self.dataset.all_segs[index].view(-1, self.dataset.all_segs.shape[-1]).to(self.rank)
            normal = torch.zeros_like(rgb)
            # depth = torch.zeros_like(rgb[..., 0])
            corres_info = torch.zeros_like(rgb)

        rays = torch.cat([rays_o, F.normalize(rays_d, p=2, dim=-1)], dim=-1)

        if stage in ['train']:
            if self.config.model.background_color == 'white':
                self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
            elif self.config.model.background_color == 'random':
                self.model.background_color = torch.rand((3,), dtype=torch.float32, device=self.rank)
            else:
                raise NotImplementedError
        else:
            self.model.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        
        if self.dataset.apply_mask:
            rgb = rgb * fg_mask[...,None] + self.model.background_color * (1 - fg_mask[...,None])
        
        batch.update({
            'rays': rays,
            'rgb': rgb,
            'fg_mask': fg_mask,
            'normal': normal,
            # 'depth': depth,
            # 'depth_scale': depth_scale,
            'seg': seg,
            'corres_info': corres_info,
            'c2w': c2w,
            'corres_c2w': corres_c2w,
            'index': index,
            'corres_index': (corres_index-self.dataset.select_win+1+index).clamp(0, self.dataset.all_images.shape[0]-1)
        })      
    
    def training_step(self, batch, batch_idx):

        opt, pose_opt = self.optimizers()
        pose_opt._on_before_step = lambda : self.trainer.profiler.start("optimizer_step")
        pose_opt._on_after_step = lambda : self.trainer.profiler.stop("optimizer_step")
        # sch, pose_sch = self.lr_schedulers()
        sch = self.lr_schedulers()
        out = self(batch)

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            if out['num_samples_full'].sum().item() != 0:
                train_num_rays = int(self.train_num_rays * (self.train_num_samples / out['num_samples_full'].sum().item()))        
                self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
            else:
                self.train_num_rays = self.config.model.max_train_num_rays

        if self.config.model.dynamic_sampleing_ratio: # 根据global_step动态变化sample rate 考虑初始化需要设置的小一些吗？
            self.sample_foreground_ratio = 0.5 + (self.config.dataset['sample_foreground_ratio'] - 0.5) * self.global_step / self.config.trainer.max_steps

        # ray_mask = torch.logical_and(out['rays_valid_full'][...,0], batch['fg_mask']) # ? just use the fg_mask?
        ray_mask = batch['fg_mask']
        mask_sum = ray_mask.sum() + 1e-5

        color_error = (out['comp_rgb_full'] - batch['rgb']) * ray_mask[:, None]
        loss_rgb_mse = F.mse_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        # loss_rgb_mse = F.mse_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb_mse', loss_rgb_mse)
        loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        loss_rgb_l1 = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
        # loss_rgb_l1 = F.l1_loss(out['comp_rgb_full'][out['rays_valid_full'][...,0]], batch['rgb'][out['rays_valid_full'][...,0]])
        self.log('train/loss_rgb', loss_rgb_l1)
        loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)        

        loss_eikonal = ((torch.linalg.norm(out['sdf_grad_samples'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', loss_eikonal)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)
        
        opacity = torch.clamp(out['opacity'].squeeze(-1), 1.e-3, 1.-1.e-3)
        weights = torch.ones_like(opacity)
        weights[batch['seg'][:, -1]==1] = 0
        loss_mask = binary_cross_entropy(opacity, batch['fg_mask'].float(), weights=weights)
        self.log('train/loss_mask', loss_mask)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        loss_opaque = binary_cross_entropy(opacity, opacity)
        self.log('train/loss_opaque', loss_opaque)
        loss += loss_opaque * self.C(self.config.system.loss.lambda_opaque)

        loss_sparsity = torch.exp(-self.config.system.loss.sparsity_scale * out['sdf_samples'].abs()).mean()
        self.log('train/loss_sparsity', loss_sparsity)
        loss += loss_sparsity * self.C(self.config.system.loss.lambda_sparsity)
        # depth loss
        # gt_depth = batch['depth'][:, None]
        # render_depth = torch.abs(batch['depth_scale']) * out['depth']
        # render_depth = self.depth_factor.weight[batch['index'], 0].view(-1, 1) * render_depth + self.depth_factor.weight[batch['index'], 1].view(-1, 1)
        # depth_error = (render_depth - gt_depth) * ray_mask[:, None]
        # loss_depth = F.l1_loss(depth_error, torch.zeros_like(depth_error), reduction='sum') / mask_sum
        # self.log('train/loss_depth', loss_depth)
        # loss += loss_depth * self.C(self.config.system.loss.lambda_depth)
        # normal loss
        gt_normal = F.normalize(batch['normal'], dim=1)
        render_normal = F.normalize(out['comp_normal'], dim=1)
        w2c = torch.inverse(batch['c2w'][:, :3, :3])
        render_normal = torch.matmul(w2c, render_normal[:, :, None]).squeeze(-1)
        cos = torch.sum(render_normal * gt_normal, dim=1, keepdim=True)
        normal_error = (1 - cos) * ray_mask[:, None]
        loss_normal = F.l1_loss(normal_error, torch.zeros_like(normal_error), reduction='sum') / mask_sum
        self.log('train/loss_normal', loss_normal)
        loss += loss_normal * self.C(self.config.system.loss.lambda_normal)
        # correspondence loss
        corres_w2c = camera.pose.invert(batch['corres_c2w'])
        corres_w2c_opencv = convert_opengl_to_opencv(corres_w2c)
        proj_pts = torch.matmul(corres_w2c_opencv[:, :3, :3], out['surface_pts'][:, :, None]).squeeze(-1) + corres_w2c_opencv[:, :3, 3]
        proj_pts = torch.matmul(self.dataset.K.unsqueeze(0).repeat(batch['corres_c2w'].shape[0], 1, 1), proj_pts[:, :, None]).squeeze(-1)
        corres_mask = torch.logical_and((out['opacity'] > 0.9).detach().squeeze(), batch['corres_info'][:, -1] > 0.0) # TODO: the threhold for opacity?
        proj_pts_ = torch.zeros((proj_pts.shape[0], 2), device=proj_pts.device, dtype=proj_pts.dtype)
        proj_pts_[corres_mask] = proj_pts[corres_mask, :2] / (proj_pts[corres_mask, 2:] + 1e-5)
        proj_pts_[:, 0] = 2 * proj_pts_[:, 0] / (self.dataset.w - 1) - 1.0
        proj_pts_[:, 1] = 2 * proj_pts_[:, 1] / (self.dataset.h - 1) - 1.0    
        reproj_error = (proj_pts_ - batch['corres_info'][:, :-1]) * corres_mask.unsqueeze(-1) * batch['corres_info'][:, -1:]
        loss_correspondence = F.l1_loss(reproj_error, torch.zeros_like(reproj_error), reduction='sum') / (corres_mask.sum() + 1e-5)
        # loss_correspondence = F.huber_loss(reproj_error, torch.zeros_like(reproj_error), reduction='sum') / (corres_mask.sum() + 1e-5) # 设计待定？感觉差别不大
        self.log('train/loss_correspondence', loss_correspondence)
        if self.global_step > self.config.system.loss.correspondence_start_iters: 
            loss += loss_correspondence * self.C(self.config.system.loss.lambda_correspondence)

        if self.C(self.config.system.loss.lambda_curvature) > 0:
            assert 'sdf_laplace_samples' in out, "Need geometry.grad_type='finite_difference' to get SDF Laplace samples"
            loss_curvature = out['sdf_laplace_samples'].abs().mean()
            self.log('train/loss_curvature', loss_curvature)
            loss += loss_curvature * self.C(self.config.system.loss.lambda_curvature)
        if self.config.model.name == 'color-neus': # regularize the relight color
            delta_rgb = out['delta_rgb'] * ray_mask[out['ray_indices'], None]
            loss_relight = F.mse_loss(torch.mean(delta_rgb.abs()), torch.tensor(0, device=delta_rgb.device, dtype=delta_rgb.dtype))
            self.log('train/loss_relight', loss_relight)
            loss += loss_relight * self.C(self.config.system.loss.lambda_relight)
        # distortion loss proposed in MipNeRF360
        # an efficient implementation from https://github.com/sunset1995/torch_efficient_distloss
        if self.C(self.config.system.loss.lambda_distortion) > 0:
            loss_distortion = flatten_eff_distloss(out['weights'], out['points'], out['intervals'], out['ray_indices'])
            self.log('train/loss_distortion', loss_distortion)
            loss += loss_distortion * self.C(self.config.system.loss.lambda_distortion)    

        if self.config.model.learned_background and self.C(self.config.system.loss.lambda_distortion_bg) > 0:
            loss_distortion_bg = flatten_eff_distloss(out['weights_bg'], out['points_bg'], out['intervals_bg'], out['ray_indices_bg'])
            self.log('train/loss_distortion_bg', loss_distortion_bg)
            loss += loss_distortion_bg * self.C(self.config.system.loss.lambda_distortion_bg)        

        losses_model_reg = self.model.regularizations(out)
        for name, value in losses_model_reg.items():
            self.log(f'train/loss_{name}', value)
            loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
            loss += loss_
        
        self.log('train/inv_s', out['inv_s'], prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        opt.zero_grad()
        pose_opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        pose_opt.step()
        sch.step()
        # pose_sch.step()
        # return {
        #     'loss': loss
        # }
    
    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        W, H = self.dataset.img_wh
        # Back-project the surface points to the image
        # w2c = camera.pose.invert(batch['c2w']).cpu()
        # from utils.camera import convert_opengl_to_opencv
        # w2c_opencv = convert_opengl_to_opencv(w2c.unsqueeze(0))[0]
        # pts = out['surface_pts'][torch.logical_and(batch['fg_mask'].cpu(), (out['opacity'] > 0.9).squeeze())].reshape(-1, 3)
        # proj_pts = torch.matmul(w2c_opencv[None, :3, :3], pts[:, :, None]).squeeze(-1) + w2c_opencv[None, :3, 3]
        # proj_pts = torch.matmul(self.dataset.K[None, :3, :3].cpu(), proj_pts[:, :, None]).squeeze(-1)
        # uv = proj_pts[:, :2] / (proj_pts[:, 2:] + 1e-5)
        # tmp_proj_mask = np.zeros((H, W))
        # tmp_proj_mask[uv[:, 1].long(), uv[:, 0].long()] = 1
        
        # corres_w2c = camera.pose.invert(batch['corres_c2w']).cpu()
        # corres_w2c_opencv = convert_opengl_to_opencv(corres_w2c.unsqueeze(0))[0]
        # pts = out['surface_pts'][torch.logical_and(batch['fg_mask'].cpu(), (out['opacity'] > 0.9).squeeze())].reshape(-1, 3)
        # proj_pts = torch.matmul(corres_w2c_opencv[None, :3, :3], pts[:, :, None]).squeeze(-1) + corres_w2c_opencv[None, :3, 3]
        # proj_pts = torch.matmul(self.dataset.K[None, :3, :3].cpu(), proj_pts[:, :, None]).squeeze(-1)
        # uv = proj_pts[:, :2] / (proj_pts[:, 2:] + 1e-5)
        # im0 = self.dataset.all_images[batch['index']][0]
        # uv_in_region = torch.logical_and(torch.logical_and(uv[:, 0] > 0, uv[:, 0] < (W - 1)), 
        #                                  torch.logical_and(uv[:, 1] > 0, uv[:, 1] < (H - 1)))
        # uv_mask = torch.logical_and(batch['fg_mask'].cpu(), (out['opacity'] > 0.9).squeeze()).reshape((H, W))
        # im1_transfer_rgb = np.zeros((H, W, 3))
        # i, j = np.meshgrid(
        #     np.arange(W, dtype=np.float32),
        #     np.arange(H, dtype=np.float32),
        #     indexing='xy'
        # )
        # origin_uv = np.concatenate([i[:, :, np.newaxis], j[:, :, np.newaxis]], axis=-1)
        # im1_transfer_rgb[uv[uv_in_region, 1].long(), uv[uv_in_region, 0].long()] = im0[torch.from_numpy(origin_uv[uv_mask][uv_in_region, 1]).long(), torch.from_numpy(origin_uv[uv_mask][uv_in_region, 0]).long()]
        # im1_transfer_rgb = (im1_transfer_rgb * 256).clip(0, 255)

        psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        normal = out['comp_normal'] # in the world coordinate
        w2c = camera.pose.invert(batch['c2w']).cpu()
        normal_trans = normal @ w2c[:3, :3].T
        normal_trans = normal_trans.reshape(H, W, 3)
        normal_trans[:, :, 0] *= -1
        render_mask = (normal_trans.sum(dim=-1) == 0)
        normal_alpha = torch.cat([normal_trans, ~render_mask[:, :, None]], dim=-1)
        normal_trans = render_mask[:, :, None] * (2 * batch['rgb'].view(H, W, 3).cpu() -1 ) + ~render_mask[:, :, None] * normal_trans
        self.save_image_grid(f"it{self.global_step}-{batch['index'][0].item()}.png", [
            {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        ] + ([
            {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        ] if self.config.model.learned_background else []) + [
            {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
            {'type': 'rgb', 'img': normal_trans, 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            {'type': 'rgb', 'img': normal_alpha, 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        ])
        # self.export()
        return {
            'psnr': psnr,
            'index': batch['index']
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)
            for idx, o in enumerate(out_set.values()):
                self.log('{}'.format(idx), o['psnr'].item())

    def test_step(self, batch, batch_idx):
        # out = self(batch)
        # psnr = self.criterions['psnr'](out['comp_rgb_full'].to(batch['rgb']), batch['rgb'])
        # W, H = self.dataset.img_wh
        # self.save_image_grid(f"it{self.global_step}-test/{batch['index'][0].item()}.png", [
        #     {'type': 'rgb', 'img': batch['rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     {'type': 'rgb', 'img': out['comp_rgb_full'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}
        # ] + ([
        #     {'type': 'rgb', 'img': out['comp_rgb_bg'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        #     {'type': 'rgb', 'img': out['comp_rgb'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
        # ] if self.config.model.learned_background else []) + [
        #     {'type': 'grayscale', 'img': out['depth'].view(H, W), 'kwargs': {}},
        #     {'type': 'rgb', 'img': out['comp_normal'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}}
        # ])
        # self.export()
        # return {
        #     'psnr': psnr,
        #     'index': batch['index']
        # }
        pass
    
    def test_epoch_end(self, out):
        """
        Synchronize devices.
        Generate image sequence using test outputs.
        """
        # out = self.all_gather(out)
        # if self.trainer.is_global_zero:
        #     out_set = {}
        #     for step_out in out:
        #         # DP
        #         if step_out['index'].ndim == 1:
        #             out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
        #         # DDP
        #         else:
        #             for oi, index in enumerate(step_out['index']):
        #                 out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
        #     psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
        #     self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True)    

        #     self.save_img_sequence(
        #         f"it{self.global_step}-test",
        #         f"it{self.global_step}-test",
        #         '(\d+)\.png',
        #         save_format='mp4',
        #         fps=30
        #     )
        if self.trainer.is_global_zero:
            self.export()
    
    def export(self):
        mesh = self.model.export(self.config.export)
        self.save_mesh(
            f"it{self.global_step}-{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}.obj",
            **mesh, post_process=self.config.export.get('post_process', False)
        )        
