import torch
import numpy as np

from utils.constants import REND_SIZE
import neural_renderer as nr

def batch_mask_iou(ref, pred, eps=0.000001):
    ref = ref.float()
    pred = pred.float()
    if ref.max() > 1 or ref.min() < 0:
        raise ValueError(
            "Ref mask should have values in [0, 1], "
            f"not [{ref.min(), ref.max()}]"
        )
    if pred.max() > 1 or pred.min() < 0:
        raise ValueError(
            "Ref mask should have values in [0, 1], "
            f"not [{pred.min(), pred.max()}]"
        )

    inter = ref * pred
    union = ref + pred - inter
    ious = inter.sum(1).sum(1).float() / (union.sum(1).sum(1).float() + eps)
    return ious

class Losses():
    def __init__(
        self,
        ref_mask_object,
        keep_mask_object,
        camintr_rois_object,
    ):
        self.ref_mask_object = ref_mask_object
        self.keep_mask_object = keep_mask_object
        self.camintr_rois_object = camintr_rois_object
        self.sil_renderer = nr.renderer.Renderer(image_size=REND_SIZE, 
                                                K=camintr_rois_object,
                                                R=torch.eye(3).unsqueeze(0).cuda(),
                                                t=torch.zeros(1, 3).cuda(),
                                                orig_size=1)

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
                                zeros).sum()  # Amount greater than 1
        upper_left = torch.max(-1 - coord_xy,
                               zeros).sum()  # Amount less than -1
        behind = torch.max(-coord_z, zeros).sum()
        too_far = torch.max(coord_z - self.sil_renderer.far, zeros).sum()
        return lower_right + upper_left + behind + too_far

    def compute_sil_loss(self, verts, faces):
        loss_sil = torch.Tensor([0.0]).float().cuda()
        rend = self.sil_renderer(verts, faces, mode="silhouettes")
        image = self.keep_mask_object * rend
        l_m = torch.sum(
            (image - self.ref_mask_object)**2) / self.keep_mask_object.sum()
        loss_sil += l_m
        ious = batch_mask_iou(image, self.ref_mask_object)
        return {
            "loss_sil_obj": loss_sil / len(verts), 
        }, {
            'iou_object': ious.mean().item()
        }
    
    def compute_smooth_loss(self, verts):
        smooth_loss_obj = ((verts[1:] - verts[:-1])**2).mean()
        return {
            "loss_smooth_obj": smooth_loss_obj,
        }
        
