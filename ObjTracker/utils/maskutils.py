import os
import torch

from detectron2.structures import BitMasks
from utils.bbox import bbox_wh_to_xy, make_bbox_square, bbox_xy_to_wh
from utils.constants import REND_SIZE

def add_occlusions(masks, occluder_mask, mask_bboxes):
    """
    Args:
        masks (list[np.ndarray]): list of object masks in [REND_SIZE, REND_SIZE] [(REND_SIZE, REND_SIZE), ...]
        mask_bboxes (list[np.ndarray]): matching list of square xy_wh bboxes [(4,), ...]
        occluder_mask (torch.Tensor): [B, IMAGE_SIZE, IMAGE_SIZE] occluder where B
            dim aggregates different one-hot encodings of occluder
            masks
    """
    occluded_masks = []
    for mask, mask_bbox in zip(masks, mask_bboxes):
        bbox_mask = bbox_wh_to_xy(torch.Tensor(mask_bbox).unsqueeze(0)).to(
            occluder_mask.device)
        occlusions = BitMasks(occluder_mask).crop_and_resize(
            bbox_mask.repeat(occluder_mask.shape[0], 1), REND_SIZE)
        # Remove occlusions
        with_occlusions = occluder_mask.new(mask).float()
        with_occlusions[occlusions.sum(0) > 0] = -1

        # Draw back original object mask in case it was removed by occlusions
        with_occlusions[mask] = 1
        occluded_masks.append(with_occlusions.numpy())
    return occluded_masks