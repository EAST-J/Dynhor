import os
import torch
import numpy as np
from glob import glob
import yaml
import argparse
import shutil

from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from detectron2.structures import BitMasks
from tensorboardX import SummaryWriter

from utils.bbox import  make_bbox_square, bbox_xy_to_wh, bbox_wh_to_xy, crop_and_resize
from utils.maskutils import add_occlusions
from utils.geometry import rot6d_to_matrix
from utils.render import batch_render, batch_random_render
from utils.constants import REND_SIZE, RENDER_H, RENDER_W
from dino import Dinov2

from pose_initializtion import find_optimal_poses
from jointopt import joint_optimize


def process_input(images, obj_masks, hand_masks):
    # process the input
    objs = []
    for i, (obj_mask, hand_mask) in enumerate(zip(obj_masks, hand_masks)):
        obj_mask = (obj_mask == 255)
        hand_mask = (hand_mask == 255)
        hand_occlusions = torch.from_numpy(hand_mask).unsqueeze(0)
        bit_masks = BitMasks(torch.from_numpy(obj_mask).unsqueeze(0))
        obj = {}
        non_zero_indices = np.nonzero(obj_mask)
        # Get tight bounding box
        min_row = max(np.min(non_zero_indices[0]) - 5., 0)
        max_row = min(np.max(non_zero_indices[0]) + 5., obj_mask.shape[0])
        min_col = max(np.min(non_zero_indices[1]) - 5., 0)
        max_col = min(np.max(non_zero_indices[1]) + 5., obj_mask.shape[1])
        box = torch.tensor([min_col, min_row, max_col, max_row]).float()
        bbox = bbox_xy_to_wh(box)  # xy_wh
        square_bbox = make_bbox_square(bbox, 0.3)
        square_boxes = torch.FloatTensor(
                np.tile(bbox_wh_to_xy(square_bbox),
                        (1, 1)))
        crop_masks = bit_masks.crop_and_resize(square_boxes,
                                                   REND_SIZE)[0]
        images_crop = crop_and_resize(torch.from_numpy((images[i]/255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(square_boxes.device), 
                                          square_boxes, REND_SIZE)[0].permute(1, 2, 0)
        images_crop[crop_masks==0] = torch.ones(3)
        obj.update({
                "bbox":
                bbox, # xywh
                "class_id":
                -1,
                "score":
                None,
                "square_bbox":
                square_bbox,  # xy_wh 与bbox的中心相同，将bbox给resize为square
                "crop_mask":
                crop_masks.cpu().numpy(),
                "crop_image":
                images_crop.cpu().permute(2, 0, 1).numpy()
        })
        # 1 for object part, 0 for background, -1 for hand part.
        target_masks = add_occlusions([obj["crop_mask"]],
                                            hand_occlusions,
                                            [obj["square_bbox"]])[0]
        obj.update({"target_crop_mask": target_masks})
        objs.append(obj)
    return objs

def load_data(image_paths):
    images = [(Image.open(image_path)) for image_path in image_paths]
    images_np = [np.array(image) for image in images]
    masks = [(Image.open(image_path.replace("rgb", "sam_seg")[:-4] + ".png")) for image_path in image_paths]
    masks_np = [np.array(mask) for mask in masks]
    hand_masks_np = []
    obj_masks_np = []
    for mask in masks_np:
        hand_mask = np.zeros((mask.shape[0], mask.shape[1]))
        obj_mask = np.zeros((mask.shape[0], mask.shape[1]))
        hand_mask[mask[:, :, -1] == 255] = 255
        obj_mask[mask[:, :, 1] == 255] = 255
        hand_masks_np.append(hand_mask)
        obj_masks_np.append(obj_mask)
    return images_np, hand_masks_np, obj_masks_np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    seq_name = config['seq_name']
    exp_name = config['exp_name']
    # Load images
    image_paths = sorted(glob(os.path.join(config['data_info']['dataroot'], 'rgb', '*.jpg')))
    print('Load {} images'.format(len(image_paths)))
    images_np, hand_masks_np, obj_masks_np = load_data(image_paths)
    annotations = process_input(images_np, obj_masks_np, hand_masks_np)
    # Init Dinov2
    dino_model = Dinov2()
    # Load 3D prior
    obj_path = config['data_info']['obj_path']
    obj_mesh = load_objs_as_meshes([obj_path])
    obj_verts = np.array(obj_mesh.verts_packed()).astype(np.float32)
    # Center and scale vertices
    if config['data_info']['normalize_mesh']:
        obj_verts = obj_verts - obj_verts.mean(0)
        obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * 1. / 2.
    else:
        obj_verts_can = obj_verts
    obj_faces = np.array(obj_mesh.faces_packed())
    obj_textures = obj_mesh.textures
    # Define camera parameters
    height, width, _ = images_np[0].shape
    image_size = max(height, width)
    focal = 1.2 * min(height, width)
    camintr = np.array([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]]).astype(np.float32)
    camintrs = [camintr for _ in range(len(images_np))]

    sample_folder = os.path.join("exps", seq_name, exp_name)
    os.makedirs(sample_folder, exist_ok=True)
    board = SummaryWriter(os.path.join(sample_folder, "board"))
    shutil.copy(args.config_path, os.path.join(sample_folder, "config.yaml"))

    if config['random_render']:
        prior_batched_renderings, prior_depths, Rs, Ts, Ks = batch_random_render(device='cuda', mesh=obj_mesh.to('cuda'), 
                                                                        view_numbers=6000, 
                                                                        H=RENDER_H, W=RENDER_W, distance_scale=3.5)
    else:
        prior_batched_renderings, prior_depths, Rs, Ts, Ks = batch_render(device='cuda', mesh=obj_mesh.to('cuda'), 
                                                                                num_azimuth=30, num_elevation=10, num_roll=13,
                                                                                H=RENDER_H, W=RENDER_W)
    prior_infos = {'prior_batched_renderings': prior_batched_renderings, 'prior_depths': prior_depths, 'Rs': Rs, 'Ts': Ts, 'Ks': Ks}
    # Step1: Initialize per-frame object poses
    object_parameters = find_optimal_poses(
        dino_model=dino_model,
        prior_infos=prior_infos,
        vertices=obj_verts_can,
        faces=obj_faces,
        textures=obj_textures,
        annotations=annotations,
        num_iterations=config['system']['init_num_iterations'],
        Ks=np.stack(camintrs),
        use_former=True,
        lr=config['system']['init_lr'], # ?adjust the learning rates
    )
    num_iterations = config['system']['joint_num_iterations']
    loss_weights = config['system']['loss']
    # Step2: Joint optimize for the temporal smoothness
    model, loss_evolution = joint_optimize(
        object_parameters=object_parameters,
        objvertices=obj_verts_can,
        objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
        optimize_object_scale=False,
        loss_weights=loss_weights,
        num_iterations=num_iterations,
        lr=config['system']['joint_lr'],
        board=board,
    )
    # Save the poses
    obj_rot = rot6d_to_matrix(model.rotations_object).transpose(1, 2) # obj_coordinate to camera coordinate
    obj_trans = model.translations_object

    obj_rot_np = obj_rot.detach().cpu().numpy()
    obj_trans_np = obj_trans.detach().cpu().numpy()
    os.makedirs(os.path.join(sample_folder, "obj_infos"), exist_ok=True)
    for i in range(len(image_paths)):
        data = {
            "R": obj_rot_np[i],
            "T": obj_trans_np[i],
            "K": camintr,
        }
        path_id = image_paths[i].split("/")[-1][:-4]
        np.savez(os.path.join(sample_folder, "obj_infos/{}.npz".format(path_id)), **data)
