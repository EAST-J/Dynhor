# Simple pose refine just use mask cues, similar to HOLD
import os
import sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
from glob import glob
import pymeshlab as ml
import trimesh
from tensorboardX import SummaryWriter
# Some func
from run_corres import load_data, process_input
from jointopt import joint_optimize
from utils.camera import get_K_crop_resize
from utils.constants import REND_SIZE
from utils.geometry import rot6d_to_matrix
from tqdm import tqdm
import yaml
import argparse
import time

def remesh_and_clean_mesh(input_path, out_path, target_face_count=5000):
    # Some mesh processing code adapted from HOLD
    ms = ml.MeshSet()
    ms.load_new_mesh(input_path)

    # Attempt to fix non-manifold edges if they exist
    ms.apply_filter("meshing_repair_non_manifold_edges")

    # 1. Remesh for target face count
    ms.apply_filter(
        "meshing_decimation_quadric_edge_collapse",
        targetfacenum=target_face_count,
        qualitythr=1.0,
        preserveboundary=True,
        preservenormal=True,
    )

    # 2. Collapse near vertices
    ms.meshing_merge_close_vertices()

    # 3. Remove identical vertices and faces
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_duplicate_faces()

    ms.save_current_mesh(out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    seq_name = config['seq_name']
    exp_name = 'refine'
    preprocess_mesh = True
    image_paths = sorted(glob(os.path.join(config['data_info']['dataroot'], 'rgb', '*.jpg')))
    obj_path = '../instant-nsr-pl/exp/sdf-hoi-{}/{}-coarse/save/it5000-mc512.obj'.format(seq_name.lower(), seq_name.lower())
    c2w = np.loadtxt(os.path.join(os.path.dirname(obj_path), 'poses.txt')).reshape(-1, 3, 4)
    w2c = np.linalg.inv(np.concatenate([c2w, 
                                        np.array([0, 0, 0, 1]).reshape(-1, 1, 4).repeat(c2w.shape[0], 0)], axis=1))
    if preprocess_mesh:
        remesh_and_clean_mesh(obj_path, obj_path[:-4]+'-clean.obj')
        obj_path = obj_path[:-4]+'-clean.obj'
    obj_mesh = trimesh.load(obj_path)
    obj_verts = np.array(obj_mesh.vertices).astype(np.float32)
    obj_faces = obj_mesh.faces

    images_np, hand_masks_np, obj_masks_np = load_data(image_paths)
    annotations = process_input(images_np, obj_masks_np, hand_masks_np)
    # TODO: Load the camera from the pred infos
    height, width, _ = images_np[0].shape
    image_size = max(height, width)
    focal = 1.2*min(height, width)
    camintr = np.array([[focal, 0, width // 2], [0, focal, height // 2], [0, 0, 1]]).astype(np.float32)

    refine_num_iterations = 1000 # Increase to give more steps to converge
    refine_loss_weights = {
            "lw_sil_obj": 1.0,
            "lw_smooth_obj": 10.0,
            "lw_sem_correspondence": 0.0,
        }

    sample_folder = os.path.join("exps/", seq_name, exp_name)
    os.makedirs(sample_folder, exist_ok=True)
    board = SummaryWriter(os.path.join(sample_folder, "board"))

    '''
    object_parameters: List
    Keys:
    rotations: torch.Size([1, 3, 3])
    translations: torch.Size([1, 1, 3])
    verts_trans: 1 * N_v * 3
    K_roi: 1, 1, 3, 3
    target_masks: 1 * REND_SIZE * REND_SIZE
    semantic_corres_infos: None for this case
    '''
    object_parameters = []
    for idx in range(c2w.shape[0]):
        rotations = torch.from_numpy(w2c[idx, :3, :3].T).unsqueeze(0).cuda().float()
        translations = torch.from_numpy(w2c[idx, :3, -1]).reshape(1, 1, 3).cuda().float()
        verts_trans = torch.from_numpy(obj_verts @ w2c[idx, :3, :3].T + w2c[idx, :3, -1].T).cuda().float()
        target_masks = torch.from_numpy(annotations[idx]["target_crop_mask"]).unsqueeze(0).cuda()
        x, y, b, _ = annotations[idx]['square_bbox']
        camintr_roi = get_K_crop_resize(
            torch.Tensor(camintr).unsqueeze(0), torch.tensor([[x, y, x + b, y + b]]),
            [REND_SIZE]).cuda()
        camintr_roi[:, :2] = camintr_roi[:, :2] / REND_SIZE
        object_parameter = {
            'rotations': rotations,
            'translations': translations,
            'verts_trans': verts_trans,
            'target_masks': target_masks,
            'K_roi': camintr_roi.reshape(1, 1, 3, 3),
            'semantic_corres_infos': None
        }
        object_parameters.append(object_parameter)

    start_time = time.time()
    model, loss_evolution = joint_optimize(
        object_parameters=object_parameters,
        objvertices=obj_verts,
        objfaces=np.stack([obj_faces for _ in range(len(images_np))]),
        optimize_object_scale=False,
        loss_weights=refine_loss_weights,
        num_iterations=refine_num_iterations,
        lr=config['system']['joint_lr'],
        board=board,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Joint-optimization time: {elapsed_time:.6f} seconds")
    '''
    Parameter Lists:
    verts_object_og: N * 3
    int_scales_object: tensor([])
    rotations_object: B * 3 * 2
    translations_object: B * 1 * 3
    faces_object: B * F * 3

    for object, R: rot6d_to_matrix(model.rotations_object).T T:model.translations_object
    K: focal, 0, width//2
        0, focal, height//2
        0, 0, 1
    '''
    obj_rot = rot6d_to_matrix(model.rotations_object).transpose(1, 2) # obj_coordinate to camera coordinate
    obj_trans = model.translations_object

    obj_rot_np = obj_rot.detach().cpu().numpy()
    obj_trans_np = obj_trans.detach().cpu().numpy()

    K = np.array([[focal, 0, width//2],
                [0, focal, height//2],
                [0, 0, 1]])
    os.makedirs(os.path.join(sample_folder, "obj_infos"), exist_ok=True)
    save_poses = []
    for i in tqdm(range(len(image_paths))):
        data = {
            "R": obj_rot_np[i],
            "T": obj_trans_np[i],
            "K": K,
        }
        path_id = image_paths[i].split("/")[-1][:-4]
        np.savez(os.path.join(sample_folder, "obj_infos/{}.npz".format(path_id)), **data)
        R = obj_rot_np[i]
        T =obj_trans_np[i]
        T = - R.T @ T.reshape(3, 1)
        R = R.T
        Rt = np.concatenate([R, T.reshape(3, 1)], axis=1).reshape(1, -1)
        save_poses.append(Rt)
    save_poses = np.concatenate(save_poses)
    os.makedirs(os.path.join(sample_folder, 'poses'), exist_ok=True)
    np.savetxt(os.path.join(sample_folder, 'poses/pose.txt'), save_poses.reshape(-1, 12))