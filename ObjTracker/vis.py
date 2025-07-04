import numpy as np
from utils.visualizer import Visualizer
import cv2
from PIL import Image
import trimesh
import os
from glob import glob
from tqdm import tqdm
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, required=True)
args = parser.parse_args()
with open(args.config_path, 'r') as f:
    config = yaml.safe_load(f)
image_paths = sorted(glob(os.path.join(config['data_info']['dataroot'], 'rgb', '*.jpg')))
sample_folder = './exps/{}/{}'.format(config['seq_name'], config['exp_name'])
assert os.path.exists(sample_folder), "Please run the pose optimizer first"
print(len(image_paths))

obj_path = config['data_info']['obj_path']
obj_scale = 1
obj_mesh = trimesh.load(obj_path, force="mesh")
obj_verts = np.array(obj_mesh.vertices).astype(np.float32)

# Center and scale vertices
obj_verts = obj_verts - obj_verts.mean(0)
obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * obj_scale / 2
# obj_verts_can = obj_verts
obj_faces = np.array(obj_mesh.faces)
# Convert images to numpy 
images = [Image.open(image_path) for image_path in image_paths]
images_np = [np.array(image) for image in images]

height, width, _ = images_np[0].shape
focal = 1.2*min(height, width)
vis = Visualizer((height, width))
output_folder = os.path.join(sample_folder, "render_res")
os.makedirs(output_folder, exist_ok=True)
for i in tqdm(range(len(image_paths))):
    id = image_paths[i].split("/")[-1][:-4]
    obj_info_path = os.path.join(sample_folder, "obj_infos/{}.npz").format(id)
    if os.path.exists(obj_info_path):
        obj_info = np.load(obj_info_path)
        R = obj_info["R"]
        T = obj_info["T"]
        if "obj_scale" in obj_info.keys():
            obj_scale = obj_info["obj_scale"]
        else:
            obj_scale = 1.0
        obj_v_trans = (obj_scale * obj_verts_can) @ R.T + T
        res_img = vis.draw_mesh(images_np[i] / 255., obj_v_trans, obj_mesh.faces, (focal, focal, width // 2, height // 2))
        # res_img = np.hstack([res_img[:, :, ::-1] * 255, images_np[i][:, :, ::-1]])
        cv2.imwrite(os.path.join(output_folder, '{}.jpg'.format(id)), res_img[:, :, ::-1] * 255)