import os
import cv2
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F

from glob import glob
import datasets
import pickle as pkl
from models.ray_utils import get_ray_directions
from utils.misc import get_rank
import torchvision.transforms.functional as TF

import trimesh
import math
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, IterableDataset
from tqdm import tqdm

import re
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_files(directory, max_value, min_value):
    # 正则表达式用于匹配文件名，如1234_5678.npz
    pattern = re.compile(r"(\d{4})_(\d{4})\.npz")
    
    # 初始化一个空列表来存储符合条件的文件内容
    file_list = []
    
    # 遍历指定文件夹中的所有文件
    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if match:
            A, B = map(int, match.groups())
            if A <= max_value and B <= max_value and A >= min_value and B>=min_value:
                file_path = os.path.join(directory, filename)
                file_list.append(file_path)
    
    return file_list

def process_correspondence_infos(info_name, idx_map, img0, idx_thres=None):
    info = np.load(info_name)
    im_idx0, im_idx1 = int(info_name.split("/")[-1].split("_")[0]), int(info_name.split("/")[-1][:-4].split("_")[1])

    if (im_idx0 not in idx_map or im_idx1 not in idx_map):
        return None
    elif idx_thres is not None and abs(im_idx1 - im_idx0) >= idx_thres:
        return None

    lis_idx0 = idx_map[im_idx0]
    lis_idx1 = idx_map[im_idx1]

    corres_0_to_1 = {
        "corres_idx": lis_idx1,
        "corres": torch.from_numpy(info["warp"][:, :img0.shape[1], 2:]),
        "certainty": torch.from_numpy(info["certainty"][:, :img0.shape[1]])
    }
    
    corres_1_to_0 = {
        "corres_idx": lis_idx0,
        "corres": torch.from_numpy(info["warp"][:, img0.shape[1]:, :2]),
        "certainty": torch.from_numpy(info["certainty"][:, img0.shape[1]:])
    }

    return (lis_idx0, corres_0_to_1), (lis_idx1, corres_1_to_0)

def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0.,0.,0.], dtype=cameras.dtype, device=cameras.device)
    mean_d = (cameras - center[None,:]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:,2].mean()
    r = (mean_d**2 - mean_h**2).sqrt()
    up = torch.as_tensor([0., 0., 1.], dtype=center.dtype, device=center.device)

    all_c2w = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, -l], dim=1), cam_pos[:,None]], axis=1)
        all_c2w.append(c2w)

    all_c2w = torch.stack(all_c2w, dim=0)
    
    return all_c2w

class HOIDatasetBase():
    initialized = False
    properties = {}
    
    def setup(self, config, split):
        self.config = config
        self.split = split
        self.rank = get_rank()
        self.apply_mask = False
        self.has_mask = True

        seq_name = self.config.seq_name
        if seq_name is None:
            seq_name = ""
        root_dir = self.config.root_dir
        obj_path = self.config.obj_path
        meta_dir = self.config.meta_dir

        if not HOIDatasetBase.initialized:
            img_lis = sorted(glob(os.path.join(root_dir, seq_name, 'rgb/*.jpg')))[self.config.start_idx:self.config.end_idx:self.config.step]
            H, W = cv2.imread(img_lis[0]).shape[:2]
            seg_lis = [im_name.replace('rgb', 'sam_seg')[:-4] + '.png' for im_name in img_lis]
            normal_lis = [im_name.replace('rgb', 'monocular_normal')[:-4] + '.png' for im_name in img_lis]

            meta_lis = [os.path.join(meta_dir, "obj_infos/{:04d}.npz".format(int(im_name.split("/")[-1][:-4]))) for im_name in img_lis]
            correspondnce_lis = load_files(os.path.dirname(img_lis[0]).replace('rgb', 'correspondence_infos'), 
                                           min_value=int(img_lis[0].split('/')[-1][:-4]), 
                                           max_value=int(img_lis[-1].split('/')[-1][:-4]))
            K = np.load(meta_lis[0])['K']
            fx, cx = K[0, 0], K[0, 2]
            fy, cy = K[1, 1], K[1, 2]
            obj_model = trimesh.load(obj_path)
            obj_verts = np.array(obj_model.vertices).astype(np.float32)
            if hasattr(self.config, 'normalize_mesh') and self.config.normalize_mesh:
                obj_verts = obj_verts - obj_verts.mean(0)
                obj_verts_can = obj_verts / np.linalg.norm(obj_verts, 2, 1).max() * 1. / 2
            else:
                obj_verts_can = obj_verts   
            if 'img_downscale' in self.config:
                w, h = W // self.config.img_downscale, H // self.config.img_downscale
                factor = w / W
                fx, cx, fy, cy = fx / factor, cx / factor, fy / factor, cy / factor
            else:
                w, h = W, H
                factor = 1
            self.w, self.h = w, h
            img_wh = (w, h)
            all_c2w, all_images, all_fg_masks, all_segs, all_normals = [], [], [], [], []
            all_fg_indexs, all_bg_indexs = [], []
            fg_indexs_dic, bg_indexs_dic = {}, {}
            directions = get_ray_directions(w, h, fx, fy, cx, cy).to(self.rank)

            for i, (img_path, seg_path, meta_path, normal_path) in enumerate(zip(img_lis, seg_lis, meta_lis, normal_lis)):
                meta_data = np.load(meta_path)
                obj_rot = meta_data['R']
                obj_trans = meta_data['T']
                obj_pose = np.concatenate([obj_rot, obj_trans.reshape(3, 1)], 1)
                obj_pose[1:] *= -1 # ! instant-nsr-pl uses the opengl coordinates
                c2o_pose = np.linalg.inv(np.concatenate([obj_pose, np.array([[0, 0, 0, 1]])]))
                c2o_pose = torch.from_numpy(c2o_pose[:3]).float()
                all_c2w.append(c2o_pose)
                img = Image.open(img_path)
                img = img.resize(img_wh, Image.BICUBIC)
                img = TF.to_tensor(img).permute(1, 2, 0)[...,:3]
                img = img.to(self.rank) if self.config.load_data_on_gpu else img.cpu()
                seg = Image.open(seg_path)
                seg = seg.resize(img_wh, Image.BICUBIC)
                seg = TF.to_tensor(seg).permute(1, 2, 0)[...,:3] # w * h * 3
                
                obj_mask = (seg[:, :, 1]==1)
                hand_mask = (seg[:, :, -1]==1)
                background_mask = torch.logical_not(torch.logical_or(obj_mask, hand_mask))

                fg_index = torch.stack(torch.nonzero(obj_mask.bool(), as_tuple=True), dim=0)
                bg_index = torch.stack(torch.nonzero(background_mask.bool(), as_tuple=True), dim=0)
                fg_index = torch.cat([torch.full((1, fg_index.shape[1]), i), fg_index], dim=0)
                bg_index = torch.cat([torch.full((1, bg_index.shape[1]), i), bg_index], dim=0)
                all_fg_indexs.append(fg_index.permute(1, 0))
                all_bg_indexs.append(bg_index.permute(1, 0))
                fg_indexs_dic[i] = fg_index.permute(1, 0)[:, 1:]
                bg_indexs_dic[i] = bg_index.permute(1, 0)[:, 1:]

                normal = Image.open(normal_path)
                normal = normal.resize(img_wh, Image.BICUBIC)
                normal = TF.to_tensor(normal).permute(1, 2, 0)[...,:3] * 2 - 1.
                normal[:, :, 0] *= -1

                all_images.append(img)
                all_fg_masks.append(obj_mask)
                all_segs.append(seg)
                all_normals.append(normal)

            all_c2w = torch.stack(all_c2w, dim=0)
            origin = obj_verts_can.mean(0).reshape(1, 3)
            scale = np.sqrt(np.sum((all_c2w[:, :3, 3].numpy() - origin)**2, axis=1)).max() * 1.1
            # scale = np.percentile(np.sqrt(np.sum((obj_verts_can - origin)**2, axis=1)), 99.9) * 4.0
            # Normalize the poses
            all_c2w[:, :3, 3] = (all_c2w[:, :3, 3] - origin) / scale
            temp_v = (obj_verts_can - origin) / scale
            # Load correspondence infos
            idx_range = list(range(int(img_lis[0].split('/')[-1][:-4]), int(img_lis[-1].split('/')[-1][:-4])+1, self.config.step))
            self.select_win = self.config.select_win
            idx_thres = self.config.step * self.select_win
            idx_map = {v: i for i, v in enumerate(idx_range)}
            correspondence_infos = torch.zeros((len(img_lis), (self.select_win-1)*2+1, all_images[0].shape[0], all_images[0].shape[1], 3)) # correspondence infos tensors
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(process_correspondence_infos, info_name, idx_map, all_images[0], idx_thres) for info_name in correspondnce_lis]
                for future in tqdm(as_completed(futures), total=len(futures), desc='reading correspondence infos'):
                    result = future.result()
                    if result is None:
                        continue
                    # import ipdb
                    # ipdb.set_trace()
                    (lis_idx0, corres_0_to_1), (lis_idx1, corres_1_to_0) = result
                    correspondence_infos[lis_idx0, self.select_win-1+corres_0_to_1['corres_idx']-lis_idx0] = torch.cat([corres_0_to_1['corres'], corres_0_to_1['certainty'].unsqueeze(-1)], dim=-1)
                    correspondence_infos[lis_idx1, self.select_win-1+corres_1_to_0['corres_idx']-lis_idx1] = torch.cat([corres_1_to_0['corres'], corres_1_to_0['certainty'].unsqueeze(-1)], dim=-1)
                    

            HOIDatasetBase.properties = {
                    'w': w,
                    'h': h,
                    'K': torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                    'img_wh': img_wh,
                    'factor': factor,
                    'directions': directions,
                    'all_c2w': all_c2w,
                    'all_segs': all_segs,
                    'all_images': all_images,
                    'all_fg_masks': all_fg_masks,
                    'all_normals': all_normals,
                    'all_fg_indexs': all_fg_indexs,
                    'all_bg_indexs': all_bg_indexs,
                    'fg_indexs_dic': fg_indexs_dic,
                    'bg_indexs_dic': bg_indexs_dic,
                    'all_correspondences': correspondence_infos,
                    'origin': origin,
                    'radius': scale,
                    'temp_v': torch.from_numpy(temp_v),
                    'select_win': self.select_win
                }

            HOIDatasetBase.initialized = True
    
        for k, v in HOIDatasetBase.properties.items():
            setattr(self, k, v)

        if self.split == 'test':
            self.all_c2w = create_spheric_poses(self.all_c2w[:,:,3], n_steps=self.config.n_test_traj_steps)
            self.all_images = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_segs = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_masks = torch.zeros((self.config.n_test_traj_steps, self.h, self.w), dtype=torch.float32)
            self.all_normals = torch.zeros((self.config.n_test_traj_steps, self.h, self.w, 3), dtype=torch.float32)
            self.all_fg_indexs, self.all_bg_indexs = torch.tensor([]), torch.tensor([])
            self.fg_indexs_dic, self.bg_indexs_dic = {}, {}
        else:
            self.all_images, self.all_fg_masks = torch.stack(self.all_images, dim=0).float(), torch.stack(self.all_fg_masks, dim=0).float()
            self.all_segs = torch.stack(self.all_segs, dim=0).float()
            self.all_normals = torch.stack(self.all_normals, dim=0).float()
            self.all_fg_indexs = torch.cat(self.all_fg_indexs, dim=0)
            self.all_bg_indexs = torch.cat(self.all_bg_indexs, dim=0)

        self.all_c2w = self.all_c2w.float().to(self.rank)
        self.K = self.K.float().to(self.rank)
        if self.config.load_data_on_gpu:
            self.all_images = self.all_images.to(self.rank) 
            self.all_fg_masks = self.all_fg_masks.to(self.rank)
            self.all_normals = self.all_normals.to(self.rank)
            self.all_segs = self.all_segs.to(self.rank)
            self.all_correspondences = self.all_correspondences.to(self.rank)
            self.all_bg_indexs = self.all_bg_indexs.to(self.rank)
            self.all_fg_indexs = self.all_fg_indexs.to(self.rank)
            self.temp_v = self.temp_v.to(self.rank)
            for k, v in self.fg_indexs_dic.items():
                self.fg_indexs_dic[k] = v.to(self.rank)
            for k, v in self.bg_indexs_dic.items():
                self.bg_indexs_dic[k] = v.to(self.rank)

class HOIDataset(Dataset, HOIDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class HOIIterableDataset(IterableDataset, HOIDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}


@datasets.register('hoi_dkm')
class HOIDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = HOIIterableDataset(self.config, 'train')
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = HOIDataset(self.config, self.config.get('val_split', 'train'))
        if stage in [None, 'test']:
            self.test_dataset = HOIDataset(self.config, self.config.get('test_split', 'test'))
        if stage in [None, 'predict']:
            self.predict_dataset = HOIDataset(self.config, 'train')         

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        sampler = None
        return DataLoader(
            dataset, 
            num_workers=os.cpu_count(), 
            batch_size=batch_size,
            pin_memory=True,
            sampler=sampler
        )
    
    def train_dataloader(self):
        return self.general_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1) 