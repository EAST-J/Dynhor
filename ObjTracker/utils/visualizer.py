import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import torch
import trimesh
import pyrender
import numpy as np

class Visualizer():
    def __init__(self, img_shape):
        self.img_shape = img_shape

    def draw_mesh(self, input_image, verts, faces, pred_camera, return_mask=False):
        # pred_camera (fx, fy, cx, cy)
        render = pyrender.OffscreenRenderer(viewport_width=self.img_shape[1],
                                    viewport_height=self.img_shape[0], point_size=1.0)
        if isinstance(verts, torch.Tensor):
            verts = verts.detach().cpu().numpy()
        # transform the coordinates
        verts[:, 1:] *= -1
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                           ambient_light=(0.3, 0.3, 0.3)
                           )
        pyrender_cam = pyrender.IntrinsicsCamera(pred_camera[0], pred_camera[1], pred_camera[2], pred_camera[3])
        scene.add(pyrender_cam, pose=np.eye(4))
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            smooth=True,
            wireframe=True,
            roughnessFactor=1.0,
            emissiveFactor=(0.1, 0.1, 0.1),
            baseColorFactor=(0.8, 0.46, 0.51, 1.0)
        )
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh, 'mesh')
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        scene.add(light, pose=light_pose)
        render_image, render_depth = render.render(scene, flags=pyrender.RenderFlags.RGBA)
        render_image = render_image / 255.
        valid_mask = (render_depth > 0)[:, :, np.newaxis]

        output_img = render_image * valid_mask + (1 - valid_mask) * input_image

        if return_mask:
            return output_img, valid_mask
        return output_img

    