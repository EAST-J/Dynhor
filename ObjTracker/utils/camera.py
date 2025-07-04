import numpy as np
import torch

def rotation_angle_difference(R1, R2):
    R_rel = R1 @ R2.transpose(1, 2)
    cos_theta = torch.clamp(0.5 * (torch.vmap(torch.trace)(R_rel) - 1), -1.0, 1.0)
    rel_theta = (180.0 / torch.pi) * (torch.acos(cos_theta))

    return rel_theta

def tensorify(array, device=None):
    if not isinstance(array, torch.Tensor):
        array = torch.tensor(array)
    if device is not None:
        array = array.to(device)
    return array

def batch_proj2d(verts, camintr, camextr=None):
    # Project 3d vertices on image plane
    if camextr is not None:
        verts = camextr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    verts_hom2d = camintr.bmm(verts.transpose(1, 2)).transpose(1, 2)
    proj_verts2d = verts_hom2d[:, :, :2] / verts_hom2d[:, :, 2:]
    return proj_verts2d

def projection(vertices, K, R, t, orig_size, dist_coeffs=torch.cuda.FloatTensor([[0., 0., 0., 0., 0.]]), eps=1e-9):
    '''
    Calculate projective transformation of vertices given a projection matrix
    Input parameters:
    K: batch_size * 3 * 3 intrinsic camera matrix
    R, t: batch_size * 3 * 3, batch_size * 1 * 3 extrinsic calibration parameters
    dist_coeffs: vector of distortion coefficients
    orig_size: original size of image captured by the camera
    Returns: For each point [X,Y,Z] in world coordinates [u,v,z] where u,v are the coordinates of the projection in
    pixels and z is the depth
    '''

    # instead of P*x we compute x'*P'
    vertices = torch.matmul(vertices, R.transpose(2,1)) + t
    x, y, z = vertices[:, :, 0], vertices[:, :, 1], vertices[:, :, 2]
    x_ = x / (z + eps)
    y_ = y / (z + eps)

    # Get distortion coefficients from vector
    k1 = dist_coeffs[:, None, 0]
    k2 = dist_coeffs[:, None, 1]
    p1 = dist_coeffs[:, None, 2]
    p2 = dist_coeffs[:, None, 3]
    k3 = dist_coeffs[:, None, 4]

    # we use x_ for x' and x__ for x'' etc.
    r = torch.sqrt(x_ ** 2 + y_ ** 2)
    x__ = x_*(1 + k1*(r**2) + k2*(r**4) + k3*(r**6)) + 2*p1*x_*y_ + p2*(r**2 + 2*x_**2)
    y__ = y_*(1 + k1*(r**2) + k2*(r**4) + k3 *(r**6)) + p1*(r**2 + 2*y_**2) + 2*p2*x_*y_
    vertices = torch.stack([x__, y__, torch.ones_like(z)], dim=-1)
    vertices = torch.matmul(vertices, K.transpose(1,2))
    u, v = vertices[:, :, 0], vertices[:, :, 1]
    v = orig_size - v
    # map u,v from [0, img_size] to [-1, 1] to use by the renderer
    u = 2 * (u - orig_size / 2.) / orig_size
    v = 2 * (v - orig_size / 2.) / orig_size
    vertices = torch.stack([u, v, z], dim=-1)
    return vertices

def compute_K_roi(upper_left, b, img_size, focal_length=1.0):
    """
    Computes the intrinsics matrix for a cropped ROI box.

    Args:
        upper_left (tuple): Top left corner (x, ytorch.Tensor(gt_verts).unsqueeze(0)).
        b (float): Square box size.
        img_size (int): Size of image in pixels.

    Returns:
        Intrinsic matrix (1 x 3 x 3).
    """
    x1, y1 = upper_left
    f = focal_length * img_size / b
    px = (img_size / 2 - x1) / b
    py = (img_size / 2 - y1) / b
    K = torch.cuda.FloatTensor([[[f, 0, px], [0, f, py], [0, 0, 1]]])
    return K

def get_K_crop_resize(K, boxes, crop_resize, invert_xy=False):
    """
    Adapted from https://github.com/BerkeleyAutomation/perception/
        blob/master/perception/camera_intrinsics.py
    Skew is not handled !
    """
    assert K.shape[1:] == (3, 3)
    assert boxes.shape[1:] == (4,)
    K = K.float()
    boxes = boxes.float()
    if invert_xy:
        boxes = torch.stack(
            [boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], 1
        )
    new_K = K.clone()

    crop_resize = torch.tensor(crop_resize, dtype=torch.float)

    final_width, final_height = max(crop_resize), min(crop_resize)
    crop_width = boxes[:, 2] - boxes[:, 0]
    crop_height = boxes[:, 3] - boxes[:, 1]
    crop_cj = (boxes[:, 0] + boxes[:, 2]) / 2
    crop_ci = (boxes[:, 1] + boxes[:, 3]) / 2

    # Crop
    cx = K[:, 0, 2] + (crop_width - 1) / 2 - crop_cj
    cy = K[:, 1, 2] + (crop_height - 1) / 2 - crop_ci

    # # Resize (upsample)
    center_x = (crop_width - 1) / 2
    center_y = (crop_height - 1) / 2
    orig_cx_diff = cx - center_x
    orig_cy_diff = cy - center_y
    scale_x = final_width / crop_width
    scale_y = final_height / crop_height
    scaled_center_x = (final_width - 1) / 2
    scaled_center_y = (final_height - 1) / 2
    fx = scale_x * K[:, 0, 0]
    fy = scale_y * K[:, 1, 1]
    cx = scaled_center_x + scale_x * orig_cx_diff
    cy = scaled_center_y + scale_y * orig_cy_diff

    new_K[:, 0, 0] = fx
    new_K[:, 1, 1] = fy
    new_K[:, 0, 2] = cx
    new_K[:, 1, 2] = cy
    return new_K

def TCO_init_from_boxes_zup_autodepth(boxes_2d, model_points_3d, K):
    # User in BOP20 challenge
    model_points_3d = tensorify(model_points_3d)
    bsz = model_points_3d.shape[0]
    device = model_points_3d.device
    K = tensorify(K).to(device)
    boxes_2d = tensorify(boxes_2d).to(device)
    if boxes_2d.dim() == 1:
        boxes_2d = boxes_2d.unsqueeze(0)
    if boxes_2d.shape[0] != bsz:
        boxes_2d = boxes_2d.repeat(bsz, 1)
    if K.dim() == 2:
        K = K.unsqueeze(0)
    if K.shape[0] != bsz:
        K = K.repeat(bsz, 1, 1)

    assert boxes_2d.shape[-1] == 4
    assert boxes_2d.dim() == 2
    # xywh to xyxy
    boxes_2d = torch.stack([
        boxes_2d[:, 0], boxes_2d[:, 1], boxes_2d[:, 0] + boxes_2d[:, 2],
        boxes_2d[:, 1] + boxes_2d[:, 3]
    ], 1)
    # Get length of reference bbox diagonal
    diag_bb = (boxes_2d[:, [2, 3]] - boxes_2d[:, [0, 1]]).norm(2, -1)
    # Get center of reference bbox
    bb_xy_centers = (boxes_2d[:, [0, 1]] + boxes_2d[:, [2, 3]]) / 2
    fxfy = K[:, [0, 1], [0, 1]]
    cxcy = K[:, [0, 1], [2, 2]]
    z = fxfy.new_ones(bsz, 1)
    xy_init = ((bb_xy_centers - cxcy) * z) / fxfy
    trans = torch.cat([xy_init, z], 1)
    for _ in range(10):
        C_pts_3d = model_points_3d + trans.unsqueeze(1)
        proj_pts = batch_proj2d(C_pts_3d, K)
        diag_proj = (proj_pts.min(1)[0] - proj_pts.max(1)[0]).norm(2, -1)
        proj_xy_centers = (proj_pts.min(1)[0] + proj_pts.max(1)[0]) / 2

        # Update z to increase/decrease size of projected bbox
        delta_z = z * (diag_proj / diag_bb - 1).unsqueeze(-1)
        z = z + delta_z
        # Update xy to shift center of projected bbox
        xy_init += ((bb_xy_centers - proj_xy_centers) * z) / fxfy
        trans = torch.cat([xy_init, z], 1)
    return trans


def compute_transformation_persp(meshes,
                                 translations,
                                 rotations=None,
                                 intrinsic_scales=None):
    """
    Computes the 3D transformation.

    Args:
        meshes (V x 3 or B x V x 3): Vertices.
        translations (B x 1 x 3).
        rotations (B x 3 x 3).
        intrinsic_scales (B).

    Returns:
        vertices (B x V x 3).
    """
    B = translations.shape[0]
    device = meshes.device
    if meshes.ndimension() == 2:
        meshes = meshes.repeat(B, 1, 1)
    if rotations is None:
        rotations = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        rotations = rotations.to(device)
    if intrinsic_scales is None:
        intrinsic_scales = torch.ones(B).to(device)
    meshes_scaled = intrinsic_scales.view(-1, 1, 1) * meshes
    verts_rot = torch.matmul(meshes_scaled, rotations)
    verts_trans = verts_rot + translations
    return verts_trans