from pytorch3d.renderer.cameras import look_at_view_transform, PerspectiveCameras
from pytorch3d.renderer.mesh.rasterizer import RasterizationSettings, MeshRasterizer
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.renderer import MeshRenderer
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer.camera_utils import rotate_on_spot
from pytorch3d.utils import opencv_from_cameras_projection
import torch
import math
import time

def cameras_from_opencv_to_pytorch3d(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
):
    # Ref: https://github.com/facebookresearch/pytorch3d/blob/57f6e79280e78b6e8308f750e64d32984ddeaba4/pytorch3d/renderer/camera_conversions.py#L19

    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)  # PyTorch3D is row-major
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=R.device,
    )

def compute_random_rotations(B=10):
    """
    Randomly samples rotation matrices.

    Args:
        B (int): Batch size.
        upright (bool): If True, samples rotations that are mostly upright. Otherwise,
            samples uniformly from rotation space.

    Returns:
        rotation_matrices (B x 3 x 3).
    """
    # Reference: J Avro. "Fast Random Rotation Matrices." (1992)
    x1, x2, x3 = torch.split(torch.rand(3 * B).cuda(), B)
    tau = 2 * math.pi
    R = torch.stack(
        (  # B x 3 x 3
            torch.stack((torch.cos(tau * x1), torch.sin(
                tau * x1), torch.zeros_like(x1)), 1),
            torch.stack((-torch.sin(tau * x1), torch.cos(
                tau * x1), torch.zeros_like(x1)), 1),
            torch.stack((torch.zeros_like(x1), torch.zeros_like(x1),
                            torch.ones_like(x1)), 1),
        ),
        1,
    )
    v = torch.stack(
        (  # B x 3
            torch.cos(tau * x2) * torch.sqrt(x3),
            torch.sin(tau * x2) * torch.sqrt(x3),
            torch.sqrt(1 - x3),
        ),
        1,
    )
    identity = torch.eye(3).repeat(B, 1, 1).cuda()
    H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
    rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices

def get_uniform_SO3_RT(num_azimuth, num_elevation, distance, center, device="cpu", add_angle_azi=0, add_angle_ele=0):
    '''
    Get a bunch of camera extrinsics centered towards center with uniform distance in polar coordinates(elevation and azimuth)
    Args:
        num_elevation: int, number of elevation angles, excluding the poles
        num_azimuth: int, number of azimuth angles
        distance: radius of those transforms
        center: center around which the transforms are generated. Needs to be torch.tensor of shape [1, 3]
    Returns:
        rotation: torch.tensor of shape [num_views, 3, 3]
        translation: torch.tensor of shape [num_views, 3]
        Weirdly in pytorch3d y-axis is for world coordinate's up axis
        pytorch3d also has a weird as convention where R is right mulplied, so its actually the inverse of the normal rotation matrix
    '''
    grid = torch.zeros((num_elevation, num_azimuth, 2)) # First channel azimuth, second channel elevation
    azimuth = torch.linspace(0, 360, num_azimuth + 1)[:-1] 
    elevation = torch.linspace(-90, 90, num_elevation + 2)[1:-1] 
    grid[:, :, 0] = azimuth[None, :]
    grid[:, :, 1] = elevation[:, None]
    grid = grid.view(-1, 2)
    top_down = torch.tensor([[0, -90], [0, 90]]) # [2, 2]
    grid = torch.cat([grid, top_down], dim=0) # [num_views, 2]
    azimuth = grid[:, 0] + add_angle_azi
    elevation = grid[:, 1] + add_angle_ele
    
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=center
    )
    return rotation, translation, azimuth, elevation

@torch.no_grad()
def run_rendering(device, mesh, num_azimuth, num_elevation, roll_rotation, H, W, add_angle_azi=0, add_angle_ele=0, use_normal_map=False, render_bs=-1):
    # 关于Roll角的设定: https://github.com/facebookresearch/pytorch3d/issues/927
    bbox = mesh.get_bounding_boxes()
    radius = bbox.abs().max()
    center = bbox.mean(2)
    distance = 2 * radius
    rotation, translation, azimuth, elevation = get_uniform_SO3_RT(num_azimuth, num_elevation, distance, center, device=device, add_angle_azi=add_angle_azi, add_angle_ele=add_angle_ele)
    rotation, translation = rotate_on_spot(rotation, translation, roll_rotation)
    normal_batched_renderings = None
    if use_normal_map:
        raise NotImplementedError
    # 增加batch_render防止超出GPU显存, e.g. for 3090
    rasterization_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    camera = PerspectiveCameras(R=rotation, T=translation, device=device)
    if render_bs == -1:
        rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
        camera_centre = camera.get_camera_center()
        lights = PointLights(
            diffuse_color=((0.4, 0.4, 0.5),),
            ambient_color=((0.6, 0.6, 0.6),),
            specular_color=((0.01, 0.01, 0.01),),
            location=camera_centre,
            device=device,
        )
        shader = HardPhongShader(device=device, cameras=camera, lights=lights)
        batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        batch_mesh = mesh.extend(len(rotation))
        batched_renderings = batch_renderer(batch_mesh)
        fragments = rasterizer(batch_mesh)
        depth = fragments.zbuf
    else:
        num_batches = rotation.shape[0] // render_bs
        num_batches = (num_batches + 1) if (rotation.shape[0] % render_bs) != 0 else num_batches
        batched_renderings = []
        depth = []
        for i_render in range(num_batches):
            start_render_idx = i_render * render_bs
            end_render_idx = start_render_idx + render_bs
            end_render_idx = end_render_idx if end_render_idx <= rotation.shape[0] else rotation.shape[0]
            rotation_ = rotation[start_render_idx:end_render_idx]
            translation_ = translation[start_render_idx:end_render_idx]
            camera_ = PerspectiveCameras(R=rotation_, T=translation_, device=device)
            rasterizer = MeshRasterizer(cameras=camera_, raster_settings=rasterization_settings)
            camera_centre = camera_.get_camera_center()
            lights = PointLights(
                diffuse_color=((0.4, 0.4, 0.5),),
                ambient_color=((0.6, 0.6, 0.6),),
                specular_color=((0.01, 0.01, 0.01),),
                location=camera_centre,
                device=device,
            )
            shader = HardPhongShader(device=device, cameras=camera_, lights=lights)
            batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            batch_mesh = mesh.extend(len(rotation_))
            batched_renderings.append(batch_renderer(batch_mesh).cpu())
            fragments_ = rasterizer(batch_mesh)
            depth.append(fragments_.zbuf.cpu())
        batched_renderings = torch.cat(batched_renderings, dim=0)
        depth = torch.cat(depth, dim=0)
    return batched_renderings, normal_batched_renderings, camera, depth

@torch.no_grad()
def run_random_rendering(device, mesh, view_nums, H, W, rotation=None, use_normal_map=False, distance_scale=2.5):
    # 关于Roll角的设定: https://github.com/facebookresearch/pytorch3d/issues/927
    bbox = mesh.get_bounding_boxes()
    radius = bbox.abs().max()
    center = bbox.mean(2)
    distance = distance_scale * radius
    _, translation = look_at_view_transform(
        dist=distance, azim=torch.zeros(1), elev=torch.zeros(1), device=device, at=center
    )
    if rotation is None:
        rotation = compute_random_rotations(view_nums)
    camera = PerspectiveCameras(R=rotation, T=translation.repeat(rotation.shape[0], 1), device=device)
    rasterization_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device,
    )
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    batch_mesh = mesh.extend(len(rotation))
    normal_batched_renderings = None
    batched_renderings = batch_renderer(batch_mesh)
    if use_normal_map:
        raise NotImplementedError
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    return batched_renderings, normal_batched_renderings, camera, depth

@torch.no_grad()
def render_mesh_opencv_pose(device, mesh, R, T, K, H, W):
    '''
    R: 1 * 3 * 3
    T: 1 * 3
    K: 1 * 3 * 3
    '''
    camera = cameras_from_opencv_to_pytorch3d(R, T, K, torch.tensor([H, W]).unsqueeze(0).to(device))
    rasterization_settings = RasterizationSettings(
        image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=camera, raster_settings=rasterization_settings)
    camera_centre = camera.get_camera_center()
    lights = PointLights(
        diffuse_color=((0.4, 0.4, 0.5),),
        ambient_color=((0.6, 0.6, 0.6),),
        specular_color=((0.01, 0.01, 0.01),),
        location=camera_centre,
        device=device,
    )
    shader = HardPhongShader(device=device, cameras=camera, lights=lights)
    batch_renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    batch_mesh = mesh
    batched_renderings = batch_renderer(batch_mesh)
    fragments = rasterizer(batch_mesh)
    depth = fragments.zbuf
    return batched_renderings,camera, depth

def batch_render(device, mesh, num_azimuth, num_elevation, num_roll, H, W, return_camera=False, use_normal_map=False, render_bs=-1):
    add_angle_azi = 0
    add_angle_ele = 0
    if num_roll == 1:
        roll_angles = torch.tensor([0])
    else:
        roll_angles = torch.linspace(-180, 180, num_roll)
    renderings = []
    Rs = []
    Ts = []
    Ks = []
    depths = []
    for roll_angle in roll_angles:
        roll_rotation = axis_angle_to_matrix(torch.FloatTensor([0, 0, math.radians(roll_angle.item())])).to(device)
        try:
            rendering, _, camera, depth =  run_rendering(device, mesh, num_azimuth, num_elevation, roll_rotation, 
                                                         H, W, add_angle_azi=add_angle_azi, add_angle_ele=add_angle_ele, 
                                                         use_normal_map=use_normal_map, render_bs=render_bs)
            R_, T_, K_ = opencv_from_cameras_projection(camera, torch.tensor([[H, W]]).repeat(rendering.shape[0], 1))
            renderings.append(rendering.cpu())
            Rs.append(R_)
            Ts.append(T_)
            Ks.append(K_)
            depths.append(depth.cpu())
        except torch.linalg.LinAlgError as e:
            raise torch.linalg.LinAlgError
    renderings = torch.cat(renderings, dim=0)
    depths = torch.cat(depths, dim=0)
    Rs = torch.cat(Rs, dim=0)
    Ts = torch.cat(Ts, dim=0)
    Ks = torch.cat(Ks, dim=0)
    if return_camera:
        return renderings, None, camera, depth
    return renderings, depths, Rs, Ts, Ks,

def batch_random_render(device, mesh, view_numbers, H, W, use_normal_map=False, distance_scale=2.5):
    renderings = []
    Rs = []
    Ts = []
    Ks = []
    depths = []
    batch_render_views = 100
    num_batches = (view_numbers + batch_render_views - 1) // batch_render_views  # 计算需要的批次数
    random_rotations = compute_random_rotations(view_numbers)
    for batch_idx in range(num_batches):
        try:
            start_idx = batch_idx * batch_render_views
            end_idx = min((batch_idx + 1) * batch_render_views, view_numbers)
            batch_view_numbers = end_idx - start_idx
            rendering, _, camera, depth = run_random_rendering(device, mesh, batch_view_numbers, rotation=random_rotations[start_idx:end_idx], 
                                                               H=H, W=W, use_normal_map=use_normal_map, distance_scale=distance_scale)
            R_, T_, K_ = opencv_from_cameras_projection(camera, torch.tensor([[H, W]]).repeat(rendering.shape[0], 1))
            renderings.append(rendering.cpu())
            Rs.append(R_)
            Ts.append(T_)
            Ks.append(K_)
            depths.append(depth.cpu())
        except torch.linalg.LinAlgError as e:
            raise torch.linalg.LinAlgError
    renderings = torch.cat(renderings, dim=0)
    depths = torch.cat(depths, dim=0)
    Rs = torch.cat(Rs, dim=0)
    Ts = torch.cat(Ts, dim=0)
    Ks = torch.cat(Ks, dim=0)
    
    return renderings, depths, Rs, Ts, Ks
