from math import isqrt
from typing import Literal
from typing import Optional
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from einops import einsum, rearrange, repeat
from torch import Tensor

from .projection import get_fov, homogenize_points


def get_projection_matrix(
    near,
    far,
    fov_x,
    fov_y,
):
    """Maps points in the viewing frustum to (-1, 1) on the X/Y axes and (0, 1) on the Z
    axis. Differs from the OpenGL version in that Z doesn't have range (-1, 1) after
    transformation and that Z is flipped.
    """
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    top = tan_fov_y * near
    bottom = -top
    right = tan_fov_x * near
    left = -right

    (b,) = near.shape
    result = torch.zeros((b, 4, 4), dtype=torch.float32, device=near.device)
    result[:, 0, 0] = 2 * near / (right - left)
    result[:, 1, 1] = 2 * near / (top - bottom)
    result[:, 0, 2] = (right + left) / (right - left)
    result[:, 1, 2] = (top + bottom) / (top - bottom)
    result[:, 3, 2] = 1
    result[:, 2, 2] = far / (far - near)
    result[:, 2, 3] = -(far * near) / (far - near)
    return result


def render_cuda_(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    semantics_sems = None,
    scale_invariant: bool = True,
    use_sh: bool = True,
):
    
    
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1



    # Make sure everything is in a range where numerical issues don't appear.


    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        semantics_sems = semantics_sems * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix


    for i in range(b):

        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True, device="cuda") + 0
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass
        include_feature = True
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
            include_feature=include_feature,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        image,sem, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            language_feature_precomp = semantics_sems[i] if include_feature else torch.zeros((1,), dtype=gaussian_opacities.dtype, device=gaussian_opacities.device),
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )


        # return  torch.stack(all_images)
    return torch.stack([image]), torch.stack([sem]), torch.stack([radii]), mean_gradients





def render_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    semantics_sems = None,
    scale_invariant: bool = True,
    use_sh: bool = True,
):
    
    
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1


    # # TODO: For DEBUG
    # num = gaussian_means.shape[1]

    # perm = torch.randperm(num)

    # gaussian_sh_coefficients[0] = gaussian_sh_coefficients[0,perm]
    # gaussian_means[0] = gaussian_means[0,perm]
    # gaussian_covariances[0] = gaussian_covariances[0,perm]
    # gaussian_opacities[0] = gaussian_opacities[0,perm]
    # semantics_sems[0] = semantics_sems[0,perm]

    # gaussian_sh_coefficients = gaussian_sh_coefficients[:,:30000,:,:]
    # gaussian_means = gaussian_means[:,:30000,...]
    # gaussian_covariances = gaussian_covariances[:,:30000,...]
    # gaussian_opacities = gaussian_opacities[:,:30000,...]
    # semantics_sems = semantics_sems[:,:30000,...]



    # Make sure everything is in a range where numerical issues don't appear.


    if scale_invariant:
        scale = 1 / near

        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        
        gaussian_means = gaussian_means * scale[:, None, None]
        semantics_sems = semantics_sems * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()
    # shs = gaussian_sh_coefficients

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_sem = []
    all_radii = []
    all_means2d = []
    all_depth = []

    all_alpha = []
    all_contri_idx = []
    all_contri_num = []

    for i in range(b):

        # Set up a tensor for the gradients of the screen-space means.
        # torch.cuda.synchronize()
        # print(gaussian_means[i],scale)
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True, device="cuda") + 0
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass
        include_feature = True
        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
            include_feature=include_feature,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        # semantics_sems = torch.zeros((gaussian_means.shape[0],gaussian_means.shape[1],16)).to(gaussian_means.device)

        image,sem, radii, depth, alpha, contri_idx, contri_num = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            language_feature_precomp = semantics_sems[i] if include_feature else torch.zeros((1,), dtype=gaussian_opacities.dtype, device=gaussian_opacities.device),
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        # print(len(torch.unique(contri_idx)),contri_num)

        # import matplotlib.pyplot as plt
        # # 去掉 batch 维度
        # depth_image = depth.squeeze().detach().cpu().numpy()  # 转成 numpy，shape: [512, 512]
        # alpha = alpha.squeeze().detach().cpu().numpy()
        # # 可视化
        # plt.figure(figsize=(6, 6))
        # plt.imshow(alpha, cmap='plasma')  # 或使用 'viridis', 'inferno', 'gray' 等
        # plt.colorbar(label='Depth')
        # plt.title("Depth Visualization")
        # plt.axis('off')
        # plt.show()
        # plt.savefig('vis/alpha.png')


        

        sem = sem

        all_images.append(image)
        all_radii.append(radii)
        all_means2d.append(mean_gradients)
        all_sem.append(sem)
        all_alpha.append(alpha)
        all_depth.append(depth)
        all_contri_idx.append(contri_idx)
        all_contri_num.append(contri_num)
        # return  torch.stack(all_images)
    return torch.stack(all_images), torch.stack(all_sem), torch.stack(all_depth), torch.stack(all_alpha), torch.stack(all_contri_idx), torch.stack(all_contri_num), torch.stack(all_radii), all_means2d


def render_cuda_orthographic(
    extrinsics,
    width,
    height,
    near,
    far,
    image_shape: tuple[int, int],
    background_color,
    gaussian_means,
    gaussian_covariances,
    gaussian_sh_coefficients,
    gaussian_opacities,
    fov_degrees,
    use_sh: bool = True,
    dump: Optional[dict] = None,
):
    b, _, _ = extrinsics.shape
    h, w = image_shape
    assert use_sh or gaussian_sh_coefficients.shape[-1] == 1

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz").contiguous()

    # Create fake "orthographic" projection by moving the camera back and picking a
    # small field of view.
    fov_x = torch.tensor(fov_degrees, device=extrinsics.device).deg2rad()
    tan_fov_x = (0.5 * fov_x).tan()
    distance_to_near = (0.5 * width) / tan_fov_x
    tan_fov_y = 0.5 * height / distance_to_near
    fov_y = (2 * tan_fov_y).atan()
    near = near + distance_to_near
    far = far + distance_to_near
    move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    move_back[2, 3] = -distance_to_near
    extrinsics = extrinsics @ move_back

    # Escape hatch for visualization/figures.
    if dump is not None:
        dump["extrinsics"] = extrinsics
        dump["fov_x"] = fov_x
        dump["fov_y"] = fov_y
        dump["near"] = near
        dump["far"] = far

    projection_matrix = get_projection_matrix(
        near, far, repeat(fov_x, "-> b", b=b), fov_y
    )
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    all_radii = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x,
            tanfovy=tan_fov_y,
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)
        # print(use_sh)
        # awdaw
        image, radii = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i] if use_sh else None,
            colors_precomp=None if use_sh else shs[i, :, 0, :],
            opacities=gaussian_opacities[i, ..., None],
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )

        all_images.append(image)
        all_radii.append(radii)
    return torch.stack(all_images)


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


def render_depth_cuda(
    extrinsics,
    intrinsics,
    near,
    far,
    image_shape: tuple[int, int],
    gaussian_means,
    gaussian_covariances,
    gaussian_opacities,
    scale_invariant: bool = True,
    mode: DepthRenderingMode = "depth",
):
    # Specify colors according to Gaussian depths.
    camera_space_gaussians = einsum(
        extrinsics.inverse(), homogenize_points(gaussian_means), "b i j, b g j -> b g i"
    )
    fake_color = camera_space_gaussians[..., 2]

    if mode == "disparity":
        fake_color = 1 / fake_color
    elif mode == "relative_disparity":
        fake_color = depth_to_relative_disparity(
            fake_color, near[:, None], far[:, None]
        )
    elif mode == "log":
        fake_color = fake_color.minimum(near[:, None]).maximum(far[:, None]).log()

    # Render using depth as color.
    b, _ = fake_color.shape
    result = render_cuda(
        extrinsics,
        intrinsics,
        near,
        far,
        image_shape,
        torch.zeros((b, 3), dtype=fake_color.dtype, device=fake_color.device),
        gaussian_means,
        gaussian_covariances,
        repeat(fake_color, "b g -> b g c ()", c=3),
        gaussian_opacities,
        scale_invariant=scale_invariant,
    )
    return result.mean(dim=1)
