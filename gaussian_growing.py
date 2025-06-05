import os
import torch
import sys
from scene import GaussianModel
import PIL
import cv2
import math
import numpy as np

sys.path.append("src/mast3r_src")
sys.path.append("src/mast3r_src/dust3r")
sys.path.append("src/pixelsplat_src")
from dust3r.utils.image import load_images
import gaussian_initialization
from pathlib import Path
import json
import torchvision
from data.data import crop_resize_if_necessary
from sklearn.decomposition import PCA
from src.mast3r_src.dust3r.dust3r.utils.image import imread_cv2
import torch.nn.functional as F
import workspace
import argparse
from einops import rearrange
from argparse import ArgumentParser
from arguments import OptimizationParams
from utils.loss_utils import l1_loss, ssim
from utils.APE_seg import get_seg_label, get_model
from utils.inpaint import get_inpainter_pipeline, inpainting
from diffusers import DDIMScheduler
from PIL import Image
from unidepth.models import UniDepthV2
from scipy import linalg
import random


def get_parser():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--config",
        type=str,
        default='configs/oggsplat_infer.yml',
    )
    return parser


def estimate_scale_from_origin(pc1: np.ndarray, pc2: np.ndarray, method="rms") -> float:

    assert pc1.shape[1] == 3 and pc2.shape[1] == 3
    if method == "mean":
        d1 = np.linalg.norm(pc1, axis=1).mean()
        d2 = np.linalg.norm(pc2, axis=1).mean()
    elif method == "rms":
        d1 = np.sqrt((pc1**2).sum(axis=1).mean())
        d2 = np.sqrt((pc2**2).sum(axis=1).mean())
    else:
        raise ValueError("method should be mean or rms'")

    scale = d2 / d1
    return scale


def transform_pointcloud(P_A, c2w_A, c2w_B):
    N = P_A.shape[0]

    P_A_hom = np.hstack([P_A, np.ones((N, 1))])

    P_world = (c2w_A @ P_A_hom.T).T

    w2c_B = np.linalg.inv(c2w_B)
    P_B_hom = (w2c_B @ P_world.T).T

    P_B = P_B_hom[:, :3] / P_B_hom[:, 3][:, None]

    return P_B


def back_project_coords(depth_map, intrinsic, extrinsic):

    H, W = depth_map.shape
    device = depth_map.device

    u = torch.arange(W, device=device)
    v = torch.arange(H, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    pixels = torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).float()

    K_inv = torch.inverse(intrinsic)
    rays = pixels @ K_inv.T

    points_3d = rays * depth_map.unsqueeze(-1)

    return points_3d


def set_seed(seed=42):

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    if sigma2 == 0:
        return diff.dot(diff)

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print("FID calculation warning: singular matrix, adding epsilon.")
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def get_3dgs(pred1, pred2):
    means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
    covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
    harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)

    opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
    semantics = torch.stack([pred1["sem"], pred2["sem"]], dim=1)
    scales = torch.stack([pred1["scales"], pred2["scales"]], dim=1)
    rotations = torch.stack([pred1["rotations"], pred2["rotations"]], dim=1)

    means = rearrange(means, "b v h w xyz -> b (v h w) xyz")

    scales = rearrange(scales, "b v h w s -> b (v h w) s")
    rotations = rearrange(rotations, "b v h w r -> b (v h w) r")

    covariances = rearrange(covariances, "b v h w i j -> b (v h w) i j")
    harmonics = rearrange(harmonics, "b v h w c d_sh -> b (v h w) c d_sh")
    opacities = rearrange(opacities, "b v h w 1 -> b (v h w)")
    semantics = rearrange(semantics, "b v h w s -> b (v h w) s")

    return (
        torch.tensor(means.squeeze(0)),
        torch.tensor(harmonics.squeeze(0)),
        torch.tensor(scales.squeeze(0)),
        torch.tensor(rotations.squeeze(0)),
        torch.tensor(opacities.squeeze(0)),
        torch.tensor(semantics.squeeze(0)),
    )


def generate_symmetric_rotations(num_samples: int, dim: int, degree: int):

    assert num_samples % 2 == 0

    half = num_samples // 2
    degrees = torch.linspace(0, degree, steps=half + 1)[1:]

    rot_list = []
    for deg in degrees:
        rot = [0.0, 0.0, 0.0]
        neg_rot = [0.0, 0.0, 0.0]
        rot[dim] = float(deg)
        neg_rot[dim] = -float(deg)

        rot_list.append(tuple(rot))
        rot_list.append(tuple(neg_rot))

    return rot_list


def generate_edge_prompt(label, edges, mapper):
    care_label = label[edges == 255]
    care_label = care_label[care_label != -100]

    unique_vals, counts = np.unique(care_label, return_counts=True)

    exclude_vals = np.array([0, 1, 2])
    exclude_indices = np.isin(unique_vals, exclude_vals)

    remaining_unique_vals = unique_vals[~exclude_indices]
    remaining_counts = counts[~exclude_indices]

    sorted_indices = np.argsort(remaining_counts)[::-1]

    remaining_unique_vals = remaining_unique_vals[sorted_indices]
    remaining_counts = remaining_counts[sorted_indices]

    final_unique_vals = np.concatenate(
        [remaining_unique_vals, unique_vals[exclude_indices]]
    )
    final_counts = np.concatenate([remaining_counts, counts[exclude_indices]])

    unique_vals = final_unique_vals.astype(np.int32)

    labels = []

    for unique_val in unique_vals:
        lable = mapper[unique_val]
        labels.append(lable)

    if len(labels) == 0:

        text = "a room"
        return text

    text = (
        ", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else labels[0]
    )
    text = "a room with " + text

    return text


def project_points(points_world, K, pose, img_size):
    H, W = img_size

    points_homo = np.concatenate(
        [points_world, np.ones((points_world.shape[0], 1))], axis=1
    )

    cam_pose_inv = np.linalg.inv(pose)
    points_cam = (cam_pose_inv @ points_homo.T).T[:, :3]

    projected = (K @ points_cam.T).T
    projected[:, 0] /= projected[:, 2]
    projected[:, 1] /= projected[:, 2]
    xy = projected[:, :2]

    visible = (
        (points_cam[:, 2] > 0)
        & (xy[:, 0] >= 0)
        & (xy[:, 0] < W)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] < H)
    )
    return xy, visible


def calculate_new_pixels(K, pose1, pose2, num_points=500000):

    points_world = np.random.uniform(
        [-20, -20, -10], [20, 20, 20], size=(num_points, 3)
    )
    img_size = (512, 512)
    xy1, vis1 = project_points(points_world, K, pose1, img_size)
    xy2, vis2 = project_points(points_world, K, pose2, img_size)

    new_visible = vis2 & ~vis1
    xy_new = xy2[new_visible].astype(int)

    mask = np.zeros(img_size, dtype=np.uint8)
    for x, y in xy_new:
        if 0 <= y < img_size[0] and 0 <= x < img_size[1]:
            mask[y, x] = 1

    kernel = np.ones((3, 3), np.uint8)

    dilated_mask = cv2.dilate(mask, kernel, iterations=5)

    mask = dilated_mask * 255

    return mask


def apply_local_transform(T, trans_xyz=(0, 0, 0), rot_deg_xyz=(0, 0, 0)):

    def rot_x(angle):
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        return torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])

    def rot_y(angle):
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        return torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_z(angle):
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        return torch.tensor([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    Rx = rot_x(rot_deg_xyz[0])
    Ry = rot_y(rot_deg_xyz[1])
    Rz = rot_z(rot_deg_xyz[2])
    R = Rz @ Ry @ Rx

    delta_T = torch.eye(4)
    delta_T[:3, :3] = R
    delta_T[:3, 3] = torch.tensor(trans_xyz)

    return T @ delta_T.to(T.device)


def get_reconstructed_scene(
    outdir,
    model,
    depth_model,
    model_vl,
    model_fid,
    device,
    silent,
    image_size,
    filelist,
    batch,
    mapper,
    inpainter,
    config,
    need_autoencoder=False,
):
    """
    Generate 3D Gaussian Splats from input images.
    """
    assert len(filelist) == 1 or len(filelist) == 2, "Please provide one or two images"

    if len(filelist) == 1:
        filelist = [filelist[0], filelist[0]]

    imgs = load_images(filelist, size=image_size, verbose=not silent)

    imgs_context = [(img["img"] + 1) / 2 for img in imgs]

    pipeline_rgb, pipeline_feat = inpainter

    rots_horizontal = generate_symmetric_rotations(4, dim=1, degree=60)

    rots_view_horizontal = generate_symmetric_rotations(60, dim=1, degree=60)

    rots_vertical = generate_symmetric_rotations(2, dim=0, degree=20)

    rots_view_vertical = generate_symmetric_rotations(20, dim=0, degree=20)

    base_pose = batch["context"][0]["camera_pose"]

    base_intrinsic = batch["context"][0]["camera_intrinsics"]

    for img in imgs:
        img["img"] = img["img"].to(device)
        img["original_img"] = img["original_img"].to(device)
        img["true_shape"] = torch.from_numpy(img["true_shape"])

    with torch.no_grad():
        output = model(imgs[0], imgs[1])

    gaussian = GaussianModel(sh_degree=0)

    parser = ArgumentParser()

    opt = OptimizationParams(parser)

    gaussian.training_setup(opt)

    pred1, pred2 = output

    pkgs = get_3dgs(pred1, pred2)

    gaussian.restore(pkgs, opt)

    targets = []
    targets.append(batch["context"][0])
    targets.append(batch["context"][1])
    targets_view = []
    targets_view_h = []
    targets_view_v = []

    view_masks = [None, None]

    for i, rot in enumerate(rots_vertical):

        new_extr = apply_local_transform(
            base_pose, trans_xyz=(0.0, 0.0, 0.0), rot_deg_xyz=rot
        )

        if i in [0, 1]:
            view_mask = calculate_new_pixels(
                base_intrinsic[0, :3, :3].detach().cpu().numpy(),
                base_pose[0].detach().cpu().numpy(),
                new_extr[0].detach().cpu().numpy(),
                num_points=1000000,
            )

        view_masks.append(view_mask)

        targets.append({"camera_intrinsics": base_intrinsic, "camera_pose": new_extr})

    assert len(rots_vertical) == 2

    v_l, v_b = targets[-2]["camera_pose"], targets[-1]["camera_pose"]

    for i, rot in enumerate(rots_horizontal):

        new_extr = apply_local_transform(
            base_pose, trans_xyz=(0.0, 0.0, 0.0), rot_deg_xyz=rot
        )

        if i == 0:
            view_mask_r = calculate_new_pixels(
                base_intrinsic[0, :3, :3].detach().cpu().numpy(),
                base_pose[0].detach().cpu().numpy(),
                new_extr[0].detach().cpu().numpy(),
                num_points=1000000,
            )

        if i == 1:
            view_mask_l = calculate_new_pixels(
                base_intrinsic[0, :3, :3].detach().cpu().numpy(),
                base_pose[0].detach().cpu().numpy(),
                new_extr[0].detach().cpu().numpy(),
                num_points=1000000,
            )

        if i % 2 == 0:
            view_mask = view_mask_r
        else:
            view_mask = view_mask_l

        view_mask_with_v_l = calculate_new_pixels(
            base_intrinsic[0, :3, :3].detach().cpu().numpy(),
            v_l[0].detach().cpu().numpy(),
            new_extr[0].detach().cpu().numpy(),
            num_points=1000000,
        )
        view_mask_with_v_b = calculate_new_pixels(
            base_intrinsic[0, :3, :3].detach().cpu().numpy(),
            v_b[0].detach().cpu().numpy(),
            new_extr[0].detach().cpu().numpy(),
            num_points=1000000,
        )

        view_mask_new = (
            (view_mask == 255)
            & (view_mask_with_v_l == 255)
            & (view_mask_with_v_b == 255)
        ).astype(np.uint8) * 255

        view_masks.append(view_mask_new)

        targets.append({"camera_intrinsics": base_intrinsic, "camera_pose": new_extr})

    for i, rot in enumerate(rots_view_horizontal):

        new_extr = apply_local_transform(
            base_pose, trans_xyz=(0.0, 0.0, 0.0), rot_deg_xyz=rot
        )

        targets_view_h.append(
            {"camera_intrinsics": base_intrinsic, "camera_pose": new_extr}
        )

    even_indices = targets_view_h[::2]
    odd_indices = targets_view_h[1::2]

    targets_view = even_indices + odd_indices

    for i, rot in enumerate(rots_view_vertical):

        new_extr = apply_local_transform(
            base_pose, trans_xyz=(0.0, 0.0, 0.0), rot_deg_xyz=rot
        )

        targets_view_v.append(
            {"camera_intrinsics": base_intrinsic, "camera_pose": new_extr}
        )

    even_indices = targets_view_v[::2]
    odd_indices = targets_view_v[1::2]

    targets_view = even_indices + odd_indices + targets_view

    batch["target"] = targets
    color_raw, _, _, _, sems_raw, _, alphas_raw, contri_idx_raw, _ = (
        model.decoder.gaussian_forward(batch, gaussian, (512, 512))
    )

    colors_raw = color_raw.squeeze(0)
    sems_raw = sems_raw.squeeze(0)
    alphas_raw = alphas_raw.squeeze(0)
    contri_idx_raw = contri_idx_raw.squeeze(0)

    alpha_list = []

    for ii, (color_raw, sem_raw, alpha_raw) in enumerate(
        zip(colors_raw, sems_raw, alphas_raw)
    ):
        alpha_raw = alpha_raw < 0.3

        alpha_raw = alpha_raw.squeeze(0).detach().cpu().numpy()
        alpha_raw = (alpha_raw * 255).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

        alpha_raw = cv2.morphologyEx(alpha_raw, cv2.MORPH_CLOSE, kernel)
        alpha_raw = cv2.dilate(
            alpha_raw, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=3
        )
        alpha_list.append(alpha_raw)
        alpha_raw = PIL.Image.fromarray(alpha_raw)

    xyzs = gaussian._xyz
    points_refer_list = []
    alpha_raw = alpha_list
    for t_id in range(len(sems_raw)):

        points_refer = contri_idx_raw[t_id][:, ~torch.tensor(alpha_raw[t_id]) == 255]

        points_refer = torch.unique(points_refer)[1:]

        points_refer = xyzs[points_refer]

        points_refer_list.append(points_refer)

    colors_backup = []
    sems_backup = []

    for t_id in range(0, len(targets), 2):

        iteration_all = opt.iterations
        imgs_targets = []
        sems_gt = []

        for iteration in range(iteration_all + 1):
            batch["target"] = []

            if t_id > 0:
                previous = t_id
                previous_a, previous_b = torch.randperm(previous)[:2]
                values, _ = torch.sort(torch.stack([previous_a, previous_b]))

                previous_a, previous_b = values[0], values[1]

                batch["target"].append(targets[previous_a])
                batch["target"].append(targets[previous_b])

            else:
                previous_a, previous_b = 0, 1

            batch["target"].append(targets[t_id])
            batch["target"].append(targets[t_id + 1])

            (
                color,
                visibility_filter,
                radii,
                viewspace_point_tensor,
                sems,
                depth,
                alpha,
                contri_idx,
                contri_num,
            ) = model.decoder.gaussian_forward(batch, gaussian, (512, 512))

            if need_autoencoder:
                batch_size, v, num_features, height, width = sems.shape
                sems = sems.permute(0, 1, 3, 4, 2).contiguous()
                sems = sems.view(-1, num_features)
                sems = model.autoencoder.decoder(sems)
                sems = sems.view(
                    batch_size, v, height, width, model.autoencoder.input_dim
                )
                sems = sems.permute(0, 1, 4, 2, 3).contiguous()

            colors = color.squeeze(0)

            sems = sems.squeeze(0)
            alphas = alpha.squeeze(0)
            contri_nums = contri_num.squeeze(0).to(torch.int64)
            contri_idxs = contri_idx.squeeze(0)

            if iteration == 0:
                imgs_targets = colors.clone()
                sems_gt = sems.clone()
                if t_id == 0:
                    sems_context = sems.clone()
            if iteration == iteration_all:
                colors_backup.append(colors[-2])
                colors_backup.append(colors[-1])
                sems_backup.append(sems[-2])
                sems_backup.append(sems[-1])

            if previous_a < 2:

                if previous_a in [0, 1]:
                    imgs_targets[0] = imgs_context[previous_a]
                    sems_gt[0] = sems_context[previous_a]
                if previous_b in [0, 1]:
                    imgs_targets[1] = imgs_context[previous_b]
                    sems_gt[1] = sems_context[previous_b]
                else:
                    imgs_targets[1] = colors_backup[previous_b]
                    sems_gt[1] = sems_backup[previous_b]

            else:

                imgs_targets[0] = colors_backup[previous_a]
                imgs_targets[1] = colors_backup[previous_b]
                sems_gt[0] = sems_backup[previous_a]
                sems_gt[1] = sems_backup[previous_b]

            Ll1 = l1_loss(colors, imgs_targets.to(colors.device).detach())
            loss_ = 0.8 * Ll1 + 0.2 * (
                1.0 - ssim(colors, imgs_targets.to(colors.device).detach())
            )
            if t_id != 0:
                gaussian.update_learning_rate(iteration)

            if iteration <= 1:

                loss = loss_
            else:

                sems_gt_ = sems_gt.permute(0, 2, 3, 1)
                sems_ = sems.permute(0, 2, 3, 1)

                cos_sim = F.cosine_similarity(
                    sems_, sems_gt_.detach().to(colors.device), dim=-1
                )

                loss = 1 - cos_sim
                mean_loss = loss.mean()
                loss = loss_ + mean_loss

            loss.backward()

            print("iteration: ", iteration, "loss:", loss)

            radii_care = radii

            for prune_densify_id in range(colors.shape[0]):
                radii_care = radii[prune_densify_id]
                visibility_filter_care = visibility_filter[prune_densify_id]
                viewspace_point_tensor_care = viewspace_point_tensor[prune_densify_id]
                if iteration < opt.densify_until_iter and iteration != 0:

                    gaussian.max_radii2D[visibility_filter_care] = torch.max(
                        gaussian.max_radii2D[visibility_filter_care],
                        radii_care[visibility_filter_care],
                    )
                    gaussian.add_densification_stats(
                        viewspace_point_tensor_care, visibility_filter_care
                    )

                    if prune_densify_id == (colors.shape[0] - 1):
                        if (iteration + 1) > opt.densify_from_iter and (
                            iteration + 1
                        ) % opt.densification_interval == 0:
                            size_threshold = (
                                20
                                if (iteration + 1) > opt.opacity_reset_interval
                                else None
                            )

                            gaussian.densify_and_prune(
                                opt.densify_grad_threshold, 0.003, 25, size_threshold
                            )

                        if (iteration + 1) % opt.opacity_reset_interval == 0:
                            gaussian.reset_opacity()

            if iteration < iteration_all - 1:
                gaussian.optimizer.step()

                gaussian.optimizer.zero_grad(set_to_none=True)
                current_lr = gaussian.optimizer.param_groups[0]["lr"]

            if iteration % 150 == 0 or iteration == iteration_all:

                for idx, (sem, color, alpha, contri_idx, contri_num) in enumerate(
                    zip(sems, colors, alphas, contri_idxs, contri_nums)
                ):

                    num_features, height, width = sem.shape

                    pca = PCA(n_components=3)
                    with torch.no_grad():
                        out_put = get_seg_label(
                            sem, color.detach().cpu() * 255, model_vl, mapper
                        )

                    sem_reduced = sem

                    sem_flat = sem_reduced.permute(1, 2, 0).reshape(-1, num_features)
                    sem_pca = pca.fit_transform(sem_flat.detach().cpu().numpy())
                    sem_reduced_ = torch.tensor(sem_pca, dtype=sem.dtype).reshape(
                        height, width, 3
                    )
                    sem_reduced_ = sem_reduced_.permute(2, 0, 1)

                    min_val = sem_reduced_.min()
                    max_val = sem_reduced_.max()

                    sem_reduced_ = (sem_reduced_ - min_val) / (max_val - min_val)

                    if t_id == 0 or idx < 2:
                        alpha = alpha < 0.07
                    else:
                        alpha = torch.logical_and(
                            torch.tensor(view_masks[t_id + (idx - 2)]) == 255,
                            torch.tensor(alpha_raw[t_id + (idx - 2)]) == 255,
                        )

                    alpha = alpha.squeeze(0).detach().cpu().numpy() * 255
                    alpha = alpha.astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)

                    if t_id == 0 or idx < 2:
                        alpha = cv2.dilate(alpha, kernel, iterations=3)

                    dilation = cv2.dilate(alpha, kernel, iterations=2)

                    erosion = cv2.erode(alpha, kernel, iterations=2)

                    edge = dilation - erosion

                    contri_num = contri_num > 1

                    text = generate_edge_prompt(
                        out_put.detach().cpu().numpy(), edge, mapper
                    )

                    if iteration == 0:

                        if t_id == 0:

                            if idx == 0:
                                gaussian.crop_all()
                                inpaint_rgb, inpaint_feat = inpainting(
                                    [text],
                                    color,
                                    sem,
                                    alpha,
                                    pipeline_rgb,
                                    pipeline_feat,
                                    model_vl,
                                    mapper,
                                    batch["scene_id"],
                                    tag=0,
                                    tid=t_id,
                                )

                            if idx == 1:
                                inpaint_rgb, inpaint_feat = inpainting(
                                    [text],
                                    color,
                                    sem,
                                    alpha,
                                    pipeline_rgb,
                                    pipeline_feat,
                                    model_vl,
                                    mapper,
                                    batch["scene_id"],
                                    tag=1,
                                    tid=t_id,
                                )

                            inpaint_rgb = np.array(inpaint_rgb)

                            inpaint_rgb = inpaint_rgb.transpose((2, 0, 1))

                            inpaint_rgb = torch.from_numpy(inpaint_rgb).float() / 255.0
                            inpaint_rgb = inpaint_rgb.to(imgs_targets.device)

                            inpaint_feat[:, alpha == 0] = sem[:, alpha == 0]

                            imgs_targets[idx] = inpaint_rgb
                            imgs_context[idx] = inpaint_rgb
                            sems_gt[idx] = inpaint_feat.to(sems_gt.device)
                            sems_context[idx] = inpaint_feat.to(sems_gt.device)

                        else:
                            if idx == 2:
                                mask_diff = torch.logical_and(
                                    torch.tensor(view_masks[t_id]) == 255,
                                    ~torch.tensor(alpha_raw[t_id]) == 255,
                                )
                                mask_r = torch.logical_and(
                                    torch.tensor(view_masks[t_id]) == 255,
                                    torch.tensor(alpha_raw[t_id]) == 255,
                                )
                                color = color.clone()
                                sem = sem.clone()

                                color[:, mask_diff] = colors_raw[t_id][:, mask_diff]
                                sem[:, mask_diff] = sems_raw[t_id][:, mask_diff]
                                mask_r = (mask_r.numpy() * 255).astype(np.uint8)

                                points_refer = points_refer_list[t_id]

                                inpaint_rgb, inpaint_feat = inpainting(
                                    [text],
                                    color,
                                    sem,
                                    mask_r,
                                    pipeline_rgb,
                                    pipeline_feat,
                                    model_vl,
                                    mapper,
                                    batch["scene_id"],
                                    tag=0,
                                    tid=t_id,
                                )

                                inpaint_rgb = np.array(inpaint_rgb)

                                inpaint_rgb = inpaint_rgb.transpose((2, 0, 1))

                                prediction = depth_model.infer(
                                    torch.from_numpy(inpaint_rgb)
                                )
                                depth = prediction["depth"].squeeze(0).squeeze(0)
                                xyz = prediction["points"].squeeze(0)
                                xyz = xyz.permute(1, 2, 0)
                                xyz = xyz.view(-1, 3).cpu().numpy()
                                extr_new = torch.inverse(base_pose[0]) @ batch[
                                    "target"
                                ][idx]["camera_pose"].squeeze(0)

                                points_3d = back_project_coords(
                                    depth,
                                    intrinsic=base_intrinsic.squeeze(0)[:3, :3],
                                    extrinsic=extr_new,
                                )

                                points_3d = points_3d.view(-1, 3).cpu().numpy()
                                points_3d = transform_pointcloud(
                                    points_3d,
                                    batch["target"][idx]["camera_pose"]
                                    .squeeze(0)
                                    .cpu()
                                    .numpy(),
                                    base_pose[0].cpu().numpy(),
                                )

                                points_new = points_3d.reshape(512, 512, -1)[
                                    ~torch.tensor(alpha_raw[t_id]) == 255, :
                                ]
                                s = estimate_scale_from_origin(
                                    points_new, points_refer.detach().cpu().numpy()
                                )

                                os.makedirs(
                                    str(
                                        Path("vis")
                                        / batch["scene_id"]
                                        / f"{config.seed}"
                                    ),
                                    exist_ok=True,
                                )

                                points_3d = points_3d * s
                                new_points_3d_r = points_3d.reshape(512, 512, -1)[
                                    mask_r == 255, :
                                ]
  
                                if depth.dtype != torch.uint8:
                                    depth_min = depth.min()
                                    depth_max = depth.max()
                                    depth = (depth - depth_min) / (
                                        depth_max - depth_min + 1e-5
                                    )
                                    depth = (depth * 255).clamp(0, 255).byte()

                                    depth = depth.detach().cpu().numpy()

                                depth = Image.fromarray(depth)

                                inpaint_rgb = (
                                    torch.from_numpy(inpaint_rgb).float() / 255.0
                                )
                                inpaint_rgb = inpaint_rgb.to(imgs_targets.device)

                                inpaint_feat[:, mask_r == 0] = sem[:, mask_r == 0]
                            if idx == 3:
                                mask_diff = torch.logical_and(
                                    torch.tensor(view_masks[t_id + 1]) == 255,
                                    ~torch.tensor(alpha_raw[t_id + 1]) == 255,
                                )
                                mask_l = torch.logical_and(
                                    torch.tensor(view_masks[t_id + 1]) == 255,
                                    torch.tensor(alpha_raw[t_id + 1]) == 255,
                                )
                                color = color.clone()
                                sem = sem.clone()
                                color[:, mask_diff] = colors_raw[t_id + 1][:, mask_diff]
                                sem[:, mask_diff] = sems_raw[t_id + 1][:, mask_diff]
                                mask_l = (mask_l.numpy() * 255).astype(np.uint8)

                                points_refer = points_refer_list[t_id + 1]

                                inpaint_rgb, inpaint_feat = inpainting(
                                    [text],
                                    color,
                                    sem,
                                    mask_l,
                                    pipeline_rgb,
                                    pipeline_feat,
                                    model_vl,
                                    mapper,
                                    batch["scene_id"],
                                    tag=1,
                                    tid=t_id,
                                )

                                inpaint_rgb = np.array(inpaint_rgb)

                                inpaint_rgb = inpaint_rgb.transpose((2, 0, 1))

                                prediction = depth_model.infer(
                                    torch.from_numpy(inpaint_rgb)
                                )
                                depth = prediction["depth"].squeeze(0).squeeze(0)
                                xyz = prediction["points"].squeeze(0)
                                xyz = xyz.permute(1, 2, 0)
                                xyz = xyz.view(-1, 3).cpu().numpy()
                                extr_new = torch.inverse(base_pose[0]) @ batch[
                                    "target"
                                ][idx]["camera_pose"].squeeze(0)
                                points_3d = back_project_coords(
                                    depth,
                                    intrinsic=base_intrinsic.squeeze(0)[:3, :3],
                                    extrinsic=extr_new,
                                )

                                points_3d = points_3d.view(-1, 3).cpu().numpy()
                                points_3d = transform_pointcloud(
                                    points_3d,
                                    batch["target"][idx]["camera_pose"]
                                    .squeeze(0)
                                    .cpu()
                                    .numpy(),
                                    base_pose[0].cpu().numpy(),
                                )

                                points_new = points_3d.reshape(512, 512, -1)[
                                    ~torch.tensor(alpha_raw[t_id + 1]) == 255, :
                                ]
                                s = estimate_scale_from_origin(
                                    points_new, points_refer.detach().cpu().numpy()
                                )


                                points_3d = points_3d * s

                                new_points_3d_l = points_3d.reshape(512, 512, -1)[
                                    mask_l == 255, :
                                ]

                                if depth.dtype != torch.uint8:
                                    depth_min = depth.min()
                                    depth_max = depth.max()
                                    depth = (depth - depth_min) / (
                                        depth_max - depth_min + 1e-5
                                    )
                                    depth = (depth * 255).clamp(0, 255).byte()

                                    depth = depth.detach().cpu().numpy()

                                depth = Image.fromarray(depth)

                                inpaint_rgb = (
                                    torch.from_numpy(inpaint_rgb).float() / 255.0
                                )
                                inpaint_rgb = inpaint_rgb.to(imgs_targets.device)

                                inpaint_feat[:, mask_l == 0] = sem[:, mask_l == 0]

                            if idx in [2, 3]:

                                imgs_targets[idx] = inpaint_rgb
                                sems_gt[idx] = inpaint_feat.to(sems_gt.device)

                    contri_idx = contri_idx[
                        :, torch.tensor(edge == 255).to(contri_idx.device)
                    ]

                    critical_idx = torch.unique(contri_idx)[1:]

                    if idx == 3 and iteration == 0:

                        new_points_3d = np.concatenate(
                            [new_points_3d_l, new_points_3d_r], axis=0
                        )

                        num_total = new_points_3d.shape[0]

                        half_indices = np.random.choice(
                            num_total, size=num_total // 4, replace=False
                        )

                        new_points_3d = new_points_3d[half_indices]

                        print(new_points_3d.shape)
                        gaussian.init_new_gaussians(new_points_3d)

                    alpha = Image.fromarray(alpha.astype("uint8"), mode="L")
                    edge = Image.fromarray(edge.astype("uint8"), mode="L")

                    sem_reduced_[color == 0] = 0

                    color_save_path = str(
                        Path(outdir)
                        / batch["scene_id"]
                        / f"rendered_color_{idx}_{iteration}_.png"
                    )
                    feat_save_path = str(
                        Path(outdir)
                        / batch["scene_id"]
                        / f"rendered_feat_{idx}_{iteration}_.png"
                    )

                    alpha_save_path = str(
                        Path(outdir)
                        / batch["scene_id"]
                        / f"rendered_alpha_{idx}_{iteration}_.png"
                    )
                    edge_save_path = str(
                        Path(outdir)
                        / batch["scene_id"]
                        / f"rendered_edge_{idx}_{iteration}_.png"
                    )
                    text_save_path = str(
                        Path(outdir)
                        / batch["scene_id"]
                        / f"rendered_text_{idx}_{iteration}_.txt"
                    )
                    os.makedirs(os.path.dirname(color_save_path), exist_ok=True)

                    alpha.save(alpha_save_path)
                    edge.save(edge_save_path)

                    os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)
                    torchvision.utils.save_image(color, color_save_path)
                    torchvision.utils.save_image(sem_reduced_, feat_save_path)
                    with open(text_save_path, "w", encoding="utf-8") as f:
                        f.write(text)

    colors = []
    alphas = []
    contri_idxs = []
    contri_nums = []
    sems_reduced = []

    mini_batch = 5
    num_elements = len(targets_view)

    num_full_batches = num_elements // mini_batch
    remainder = num_elements % mini_batch
    total_batches = num_full_batches + (1 if remainder else 0)

    for i in range(total_batches):
        start = i * mini_batch
        end = start + mini_batch
        targets_view_batch = targets_view[start:end]

        batch["target"] = targets_view_batch
        (
            color,
            visibility_filter,
            radii,
            viewspace_point_tensor,
            sems,
            depth,
            alpha,
            contri_idx,
            contri_num,
        ) = model.decoder.gaussian_forward(batch, gaussian, (512, 512))

        colors.append(color.squeeze(0).detach().cpu())
        sems = sems.squeeze(0)
        alphas.append(alpha.squeeze(0).detach().cpu())

        sems_reduced.append(sems.detach().cpu())

    sems_reduced = torch.cat(sems_reduced)

    alphas = torch.cat(alphas)
    colors = torch.cat(colors)
    bs, num_features, height, width = sems_reduced.shape

    pca = PCA(n_components=3)

    sem_flat = sems_reduced.permute(0, 2, 3, 1).reshape(-1, num_features)
    sem_pca = pca.fit_transform(sem_flat.detach().cpu().numpy())
    sem_reduced_ = torch.tensor(sem_pca, dtype=colors.dtype).reshape(
        len(sems_reduced), height, width, 3
    )
    sem_reduced_ = sem_reduced_.permute(0, 3, 1, 2)

    min_val = sem_reduced_.min()
    max_val = sem_reduced_.max()

    sems = (sem_reduced_ - min_val) / (max_val - min_val)

    gs_path = str(Path("vis") / batch["scene_id"] / f"{config.seed}" / "gaussian.pth")
    os.makedirs(os.path.dirname(gs_path), exist_ok=True)
    torch.save(gaussian.capture(True), gs_path)

    for idx, (sem, color, alpha) in enumerate(zip(sems, colors, alphas)):

        num_features, height, width = sem.shape

        sem_reduced_ = sem
        sem_reduced_[color == 0] = 0

        color_save_path = str(
            Path("vis")
            / batch["scene_id"]
            / f"{config.seed}"
            / "color"
            / f"rendered_color_{idx}.png"
        )
        feat_save_path = str(
            Path("vis")
            / batch["scene_id"]
            / f"{config.seed}"
            / "feat"
            / f"rendered_feat_{idx}.png"
        )

        os.makedirs(os.path.dirname(color_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(feat_save_path), exist_ok=True)
        torchvision.utils.save_image(color, color_save_path)
        torchvision.utils.save_image(sem_reduced_, feat_save_path)


if __name__ == "__main__":

    args = get_parser().parse_args()
    P = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]).astype(
        np.float32
    )
    image_size = 512
    silent = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = workspace.load_config(args.config)
    config.seed = args.seed
    set_seed(config.seed)
    root_path = Path(config.data_root)

    weights_path = config.weights_path_gs_init
    save_path = os.path.dirname(os.path.dirname(config.weights_path_gs_init))
    save_path = os.path.join('vis', "result")

    all_batch = []
    for select_frame in config.select_frames:
        scene_id = select_frame

        input_images = [
            str(
                root_path
                / scene_id
                / "dslr"
                / "undistorted_images"
                / (config.select_frames[scene_id][0] + ".JPG")
            ),
            str(
                root_path
                / scene_id
                / "dslr"
                / "undistorted_images"
                / (config.select_frames[scene_id][1] + ".JPG")
            ),
        ]

        cam_pose_path = str(
            root_path / scene_id / "dslr" / "nerfstudio" / "transforms_undistorted.json"
        )
        with open(cam_pose_path, "r") as f:
            cams_metadata = json.load(f)
        file_path_to_frame_metadata = {}

        for frame in cams_metadata["frames"]:
            file_path_to_frame_metadata[frame["file_path"]] = frame

        batches = []

        K = np.eye(4, dtype=np.float32)
        K[0, 0] = cams_metadata["fl_x"]
        K[1, 1] = cams_metadata["fl_y"]
        K[0, 2] = cams_metadata["cx"]
        K[1, 2] = cams_metadata["cy"]

        for target_id in config.select_frames[scene_id][:2]:

            target_id = target_id + ".JPG"

            target_path = str(
                root_path / scene_id / "dslr" / "undistorted_images" / target_id
            )

            target_feats_path = target_path.replace(
                "scannetpp/data", "scannetpp/APEfeat_low"
            ).replace(".JPG", ".pt")

            sem = torch.load(target_feats_path)

            resize_height, resize_width = 512, 768

            sem = F.interpolate(
                sem.unsqueeze(0), size=(resize_height, resize_width), mode="nearest"
            ).squeeze(0)

            start_x = (resize_width - 512) // 2
            start_y = (resize_height - 512) // 2
            sem = sem[:, start_y : start_y + 512, start_x : start_x + 512]

            frame_metadata = file_path_to_frame_metadata[target_id]
            c2w = np.array(frame_metadata["transform_matrix"], dtype=np.float32)

            c2w = P @ c2w @ P.T
            c2w = torch.tensor(c2w)

            intrinsics = K
            rgb_image = str(
                root_path / scene_id / "dslr" / "undistorted_images" / target_id
            )

            rgb_image = imread_cv2(rgb_image)
            _, _, intrinsics = crop_resize_if_necessary(
                rgb_image, rgb_image, intrinsics, [512, 512]
            )

            batches.append(
                {
                    "camera_intrinsics": torch.tensor(intrinsics)
                    .to(device)
                    .unsqueeze(0),
                    "camera_pose": c2w.to(device).unsqueeze(0),
                }
            )

        batch = {
            "context": [batches[0], batches[1]],
            "context_images": input_images,
            "target": [],
            "scene_id": scene_id,
        }

        all_batch.append(batch)

    model = gaussian_initialization.MAST3RGaussians.load_from_checkpoint(
        weights_path, device, strict=False
    )

    autoencoder_ckpt = torch.load(
        config.autoencoder_path,
        map_location=torch.device("cpu"),
    )["state_dict"]
    new_state_dict = {}
    for key, value in autoencoder_ckpt.items():
        new_key = key.replace("model.", "")
        new_state_dict[new_key] = value
    model.autoencoder.load_state_dict(new_state_dict)
    for param in model.parameters():
        param.requires_grad = False

    depth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    depth_model = depth_model.to(device)

    model.eval()
    model_vl = get_model(model.autoencoder,config)
    model_vl.eval()
    inpainter = get_inpainter_pipeline(config.pretrained_unet_name,config.controlnet_model_feat_name_or_path,config.pretrained_feat_model_name_or_path,config.pretrained_vae_feat_path)
    print("Model loaded successfully.")

    with open(
        config.semantic_label, "r", encoding="utf-8"
    ) as file:
        mapper = [line.strip() for line in file]

    os.makedirs(save_path, exist_ok=True)

    for batch in all_batch:

        inpainter[0].scheduler = DDIMScheduler.from_config(
            inpainter[0].scheduler.config
        )

        inpainter[1].scheduler = DDIMScheduler.from_config(
            inpainter[1].scheduler.config
        )

        input_images = batch["context_images"]
        get_reconstructed_scene(
            save_path,
            model,
            depth_model,
            model_vl,
            None,
            device,
            silent,
            image_size,
            input_images,
            batch,
            mapper,
            inpainter,
            config,
        )
