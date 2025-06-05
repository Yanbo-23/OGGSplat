#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, build_quaternion
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH

from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_quaternion
# from torch_cluster import radius_graph
# from pytorch3d.ops import ball_query
from scipy.spatial import ConvexHull

class GaussianModel_LangSplat:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)

            # symm = strip_symmetric(actual_covariance)
            # print(symm.shape)
            return actual_covariance
        
        def decompose_covariance_to_scaling_rotation(covariance):
            """
            从协方差矩阵反解 scaling 和 rotation
            """
            # torch.linalg.eigh 返回升序排列的特征值和对应特征向量
            eigvals, eigvecs = torch.linalg.eigh(covariance)  # eigvals: (N, 3), eigvecs: (N, 3, 3)

            # 计算 scaling（标准差），确保数值稳定
            scaling = torch.sqrt(eigvals.clamp(min=1e-8))  # 避免负值或0

            # rotation 的每一列为一个特征向量
            rotation = eigvecs

            rotation = build_quaternion(rotation)

            return scaling, rotation

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation
        self.inverse_covariance_activation = decompose_covariance_to_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._semantics = torch.empty(0)
        self._opacity = torch.empty(0)
        self._language_feature = None
        

        # self._xyz_frozen = torch.empty(0)
        # self._features_dc_frozen = torch.empty(0)
        # self._features_rest_frozen = torch.empty(0)
        # self._scaling_frozen = torch.empty(0)
        # self._rotation_frozen = torch.empty(0)
        # self._opacity_frozen = torch.empty(0)
        # self._language_feature_frozen = None

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 1
        self.setup_functions()

    def capture(self, include_feature=False):
        if include_feature:
            assert self._semantics is not None, "没有设置language feature"
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self._semantics,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )
        else:
            return (
                self.active_sh_degree,
                self._xyz,
                self._features_dc,
                self._features_rest,
                self._scaling,
                self._rotation,
                self._opacity,
                self.max_radii2D,
                self.xyz_gradient_accum,
                self.denom,
                self.optimizer.state_dict(),
                self.spatial_lr_scale,
            )            
    
    def restore(self, model_args, training_args,need_remove_outlier=True, mode='train'):
        if len(model_args) == 13: # 这是一个feature训练时保存的ckpt
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self._semantics,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
        elif len(model_args) == 12: # 这是一个不训练feature保存的ckpt
            (self.active_sh_degree, 
            self._xyz, 
            self._features_dc, 
            self._features_rest,
            self._scaling, 
            self._rotation, 
            self._opacity,
            self.max_radii2D, 
            xyz_gradient_accum, 
            denom,
            opt_dict, 
            self.spatial_lr_scale) = model_args
            if not training_args.include_feature: # 如果是以原始gs为初始化来训练feature的话，就不需要restore optimizer
                self.optimizer.load_state_dict(opt_dict)

        elif len(model_args) == 6:
            (self._xyz, 
            self._features_dc, 
            self._scaling,
            self._rotation,
            self._opacity,
            self._semantics) = model_args 
            self._semantics = torch.zeros((self._semantics.shape[0],3)).to(self._semantics.device)


            fused_point_cloud = torch.tensor(self._xyz).float().cuda()
            fused_color = RGB2SH(torch.tensor(self._xyz).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0 ] = fused_color
            features[:, 3:, 1:] = 0.0

            print("Number of points at initialisation : ", fused_point_cloud.shape[0])

            dist2 = torch.clamp_min(distCUDA2(self._xyz.float().cuda()), 0.0000001)
            scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
            rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
            rots[:, 0] = 1

            opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
            # language_feature = torch.zeros((fused_point_cloud.shape[0], 512), device="cuda")
            opacities = opacities.squeeze(-1)
            self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
            # self._xyz = nn.Parameter(self._xyz.clone().detach().requires_grad_(True))
            # print(self._features_dc.shape)
            self._features_dc = nn.Parameter(features[:,:,0:1].contiguous().requires_grad_(True))
            # print(self._features_dc.shape)
            # awdwa
            # self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
            self._scaling = nn.Parameter(scales.requires_grad_(True))
            self._rotation = nn.Parameter(rots.requires_grad_(True))
            self._opacity = nn.Parameter(opacities.requires_grad_(True))
            # self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
            # 在从pointcloud初始化的时候是再训练原始gs的时候，这个时候不需要进行feature的初始化
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            
            # self._features_rest = self._features_rest.to(self._features_dc.device)

            # self._opacity = self.inverse_opacity_activation(self._opacity)

            # self._scaling = self.scaling_inverse_activation(self._scaling)

            # self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")


            # self._xyz = nn.Parameter(self._xyz.clone().detach().requires_grad_(True))
            self._features_dc = nn.Parameter(self._features_dc.clone().detach().requires_grad_(True))
            
            # self._scaling = nn.Parameter(self._scaling.clone().detach().requires_grad_(True))
            # self._rotation = nn.Parameter(self._rotation.clone().detach().requires_grad_(True))
            # self._opacity = nn.Parameter(self._opacity.clone().detach().requires_grad_(True))
            self._semantics = nn.Parameter(self._semantics.clone().detach().requires_grad_(True))
            

        

        if mode == 'train':
            self.training_setup(training_args)
            # 
            if len(model_args) != 6:
                self.xyz_gradient_accum = xyz_gradient_accum
                self.denom = denom
    @property
    def get_scaling_frozen(self):

        return self.scaling_activation(self._scaling_frozen)
    
    @property
    def get_rotation_frozen(self):
        return self.rotation_activation(self._rotation_frozen)
    
    @property
    def get_xyz_frozen(self):
        return self._xyz_frozen
    
    @property
    def get_features_frozen(self):
        features_dc = self._features_dc_frozen
        features_rest = self._features_rest_frozen
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity_frozen(self):
        return self.opacity_activation(self._opacity_frozen)





    @property
    def get_scaling(self):

        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        # features_rest = self._features_rest
        # return torch.cat((features_dc), dim=1)
        return features_dc
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_language_feature(self):
        if self._language_feature is not None:
            return self._language_feature
        else:
            raise ValueError('没有设置language feature')
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1


    def remove_outlier_gaussians_raddi(self,raddi):
    
        # raddi = 

        xyz = self._xyz.contiguous()  # Nx3 tensor

        _, idx, nn = ball_query(xyz.unsqueeze(0), xyz.unsqueeze(0), K=max_neighbors, radius=r, return_nn=True)
        idx = idx.squeeze(0)
        nn = nn.squeeze(0)
        scalings = torch.max(self.get_scaling, dim=1).values

        max_scale =  torch.quantile(scalings, 0.85)


        valid_neighbors_count = (idx != -1).sum(dim=1)



        valid_counts = torch.clamp(valid_neighbors_count, min=1)

        nn_mean = nn.sum(dim=1, keepdim=False) / valid_counts.unsqueeze(1)  # 形状为 (N, P1, 1, D)

        dist = torch.norm(nn_mean - xyz, dim=1)






        mask = torch.logical_and( valid_neighbors_count < max_neighbors, scalings > max_scale)
        print('Points num been removed:', torch.sum(mask),'0')
        # mask = torch.logical_and( mask, mask_dist)


        if radii is not None:
            max_radii = torch.quantile(radii.float(), 0.95)
            mask = torch.logical_and(mask, radii>max_radii)


        

        print('Points num been removed:', torch.sum(mask))

        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._semantics = optimizable_tensors["semantics"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return valid_points_mask

    def copy_edge_gaussians(self, idx, N=5):
        
        opacity = self._opacity
        mask = torch.zeros_like(opacity)
        mask[idx] = 1
        selected_pts_mask = (mask==1)


        print("Clone Edge Points Num: ", torch.sum(selected_pts_mask))
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest
        new_opacities = self._opacity[selected_pts_mask]
        # new_language_feature = self._language_feature[selected_pts_mask]
        new_language_feature = self._semantics[selected_pts_mask]


        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_language_feature)
    def crop_all(self):
        
        N = len(self._xyz)
        # 创建一半True一半False的列表

        num_true = N // 4
        
        num_false = N - num_true  # 处理奇数N时False稍多

        mask = torch.tensor([True] * num_true + [False] * num_false)

        valid_points_mask = mask[torch.randperm(N)].to(self._xyz.device)

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._semantics = optimizable_tensors["semantics"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]


        # self._xyz = self._xyz[valid_points_mask]
        # self._features_dc = self._features_dc[valid_points_mask]
        # # self._features_rest = self._features_rest[valid_points_mask]
        # self._opacity = self._opacity[valid_points_mask]
        # self._semantics = self._semantics[valid_points_mask]
        # self._scaling = self._scaling[valid_points_mask]
        # self._rotation = self._rotation[valid_points_mask]


        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        torch.cuda.empty_cache()
    def init_new_gaussians(self, points):
        
        device = self._xyz.device
        print('Start Init...')
        new_xyz = torch.tensor(points, dtype=torch.float32).to(device)

        N = new_xyz.shape[0]  # 新高斯的数量

        # # 初始化为全0，保持形状一致
        # new_features_dc = torch.zeros((N, *self._features_dc.shape[1:]), device=self._features_dc.device, dtype=self._features_dc.dtype)
        # new_features_rest = torch.zeros((N, *self._features_rest.shape[1:]), device=self._features_rest.device, dtype=self._features_rest.dtype)
        # new_opacities = torch.zeros((N, *self._opacity.shape[1:]), device=self._opacity.device, dtype=self._opacity.dtype)
        # new_language_feature = torch.zeros((N, *self._semantics.shape[1:]), device=self._semantics.device, dtype=self._semantics.dtype)
        # new_scaling = torch.zeros((N, *self._scaling.shape[1:]), device=self._scaling.device, dtype=self._scaling.dtype)
        # new_rotation = torch.zeros((N, *self._rotation.shape[1:]), device=self._rotation.device, dtype=self._rotation.dtype)


        # new_xyz = self._xyz[:N]
        new_features_dc = self._features_dc[:N ]
        new_features_rest = self._features_rest
        new_opacities = self._opacity[:N]
        # new_language_feature = self._language_feature[selected_pts_mask]
        new_language_feature = self._semantics[:N]


        new_scaling = self._scaling[:N]
        new_rotation = self._rotation[:N]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_language_feature)
    # def init_new_gaussians(self, points):
    #     device = self._xyz.device
    #     print('Start Init...')
    #     new_xyz = torch.tensor(points, dtype=torch.float32).to(device)
    #     N = new_xyz.shape[0]  # 新高斯的数量

    #     # --- 合理初始化各参数 ---

    #     # 1. DC SH coefficients 初始化为灰色环境光（常用 0.5 或 0.7）
    #     new_features_dc = torch.full(
    #         (N, *self._features_dc.shape[1:]), 
    #         0.5,  # 可选：使用 0.7 以提高亮度
    #         device=self._features_dc.device,
    #         dtype=self._features_dc.dtype
    #     )

    #     # 2. 其余 SH 系数为 0（表示没有方向性）
    #     new_features_rest = torch.zeros(
    #         (N, *self._features_rest.shape[1:]), 
    #         device=self._features_rest.device, 
    #         dtype=self._features_rest.dtype
    #     )

    #     # 3. 不要设为 0，设为小值如 0.1
    #     new_opacities = torch.full(
    #         (N, *self._opacity.shape[1:]), 
    #         0.1, 
    #         device=self._opacity.device, 
    #         dtype=self._opacity.dtype
    #     )

    #     # 4. 如果 language feature 没用，可以设为 0；否则设为均值向量或正态分布随机值
    #     new_language_feature = torch.zeros(
    #         (N, *self._semantics.shape[1:]), 
    #         device=self._semantics.device, 
    #         dtype=self._semantics.dtype
    #     )

    #     # 5. 初始缩放不能为 0，推荐小常数，如 0.01（各向同性或加点随机扰动）
    #     new_scaling = torch.full(
    #         (N, *self._scaling.shape[1:]), 
    #         0.01, 
    #         device=self._scaling.device, 
    #         dtype=self._scaling.dtype
    #     )

    #     # 6. 四元数初始设为单位四元数 [1, 0, 0, 0]，表示无旋转
    #     new_rotation = torch.zeros(
    #         (N, *self._rotation.shape[1:]), 
    #         device=self._rotation.device, 
    #         dtype=self._rotation.dtype
    #     )
    #     new_rotation[:, 0] = 1.0  # w=1, x=y=z=0

    #     self.new_xyz = nn.Parameter(new_xyz)
    #     self.new_features_dc = nn.Parameter(new_features_dc)
    #     self.new_features_rest = nn.Parameter(new_features_rest)
    #     self.new_opacity = nn.Parameter(new_opacities)
    #     self.new_scaling = nn.Parameter(new_scaling)
    #     self.new_rotation = nn.Parameter(new_rotation)
    #     self.new_semantics = nn.Parameter(new_language_feature)

    #     # --- 调用 densification 函数 ---
    #     self.densification_postfix(
    #         new_xyz, 
    #         new_features_dc, 
    #         new_features_rest, 
    #         new_opacities, 
    #         new_scaling, 
    #         new_rotation, 
    #         new_language_feature
    #     )




    def remove_outlier_gaussians_ballquery(self, r, max_neighbors=5,radii=None):
        


        xyz = self._xyz.contiguous()  # Nx3 tensor

        _, idx, nn = ball_query(xyz.unsqueeze(0), xyz.unsqueeze(0), K=max_neighbors, radius=r, return_nn=True)
        idx = idx.squeeze(0)
        nn = nn.squeeze(0)
        scalings = torch.max(self.get_scaling, dim=1).values



        valid_neighbors_count = (idx != -1).sum(dim=1)



        valid_counts = torch.clamp(valid_neighbors_count, min=1)

        nn_mean = nn.sum(dim=1, keepdim=False) / valid_counts.unsqueeze(1)  # 形状为 (N, P1, 1, D)


        
        # mask_dist = torch.logical_or(torch.logical_and((dist > 0.03),valid_counts>=5),valid_counts<5)

        threshold = torch.quantile(scalings, 0.5)

        mask = torch.logical_and( valid_neighbors_count < max_neighbors, scalings < threshold)
        # mask = valid_neighbors_count < max_neighbors
        print('Points num been removed:', torch.sum(mask))
        # mask = torch.logical_and( mask, mask_dist)


        if radii is not None:
            max_radii = torch.quantile(radii.float(), 0.95)
            mask = torch.logical_and(mask, radii>max_radii)


        

        print('Points num been removed:', torch.sum(mask))

        valid_points_mask = ~mask

        # optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._semantics = optimizable_tensors["semantics"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]

        self._xyz = self._xyz[valid_points_mask]
        self._features_dc = self._features_dc[valid_points_mask]
        # self._features_rest = self._features_rest[valid_points_mask]
        self._opacity = self._opacity[valid_points_mask]
        self._semantics = self._semantics[valid_points_mask]
        self._scaling = self._scaling[valid_points_mask]
        self._rotation = self._rotation[valid_points_mask]


        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return valid_points_mask


    def remove_outlier_gaussians(self, radius=0.001, max_scale=0.03):
        """
        使用 distCUDA2 移除离群 Gaussians。

        参数:
            radius (float): 半径范围内才认为是邻居。
            min_neighbors (int): 小于这个邻居数的点被视为离群点。
        """
        xyz = self._xyz.contiguous()  # Nx3 tensor
        N = xyz.shape[0]

        scalings = torch.max(self.get_scaling, dim=1).values


        neighbors_np = distCUDA2(xyz) 
        mask = torch.logical_and( neighbors_np > radius, scalings > max_scale)
        print('Points num been removed:', torch.sum(mask))
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._semantics = optimizable_tensors["semantics"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        return valid_points_mask


    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))
        # language_feature = torch.zeros((fused_point_cloud.shape[0], 512), device="cuda")
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        # self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
        # 在从pointcloud初始化的时候是再训练原始gs的时候，这个时候不需要进行feature的初始化
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        
        if training_args.include_feature:
            if self._language_feature is None or self._language_feature.shape[0] != self._xyz.shape[0]:
                # 开始feature训练的时候，往模型中加入language feature参数
                language_feature = torch.zeros((self._xyz.shape[0], 3), device="cuda")
                self._language_feature = nn.Parameter(language_feature.requires_grad_(True))
                
            l = [
                {'params': [self._language_feature], 'lr': training_args.language_feature_lr, "name": "language_feature"}, # TODO: training_args.language_feature_lr
            ]
            self._xyz.requires_grad_(True)
            self._features_dc.requires_grad_(False)
            self._features_rest.requires_grad_(False)
            self._scaling.requires_grad_(False)
            self._rotation.requires_grad_(False)
            self._opacity.requires_grad_(False)
        else:
            mul_scale = 0.01
            l = [
                {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
                {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
                {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
                {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
                {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
                {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
                {'params': [self._semantics], 'lr': training_args.language_feature_lr, "name": "semantics"},
                # # Origianl Gaussian Part
                # {'params': [self._xyz_frozen], 'lr': training_args.position_lr_init * self.spatial_lr_scale * mul_scale, "name": "xyz"},
                # {'params': [self._features_dc_frozen], 'lr': training_args.feature_lr * mul_scale, "name": "f_dc"},
                # {'params': [self._features_rest_frozen], 'lr': training_args.feature_lr / 20.0 * mul_scale, "name": "f_rest"},
                # {'params': [self._opacity_frozen], 'lr': training_args.opacity_lr * mul_scale, "name": "opacity"},
                # {'params': [self._scaling_frozen], 'lr': training_args.scaling_lr * mul_scale, "name": "scaling"},
                # {'params': [self._rotation_frozen], 'lr': training_args.rotation_lr * mul_scale, "name": "rotation"},
            ]
            # assert self._language_feature is None, "在训练原始gs的时候language feature应该始终为None"




        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def set_scale_location_lr(self,ratio,opt):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = opt.position_lr_init*ratio
            if param_group["name"] == "scale":
                param_group['lr'] = opt.scaling_lr*ratio


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def reset_learning_rate(self,opt):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                param_group['lr'] = opt.position_lr_init
            if param_group["name"] == "scale":
                param_group['lr'] = opt.scaling_lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        if self._features_rest.numel() > 0:
            for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
                l.append('f_rest_{}'.format(i))
        l.append('opacity')
        # l.append('language_feature')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        if self._features_rest.numel() > 0:
            f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  
        else:
            f_rest = np.array([])
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        if self._features_rest.numel() > 0:
            attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        else:
            opacities = np.expand_dims(opacities, axis=1)
            attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01)) #0.35
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_(self):


        opacity = self.get_opacity
        mask = (opacity < 0.2)

        opacity[mask] = 0

        optimizable_tensors = self.replace_tensor_to_optimizer(opacity, "opacity")
        self._opacity = optimizable_tensors["opacity"]
        
        # awdaw


        # opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        # optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        # self._opacity = optimizable_tensors["opacity"]


    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        print(self._xyz.shape[0])
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)

                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        # self._language_feature = optimizable_tensors["language_feature"]
        self._semantics = optimizable_tensors["semantics"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}

        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                print(group["params"][0])
                print(group["params"][0].shape,extension_tensor.shape)
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation,new_language_feature):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "semantics": new_language_feature,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        print(new_xyz.shape,new_features_dc.shape,new_features_rest.shape,new_opacities.shape,new_language_feature.shape,new_scaling.shape,new_rotation.shape)
        self._xyz = optimizable_tensors["xyz"]
        # print(self._xyz.shape[0])
        self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._semantics = optimizable_tensors["semantics"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest.repeat(N,1,1)
        # new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N)
        new_language_feature = self._semantics[selected_pts_mask].repeat(N,1)

        print("Split Points Num: ", torch.sum(selected_pts_mask))

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_language_feature)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        print("Clone Points Num: ", torch.sum(selected_pts_mask))
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest
        new_opacities = self._opacity[selected_pts_mask]
        # new_language_feature = self._language_feature[selected_pts_mask]
        new_language_feature = self._semantics[selected_pts_mask]


        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_language_feature)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        print('Prune Points Num: ', torch.sum(prune_mask))
        self.prune_points(prune_mask)
        print("Num points: ", self._xyz.shape)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):

        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1