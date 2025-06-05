import torch
from einops import rearrange, repeat

from .cuda_splatting import render_cuda,render_cuda_
from utils.geometry import normalize_intrinsics
import utils.geometry as geometry
import math
class DecoderSplattingCUDA(torch.nn.Module):

    def __init__(self, background_color):
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )
    


    def generate_camera_poses(self, center, radius, num_views=20, height=0.0):
        poses = []
        for i in range(num_views):
            theta = 2 * math.pi * i / num_views
            cam_position = center + torch.tensor([
                radius * math.cos(theta),
                height,
                radius * math.sin(theta)
            ])
            forward = (center - cam_position)
            forward = forward / forward.norm()

            # Create right and up vectors (basic look-at construction)
            up = torch.tensor([0.0, 1.0, 0.0])
            right = torch.cross(up, forward)
            right = right / right.norm()
            up = torch.cross(forward, right)

            # Construct rotation matrix
            R = torch.stack([right, up, forward], dim=1)  # [3, 3]
            T = cam_position.view(3, 1)
            extrinsic = torch.eye(4)
            extrinsic[:3, :3] = R
            extrinsic[:3, 3] = T.squeeze()
            poses.append(extrinsic)
        return torch.stack(poses)  # [num_views, 4, 4]



    def get_scene_stats(self,means):
        """
        获取场景的中心和半径
        """
        center = means.mean(dim=0)  # 场景中心
        bbox_min = means.min(dim=0)[0]
        bbox_max = means.max(dim=0)[0]
        radius = (bbox_max - bbox_min).norm() / 2
        return center, radius

    def look_at_matrix(self, eye, center, up=torch.tensor([0.0, 1.0, 0.0])):
        """
        构建 LookAt 相机外参矩阵（world-to-camera）
        """
        forward = (center - eye)
        forward = forward / forward.norm()

        right = torch.cross(up, forward)
        right = right / right.norm()
        true_up = torch.cross(forward, right)

        R = torch.stack([right, true_up, forward], dim=1)  # 3x3
        T = -R.T @ eye  # 平移部分

        extrinsic = torch.eye(4)
        extrinsic[:3, :3] = R.T  # 注意是 world-to-camera 方向，需要转置
        extrinsic[:3, 3] = T
        return extrinsic

    def generate_orbit_camera_poses(self, center, radius, num_views=20, height_offset=0.0):
        poses = []
        for i in range(num_views):
            theta = 2 * math.pi * i / num_views
            cam_position = center + torch.tensor([
                radius * math.cos(theta),
                height_offset,
                radius * math.sin(theta)
            ])
            pose = self.look_at_matrix(cam_position, center)
            poses.append(pose)
        return torch.stack(poses)  # [num_views, 4, 4]

    def get_default_intrinsics(self, W=800, H=800, fov_degrees=50.0):
        """
        构建简化版 pinhole 模型的内参矩阵（支持4x4）
        """
        f = 0.5 * W / math.tan(0.5 * math.radians(fov_degrees))
        K = torch.tensor([
            [f, 0, W / 2, 0],
            [0, f, H / 2, 0],
            [0, 0,    1, 0],
            [0, 0,    0, 1]
        ])
        return K

    def simulate_camera_parameters_from_gaussians(self, means, num_views=20, image_size=(800, 800), fov=50.0):
        center, radius = self.get_scene_stats(means)
        extrinsics = self.generate_orbit_camera_poses(center, radius, num_views)
        intrinsics = torch.stack([self.get_default_intrinsics(*image_size, fov) for _ in range(num_views)])
        return extrinsics, intrinsics



    def apply_local_transform(self, T, trans_xyz=(0,0,0), rot_deg_xyz=(0,0,0)):


        def rot_x(angle):  # deg
            rad = math.radians(angle)
            c, s = math.cos(rad), math.sin(rad)
            return torch.tensor([[1,0,0],[0,c,-s],[0,s,c]])

        def rot_y(angle):
            rad = math.radians(angle)
            c, s = math.cos(rad), math.sin(rad)
            return torch.tensor([[c,0,s],[0,1,0],[-s,0,c]])

        def rot_z(angle):
            rad = math.radians(angle)
            c, s = math.cos(rad), math.sin(rad)
            return torch.tensor([[c,-s,0],[s,c,0],[0,0,1]])

        Rx = rot_x(rot_deg_xyz[0])
        Ry = rot_y(rot_deg_xyz[1])
        Rz = rot_z(rot_deg_xyz[2])
        R = Rz @ Ry @ Rx

        delta_T = torch.eye(4)
        delta_T[:3, :3] = R
        delta_T[:3, 3] = torch.tensor(trans_xyz)

        return T @ delta_T.to(T.device)


    def generate_symmetric_rotations(self, num_samples: int, dim: int):
        """
        生成绕 X 轴旋转的对称角度列表，范围为 ±90°
        
        Args:
            num_samples (int): 生成的总样本数，必须是偶数

        Returns:
            List[Tuple[float, float, float]]: 每个样本的旋转角度 (X_deg, Y_deg, Z_deg)
        """
        assert num_samples % 2 == 0, "num_samples 必须是偶数，用于构造对称角度对"

        half = num_samples // 2  # 一共多少对
        degrees = torch.linspace(0, 90, steps=half + 1)[1:]  # 去除0°
        
        rot_list = []
        for deg in degrees:
            rot = [0.0,0.0,0.0]
            neg_rot = [0.0,0.0,0.0]
            rot[dim] = float(deg)
            neg_rot[dim] = -float(deg)

            rot_list.append( tuple(rot) )   
            rot_list.append( tuple(neg_rot) ) 
            
        return rot_list


    def gaussian_forward(self,batch,gs,image_shape):
        
        # batch['target'] = batch['target'][:2]

        base_pose = batch['context'][0]['camera_pose'] # [b, 4, 4]
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
        intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
        # print(intrinsics)
        # awdaw
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]

        # Rotate the ground truth extrinsics into the coordinate system used by MAST3R
        # --i.e. in the coordinate system of the first context view, normalized by the scene scale



        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics
        # print(torch.broadcast_shapes(inv_base_pose[:, None, :, :].shape, extrinsics.shape))
        

        means = gs.get_xyz
        scales =  gs.get_scaling
        rotation = gs.get_rotation
        semantics = gs._semantics
        harmonics = gs.get_features
        opacities = gs.get_opacity


        # means_frozen = gs.get_xyz_frozen
        # scales_frozen =  gs.get_scaling_frozen
        # rotation_frozen = gs.get_rotation_frozen
        # semantics_frozen = gs._semantics_frozen
        # harmonics_frozen = gs.get_features_frozen
        # opacities_frozen = gs.get_opacity_frozen

        # means = torch.cat([means,means_frozen])
        # scales = torch.cat([scales,scales_frozen])
        # rotation = torch.cat([rotation,rotation_frozen])
        # semantics = torch.cat([semantics,semantics_frozen])
        # harmonics = torch.cat([harmonics,harmonics_frozen])
        # opacities = torch.cat([opacities,opacities_frozen])


        covariances = geometry.build_covariance(scales,rotation)


        # print(opacities.shape,semantics.shape)
        # awdwa
        b, v, _, _ = extrinsics.shape





        # rots = self.generate_symmetric_rotations(20,dim=1)
        # for i in range(b):
        #     for j in range(v):
        #         extrinsics[i][j] =  self.apply_local_transform(extrinsics[i][j], trans_xyz=(0.0,0.0,0.0), rot_deg_xyz=rots[j])
        


        near = torch.full((b, v), 0.1, device=means.device)
        far = torch.full((b, v), 1000.0, device=means.device)

        # print(covariances.shape)
        # covariances = rearrange(covariances, "b v h w i j -> b (v h w) i j")
        # print(covariances.shape)
        # covariances = repeat(covariances, "b g i j -> (b v) g i j", v=v)
        # print(covariances.shape)



        # extrinsics = self.generate_camera_poses(center.detach().cpu(),radius.detach().cpu())

        # extrinsics, _ = self.simulate_camera_parameters_from_gaussians(means.detach().cpu(), num_views=20)
        # extrinsics = extrinsics.unsqueeze(0).to(near.device)
        # extrinsics = inv_base_pose[:, None, :, :] @ extrinsics
        # print(intrinsics)
        # dawda
        # intrinsics = intrinsics.unsqueeze(0).to(near.device)
        # intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]

        # print(means.unsqueeze(0))
        # awdaw

        color, sems, depth, alpha, contri_idx, contri_num, radii, means2d = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(means.unsqueeze(0), "b g xyz -> (b v) g xyz", v=v),
            repeat(covariances.unsqueeze(0), "b g i j -> (b v) g i j", v=v),
            repeat(harmonics.unsqueeze(0), "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(opacities.unsqueeze(0), "b g -> (b v) g", v=v),
            repeat(semantics.unsqueeze(0), "b g s -> (b v) g s", v=v)
        )

        contri_idx = rearrange(contri_idx, "(b v) c h w -> b v c h w", b=b, v=v)
        contri_num = rearrange(contri_num, "(b v) c h w -> b v c h w", b=b, v=v)
        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        depth = rearrange(depth, "(b v) c h w -> b v c h w", b=b, v=v)
        alpha = rearrange(alpha, "(b v) c h w -> b v c h w", b=b, v=v)
        if sems.shape[-1] != 1:
            sems = rearrange(sems, "(b v) c h w -> b v c h w", b=b, v=v)
        visibility_filter = radii>0
        
        return color, visibility_filter, radii, means2d, sems, depth, alpha, contri_idx, contri_num



    def forward(self, batch, pred1, pred2, image_shape):

        base_pose = batch['context'][0]['camera_pose'] # [b, 4, 4]
        inv_base_pose = torch.inverse(base_pose)

        extrinsics = torch.stack([target_view['camera_pose'] for target_view in batch['target']], dim=1)
        intrinsics = torch.stack([target_view['camera_intrinsics'] for target_view in batch['target']], dim=1)
        intrinsics = normalize_intrinsics(intrinsics, image_shape)[..., :3, :3]

        # Rotate the ground truth extrinsics into the coordinate system used by MAST3R
        # --i.e. in the coordinate system of the first context view, normalized by the scene scale
        extrinsics = inv_base_pose[:, None, :, :] @ extrinsics

        means = torch.stack([pred1["means"], pred2["means_in_other_view"]], dim=1)
        covariances = torch.stack([pred1["covariances"], pred2["covariances"]], dim=1)
        harmonics = torch.stack([pred1["sh"], pred2["sh"]], dim=1)

        opacities = torch.stack([pred1["opacities"], pred2["opacities"]], dim=1)
        semantics = torch.stack([pred1["sem"], pred2["sem"]], dim=1)
        # print(opacities.shape,semantics.shape)
        # awdwa
        b, v, _, _ = extrinsics.shape
        near = torch.full((b, v), 0.1, device=means.device)
        far = torch.full((b, v), 1000.0, device=means.device)

        # print(covariances.shape)
        # covariances = rearrange(covariances, "b v h w i j -> b (v h w) i j")
        # print(covariances.shape)
        # covariances = repeat(covariances, "b g i j -> (b v) g i j", v=v)
        # print(covariances.shape)



        color, sems = render_cuda(
            rearrange(extrinsics, "b v i j -> (b v) i j"),
            rearrange(intrinsics, "b v i j -> (b v) i j"),
            rearrange(near, "b v -> (b v)"),
            rearrange(far, "b v -> (b v)"),
            image_shape,
            repeat(self.background_color, "c -> (b v) c", b=b, v=v),
            repeat(rearrange(means, "b v h w xyz -> b (v h w) xyz"), "b g xyz -> (b v) g xyz", v=v),
            repeat(rearrange(covariances, "b v h w i j -> b (v h w) i j"), "b g i j -> (b v) g i j", v=v),
            repeat(rearrange(harmonics, "b v h w c d_sh -> b (v h w) c d_sh"), "b g c d_sh -> (b v) g c d_sh", v=v),
            repeat(rearrange(opacities, "b v h w 1 -> b (v h w)"), "b g -> (b v) g", v=v),
            repeat(rearrange(semantics, "b v h w s -> b (v h w) s"), "b g s -> (b v) g s", v=v)
        )

        color = rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)
        sems = rearrange(sems, "(b v) c h w -> b v c h w", b=b, v=v)
        return color, None, sems