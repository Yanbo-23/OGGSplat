import json
import os
import sys
import einops
import lightning as L
import lpips
import omegaconf
import torch
import wandb
import os

sys.path.append("src/pixelsplat_src")
sys.path.append("src/mast3r_src")
sys.path.append("src/mast3r_src/dust3r")
from src.mast3r_src.dust3r.dust3r.losses import L21
from src.mast3r_src.mast3r.losses import ConfLoss, Regr3D
import data.scannetpp.scannetpp as scannetpp
import src.mast3r_src.mast3r.model as mast3r_model
from src.mast3r_src.mast3r.fast_nn import fast_reciprocal_NNs
import src.pixelsplat_src.benchmarker as benchmarker
import src.pixelsplat_src.decoder_splatting_cuda as pixelsplat_decoder
import utils.compute_ssim as compute_ssim
import utils.export as export
import utils.geometry as geometry
import utils.loss_mask as loss_mask
import utils.sh_utils as sh_utils
import torch.nn.functional as F
import workspace
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )


def cal_consistency_loss(gs_sem, sem_gt):

    all_idx = torch.unique(sem_gt)

    feat_mean_stack = []
    mask_feat_loss = []

    for idx in all_idx:

        if idx == -100 and len(all_idx) > 1:
            continue
        gs_sem_this = gs_sem[sem_gt == idx]
        mean_feat = gs_sem_this.mean(dim=0)
        feat_mean_stack.append(mean_feat)
        mean_feat = mean_feat.unsqueeze(0).expand(gs_sem_this.shape[0], -1)

        cos_sim = F.cosine_similarity(gs_sem_this, mean_feat.detach(), dim=1)
        cosine_similarity_loss = 1 - cos_sim

        dist = cosine_similarity_loss.mean()

        mask_feat_loss.append(dist)

    loss_cohesion = sum(mask_feat_loss) / len(mask_feat_loss)

    feat_mean_stack = torch.stack(feat_mean_stack)

    return loss_cohesion


def l2_distance_loss_min(X):

    N = X.size(0)

    X_expanded = X.unsqueeze(0) - X.unsqueeze(1)

    distances = torch.norm(X_expanded, p=2, dim=2)

    mask = torch.eye(N, device=X.device)
    distances = distances * (1 - mask)

    positive_distances = distances[distances > 0]
    if positive_distances.numel() > 0:
        min_distance = torch.min(positive_distances)
        result = 1 / (min_distance * 20)

    else:

        result = distances[0][0] * 0

    return result


def cosine_similarity_loss(X, Y, margin=-0.5, lambda_weight=0.5):

    similarity_matrix = F.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=-1)

    diagonal_similarity = torch.diag(similarity_matrix)
    similarity_loss = 1 - diagonal_similarity.mean()

    if similarity_matrix.shape[0] > 1:
        off_diagonal_mask = ~torch.eye(
            similarity_matrix.size(0), dtype=bool, device=X.device
        )
        off_diagonal_similarity = similarity_matrix[off_diagonal_mask]

        dissimilarity_loss = F.relu(off_diagonal_similarity + margin).mean()

        total_loss = similarity_loss + lambda_weight * dissimilarity_loss
    else:
        total_loss = similarity_loss
    return total_loss


class MAST3RGaussians(L.LightningModule):

    def __init__(self, config):

        super().__init__()

        self.config = config
        self.loss_mse = []
        self.loss_lpips = []
        self.loss_sim = []
        self.loss_pixel_sem = []
        self.loss_mast3r = []
        self.loss_consistency = []

        self.encoder = mast3r_model.AsymmetricMASt3R(
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            head_type="gaussian_head",
            output_mode="pts3d+gaussian+desc24",
            depth_mode=("exp", -mast3r_model.inf, mast3r_model.inf),
            conf_mode=("exp", 1, mast3r_model.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            use_offsets=config.use_offsets,
            sh_degree=config.sh_degree if hasattr(config, "sh_degree") else 1,
        )
        if self.config.pretrained_path is not None:
            self.encoder.requires_grad_(True)
        else:
            self.encoder.requires_grad_(False)

        self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head1.gaussian_sem_dpt.dpt.requires_grad_(True)
        self.encoder.downstream_head2.gaussian_sem_dpt.dpt.requires_grad_(True)

        self.decoder = pixelsplat_decoder.DecoderSplattingCUDA(
            background_color=[0.0, 0.0, 0.0]
        )

        self.benchmarker = benchmarker.Benchmarker()

        if config.loss.average_over_mask:
            self.lpips_criterion = lpips.LPIPS("vgg", spatial=True)
        else:
            self.lpips_criterion = lpips.LPIPS("vgg")

        if config.loss.mast3r_loss_weight is not None:
            self.mast3r_criterion = ConfLoss(
                Regr3D(L21, norm_mode="?avg_dis"), alpha=0.2
            )
            self.encoder.downstream_head1.requires_grad_(True)
            self.encoder.downstream_head2.requires_grad_(True)

        self.save_hyperparameters()

        self.autoencoder = Autoencoder(input_dim=256, latent_dim=16)

    def forward(self, view1, view2):

        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = (
                self.encoder._encode_symmetrized(view1, view2)
            )
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        pred1 = self.encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)

        pred1["covariances"] = geometry.build_covariance(
            pred1["scales"], pred1["rotations"]
        )
        pred2["covariances"] = geometry.build_covariance(
            pred2["scales"], pred2["rotations"]
        )

        learn_residual = True
        if learn_residual:
            new_sh1 = torch.zeros_like(pred1["sh"])
            new_sh2 = torch.zeros_like(pred2["sh"])
            new_sh1[..., 0] = sh_utils.RGB2SH(
                einops.rearrange(view1["original_img"], "b c h w -> b h w c")
            )
            new_sh2[..., 0] = sh_utils.RGB2SH(
                einops.rearrange(view2["original_img"], "b c h w -> b h w c")
            )
            pred1["sh"] = pred1["sh"] + new_sh1
            pred2["sh"] = pred2["sh"] + new_sh2

        pred2["pts3d_in_other_view"] = pred2.pop("pts3d")
        pred2["means_in_other_view"] = pred2.pop("means")

        return pred1, pred2

    def training_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape

        view1, view2 = batch["context"]

        pred1, pred2 = self.forward(view1, view2)
        matches_im0s = None
        matches_im1s = None

        if self.config.use_desc:
            matches_im0s = []
            matches_im1s = []
            desc1s, desc2s = pred1["desc"].detach(), pred2["desc"].detach()

            for desc1, desc2 in zip(desc1s, desc2s):

                matches_im0, matches_im1 = fast_reciprocal_NNs(
                    desc1,
                    desc2,
                    subsample_or_initxy1=8,
                    device=desc1.device,
                    dist="dot",
                    block_size=2**13,
                )

                H0, W0 = h, w
                valid_matches_im0 = (
                    (matches_im0[:, 0] >= 3)
                    & (matches_im0[:, 0] < int(W0) - 3)
                    & (matches_im0[:, 1] >= 3)
                    & (matches_im0[:, 1] < int(H0) - 3)
                )

                H1, W1 = h, w
                valid_matches_im1 = (
                    (matches_im1[:, 0] >= 3)
                    & (matches_im1[:, 0] < int(W1) - 3)
                    & (matches_im1[:, 1] >= 3)
                    & (matches_im1[:, 1] < int(H1) - 3)
                )

                valid_matches = valid_matches_im0 & valid_matches_im1
                matches_im0, matches_im1 = (
                    matches_im0[valid_matches],
                    matches_im1[valid_matches],
                )
                matches_im0s.append(matches_im0)
                matches_im1s.append(matches_im1)

        color, _, sems = self.decoder(batch, pred1, pred2, (h, w))

        feats = torch.stack(
            [target_view["sem_feat"] for target_view in batch["target"]], dim=1
        )
        feats_path = [target_view["feat_path"] for target_view in batch["target"]]

        feats = feats.view(-1, feats.shape[2], feats.shape[3], feats.shape[4])

        scale_final = (
            max(
                torch.tensor(self.config.data.resolution)
                / torch.tensor(self.config.data.input_resolution)
            )
            + 1e-8
        )
        output_resolution = torch.floor(
            torch.tensor(self.config.data.input_resolution) * scale_final
        ).int()

        feats = feats.to(torch.float32)
        feats = F.interpolate(
            feats, size=(output_resolution[0], output_resolution[1]), mode="nearest"
        ).squeeze(0)
        l, t, r, b = tuple(self.config.data.ltrb)
        feats = feats[:, :, t:b, l:r]

        if batch["context"][0].get("sem_gt") is not None:
            sem_gt = torch.stack(
                [target_view["sem_gt"] for target_view in batch["context"]], dim=1
            )

            sem_gt = sem_gt.view(-1, sem_gt.shape[2], sem_gt.shape[3], sem_gt.shape[4])

            sem_gt = F.interpolate(
                sem_gt,
                size=(output_resolution[0], output_resolution[1]),
                mode="nearest",
            ).squeeze(0)
            l, t, r, b = tuple(self.config.data.ltrb)
            sem_gt = sem_gt[:, :, t:b, l:r]

        sem_gt = sem_gt.contiguous()

        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips, sim, loss_consistency = self.calculate_loss(
            batch,
            view1,
            view2,
            pred1,
            pred2,
            color,
            sems,
            feats,
            mask,
            matches_im0s,
            matches_im1s,
            sem_gt,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False,
        )

        if (
            batch_idx % self.config.data.log_interval == 0
            and color.device == torch.device("cuda:0")
        ):
            print(
                f"sim : {sum(self.loss_sim) / len(self.loss_sim)}, mse: { sum(self.loss_mse) / len(self.loss_mse)}, lpips: {sum(self.loss_lpips) / len(self.loss_lpips)} "
            )
            if self.config.loss.mast3r_loss_weight is not None:
                print(f"mast3r: {sum(self.loss_mast3r) / len(self.loss_mast3r)}")
            if self.config.loss.consistency_loss_weight is not None:
                print(
                    f"consistency: {sum(self.loss_consistency) / len(self.loss_consistency)}"
                )
            self.loss_mse = []
            self.loss_lpips = []
            self.loss_sim = []
            self.loss_pixel_sem = []
            self.loss_consistency = []
        self.log_metrics(
            "train", loss, mse, lpips, sim, loss_consistency=loss_consistency
        )
        return loss

    def validation_step(self, batch, batch_idx):
        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch["context"]

        pred1, pred2 = self.forward(view1, view2)

        matches_im0s = None
        matches_im1s = None

        if self.config.use_desc:
            matches_im0s = []
            matches_im1s = []
            desc1s, desc2s = pred1["desc"].detach(), pred2["desc"].detach()

            for desc1, desc2 in zip(desc1s, desc2s):
                matches_im0, matches_im1 = fast_reciprocal_NNs(
                    desc1,
                    desc2,
                    subsample_or_initxy1=8,
                    device=desc1.device,
                    dist="dot",
                    block_size=2**13,
                )

                H0, W0 = h, w
                valid_matches_im0 = (
                    (matches_im0[:, 0] >= 3)
                    & (matches_im0[:, 0] < int(W0) - 3)
                    & (matches_im0[:, 1] >= 3)
                    & (matches_im0[:, 1] < int(H0) - 3)
                )

                H1, W1 = h, w
                valid_matches_im1 = (
                    (matches_im1[:, 0] >= 3)
                    & (matches_im1[:, 0] < int(W1) - 3)
                    & (matches_im1[:, 1] >= 3)
                    & (matches_im1[:, 1] < int(H1) - 3)
                )

                valid_matches = valid_matches_im0 & valid_matches_im1
                matches_im0, matches_im1 = (
                    matches_im0[valid_matches],
                    matches_im1[valid_matches],
                )
                matches_im0s.append(matches_im0)
                matches_im1s.append(matches_im1)

        color, _, sems = self.decoder(batch, pred1, pred2, (h, w))
        feats = torch.stack(
            [target_view["sem_feat"] for target_view in batch["target"]], dim=1
        )
        feats_path = [target_view["feat_path"] for target_view in batch["target"]]

        feats = feats.view(-1, feats.shape[2], feats.shape[3], feats.shape[4])

        scale_final = (
            max(
                torch.tensor(self.config.data.resolution)
                / torch.tensor(self.config.data.input_resolution)
            )
            + 1e-8
        )
        output_resolution = torch.floor(
            torch.tensor(self.config.data.input_resolution) * scale_final
        ).int()

        feats = feats.to(torch.float32)
        feats = F.interpolate(
            feats, size=(output_resolution[0], output_resolution[1]), mode="nearest"
        )

        l, t, r, b = tuple(self.config.data.ltrb)

        feats = feats[:, :, t:b, l:r]

        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips, sim = self.calculate_loss(
            batch,
            view1,
            view2,
            pred1,
            pred2,
            color,
            sems,
            feats,
            mask,
            matches_im0s,
            matches_im1s,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=False,
        )

        self.log_metrics("val", loss, mse, lpips, sim)
        return loss

    def test_step(self, batch, batch_idx):

        _, _, h, w = batch["context"][0]["img"].shape
        view1, view2 = batch["context"]
        num_targets = len(batch["target"])

        with self.benchmarker.time("encoder"):
            pred1, pred2 = self.forward(view1, view2)
        with self.benchmarker.time("decoder", num_calls=num_targets):
            color, _, sems = self.decoder(batch, pred1, pred2, (h, w))

        mask = loss_mask.calculate_loss_mask(batch)
        loss, mse, lpips, ssim = self.calculate_loss(
            batch,
            view1,
            view2,
            pred1,
            pred2,
            color,
            sems,
            None,
            mask,
            apply_mask=self.config.loss.apply_mask,
            average_over_mask=self.config.loss.average_over_mask,
            calculate_ssim=True,
        )

        self.log_metrics("test", loss, mse, lpips, ssim=ssim)
        return loss

    def on_test_end(self):
        benchmark_file_path = os.path.join(self.config.save_dir, "benchmark.json")
        self.benchmarker.dump(os.path.join(benchmark_file_path))

    def calculate_loss(
        self,
        batch,
        view1,
        view2,
        pred1,
        pred2,
        color,
        sems,
        feats,
        mask,
        matches_im0s=None,
        matches_im1s=None,
        sem_gt=None,
        apply_mask=True,
        average_over_mask=True,
        calculate_ssim=False,
    ):

        target_color = torch.stack(
            [target_view["original_img"] for target_view in batch["target"]], dim=1
        )

        context_color = torch.stack(
            [target_view["original_img"] for target_view in batch["context"]], dim=1
        )
        predicted_color = color

        if apply_mask:
            assert mask.sum() > 0, "There are no valid pixels in the mask!"
            target_color = target_color * mask[..., None, :, :]
            predicted_color = predicted_color * mask[..., None, :, :]

        flattened_color = einops.rearrange(predicted_color, "b v c h w -> (b v) c h w")
        flattened_sem = einops.rearrange(sems, "b v c h w -> (b v) c h w")

        flattened_target_color = einops.rearrange(
            target_color, "b v c h w -> (b v) c h w"
        )
        flattened_mask = einops.rearrange(mask, "b v h w -> (b v) h w")

        rgb_l2_loss = (predicted_color - target_color) ** 2
        if average_over_mask:
            mse_loss = (rgb_l2_loss * mask[:, None, ...]).sum() / mask.sum()
        else:
            mse_loss = rgb_l2_loss.mean()

        lpips_loss = self.lpips_criterion(
            flattened_target_color, flattened_color, normalize=True
        )
        if average_over_mask:
            lpips_loss = (
                lpips_loss * flattened_mask[:, None, ...]
            ).sum() / flattened_mask.sum()
        else:
            lpips_loss = lpips_loss.mean()

        loss = 0
        loss_consistency = 0
        loss += self.config.loss.mse_loss_weight * mse_loss
        loss += self.config.loss.lpips_loss_weight * lpips_loss

        if self.config.loss.mast3r_loss_weight is not None:
            mast3r_loss = self.mast3r_criterion(view1, view2, pred1, pred2)[0]
            loss += self.config.loss.mast3r_loss_weight * mast3r_loss
            self.loss_mast3r.append(
                self.config.loss.mast3r_loss_weight * mast3r_loss.item()
            )

        if all(
            [
                self.config.loss.pixel_sem_loss_weight is not None,
                matches_im0s is not None,
                matches_im1s is not None,
            ]
        ):

            gs_sems0 = pred1["sem"]
            gs_sems1 = pred2["sem"]
            gs_sem0_matchs = []
            gs_sem1_matchs = []
            for gs_sem0, gs_sem1, matches_im0, matches_im1 in zip(
                gs_sems0, gs_sems1, matches_im0s, matches_im1s
            ):

                gs_sem0_match = gs_sem0[matches_im0[:, 0], matches_im0[:, 1]]
                gs_sem1_match = gs_sem1[matches_im1[:, 0], matches_im1[:, 1]]
                gs_sem0_matchs.append(gs_sem0_match)
                gs_sem1_matchs.append(gs_sem1_match)

            gs_sem0_matchs = torch.cat(gs_sem0_matchs)
            gs_sem1_matchs = torch.cat(gs_sem1_matchs)
            cosine_sim = F.cosine_similarity(gs_sem0_matchs, gs_sem1_matchs, dim=1)

            pixel_sem_loss = 1 - cosine_sim.mean()

            loss += self.config.loss.pixel_sem_loss_weight * pixel_sem_loss

            self.loss_pixel_sem.append(
                self.config.loss.pixel_sem_loss_weight * pixel_sem_loss.item()
            )

        if all(
            [self.config.loss.consistency_loss_weight is not None, sem_gt is not None]
        ):
            gs_sems0 = pred1["sem"]
            gs_sems1 = pred2["sem"]

            sem_gts = sem_gt.squeeze(1).view(
                gs_sems0.shape[0], -1, sem_gt.shape[-2], sem_gt.shape[-1]
            )
            loss_consistency = []
            for gs_sem0, gs_sem1, sem_gt, color_c in zip(
                gs_sems0, gs_sems1, sem_gts, context_color
            ):

                sem_gt0 = sem_gt[0]
                sem_gt1 = sem_gt[1]

                sem_gt = sem_gt.view(2, -1)
                sem_gt = torch.cat((sem_gt[0], sem_gt[1]))
                gs_sem0 = gs_sem0.view(-1, gs_sem0.shape[-1])
                gs_sem1 = gs_sem1.view(-1, gs_sem1.shape[-1])
                gs_sem = torch.cat((gs_sem0, gs_sem1), dim=0)

                consistency = cal_consistency_loss(gs_sem, sem_gt)

                loss_consistency.append(consistency)

            loss_consistency = sum(loss_consistency) / len(loss_consistency)

            loss += self.config.loss.consistency_loss_weight * loss_consistency

            self.loss_consistency.append(
                self.config.loss.consistency_loss_weight * loss_consistency.item()
            )

        if calculate_ssim:
            if average_over_mask:
                ssim_val = compute_ssim.compute_ssim(
                    flattened_target_color, flattened_color, full=True
                )
                ssim_val = (
                    ssim_val * flattened_mask[:, None, ...]
                ).sum() / flattened_mask.sum()
            else:
                ssim_val = compute_ssim.compute_ssim(
                    flattened_target_color, flattened_color, full=False
                )
                ssim_val = ssim_val.mean()
            return loss, mse_loss, lpips_loss, ssim_val

        if torch.isnan(feats).any():
            with open("output_log.txt", "a") as log_file:
                log_file.write("FIND NAN!\n")
            sim = F.cosine_similarity(feats.detach(), feats.detach(), dim=1)
            sim = 1 - sim.mean()

        else:

            norm_flattened_sem = torch.norm(flattened_sem, p=2, dim=1, keepdim=True)
            norm_feats = torch.norm(feats, p=2, dim=1, keepdim=True)

            flattened_sem = flattened_sem / norm_flattened_sem.clamp(min=1e-8)
            feats = feats / norm_feats.clamp(min=1e-8)

            cosine_similarity = torch.sum(flattened_sem * feats, dim=1)

            sim = 1 - cosine_similarity

            if average_over_mask:

                sim = (sim * flattened_mask[:, None, ...]).sum() / mask.sum()
            else:
                sim = sim.mean()

        if self.current_epoch >= self.config.loss.start_sem_loss:
            if self.current_epoch < 10:
                loss += (
                    self.config.loss.sim_loss_weight
                    * sim
                    * (self.current_epoch + 1)
                    / 10
                )
            else:
                loss += self.config.loss.sim_loss_weight * sim

        self.loss_sim.append(self.config.loss.sim_loss_weight * sim.item())
        self.loss_mse.append(self.config.loss.mse_loss_weight * mse_loss.item())
        self.loss_lpips.append(self.config.loss.lpips_loss_weight * lpips_loss.item())

        if sem_gt is not None:
            return loss, mse_loss, lpips_loss, sim, loss_consistency
        return loss, mse_loss, lpips_loss, sim

    def log_metrics(
        self, prefix, loss, mse, lpips, sim=None, ssim=None, loss_consistency=None
    ):
        values = {}
        if sim is not None:
            values[f"{prefix}/sim"] = sim

        values[f"{prefix}/mse"] = mse
        values[f"{prefix}/loss"] = loss
        values[f"{prefix}/psnr"] = -10.0 * mse.log10()
        values[f"{prefix}/lpips"] = lpips

        if ssim is not None:
            values[f"{prefix}/ssim"] = ssim
        if loss_consistency is not None:
            values[f"{prefix}/loss_consistency"] = loss_consistency

        prog_bar = prefix != "val"
        sync_dist = prefix != "train"
        self.log_dict(
            values,
            prog_bar=prog_bar,
            sync_dist=sync_dist,
            batch_size=self.config.data.batch_size,
        )

    def configure_optimizers(self):
        params = list(self.encoder.parameters())
        optimizer = torch.optim.Adam(params, lr=self.config.opt.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [self.config.opt.epochs // 2], gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


def run_experiment(config):

    L.seed_everything(config.seed, workers=True)

    os.makedirs(os.path.join(config.save_dir, config.name), exist_ok=True)
    loggers = []
    if config.loggers.use_csv_logger:
        csv_logger = L.pytorch.loggers.CSVLogger(
            save_dir=config.save_dir, name=config.name
        )
        loggers.append(csv_logger)
    if config.loggers.use_wandb:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            project="splatt3r",
            name=config.name,
            save_dir=config.save_dir,
            config=omegaconf.OmegaConf.to_container(config),
        )
        if wandb.run is not None:
            wandb.run.log_code(".")
        loggers.append(wandb_logger)

    if config.use_profiler:
        profiler = L.pytorch.profilers.PyTorchProfiler(
            dirpath=config.save_dir,
            filename="trace",
            export_to_chrome=True,
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(config.save_dir),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            profile_memory=True,
            with_stack=True,
        )
    else:
        profiler = None

    print("Loading Model")
    model = MAST3RGaussians(config)
    if config.pretrained_path is not None:
        ckpt = torch.load(config.pretrained_path, map_location=torch.device("cpu"))
        _ = model.load_state_dict(ckpt["state_dict"], strict=False)

        del ckpt
    elif config.use_pretrained:

        ckpt = torch.load(
            config.pretrained_mast3r_path, map_location=torch.device("cpu")
        )

        _ = model.load_state_dict(ckpt["state_dict"], strict=False)

        for name, param in model.named_parameters():
            if name in ckpt["state_dict"]:
                param.requires_grad = False
        del ckpt

    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Trainable:", name)

    print(f"Building Datasets")
    train_dataset = scannetpp.get_scannet_dataset(
        config.data.root,
        "train",
        config.data.resolution,
        num_epochs_per_epoch=config.data.epochs_per_train_epoch,
    )
    data_loader_train = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
    )

    if config.val:
        val_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=0.5,
            beta=0.5,
            resolution=config.data.resolution,
            use_every_n_sample=100,
        )
        data_loader_val = torch.utils.data.DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )
    else:
        data_loader_val = None
    if config.need_train:

        print("Training")

        trainer = L.Trainer(
            accelerator="gpu",
            benchmark=True,
            callbacks=[
                L.pytorch.callbacks.LearningRateMonitor(
                    logging_interval="epoch", log_momentum=True
                ),
                export.SaveBatchData(save_dir=config.save_dir),
            ],
            check_val_every_n_epoch=1,
            default_root_dir=config.save_dir,
            devices=config.devices,
            gradient_clip_val=config.opt.gradient_clip_val,
            log_every_n_steps=10,
            logger=loggers,
            max_epochs=config.opt.epochs,
            profiler=profiler,
            strategy=(
                "ddp_find_unused_parameters_true" if len(config.devices) > 1 else "auto"
            ),
            accumulate_grad_batches=config.accumulate_grad_batches,
        )
        trainer.fit(
            model, train_dataloaders=data_loader_train, val_dataloaders=data_loader_val
        )

    original_save_dir = config.save_dir
    results = {}
    for alpha, beta in ((0.9, 0.9), (0.7, 0.7), (0.5, 0.5), (0.3, 0.3)):

        test_dataset = scannetpp.get_scannet_test_dataset(
            config.data.root,
            alpha=alpha,
            beta=beta,
            resolution=config.data.resolution,
            use_every_n_sample=10,
        )
        data_loader_test = torch.utils.data.DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
        )

        masking_configs = ((True, False), (True, True))
        for apply_mask, average_over_mask in masking_configs:

            new_save_dir = os.path.join(
                original_save_dir,
                f"alpha_{alpha}_beta_{beta}_apply_mask_{apply_mask}_average_over_mask_{average_over_mask}",
            )
            os.makedirs(new_save_dir, exist_ok=True)
            model.config.save_dir = new_save_dir

            L.seed_everything(config.seed, workers=True)

            trainer = L.Trainer(
                accelerator="gpu",
                benchmark=True,
                callbacks=[
                    export.SaveBatchData(save_dir=config.save_dir),
                ],
                default_root_dir=config.save_dir,
                devices=config.devices,
                log_every_n_steps=10,
                strategy=(
                    "ddp_find_unused_parameters_true"
                    if len(config.devices) > 1
                    else "auto"
                ),
            )

            model.lpips_criterion = lpips.LPIPS("vgg", spatial=average_over_mask)
            model.config.loss.apply_mask = apply_mask
            model.config.loss.average_over_mask = average_over_mask

            res = trainer.test(model, dataloaders=data_loader_test)

            results[
                f"alpha: {alpha}, beta: {beta}, apply_mask: {apply_mask}, average_over_mask: {average_over_mask}"
            ] = res

            save_path = os.path.join(original_save_dir, "results.json")
            with open(save_path, "w") as f:
                json.dump(results, f)


if __name__ == "__main__":

    print(sys.argv[1], sys.argv[2:])

    config = workspace.load_config(sys.argv[1], sys.argv[2:])

    if os.getenv("LOCAL_RANK", "0") == "0":
        config = workspace.create_workspace(config)

    run_experiment(config)
