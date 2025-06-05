from ape.layers.vision_language_align import VisionLanguageAlign
from ape.modeling.text import EVA02CLIP
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tools import colormaps
from pathlib import Path
from tools.utils import colormap_saving
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

@torch.no_grad()
def get_relevancy(
    embed: torch.Tensor, positive_id: int, phrases_embeds, positives, negatives, bias
) -> torch.Tensor:
    p = phrases_embeds.to(embed.dtype)

    embed = embed / (embed.norm(dim=-1, keepdim=True) + 1e-6)

    output = torch.mm(embed, p.T) + bias
    positive_vals = output[..., positive_id : positive_id + 1]
    negative_vals = output[..., len(positives) :]
    repeated_pos = positive_vals.repeat(1, len(negatives))

    sims = torch.stack((repeated_pos, negative_vals), dim=-1)
    softmax = torch.softmax(10 * sims, dim=-1)

    best_id = softmax[..., 0].argmin(dim=1)

    return torch.gather(
        softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(negatives), 2)
    )[:, 0, :]


def generate_colormap(num_classes):

    cmap = plt.cm.get_cmap("tab20", num_classes)
    colormap = {
        i: tuple((np.array(cmap(i)[:3]) * 255).astype(int)) for i in range(num_classes)
    }
    return colormap


class TestAlign(nn.Module):
    def __init__(self, autoencoder, config, need_log_scale=True, return_all=False):
        super(
            TestAlign,
            self,
        ).__init__()
        self.class_embed = VisionLanguageAlign(
            256, 1024, need_log_scale=need_log_scale, return_all=return_all
        )
        self.model_language = EVA02CLIP(
            clip_model="EVA02-CLIP-bigE-14-plus",
            cache_dir=None,
            dtype="float16",
        )

        checkpoint_vl = torch.load(config.vl_algin_weights, map_location="cpu")
        checkpoint_lang = torch.load(config.lang_weights, map_location="cpu")

        self.class_embed.load_state_dict(checkpoint_vl)
        self.model_language.load_state_dict(checkpoint_lang)
        self.autoencoder = autoencoder
        self.dtype = torch.float32

    def forward(self, x, text):
        x = x.to(self.dtype)
        x = self.autoencoder.decoder(x)

        langmodel_feature = self.model_language.forward_text(text)[
            "last_hidden_state_eot"
        ]

        langmodel_feature = langmodel_feature.unsqueeze(0)

        outputs = self.class_embed(x.to(self.dtype), langmodel_feature.to(self.dtype))

        return outputs




def get_model(autoencoder, config, need_log_scale=True, return_all=False):

    model = TestAlign(autoencoder, config, need_log_scale, return_all).cuda()

    return model


def get_seg_label(x, image, model, mapper):

    image = image.permute(1, 2, 0).contiguous()

    image = np.array(image.to(torch.uint8))

    image = Image.fromarray(image)

    x = x.permute(1, 2, 0).contiguous()
    h, w, c = x.shape
    x = x.view(-1, x.shape[-1]).contiguous().unsqueeze(0)

    text = mapper

    out_put = model(x, text).squeeze(0)

    out_put = out_put.view(h, w, -1).detach().cpu()

    out_put = out_put.permute(2, 0, 1)

    metadata = MetadataCatalog.get("__unused_ape_" + "123")
    metadata.thing_classes = text
    metadata.stuff_classes = text
    image = np.array(image)

    visualizer = Visualizer(image, metadata, instance_mode=ColorMode.IMAGE)

    sem_seg = out_put

    sem_seg = sem_seg.argmax(dim=0)

    vis_output = visualizer.draw_sem_seg(sem_seg)

    out_put = torch.argmax(out_put, dim=0)

    out_put = out_put.view(h, w).detach().cpu()

    return out_put


def get_iou_list_per_image(gt, pred_masks, alpha, v_mask_self):

    v_gt_mask = gt != len(pred_masks)
    iou_list = []

    for idx, pred_mask in enumerate(pred_masks):

        gt_mask = gt == idx

        v_mask = (alpha == 255) & v_gt_mask

        intersection = np.logical_and(
            pred_mask * v_mask * v_mask_self, gt_mask * v_mask
        ).sum()
        union = np.logical_or(pred_mask * v_mask * v_mask_self, gt_mask * v_mask).sum()

        if union > 0:
            iou = intersection / union
        else:
            iou = None

        iou_list.append(iou)
    return iou_list


def get_seg_query_vis(x, image, model, scene_id, idx, config):

    image = image.permute(1, 2, 0).contiguous()

    image = np.array(image.to(torch.uint8))

    image = (image / 255).astype(np.float32)
    image = torch.from_numpy(image).to(x.device)

    pred_masks = []

    x = x.permute(1, 2, 0).contiguous()
    h, w, c = x.shape
    x = x.view(-1, x.shape[-1]).contiguous().unsqueeze(0)

    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=0,
        colormap_max=1.0,
    )

    CLS_LIST_POS = [
        "wall",
        "ceiling",
        "floor",
        "table",
        "door",
        "storage cabinet",
        "chair",
        "bookshelf",
        "box",
        "bed",
    ]

    CLS_LIST_NEG = ["object", "texture", "stuff", "things"]

    text = CLS_LIST_POS + CLS_LIST_NEG

    img_emb, query_emb, bias = model(x, text)

    n_phrases = len(CLS_LIST_POS)
    n_phrases_sims = [None for _ in range(n_phrases)]
    for j in range(n_phrases):
        probs = get_relevancy(
            img_emb.squeeze(0),
            j,
            query_emb.squeeze(0),
            CLS_LIST_POS,
            CLS_LIST_NEG,
            bias.squeeze(0),
        )

        pos_prob = probs[..., 0:1]

        n_phrases_sims[j] = pos_prob

    n_phrases_sims = torch.stack(n_phrases_sims).squeeze(-1)
    valid_map = n_phrases_sims.view(n_phrases, h, w).detach()

    for i in range(len(CLS_LIST_POS)):

        scale = 30
        kernel = np.ones((scale, scale)) / (scale**2)
        np_relev = valid_map[i].cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel)
        avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
        valid_map[i] = 0.5 * (avg_filtered + valid_map[i])

        output_path_relev = (
            Path("vis")
            / f"{scene_id}"
            / f"{config.seed}"
            / "heatmap"
            / f"{CLS_LIST_POS[i]}_{idx}"
        )
        output_path_relev.parent.mkdir(exist_ok=True, parents=True)

        colormap_saving(valid_map[i].unsqueeze(-1), colormap_options, output_path_relev)

        p_i = torch.clip(valid_map[i] - 0.5, 0, 1).unsqueeze(-1)

        valid_composited = colormaps.apply_colormap(
            p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo")
        )
        mask = (valid_map[i] < 0.5).squeeze()
        pred_mask = (valid_map[i] >= 0.5).squeeze()

        pred_mask = pred_mask.to(torch.uint8) * 255
        pred_masks.append(pred_mask.detach().cpu().numpy())
        pred_mask = Image.fromarray(pred_mask.detach().cpu().numpy())

        valid_composited[mask, :] = image[mask, :] * 0.5
        output_path_compo = (
            Path("vis")
            / f"{scene_id}"
            / f"{config.seed}"
            / "composited"
            / f"{CLS_LIST_POS[i]}_{idx}"
        )

        output_path_compo.parent.mkdir(exist_ok=True, parents=True)
        colormap_saving(valid_composited, colormap_options, output_path_compo)
        output_path_pred = (
            Path("vis")
            / f"{scene_id}"
            / f"{config.seed}"
            / "pred"
            / f"{CLS_LIST_POS[i]}_{idx}.png"
        )

        output_path_pred.parent.mkdir(exist_ok=True, parents=True)
        pred_mask.save(output_path_pred)

    return pred_masks


def get_seg_query(x, image, model, scene_id, idx, config):

    image = image.permute(1, 2, 0).contiguous()

    image = np.array(image.to(torch.uint8))

    image = (image / 255).astype(np.float32)
    image = torch.from_numpy(image).to(x.device)

    pred_masks = []

    x = x.permute(1, 2, 0).contiguous()
    h, w, c = x.shape
    x = x.view(-1, x.shape[-1]).contiguous().unsqueeze(0)

    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=0,
        colormap_max=1.0,
    )

    CLS_LIST_POS = [
        "wall",
        "ceiling",
        "floor",
        "table",
        "door",
        "storage cabinet",
        "chair",
        "bookshelf",
        "box",
        "bed",
    ]

    CLS_LIST_NEG = ["object", "texture", "stuff", "things"]

    text = CLS_LIST_POS + CLS_LIST_NEG

    img_emb, query_emb, bias = model(x, text)

    n_phrases = len(CLS_LIST_POS)
    n_phrases_sims = [None for _ in range(n_phrases)]
    for j in range(n_phrases):
        probs = get_relevancy(
            img_emb.squeeze(0),
            j,
            query_emb.squeeze(0),
            CLS_LIST_POS,
            CLS_LIST_NEG,
            bias.squeeze(0),
        )

        pos_prob = probs[..., 0:1]

        n_phrases_sims[j] = pos_prob

    n_phrases_sims = torch.stack(n_phrases_sims).squeeze(-1)
    valid_map = n_phrases_sims.view(n_phrases, h, w).detach()

    for i in range(len(CLS_LIST_POS)):

        scale = 30
        kernel = np.ones((scale, scale)) / (scale**2)
        np_relev = valid_map[i].cpu().numpy()
        avg_filtered = cv2.filter2D(np_relev, -1, kernel)
        avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
        valid_map[i] = 0.5 * (avg_filtered + valid_map[i])

        output_path_relev = (
            Path("vis")
            / f"{scene_id}"
            / f"{config.seed}"
            / "heatmap"
            / f"{CLS_LIST_POS[i]}_{idx}"
        )
        output_path_relev.parent.mkdir(exist_ok=True, parents=True)

        colormap_saving(valid_map[i].unsqueeze(-1), colormap_options, output_path_relev)

        p_i = torch.clip(valid_map[i] - 0.5, 0, 1).unsqueeze(-1)

        valid_composited = colormaps.apply_colormap(
            p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo")
        )
        mask = (valid_map[i] < 0.5).squeeze()
        pred_mask = (valid_map[i] >= 0.5).squeeze()

        pred_mask = pred_mask.to(torch.uint8) * 255
        pred_masks.append(pred_mask.detach().cpu().numpy())
        pred_mask = Image.fromarray(pred_mask.detach().cpu().numpy())

        valid_composited[mask, :] = image[mask, :] * 0.5
        output_path_compo = (
            Path("vis")
            / f"{scene_id}"
            / f"{config.seed}"
            / "composited"
            / f"{CLS_LIST_POS[i]}_{idx}"
        )

        output_path_compo.parent.mkdir(exist_ok=True, parents=True)
        colormap_saving(valid_composited, colormap_options, output_path_compo)
        output_path_pred = (
            Path("vis")
            / f"{scene_id}"
            / f"{config.seed}"
            / "pred"
            / f"{CLS_LIST_POS[i]}_{idx}.png"
        )

        output_path_pred.parent.mkdir(exist_ok=True, parents=True)
        pred_mask.save(output_path_pred)

    return pred_masks
