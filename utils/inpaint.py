from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from diffusers import ControlNetModel, AutoencoderKL, UNet2DConditionModel
import random
import cv2
import os
import contextlib
from sklearn.decomposition import PCA
from transformers import AutoTokenizer
from diffusers import (
    StableDiffusionControlNetInpaintPipeline,
    DDIMScheduler,
    StableDiffusionInpaintPipeline,
)
from src.diffusers.image_processor import VaeImageProcessor


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


def get_inpainter_pipeline(
    pretrained_unet_name,
    controlnet_model_feat_name_or_path,
    pretrained_feat_model_name_or_path,
    pretrained_vae_feat_path,
):
    pretrained_model_name_or_path = "runwayml/stable-diffusion-inpainting"

    pretrained_unet_name = pretrained_unet_name

    controlnet_model_feat_name_or_path = controlnet_model_feat_name_or_path

    pretrained_feat_model_name_or_path = pretrained_feat_model_name_or_path

    pretrained_vae_feat_path = pretrained_vae_feat_path

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path, subfolder="vae", revision=None, variant=None
    )

    vae_feat = AutoencoderKL.from_pretrained(
        pretrained_vae_feat_path, use_safetensors=True
    )

    controlnet_feat = ControlNetModel.from_pretrained(
        controlnet_model_feat_name_or_path
    )

    unet = UNet2DConditionModel.from_pretrained(pretrained_unet_name)

    unet_feat = UNet2DConditionModel.from_pretrained(
        pretrained_feat_model_name_or_path,
        subfolder="unet",
        revision=None,
        variant=None,
    )

    pipeline = StableDiffusionInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae,
        tokenizer=tokenizer,
        unet=unet,
        safety_checker=None,
        revision=None,
        variant=None,
    ).to("cuda")

    pipeline_feat = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        pretrained_model_name_or_path,
        vae=vae_feat,
        tokenizer=tokenizer,
        unet=unet_feat,
        controlnet=controlnet_feat,
        safety_checker=None,
        revision=None,
        variant=None,
    ).to("cuda")

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)

    pipeline_feat.image_processor = VaeImageProcessor(
        vae_scale_factor=vae_scale_factor, do_normalize=False
    )

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    pipeline_feat.scheduler = DDIMScheduler.from_config(pipeline_feat.scheduler.config)

    return pipeline, pipeline_feat


def generate_random_mask():
    height, width = 512, 512
    max_white_area = int((height * width) * 0.4)
    mask = np.zeros((height, width), dtype=np.uint8)

    numbers = [1, 2, 3, 4, 5]
    weights = [0.75, 0.15, 0.05, 0.04, 0.01]
    num_polygons = random.choices(numbers, weights=weights, k=1)[0]

    for _ in range(num_polygons):
        type = random.random()

        if type < 0.2:
            height_h = random.randint(1, height // 2)
            width_h = width
            height_l = 0
            width_l = 0
        elif type < 0.4:
            height_h = height
            width_h = random.randint(1, width // 2)
            height_l = 0
            width_l = 0
        elif type < 0.6:
            height_h = height
            width_h = width
            height_l = random.randint(height // 2, height - 2)
            width_l = 0
        elif type < 0.8:
            height_h = height
            width_h = width
            height_l = 0
            width_l = random.randint(width // 2, width - 2)
        else:
            height_h = height
            height_l = 0
            width_h = width
            width_l = 0

        num_vertices = random.randint(3, 15)
        vertices = np.array(
            [
                [
                    (
                        0
                        if (rand_x := random.randint(width_l, width_h - 1)) < 50
                        and random.random() < 0.5
                        else (
                            width - 1
                            if rand_x > 460 and random.random() < 0.5
                            else rand_x
                        )
                    ),
                    (
                        0
                        if (rand_y := random.randint(height_l, height_h - 1)) < 50
                        and random.random() < 0.5
                        else (
                            height - 1
                            if rand_y > 460 and random.random() < 0.5
                            else rand_y
                        )
                    ),
                ]
                for _ in range(num_vertices)
            ],
            dtype=np.int32,
        )

        vertices = cv2.convexHull(vertices)

        current_white_area = np.count_nonzero(mask)
        remaining_area = max_white_area - current_white_area

        if remaining_area <= 0:
            break

        temp_mask = mask.copy()
        cv2.fillPoly(temp_mask, [vertices], 255)

        new_white_area = np.count_nonzero(temp_mask)

        if new_white_area > max_white_area:
            scale_factor = remaining_area / (new_white_area - current_white_area)
            if scale_factor < 1.0:
                center = np.mean(vertices, axis=0)
                vertices = center + (vertices - center) * scale_factor
                vertices = vertices.astype(np.int32)
                temp_mask = mask.copy()
                cv2.fillPoly(temp_mask, [vertices], 255)
                new_white_area = np.count_nonzero(temp_mask)

        if new_white_area <= max_white_area:
            mask = temp_mask.copy()
        else:
            break

    return mask


def get_pca_vis(sem, mask=None):
    if len(sem.shape) == 3:
        sem = sem.unsqueeze(0)
    bs, num_features, height, width = sem.shape
    pca = PCA(n_components=3)
    sem_flat = sem.permute(0, 2, 3, 1).reshape(-1, num_features)
    sem_pca = pca.fit_transform(sem_flat.cpu().numpy())
    sem_reduced_ = torch.tensor(sem_pca, dtype=sem.dtype).reshape(bs, height, width, 3)

    min_val = sem_reduced_.min()
    max_val = sem_reduced_.max()

    sem_reduced_ = (sem_reduced_ - min_val) / (max_val - min_val)
    sem_reduceds = []
    for single in sem_reduced_:

        sem_reduced = Image.fromarray((single.numpy() * 255).astype(np.uint8)).convert(
            "RGB"
        )

        sem_reduceds.append(sem_reduced)

    return sem_reduceds


def inpainting(
    prompt,
    image,
    sem,
    mask,
    pipeline,
    pipeline_feat,
    model,
    mapper,
    scene_id,
    tag=0,
    tid=0,
):

    validation_prompt = prompt
    image = image.clamp(0, 1)
    image = image.detach().cpu()

    image = (image * 255).to(torch.uint8).permute(1, 2, 0).contiguous()

    num_features, height, width = sem.shape

    validation_sem = sem.unsqueeze(0).to(torch.float32).detach().cpu()

    validation_image_array = np.array(image)
    image = Image.fromarray(validation_image_array)
    validation_image_array[mask == 255, :] = 0
    validation_image_array = Image.fromarray(validation_image_array)

    control_image = image

    generator = None
    inference_ctx = contextlib.nullcontext()
    with inference_ctx:

        image = pipeline(
            validation_prompt,
            num_inference_steps=20,
            negative_prompt=[
                "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW"
            ],
            generator=generator,
            image=image,
            mask_image=mask,
        ).images[0]

        feat = pipeline_feat(
            validation_prompt,
            num_inference_steps=20,
            generator=generator,
            image=validation_sem,
            mask_image=mask,
            control_image=image,
            controlnet_conditioning_scale=2.0,
            use_feat=True,
        ).images[0]

        feat_vis = get_pca_vis(feat)[0]

    mask_save = Image.fromarray(mask)
    os.makedirs(f"vis/{scene_id}/images", exist_ok=True)
    validation_image_array.save(f"vis/{scene_id}/images/init_image_{tid}_{tag}.png")
    image.save(f"vis/{scene_id}/images/image_mid_{tid}_{tag}.png")
    mask_save.save(f"vis/{scene_id}/images/mask_{tid}_{tag}.png")
    feat_vis.save(f"vis/{scene_id}/images/feat_{tid}_{tag}.png")
    validation_prompt = validation_prompt[0]
    with open(
        f"vis/{scene_id}/images/text_{tid}_{tag}.txt", "w", encoding="utf-8"
    ) as f:
        f.write(validation_prompt)

    return image, feat
