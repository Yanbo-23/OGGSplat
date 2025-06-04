
# OGGSplat: Open Gaussian Growing for Generalizable Reconstruction with Expanded Field-of-View

![License](https://img.shields.io/badge/license-MIT-blue.svg)

> Official codebase for **OGGSplat**, presented in the paper:
> **OGGSplat: Open Gaussian Growing for Generalizable Reconstruction with Expanded Field-of-View**



## 🧠 Overview

**OGGSplat** is a generalizable 3D reconstruction framework that expands the field-of-view from sparse input views by leveraging semantic priors. It introduces **Open Gaussian Growing**, which uses RGB-semantic joint inpainting guided by bidirectional diffusion to extrapolate beyond visible regions. The inpainted views are lifted back into 3D space for progressive Gaussian optimization. To evaluate performance, we propose the **Gaussian Outpainting (GO) Benchmark** assessing both semantic consistency and generative quality. OGGSplat enables high-quality, semantic-aware scene reconstruction from as few as two smartphone images, making it suitable for real-world applications in VR and embodied AI.


## 📝 Paper Abstract

> Reconstructing semantic-aware 3D scenes from sparse views is a challenging yet essential research direction, driven by the demands of emerging applications such as virtual reality and embodied AI. Existing per-scene optimization methods require dense input views and incur high computational costs, while generalizable approaches often struggle to reconstruct regions outside the input view cone.
>
> In this paper, we propose **OGGSplat**, an open Gaussian growing method that expands the field-of-view in generalizable 3D reconstruction. Our key insight is that the semantic attributes of open Gaussians provide strong priors for image extrapolation, enabling both semantic consistency and visual plausibility. Specifically, once open Gaussians are initialized from sparse views, we introduce an RGB-semantic consistent inpainting module applied to selected rendered views. This module enforces bidirectional control between an image diffusion model and a semantic diffusion model. The inpainted regions are then lifted back into 3D space for efficient and progressive Gaussian parameter optimization.
>
> To evaluate our method, we establish a **Gaussian Outpainting (GO)** benchmark that assesses both semantic and generative quality of reconstructed open-vocabulary scenes. OGGSplat also demonstrates promising semantic-aware scene reconstruction capabilities when provided with two view images captured directly from a smartphone camera.



## 🛠️ Installation

Requirements and installation instructions will be added here soon.

```bash
# Clone this repo
git clone https://github.com/your-org/OGGSplat.git
cd OGGSplat

# Set up the environment
conda create -n oggsplat python=3.10
conda activate oggsplat
pip install -r requirements.txt
```

## ✅ TO DO List

* [ ] **Inference Code**
  Load pretrained models and run scene extrapolation from sparse input views.

* [ ] **Evaluation Benchmark**
  Release the Gaussian Outpainting (GO) benchmark suite with metrics for semantic and visual plausibility.

* [ ] **Training Code**
  Provide full training pipeline including Gaussian initialization, diffusion-based inpainting, and progressive optimization.


## 📄 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{your2025oggsplat,
  title={OGGSplat: Open Gaussian Growing for Generalizable Reconstruction with Expanded Field-of-View},
  author={Your Name and Co-authors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2025}
}
```

