
# OGGSplat: Open Gaussian Growing for Generalizable Reconstruction with Expanded Field-of-View

### [Paper]()  | [Project Page](https://yanbo-23.github.io/OGGSplat/)  | [Code](https://github.com/Yanbo-23/OGGSplat) 
> Official codebase for **OGGSplat**, presented in the paper:
> **OGGSplat: Open Gaussian Growing for Generalizable Reconstruction with Expanded Field-of-View**

Created by [Yanbo Wang*](https://Yanbo-23.github.io/), [Ziyi Wang*](https://wangzy22.github.io/), [Wenzhao Zheng](https://wzzheng.net/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu†](https://scholar.google.com/citations?user=TN8uDQoAAAAJ&hl=zh-CN).


## 🧠 Overview

![intro](fig/overview.png)

**OGGSplat** is a novel generalizable 3D reconstruction framework that extends the field-of-view from sparse input images by leveraging semantic priors. It introduces an **Open Gaussian Growing** approach that combines RGB-semantic joint inpainting guided by bidirectional diffusion models to extrapolate unseen regions beyond the input views. The inpainted images are then lifted back into 3D space for progressive optimization of Gaussian parameters, enabling efficient and high-quality semantic-aware scene reconstruction. 



## 🛠️ Installation

- Follow the [installation.md](installation.md) to install all required packages so you can do the training & evaluation afterwards.

## ✅ TO DO List

* [x] **Inference Code**
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

