# Deepfake Detection with Self-Supervised Learning

> **Self-Supervised Learning for Robust Deepfake Detection Under Domain Shift**
>
> CS671 — Deep Learning & Applications | Group 34 | Project P20
> IIT Mandi

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)
![License](https://img.shields.io/badge/license-MIT-yellow.svg)

---

## Overview

A two-phase deepfake detection pipeline that reduces dependency on labelled fake data by learning structural priors from real faces using self-supervised learning (SSL). The system combines a frozen **DINOv2-ViT-L/14** backbone, an **SRM frequency branch**, and a **Fused Gromov-Wasserstein (FGW)** optimal transport module for cross-domain generalisation.

### Key Results

| Metric | Value |
|--------|-------|
| Cross-domain AUC (WildDeepfake) | **0.884** |
| In-distribution Val AUC | **0.946** |
| Improvement over XceptionNet baseline | **+5.3% AUC** |
| Region-level heatmap localisation | ✅ Achieved |
| Test datasets | WildDeepfake + Faceshifter (140k) + VGGFace2 |

---

## Problem Statement

Deepfakes — synthetically generated or manipulated face videos and images — pose serious risks to digital trust, enabling misinformation, identity fraud, and reputational harm. Modern deepfake generation methods (GANs, diffusion-based face swapping, reenactment, and neural rendering) are improving rapidly, making artifacts harder to detect.

While supervised deepfake detectors achieve strong performance on known datasets, they often fail to generalize to:
- **(i)** unseen manipulation methods,
- **(ii)** new camera/compression pipelines (social media),
- **(iii)** different lighting/pose demographics, and
- **(iv)** low-quality, heavily compressed videos.

Additionally, collecting and labeling large-scale deepfake datasets for every new manipulation type is costly and quickly becomes outdated.

This project addresses the challenge of building a **generalizable deepfake detection system with reduced dependency on labeled deepfake data**. The key idea is to leverage self-supervised learning (SSL) on large-scale unlabeled real videos/images to learn manipulation-invariant but forensic-sensitive representations. By learning robust spatio-temporal and frequency-domain features through SSL pretext tasks (e.g., masked modeling, contrastive learning, temporal consistency), the detector can better identify subtle inconsistencies in face dynamics, texture, and compression — improving performance on unseen deepfake types and distribution shifts.

---

## Objectives

1. **Learn strong forensic representations** using SSL on unlabeled real face videos/images.
2. **Improve cross-dataset generalization** for deepfake detection (train on one dataset, test on another).
3. **Detect deepfakes across diverse conditions:** compression, low resolution, occlusion, motion blur, and varied demographics.
4. **Provide localization cues** (frame-level or region-level heatmaps) indicating manipulated regions.

---

## Architecture

```
Input Image (224×224)
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
DINOv2 Backbone (Frozen)         SRM Frequency Branch
ViT-L/14 + registers             30 noise-residue kernels
       │                                 │
  CLS Token (1024-d)             128-d frequency vector
  Patch Tokens (256×1024)                │
       │                                 │
       ▼                                 │
 Patch Graph (256×256)                   │
 cosine cost matrix                      │
       │                                 │
       ▼                                 │
 FGW Distance Module                     │
 Sinkhorn solver (20 iter)               │
 Top-K prototype matching (K=3)          │
       │                                 │
  FGW scalar + T matrix                  │
       │                                 │
       └──────────────┬──────────────────┘
                      ▼
              Detection Head
         CLS(1024) + Freq(128) + FGW(1)
         LayerNorm → 512 → 128 → 2
                      │
              ┌───────┴───────┐
              ▼               ▼
        Real/Fake         Patch Anomaly
          Score              Map (256)
                              │
                              ▼
                       Heatmap (224×224)
```

---

## Repository Structure

```
.
├── newphase1.py              ← Phase 1: SSL prototype bank building
├── newphase2.py              ← Phase 2: Supervised training + evaluation
├── dinov2_import.py          ← DINOv2 backbone import helper
├── ff_download.py            ← FaceForensics++ download script
├── get_data.sh               ← Bash script for dataset downloads
├── heatmaps_v5/              ← Generated anomaly heatmaps (results)
│   ├── real/                 ← Correctly classified real faces
│   └── fake/                 ← Correctly classified fakes (manipulation localised)
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

**Not included** (download separately, see [Data Setup](#data-setup) below):
- `data/` — All training datasets
- `extracted_frames/` — FF++ extracted real/fake frames
- `checkpoints/` — Trained model checkpoints (~1.2 GB)
- `phase1_output/phase1_result.pt` — Prototype bank (~400 MB)

---

## Datasets

### Phase 1 — SSL Prototype Bank (real images only, no labels)

| Dataset | Source | Augmentation | Notes |
|---------|--------|--------------|-------|
| FaceForensics++ real frames | YouTube videos | Light | Pre-extracted frames |
| CelebA | Aligned celebrity JPEGs | Medium | All available used |
| FFHQ | High-quality 1024px PNGs | Strong | All available used |
| Faceshifter / 140k Real | FFHQ-sourced | Strong | All splits combined |

### Phase 2 — Supervised Training

| Dataset | Labels | Augmentation |
|---------|--------|--------------|
| FF++ frames (real + 6 manipulation types) | Real + Fake | Light + JPEG sim |
| WildDeepfake (train + valid) | Real + Fake | Strong + JPEG + Affine |
| Faceshifter (train + valid) | Real + Fake | Strong + JPEG sim |

### Phase 2 — Cross-Domain Evaluation

| Dataset | Labels | Purpose |
|---------|--------|---------|
| VGGFace2 (val/) | Real only | Unseen real domain |
| WildDeepfake (test/) | Fake | In-the-wild video deepfakes |
| Faceshifter (test/) | Fake | StyleGAN synthesis fakes |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/deepfake-detection-ssl.git
cd deepfake-detection-ssl

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

DINOv2 is loaded automatically via `torch.hub` on first run — requires internet access.

---

## Data Setup

The datasets are not included in this repository due to size and licensing. Download them separately:

| Dataset | Link | Size |
|---------|------|------|
| FaceForensics++ | https://github.com/ondyari/FaceForensics | ~50 GB |
| WildDeepfake | https://github.com/OpenTAI/wild-deepfake | ~10 GB |
| VGGFace2 | https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/ | ~36 GB |
| CelebA | https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html | ~1.4 GB |
| FFHQ | https://github.com/NVlabs/ffhq-dataset | ~90 GB |
| Faceshifter (140k) | https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces | ~3 GB |

Place them in `data/` following this structure:

```
data/
├── celeba/img_align_celeba/img_align_celeba/   ← *.jpg files
├── ffhq/ffhq-dataset/images1024x1024/          ← *.png recursive
├── wilddeepfake/{train,valid,test}/{real,fake}/
├── vggface2/{train,val}/<id>/*.jpg
└── faceshifter/real_vs_fake/real_vs_fake/{train,valid,test}/{real,fake}/

extracted_frames/
├── real/   ← FF++ extracted real frames
└── fake/   ← FF++ extracted fake frames
```

---

## Usage

### Step 1 — Phase 1: Build Prototype Bank (SSL, real images only)

Builds the prototype bank from real face images — runs **once** before training.

```bash
python newphase1.py \
  --ffpp_frames_real   ./extracted_frames/real \
  --celeba_root        ./data/celeba/img_align_celeba/img_align_celeba \
  --ffhq_root          ./data/ffhq/ffhq-dataset/images1024x1024 \
  --faceshifter_root   ./data/faceshifter/real_vs_fake/real_vs_fake \
  --num_protos         32 \
  --batch_size         8 \
  --num_workers        4 \
  --output_dir         ./phase1_output
```

**Expected output:**
- `phase1_output/phase1_result.pt` (~400 MB)
- fgw_scale ≈ 1.0–3.0 (healthy range)
- Runtime: 3–5 hours on RTX A5000

### Step 2 — Phase 2: Train + Evaluate

```bash
python newphase2.py \
  --phase1_result    ./phase1_output/phase1_result.pt \
  --frames_out       ./extracted_frames \
  --wild_root        ./data/wilddeepfake \
  --vggface2_root    ./data/vggface2 \
  --faceshifter_root ./data/faceshifter/real_vs_fake/real_vs_fake \
  --max_per_class    5000 \
  --num_protos       32 \
  --k_match          3 \
  --lambda_fgw       1.0 \
  --lambda_con       0.3 \
  --lambda_proto     0.2 \
  --focal_gamma      2.0 \
  --focal_alpha      0.75 \
  --epochs           30 \
  --batch_size       4 \
  --num_workers      4 \
  --ckpt_dir         ./checkpoints \
  --save_heatmaps \
  --heatmap_dir      ./heatmaps \
  --heatmap_n        300 \
  --use_tta
```

**Expected output:**
- `checkpoints/best.pt` — best model checkpoint
- `heatmaps/real/` and `heatmaps/fake/` — anomaly heatmaps
- Final cross-domain AUC reported in stdout

---

## Model Components

### DINOv2 Backbone
- Model: `dinov2_vitl14_reg` (ViT-Large with registers)
- Parameters: ~307M, **completely frozen**
- Outputs: CLS token (1024-d) + 256 patch tokens (1024-d each)
- Patch tokens L2-normalised

### SRM Frequency Branch
- 30 fixed 5×5 high-pass kernels (Steganalysis Rich Model)
- Suppresses image content, amplifies noise residue
- Architecture: Conv(30→64) → BN → ReLU → Conv(64→128) → AvgPool(4×4) → Linear + Dropout(0.5)
- Output: 128-d frequency feature

### FGW Distance Module
- Builds 256×256 cosine cost matrix from patch tokens
- Sinkhorn solver: 20 iterations, regularisation=0.05
- FGW refinement: 3 iterations, α=0.5 (equal feature + structural weight)
- Top-K matching: k=3 nearest prototypes, minimum distance used

### Prototype Bank
- K=32 clusters of real face patch graphs
- Built entirely from real images in Phase 1 (no fake data ever seen)
- EMA update during Phase 2 training (momentum=0.99)
- Per-prototype: patch features (256×1024) + cost matrix (256×256)

### Detection Head
- Input: 1153-d fused vector (CLS + frequency + FGW scalar)
- MLP: LayerNorm → Linear(1153→512) → GELU → Dropout(0.5) → Linear(512→128) → GELU → Dropout(0.3) → Linear(128→2)
- Patch scorer: Linear(1024→256) → GELU → Linear(256→1) per patch
- Heatmap: patch scores + transport entropy, upsampled 16×16 → 224×224

---

## Loss Functions

The model is trained with a combined loss with four complementary components:

```
Total Loss = FocalLoss(γ=2, α=0.75)            ← class imbalance
           + λ_fgw   × FGW Margin Loss          ← structural separation
           + λ_con   × SupCon Loss (CLS)        ← embedding separation
           + λ_proto × Proto Contrastive Loss   ← prototype alignment
```

| Loss | Purpose |
|------|---------|
| **Focal Loss** | Focuses training on hard fake examples, fixes low recall |
| **FGW Margin Loss** | Pulls real FGW distance low, pushes fake FGW distance high |
| **SupCon Loss** | Real CLS embeddings cluster together, fakes pushed apart |
| **Proto Contrastive Loss** | Real patches align with prototypes, fakes diverge |

---

## Results

### Cross-domain test (WildDeepfake test split)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| DINOv2 only (baseline) | 68.4% | 0.66 | 0.58 | 0.62 | 0.791 |
| DINOv2 + SRM (no FGW) | 72.3% | 0.74 | 0.66 | 0.70 | 0.841 |
| **Ours: DINOv2 + SRM + FGW** | **76.5%** | **0.79** | **0.74** | **0.76** | **0.884** |
| XceptionNet (published) | 72.3% | 0.76 | 0.48 | 0.59 | 0.831 |

### In-distribution validation

| Metric | Value |
|--------|-------|
| AUC | 0.946 |
| Accuracy | 88.7% |
| Best epoch | 17/20 |
| FGW scale | 1.44 ✓ |

---

## Heatmap Visualisations

Sample heatmaps are included in `heatmaps_v5/` showing region-level localisation:

```
heatmaps_v5/
├── real/   ← Real faces (low scores, p≈0)
└── fake/   ← Fake faces (high scores, p≈1)
```

Each heatmap is a 3-panel PNG: `[original face | jet overlay blend | pure jet heatmap]`

### Reading the Heatmaps

| Color | Meaning |
|-------|---------|
| 🔵 Blue | Low anomaly — natural face region or background |
| 🟡 Yellow/Green | Medium anomaly |
| 🔴 Red | High anomaly — likely manipulation zone |

**Real face pattern:** Diffuse scattered activations across face, no concentrated hotspots, background uniformly cold blue.

**Fake face pattern:** Concentrated hotspots at manipulation seams — eye blend boundaries, jawline, hairline (GAN checkerboard), mouth corners. Forms a characteristic perimeter ring.

---

## Augmentation Strategy

| Dataset | Strength | Operations |
|---------|----------|------------|
| FF++ frames | Light | Resize(232) → RandomCrop(224), HFlip, mild ColorJitter, JPEG sim |
| CelebA | Medium | Resize(256) → RandomCrop(224), HFlip, ColorJitter, GaussianBlur, RandomAffine |
| FFHQ | Strong | Resize(288) → RandomCrop(224), aggressive ColorJitter, GaussianBlur, RandomAffine |
| WildDeepfake | Strong | Heavy ColorJitter + GaussianBlur + Affine + JPEG sim |
| Faceshifter | Strong | ColorJitter + GaussianBlur + RandomAffine + JPEG sim |
| Val / Test | None | Resize(224) → CenterCrop(224), normalise only |

### Test-Time Augmentation (TTA)

When `--use_tta` is set, each test image is evaluated with 5 augmentation views:
1. Original (center crop)
2. Horizontal flip
3. Scale up + crop
4. Scale down + pad + crop
5. Mild brightness/contrast shift

Predictions are averaged → improves cross-domain AUC by ~1-2%.

---

## Hardware & Environment

| Component | Specification |
|-----------|---------------|
| GPU | NVIDIA RTX A5000 (24 GB VRAM) |
| CUDA | 11.8 |
| Framework | PyTorch 2.x |
| Phase 1 runtime | ~3–5 hours |
| Phase 2 training | ~40 min/epoch (~13 hrs full run) |

---

## Team

| Name | Roll Number |
|------|-------------|
| Shivam Goyal | B23231 |
| Prakul Garg | B23223 |
| Lakshya Goyal | B23212 |
| Vidit Tank | B24409 |
| Krupal Butala | B24315 |
| Gaurav Girish Rathod | B24076 |
| Ridhi Garg | B24348 |
| Vaishnavi Garg | B24173 |
| Tvisha Jaiswal | B24169 |

**Mentors:** Bhavesh Kapil (d24023@students.iitmandi.ac.in) | Parul Chaudhary (s23109@students.iitmandi.ac.in)

---

## References

1. Rossler et al. *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV 2019. [arXiv:1901.08971](https://arxiv.org/abs/1901.08971)
2. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2023. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
3. Zi et al. *WildDeepfake: A Challenging Real-World Dataset for Deepfake Detection.* ACM MM 2020.
4. Cao et al. *VGGFace2: A Dataset for Recognising Faces Across Pose and Age.* IEEE FG 2018.
5. Liu et al. *Large-scale CelebFaces Attributes (CelebA) Dataset.* ICCV 2015.
6. Karras et al. *A Style-Based Generator Architecture for GANs (FFHQ).* CVPR 2019.
7. Khosla et al. *Supervised Contrastive Learning.* NeurIPS 2020.
8. Lin et al. *Focal Loss for Dense Object Detection.* ICCV 2017.

---

## License

This project is released under the MIT License — see [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cs671group34_2026,
  author = {Group 34, CS671},
  title  = {Self-Supervised Learning for Robust Deepfake Detection Under Domain Shift},
  year   = {2026},
  publisher = {GitHub},
  url    = {https://github.com/YOUR_USERNAME/deepfake-detection-ssl}
}
```
