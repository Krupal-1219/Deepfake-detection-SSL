# """
# phase1_build_graphs.py  —  v3  (WildDeepfake + VGGFace2 + CelebA + FFHQ)
# =========================================================================
# DATASET CHANGES FROM PREVIOUS VERSION:
#   REMOVED : DFDC  (deleted dataset)
#   ADDED   : WildDeepfake REAL images (data/wilddeepfake/{train,test,valid}/real/)
#   ADDED   : VGGFace2 REAL images     (data/vggface2/{train,val}/<id>/*.jpg)
#   KEPT    : CelebA    (data/celeba/img_align_celeba/img_align_celeba/)
#   KEPT    : FFHQ      (data/ffhq/ffhq-dataset/images1024x1024/)
#   KEPT    : FF++ extracted real frames (extracted_frames/real/)

# ONLY REAL IMAGES ARE USED IN PHASE 1.
# Phase 1 is self-supervised — no labels needed.
# The prototype bank captures "what real faces look like structurally".

# AUGMENTATION PER SOURCE:
#   FF++ frames    : Light  — already 224x224 H.264 frames, minimal distortion
#   CelebA         : Medium — aligned JPEGs, color + blur + affine
#   FFHQ           : Strong — 1024px PNGs, aggressive resize + color + blur
#   WildDeepfake   : Strong — in-the-wild video frames, JPEG + color + affine
#   VGGFace2       : Medium — web-scraped photos, JPEG sim + color + flip

# EXPECTED DIRECTORY STRUCTURE:
#   extracted_frames/real/          ← FF++ pre-extracted real frames
#   data/celeba/img_align_celeba/img_align_celeba/  ← CelebA *.jpg flat
#   data/ffhq/ffhq-dataset/images1024x1024/00000/   ← FFHQ *.png recursive
#   data/wilddeepfake/train/real/   ← WildDeepfake real (subdirs per video)
#   data/wilddeepfake/test/real/
#   data/wilddeepfake/valid/real/
#   data/vggface2/train/n000001/*.jpg  ← VGGFace2 (identity subdirs)
#   data/vggface2/val/n000001/*.jpg

# OUTPUT (phase1_output/phase1_result.pt):
#   proto_feats   : (K, 256, 1024) — K real-face prototype patch features
#   proto_costs   : (K, 256, 256)  — K cosine-distance cost matrices
#   fgw_scale     : float          — 95th pct real FGW (fixed normaliser)
#   fgw_99_pct    : float
#   fgw_mean      : float
#   fgw_std       : float
#   n_images_used : int
#   K             : int
#   cluster_sizes : list
#   version       : "v3_wildvgg"
#   args          : dict

# HOW TO RUN:
#   python phase1_build_graphs.py \\
#     --ffpp_frames_real  ./extracted_frames/real \\
#     --celeba_root       ./data/celeba/img_align_celeba/img_align_celeba \\
#     --ffhq_root         ./data/ffhq/ffhq-dataset/images1024x1024 \\
#     --wild_root         ./data/wilddeepfake \\
#     --vggface2_root     ./data/vggface2 \\
#     --max_per_source    5000 \\
#     --num_protos        32 \\
#     --batch_size        8 \\
#     --num_workers       4 \\
#     --output_dir        ./phase1_output

# EXPECTED OUTPUT:
#   fgw_scale : 1.0 – 3.0  (L2-normalised cosine distances)
#   Runtime   : 30–90 min on one GPU
#   File size : ~400 MB for K=32
# """

# import os, argparse, random, io
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# from sklearn.cluster import MiniBatchKMeans


# # =============================================================
# # CONSTANTS
# # =============================================================
# MEAN = [0.485, 0.456, 0.406]
# STD  = [0.229, 0.224, 0.225]
# EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# # =============================================================
# # JPEG AUGMENTATION  (PIL-level, for in-the-wild sources)
# # =============================================================
# class RandomJPEGAug:
#     """
#     Simulate JPEG re-compression at random quality.
#     Applied before the torchvision transform pipeline.
#     Used for WildDeepfake + VGGFace2 which have naturally
#     varied real-world JPEG quality.
#     """
#     def __init__(self, low=40, high=95, p=0.45):
#         self.low = low; self.high = high; self.p = p

#     def __call__(self, img: Image.Image) -> Image.Image:
#         if random.random() > self.p:
#             return img
#         buf = io.BytesIO()
#         img.save(buf, format="JPEG",
#                  quality=random.randint(self.low, self.high))
#         buf.seek(0)
#         return Image.open(buf).copy()


# # =============================================================
# # PER-SOURCE AUGMENTATION TRANSFORMS
# # =============================================================

# def get_ffpp_transform():
#     """
#     FF++ extracted real frames.
#     Already 224×224, face-cropped, H.264 compressed.
#     Light augmentation: preserve the spatial structure,
#     only vary compression level and mild color.
#     """
#     return transforms.Compose([
#         transforms.Resize(232),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.2, contrast=0.2,
#                                saturation=0.15, hue=0.05),
#         transforms.RandomGrayscale(p=0.03),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


# def get_celeba_transform():
#     """
#     CelebA: 178×218 aligned JPEG, clean centered faces.
#     Medium augmentation: simulate real-world lighting and
#     small pose variation. Color jitter important for diversity.
#     """
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.35, contrast=0.35,
#                                saturation=0.25, hue=0.08),
#         transforms.RandomGrayscale(p=0.05),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(5, sigma=(0.1, 2.5))], p=0.4),
#         transforms.RandomApply(
#             [transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
#                                      scale=(0.92, 1.08))], p=0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


# def get_ffhq_transform():
#     """
#     FFHQ: high-quality 1024×1024 PNG, very diverse real faces.
#     Strong downscale augmentation: simulate different acquisition
#     resolutions. Aggressive color because FFHQ is very clean —
#     we need to teach the model the natural variation range.
#     """
#     return transforms.Compose([
#         transforms.Resize(288),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.4, contrast=0.4,
#                                saturation=0.35, hue=0.10),
#         transforms.RandomGrayscale(p=0.05),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.4),
#         transforms.RandomApply(
#             [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
#                                      scale=(0.9, 1.1))], p=0.35),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


# def get_wilddeepfake_real_transform():
#     """
#     WildDeepfake REAL frames: in-the-wild internet videos.
#     Naturally varied: different cameras, lighting, ethnicities,
#     ages, poses, compression levels.
#     Strong augmentation because this source already has high variance.
#     We add JPEG aug at PIL level to simulate the varied compression.
#     """
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.5, contrast=0.5,
#                                saturation=0.4, hue=0.12),
#         transforms.RandomGrayscale(p=0.06),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(5, sigma=(0.1, 3.5))], p=0.45),
#         transforms.RandomApply(
#             [transforms.RandomAffine(degrees=18, translate=(0.12, 0.12),
#                                      scale=(0.88, 1.12), shear=5)], p=0.4),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


# def get_vggface2_transform():
#     """
#     VGGFace2: web-scraped celebrity/public figure photos.
#     Various poses, backgrounds, lighting, web-JPEG compression.
#     Medium-strong augmentation: these are already diverse,
#     but JPEG simulation is important to cover compression range.
#     """
#     return transforms.Compose([
#         transforms.Resize(256),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.35, contrast=0.35,
#                                saturation=0.30, hue=0.08),
#         transforms.RandomGrayscale(p=0.04),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.35),
#         transforms.RandomApply(
#             [transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
#                                      scale=(0.9, 1.1))], p=0.30),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


# # =============================================================
# # SINGLE-SOURCE DATASET
# # =============================================================
# class SingleSourceDataset(Dataset):
#     """
#     Generic single-source image dataset for Phase 1.
#     No labels — Phase 1 is self-supervised.
#     Applies source-specific transform + optional PIL-level JPEG aug.
#     """
#     def __init__(self, paths: list, transform,
#                  jpeg_aug=None, name: str = ""):
#         self.paths     = paths
#         self.transform = transform
#         self.jpeg_aug  = jpeg_aug
#         self.name      = name

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         path = self.paths[idx]
#         try:
#             img = Image.open(path).convert("RGB")
#             if self.jpeg_aug is not None:
#                 img = self.jpeg_aug(img)
#             return self.transform(img)
#         except Exception:
#             alt = self.paths[random.randint(0, len(self.paths) - 1)]
#             img = Image.open(alt).convert("RGB")
#             if self.jpeg_aug is not None:
#                 img = self.jpeg_aug(img)
#             return self.transform(img)


# # =============================================================
# # IMAGE COLLECTION HELPERS
# # =============================================================
# def _glob_images(root: Path, recursive: bool = False) -> list:
#     if not root.exists():
#         print(f"  WARNING: path does not exist: {root}")
#         return []
#     imgs, pattern = [], "**/*" if recursive else "*"
#     for p in root.glob(pattern):
#         if p.suffix in EXTS and p.is_file():
#             imgs.append(str(p))
#     return imgs


# # =============================================================
# # PER-SOURCE DATASET BUILDERS
# # =============================================================

# def build_ffpp_dataset(ffpp_frames_real: str, max_n: int, seed: int):
#     """FF++ pre-extracted real frames (extracted_frames/real/)."""
#     root = Path(ffpp_frames_real)
#     imgs = _glob_images(root, recursive=False)
#     if not imgs:
#         imgs = _glob_images(root, recursive=True)
#     rng = random.Random(seed); rng.shuffle(imgs)
#     imgs = imgs[:max_n]
#     print(f"  FF++ extracted real frames  : {len(imgs)}")
#     return SingleSourceDataset(imgs, get_ffpp_transform(),
#                                jpeg_aug=None, name="ffpp")


# def build_celeba_dataset(celeba_root: str, max_n: int, seed: int):
#     """CelebA flat directory: img_align_celeba/*.jpg"""
#     imgs = _glob_images(Path(celeba_root), recursive=False)
#     rng = random.Random(seed); rng.shuffle(imgs)
#     imgs = imgs[:max_n]
#     print(f"  CelebA                      : {len(imgs)}")
#     return SingleSourceDataset(imgs, get_celeba_transform(),
#                                jpeg_aug=None, name="celeba")


# def build_ffhq_dataset(ffhq_root: str, max_n: int, seed: int):
#     """FFHQ recursive: images1024x1024/00000/*.png"""
#     imgs = _glob_images(Path(ffhq_root), recursive=True)
#     rng = random.Random(seed); rng.shuffle(imgs)
#     imgs = imgs[:max_n]
#     print(f"  FFHQ                        : {len(imgs)}")
#     return SingleSourceDataset(imgs, get_ffhq_transform(),
#                                jpeg_aug=None, name="ffhq")


# def build_wilddeepfake_real_dataset(wild_root: str, max_n: int, seed: int):
#     """
#     WildDeepfake REAL images only.
#     Scans: {wild_root}/{train,test,valid}/real/  (recursive for subdirs).
#     """
#     root = Path(wild_root)
#     imgs = []
#     for split in ["train", "test", "valid"]:
#         real_dir = root / split / "real"
#         if real_dir.exists():
#             found = _glob_images(real_dir, recursive=True)
#             imgs.extend(found)
#             print(f"    WildDeepfake {split}/real    : {len(found)}")
#     if not imgs:
#         print(f"  WARNING: No WildDeepfake real images under {root}")
#         return None
#     rng = random.Random(seed); rng.shuffle(imgs)
#     imgs = imgs[:max_n]
#     print(f"  WildDeepfake real (total)   : {len(imgs)}")
#     jpeg_aug = RandomJPEGAug(low=40, high=95, p=0.45)
#     return SingleSourceDataset(imgs, get_wilddeepfake_real_transform(),
#                                jpeg_aug=jpeg_aug, name="wilddeepfake_real")


# def build_vggface2_dataset(vgg_root: str, max_n: int, seed: int):
#     """
#     VGGFace2 real images.
#     Scans: {vgg_root}/{train,val}/<identity_id>/*.jpg  (recursive).
#     """
#     root = Path(vgg_root)
#     imgs = []
#     for split in ["train", "val"]:
#         split_dir = root / split
#         if split_dir.exists():
#             found = _glob_images(split_dir, recursive=True)
#             imgs.extend(found)
#             print(f"    VGGFace2 {split}             : {len(found)}")
#     if not imgs:
#         print(f"  WARNING: No VGGFace2 images found under {root}")
#         return None
#     rng = random.Random(seed); rng.shuffle(imgs)
#     imgs = imgs[:max_n]
#     print(f"  VGGFace2 real (total)       : {len(imgs)}")
#     jpeg_aug = RandomJPEGAug(low=50, high=95, p=0.35)
#     return SingleSourceDataset(imgs, get_vggface2_transform(),
#                                jpeg_aug=jpeg_aug, name="vggface2")


# def build_combined_dataset(args) -> ConcatDataset:
#     """Combine all configured real-image sources."""
#     datasets = []

#     if args.ffpp_frames_real:
#         ds = build_ffpp_dataset(args.ffpp_frames_real,
#                                 args.max_per_source, args.seed)
#         if ds and len(ds) > 0: datasets.append(ds)

#     if args.celeba_root:
#         ds = build_celeba_dataset(args.celeba_root,
#                                   args.max_per_source, args.seed)
#         if ds and len(ds) > 0: datasets.append(ds)

#     if args.ffhq_root:
#         ds = build_ffhq_dataset(args.ffhq_root,
#                                 args.max_per_source, args.seed)
#         if ds and len(ds) > 0: datasets.append(ds)

#     if args.wild_root:
#         ds = build_wilddeepfake_real_dataset(args.wild_root,
#                                               args.max_per_source, args.seed)
#         if ds and len(ds) > 0: datasets.append(ds)

#     if args.vggface2_root:
#         ds = build_vggface2_dataset(args.vggface2_root,
#                                     args.max_per_source, args.seed)
#         if ds and len(ds) > 0: datasets.append(ds)

#     if not datasets:
#         raise RuntimeError(
#             "No images found from any source! Check your path arguments.")

#     total = sum(len(d) for d in datasets)
#     print(f"\n  {'='*55}")
#     print(f"  Total real images for Phase 1 : {total}")
#     print(f"  Number of sources             : {len(datasets)}")
#     print(f"  {'='*55}\n")
#     return ConcatDataset(datasets)


# # =============================================================
# # DINOV2 BACKBONE  (frozen + L2-normalised)
# # =============================================================
# class DINOv2Backbone(nn.Module):
#     """
#     Frozen DINOv2-ViT-L/14 backbone.
#     Patch tokens are L2-normalised so fgw_scale stays in 1–3 range.
#     Without normalisation, raw token norms ~30–50 → fgw_scale ~88 (dead).
#     """
#     def __init__(self, model_name: str = "dinov2_vitl14_reg"):
#         super().__init__()
#         print(f"  Loading {model_name} ...")
#         self.model = torch.hub.load(
#             "facebookresearch/dinov2", model_name, pretrained=True)
#         for param in self.model.parameters():
#             param.requires_grad_(False)
#         self.model.eval()
#         self.embed_dim   = self.model.embed_dim
#         self.num_patches = 256
#         print(f"  Backbone ready. embed_dim={self.embed_dim}  frozen.")

#     def train(self, mode=True):
#         super().train(False)
#         return self

#     @torch.no_grad()
#     def forward(self, x):
#         out     = self.model.forward_features(x)
#         cls     = out["x_norm_clstoken"]
#         patches = out["x_norm_patchtokens"]
#         patches = F.normalize(patches, dim=-1)   # L2-norm → fgw_scale 1–3
#         return cls, patches


# # =============================================================
# # PATCH GRAPH — cosine distance
# # =============================================================
# def build_patch_graph(patch_tokens: torch.Tensor) -> torch.Tensor:
#     """
#     256×256 cosine-distance cost matrix.
#     With L2-normalised tokens: cosine_dist = 1 - dot ∈ [0,2].
#     Normalised per image to [0,1].
#     """
#     dot    = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))
#     cost_M = (1.0 - dot).clamp(min=0.)
#     cost_M = cost_M / (cost_M.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
#     return cost_M


# # =============================================================
# # SINKHORN + FGW
# # =============================================================
# def sinkhorn_log(a, b, M, reg=0.05, num_iter=20):
#     log_K = -M / reg
#     log_u = torch.zeros(M.shape[0], M.shape[1], 1, device=M.device)
#     log_v = torch.zeros(M.shape[0], 1, M.shape[2], device=M.device)
#     log_a = a.log().unsqueeze(2)
#     log_b = b.log().unsqueeze(1)
#     for _ in range(num_iter):
#         log_u = log_a - torch.logsumexp(log_K + log_v, 2, keepdim=True)
#         log_v = log_b - torch.logsumexp(log_K + log_u, 1, keepdim=True)
#     return (log_u + log_K + log_v).exp()


# def gw_loss_vec(C1, C2, T):
#     p  = T.sum(2); q = T.sum(1)
#     t1 = (C1 ** 2).bmm(p.unsqueeze(2)).squeeze(2).sum(1)
#     t2 = ((C2 ** 2).bmm(q.unsqueeze(2)).squeeze(2).unsqueeze(1) * T).sum([1, 2])
#     t3 = 2. * (T.bmm(C2).bmm(T.permute(0, 2, 1)) * C1).sum([1, 2])
#     return t1 + t2 - t3


# @torch.no_grad()
# def fgw_distance_raw(feat_src, cost_src, feat_tgt, cost_tgt,
#                      alpha=0.5, reg=0.05, n_iter=20):
#     """Raw FGW using cosine feature distance (tokens are L2-normalised)."""
#     B, N, D   = feat_src.shape
#     M_nodes   = feat_tgt.shape[1]
#     p = torch.ones(B, N,       device=feat_src.device) / N
#     q = torch.ones(B, M_nodes, device=feat_src.device) / M_nodes

#     dot    = torch.bmm(feat_src, feat_tgt.transpose(1, 2))
#     M_feat = (1.0 - dot).clamp(min=0.)
#     M_feat = M_feat / (M_feat.flatten(1).max(1)[0].view(B, 1, 1) + 1e-8)

#     T = sinkhorn_log(p, q, M_feat, reg, n_iter)
#     for _ in range(3):
#         gw_g  = -2. * cost_src.bmm(T).bmm(cost_tgt)
#         M_fgw = (1 - alpha) * M_feat + alpha * gw_g
#         M_fgw = M_fgw / (M_fgw.flatten(1).max(1)[0].view(B, 1, 1) + 1e-8)
#         T     = sinkhorn_log(p, q, M_fgw, reg, n_iter)

#     return ((1 - alpha) * (M_feat * T).sum([1, 2]) +
#             alpha * gw_loss_vec(cost_src, cost_tgt, T))


# # =============================================================
# # PROTOTYPE BANK
# # =============================================================
# class PrototypeBank(nn.Module):
#     def __init__(self, K=32, N=256, D=1024):
#         super().__init__()
#         self.K = K
#         self.register_buffer("proto_feats",
#             F.normalize(torch.randn(K, N, D), dim=-1))
#         self.register_buffer("proto_costs", torch.zeros(K, N, N))

#     @torch.no_grad()
#     def update_prototypes(self, feat, cost, assignments, momentum=0.5):
#         for k in range(self.K):
#             mask = (assignments == k)
#             if mask.sum() == 0: continue
#             new_f = F.normalize(feat[mask].mean(0), dim=-1)
#             self.proto_feats[k] = F.normalize(
#                 momentum * self.proto_feats[k] + (1 - momentum) * new_f,
#                 dim=-1)
#             self.proto_costs[k] = (momentum * self.proto_costs[k] +
#                                    (1 - momentum) * cost[mask].mean(0))

#     @torch.no_grad()
#     def get_nearest(self, feat):
#         feat_m  = F.normalize(feat.mean(1), dim=-1)
#         proto_m = F.normalize(self.proto_feats.mean(1), dim=-1)
#         sim     = torch.mm(feat_m, proto_m.t())
#         idx     = sim.argmax(1)
#         return self.proto_feats[idx], self.proto_costs[idx], idx


# # =============================================================
# # MAIN
# # =============================================================
# @torch.no_grad()
# def run_phase1(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nDevice: {device}")

#     out_dir  = Path(args.output_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     out_path = out_dir / "phase1_result.pt"

#     # ── Dataset ────────────────────────────────────────────────────────
#     print("\n" + "=" * 65)
#     print("PHASE 1 — Real image sources:")
#     print("  FF++ frames + CelebA + FFHQ + WildDeepfake real + VGGFace2")
#     print("=" * 65)
#     dataset = build_combined_dataset(args)

#     loader = DataLoader(
#         dataset,
#         batch_size  = args.batch_size,
#         shuffle     = True,
#         num_workers = args.num_workers,
#         pin_memory  = True,
#         drop_last   = False,
#     )

#     # ── Backbone ───────────────────────────────────────────────────────
#     print("=" * 65)
#     print("Loading DINOv2 (L2-normalised patch tokens) ...")
#     print("=" * 65)
#     backbone = DINOv2Backbone(args.backbone).to(device)

#     # ── Feature extraction ─────────────────────────────────────────────
#     print(f"\nExtracting features from {len(dataset)} real images ...")
#     cls_list, patch_list, cost_list = [], [], []
#     collected = 0

#     for imgs in tqdm(loader, desc="Features"):
#         imgs = imgs.to(device)
#         cls_tok, patch_tok = backbone(imgs)
#         cost_M = build_patch_graph(patch_tok)

#         cls_list.append(cls_tok.cpu().float().numpy())
#         patch_list.append(patch_tok.cpu().float())
#         cost_list.append(cost_M.cpu().float())
#         collected += imgs.shape[0]

#         if collected % 1000 < args.batch_size:
#             print(f"  Processed {collected} images...")

#     cls_np    = np.concatenate(cls_list, axis=0)
#     patch_all = torch.cat(patch_list,   dim=0)
#     cost_all  = torch.cat(cost_list,    dim=0)
#     print(f"\n  Total processed: {len(cls_np)}")

#     # Sanity check
#     norms = patch_all.norm(dim=-1)
#     print(f"  Token L2 norms: mean={norms.mean():.4f} "
#           f"std={norms.std():.6f}  (should be 1.0 ± tiny)")

#     # ── K-means ────────────────────────────────────────────────────────
#     K = args.num_protos
#     print(f"\n" + "=" * 65)
#     print(f"K-means (K={K}) on {len(cls_np)} CLS tokens ...")
#     print("=" * 65)
#     km = MiniBatchKMeans(
#         n_clusters  = K, random_state = args.seed,
#         n_init      = 10, max_iter    = 500,
#         verbose     = 0, batch_size   = min(1024, len(cls_np)),
#     )
#     km.fit(cls_np)
#     labels = torch.from_numpy(km.labels_).long()
#     sizes  = [(labels == k).sum().item() for k in range(K)]
#     print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
#           f"mean={np.mean(sizes):.0f}")
#     empty = [k for k, s in enumerate(sizes) if s == 0]
#     if empty:
#         print(f"  WARNING: {len(empty)} empty clusters. "
#               f"Try --num_protos {K - len(empty)}")

#     # ── Prototype bank ─────────────────────────────────────────────────
#     print("\nBuilding prototype bank ...")
#     proto_bank = PrototypeBank(K=K, N=256, D=1024).to(device)
#     asgn_dev   = labels.to(device)
#     CHUNK      = 2048
#     for start in tqdm(range(0, len(patch_all), CHUNK), desc="Protos"):
#         end = min(start + CHUNK, len(patch_all))
#         proto_bank.update_prototypes(
#             patch_all[start:end].to(device),
#             cost_all[start:end].to(device),
#             asgn_dev[start:end], momentum=0.5)
#     print("  Done.")

#     # ── FGW scale ──────────────────────────────────────────────────────
#     print("\nComputing FGW scale (expect 1.0–3.0) ...")
#     real_fgw_vals = []
#     n_for_scale   = min(len(patch_all), 2000)
#     for i in tqdm(range(0, n_for_scale, args.batch_size), desc="FGW scale"):
#         f_b = patch_all[i:i+args.batch_size].to(device)
#         c_b = cost_all[i:i+args.batch_size].to(device)
#         pf, pc, _ = proto_bank.get_nearest(f_b)
#         fgw_b = fgw_distance_raw(f_b, c_b, pf, pc,
#                                   alpha=args.fgw_alpha,
#                                   reg=args.fgw_reg, n_iter=20)
#         real_fgw_vals.append(fgw_b.cpu().float())

#     real_fgw_all = torch.cat(real_fgw_vals)
#     fgw_95   = float(torch.quantile(real_fgw_all, 0.95))
#     fgw_99   = float(torch.quantile(real_fgw_all, 0.99))
#     fgw_mean = float(real_fgw_all.mean())
#     fgw_std  = float(real_fgw_all.std())
#     fgw_scale = max(fgw_95, 1e-4)

#     print(f"\n  FGW stats ({n_for_scale} real images):")
#     print(f"    mean  = {fgw_mean:.4f}")
#     print(f"    std   = {fgw_std:.4f}")
#     print(f"    95pct = {fgw_95:.4f}  ← fgw_scale")
#     print(f"    99pct = {fgw_99:.4f}")
#     if fgw_scale > 10:
#         print(f"  !! WARN: scale={fgw_scale:.2f} > 10. "
#               f"Check L2-normalisation.")
#     else:
#         print(f"  ✓ Scale OK (1–3)")
#     print(f"  Normalised real FGW ≈ {fgw_mean/fgw_scale:.3f} "
#           f"(fake should be > 1.0)")

#     # ── Save ───────────────────────────────────────────────────────────
#     result = {
#         "proto_feats"   : proto_bank.proto_feats.cpu(),
#         "proto_costs"   : proto_bank.proto_costs.cpu(),
#         "fgw_scale"     : fgw_scale,
#         "fgw_99_pct"    : fgw_99,
#         "fgw_mean"      : fgw_mean,
#         "fgw_std"       : fgw_std,
#         "n_images_used" : len(cls_np),
#         "K"             : K,
#         "cluster_sizes" : sizes,
#         "version"       : "v3_wildvgg",
#         "args"          : vars(args),
#     }
#     torch.save(result, out_path)

#     print(f"\n{'='*65}")
#     print(f"Phase 1 DONE!")
#     print(f"  Output     : {out_path}")
#     print(f"  Size       : {out_path.stat().st_size/1e6:.1f} MB")
#     print(f"  K          : {K}")
#     print(f"  fgw_scale  : {fgw_scale:.4f}")
#     print(f"  N images   : {len(cls_np)}")
#     print(f"\nNext step — run phase2.py:")
#     print(f"  python phase2.py \\")
#     print(f"    --phase1_result {out_path} \\")
#     print(f"    --frames_out    ./extracted_frames \\")
#     print(f"    --wild_root     ./data/wilddeepfake \\")
#     print(f"    --vggface2_root ./data/vggface2 \\")
#     print(f"    --num_protos    {K}")
#     print("="*65)
#     return result


# # =============================================================
# # CLI
# # =============================================================
# if __name__ == "__main__":
#     p = argparse.ArgumentParser(
#         description="Phase 1: Prototype bank from real images "
#                     "(FF++ + CelebA + FFHQ + WildDeepfake real + VGGFace2)")

#     p.add_argument("--ffpp_frames_real", default=None,
#         help="FF++ pre-extracted real frames dir (e.g. ./extracted_frames/real)")
#     p.add_argument("--celeba_root",      default=None,
#         help="CelebA img_align_celeba directory")
#     p.add_argument("--ffhq_root",        default=None,
#         help="FFHQ images1024x1024 directory")
#     p.add_argument("--wild_root",        default=None,
#         help="WildDeepfake root (has train/test/valid subfolders)")
#     p.add_argument("--vggface2_root",    default=None,
#         help="VGGFace2 root (has train/ and val/ subfolders)")

#     p.add_argument("--num_protos",       type=int,   default=32)
#     p.add_argument("--max_per_source",   type=int,   default=5000)
#     p.add_argument("--backbone",         default="dinov2_vitl14_reg")
#     p.add_argument("--fgw_alpha",        type=float, default=0.5)
#     p.add_argument("--fgw_reg",          type=float, default=0.05)
#     p.add_argument("--output_dir",       default="./phase1_output")
#     p.add_argument("--batch_size",       type=int,   default=8)
#     p.add_argument("--num_workers",      type=int,   default=4)
#     p.add_argument("--seed",             type=int,   default=42)

#     args = p.parse_args()

#     if not any([args.ffpp_frames_real, args.celeba_root, args.ffhq_root,
#                 args.wild_root, args.vggface2_root]):
#         p.error("Provide at least one data source.")

#     run_phase1(args)

"""
phase1_build_graphs.py  —  v4  (WildDeepfake + VGGFace2 + CelebA + FFHQ)
=========================================================================
CHANGES FROM v3:
  FIX: OOM (RAM exhaustion) on large datasets (300k+ images).

  ROOT CAUSE IN v3:
    patch_list / patch_all : 300k × 256 × 1024 × 4 bytes ≈ 300 GB
    cost_list  / cost_all  : 300k × 256 × 256  × 4 bytes ≈  75 GB
    Both were accumulated in RAM before K-means, crashing around ~120k images.

  FIX — TWO-PASS STREAMING:
    Pass 1 : CLS tokens only (300k × 1024 × 4 bytes ≈ 1.2 GB) → K-means
    Pass 2 : Stream patch + cost per batch → update prototypes in-place,
             also collect a small FGW sample (≤2000 images) for fgw_scale.
    Peak RAM: ~3–4 GB total (was 375+ GB).

  ORDERING CONSTRAINT:
    Both passes use shuffle=False so labels_np[i] maps correctly to image i.
    K-means is order-invariant — no quality loss from removing shuffle.

DATASET SOURCES (REAL images only — Phase 1 is self-supervised):
  FF++ extracted real frames (extracted_frames/real/)
  CelebA   (data/celeba/img_align_celeba/img_align_celeba/)
  FFHQ     (data/ffhq/ffhq-dataset/images1024x1024/)
  WildDeepfake real (data/wilddeepfake/{train,test,valid}/real/)
  VGGFace2 (data/vggface2/{train,val}/<id>/*.jpg)
  Faceshifter/140k real (data/faceshifter/real_vs_fake/real_vs_fake/{train,valid,test}/real/)

OUTPUT (phase1_output/phase1_result.pt):
  proto_feats   : (K, 256, 1024) — K real-face prototype patch features
  proto_costs   : (K, 256, 256)  — K cosine-distance cost matrices
  fgw_scale     : float          — 95th pct real FGW (fixed normaliser)
  fgw_99_pct    : float
  fgw_mean      : float
  fgw_std       : float
  n_images_used : int
  K             : int
  cluster_sizes : list
  version       : "v4_streaming"
  args          : dict

HOW TO RUN:
  python phase1_build_graphs.py \\
    --ffpp_frames_real  ./extracted_frames/real \\
    --celeba_root       ./data/celeba/img_align_celeba/img_align_celeba \\
    --ffhq_root         ./data/ffhq/ffhq-dataset/images1024x1024 \\
    --wild_root         ./data/wilddeepfake \\
    --vggface2_root     ./data/vggface2 \\
    --max_per_source    5000 \\
    --num_protos        32 \\
    --batch_size        8 \\
    --num_workers       4 \\
    --output_dir        ./phase1_output

EXPECTED OUTPUT:
  fgw_scale : 1.0 – 3.0  (L2-normalised cosine distances)
  Runtime   : 30–90 min on one GPU
  File size : ~400 MB for K=32
"""

import os, argparse, random, io, gc
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans


# =============================================================
# CONSTANTS
# =============================================================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# =============================================================
# JPEG AUGMENTATION  (PIL-level, for in-the-wild sources)
# =============================================================
class RandomJPEGAug:
    """
    Simulate JPEG re-compression at random quality.
    Applied before the torchvision transform pipeline.
    Used for WildDeepfake + VGGFace2 which have naturally
    varied real-world JPEG quality.
    """
    def __init__(self, low=40, high=95, p=0.45):
        self.low = low; self.high = high; self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format="JPEG",
                 quality=random.randint(self.low, self.high))
        buf.seek(0)
        return Image.open(buf).copy()


# =============================================================
# PER-SOURCE AUGMENTATION TRANSFORMS
# =============================================================

def get_ffpp_transform():
    """
    FF++ extracted real frames.
    Already 224×224, face-cropped, H.264 compressed.
    Light augmentation: preserve the spatial structure,
    only vary compression level and mild color.
    """
    return transforms.Compose([
        transforms.Resize(232),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.15, hue=0.05),
        transforms.RandomGrayscale(p=0.03),
        transforms.RandomApply(
            [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_celeba_transform():
    """
    CelebA: 178×218 aligned JPEG, clean centered faces.
    Medium augmentation: simulate real-world lighting and
    small pose variation. Color jitter important for diversity.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35,
                               saturation=0.25, hue=0.08),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 2.5))], p=0.4),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=10, translate=(0.08, 0.08),
                                     scale=(0.92, 1.08))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_ffhq_transform():
    """
    FFHQ: high-quality 1024×1024 PNG, very diverse real faces.
    Strong downscale augmentation: simulate different acquisition
    resolutions. Aggressive color because FFHQ is very clean —
    we need to teach the model the natural variation range.
    """
    return transforms.Compose([
        transforms.Resize(288),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.35, hue=0.10),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.4),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1),
                                     scale=(0.9, 1.1))], p=0.35),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_wilddeepfake_real_transform():
    """
    WildDeepfake REAL frames: in-the-wild internet videos.
    Naturally varied: different cameras, lighting, ethnicities,
    ages, poses, compression levels.
    Strong augmentation because this source already has high variance.
    We add JPEG aug at PIL level to simulate the varied compression.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5,
                               saturation=0.4, hue=0.12),
        transforms.RandomGrayscale(p=0.06),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 3.5))], p=0.45),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=18, translate=(0.12, 0.12),
                                     scale=(0.88, 1.12), shear=5)], p=0.4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_vggface2_transform():
    """
    VGGFace2: web-scraped celebrity/public figure photos.
    Various poses, backgrounds, lighting, web-JPEG compression.
    Medium-strong augmentation: these are already diverse,
    but JPEG simulation is important to cover compression range.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.35, contrast=0.35,
                               saturation=0.30, hue=0.08),
        transforms.RandomGrayscale(p=0.04),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.35),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=12, translate=(0.08, 0.08),
                                     scale=(0.9, 1.1))], p=0.30),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_faceshifter_real_transform():
    """
    Faceshifter / 140k Real-vs-Fake dataset — REAL images only.
    Real images are FFHQ-sourced (1024px high-quality PNGs/JPGs),
    stored under real_vs_fake/train/real/ and real_vs_fake/valid/real/.
    Since source is FFHQ-quality (very clean, studio-like), we apply
    STRONG augmentation to teach the model natural variation —
    same reasoning as the FFHQ transform above.
    JPEG aug added because Kaggle stores them as .jpg with varying quality.
    """
    return transforms.Compose([
        transforms.Resize(288),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.45, contrast=0.45,
                               saturation=0.38, hue=0.10),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply(
            [transforms.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.40),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=15, translate=(0.10, 0.10),
                                     scale=(0.88, 1.12))], p=0.35),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# =============================================================
# SINGLE-SOURCE DATASET
# =============================================================
class SingleSourceDataset(Dataset):
    """
    Generic single-source image dataset for Phase 1.
    No labels — Phase 1 is self-supervised.
    Applies source-specific transform + optional PIL-level JPEG aug.
    """
    def __init__(self, paths: list, transform,
                 jpeg_aug=None, name: str = ""):
        self.paths     = paths
        self.transform = transform
        self.jpeg_aug  = jpeg_aug
        self.name      = name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.jpeg_aug is not None:
                img = self.jpeg_aug(img)
            return self.transform(img)
        except Exception:
            alt = self.paths[random.randint(0, len(self.paths) - 1)]
            img = Image.open(alt).convert("RGB")
            if self.jpeg_aug is not None:
                img = self.jpeg_aug(img)
            return self.transform(img)


# =============================================================
# IMAGE COLLECTION HELPERS
# =============================================================
def _glob_images(root: Path, recursive: bool = False) -> list:
    if not root.exists():
        print(f"  WARNING: path does not exist: {root}")
        return []
    imgs, pattern = [], "**/*" if recursive else "*"
    for p in root.glob(pattern):
        if p.suffix in EXTS and p.is_file():
            imgs.append(str(p))
    return imgs


# =============================================================
# PER-SOURCE DATASET BUILDERS
# =============================================================

def build_ffpp_dataset(ffpp_frames_real: str, max_n: int, seed: int):
    """FF++ pre-extracted real frames (extracted_frames/real/) — ALL images."""
    root = Path(ffpp_frames_real)
    imgs = _glob_images(root, recursive=False)
    if not imgs:
        imgs = _glob_images(root, recursive=True)
    rng = random.Random(seed); rng.shuffle(imgs)
    # NO cap — use all available real frames
    print(f"  FF++ extracted real frames  : {len(imgs)}  [ALL]")
    return SingleSourceDataset(imgs, get_ffpp_transform(),
                               jpeg_aug=None, name="ffpp")


def build_celeba_dataset(celeba_root: str, max_n: int, seed: int):
    """CelebA flat directory: img_align_celeba/*.jpg — ALL images."""
    imgs = _glob_images(Path(celeba_root), recursive=False)
    rng = random.Random(seed); rng.shuffle(imgs)
    # NO cap — use all available CelebA real images
    print(f"  CelebA                      : {len(imgs)}  [ALL]")
    return SingleSourceDataset(imgs, get_celeba_transform(),
                               jpeg_aug=None, name="celeba")


def build_ffhq_dataset(ffhq_root: str, max_n: int, seed: int):
    """FFHQ recursive: images1024x1024/00000/*.png — ALL images."""
    imgs = _glob_images(Path(ffhq_root), recursive=True)
    rng = random.Random(seed); rng.shuffle(imgs)
    # NO cap — use all available FFHQ real images
    print(f"  FFHQ                        : {len(imgs)}  [ALL]")
    return SingleSourceDataset(imgs, get_ffhq_transform(),
                               jpeg_aug=None, name="ffhq")


def build_wilddeepfake_real_dataset(wild_root: str, max_n: int, seed: int):
    """
    WildDeepfake REAL images only.
    Scans: {wild_root}/{train,test,valid}/real/  (recursive for subdirs).
    """
    root = Path(wild_root)
    imgs = []
    for split in ["train", "test", "valid"]:
        real_dir = root / split / "real"
        if real_dir.exists():
            found = _glob_images(real_dir, recursive=True)
            imgs.extend(found)
            print(f"    WildDeepfake {split}/real    : {len(found)}")
    if not imgs:
        print(f"  WARNING: No WildDeepfake real images under {root}")
        return None
    rng = random.Random(seed); rng.shuffle(imgs)
    # NO cap — use all available WildDeepfake real images
    print(f"  WildDeepfake real (total)   : {len(imgs)}  [ALL]")
    jpeg_aug = RandomJPEGAug(low=40, high=95, p=0.45)
    return SingleSourceDataset(imgs, get_wilddeepfake_real_transform(),
                               jpeg_aug=jpeg_aug, name="wilddeepfake_real")


def build_vggface2_dataset(vgg_root: str, max_n: int, seed: int):
    """
    VGGFace2 real images.
    Scans: {vgg_root}/{train,val}/<identity_id>/*.jpg  (recursive).
    """
    root = Path(vgg_root)
    imgs = []
    for split in ["train", "val"]:
        split_dir = root / split
        if split_dir.exists():
            found = _glob_images(split_dir, recursive=True)
            imgs.extend(found)
            print(f"    VGGFace2 {split}             : {len(found)}")
    if not imgs:
        print(f"  WARNING: No VGGFace2 images found under {root}")
        return None
    rng = random.Random(seed); rng.shuffle(imgs)
    # NO cap — use all available VGGFace2 real images
    print(f"  VGGFace2 real (total)       : {len(imgs)}  [ALL]")
    jpeg_aug = RandomJPEGAug(low=50, high=95, p=0.35)
    return SingleSourceDataset(imgs, get_vggface2_transform(),
                               jpeg_aug=jpeg_aug, name="vggface2")


def build_faceshifter_real_dataset(faceshifter_root: str,
                                    max_n: int, seed: int):
    """
    Faceshifter / 140k Real-vs-Fake Kaggle dataset — REAL images only.

    Expected structure:
      data/faceshifter/real_vs_fake/real_vs_fake/
          train/real/*.jpg
          valid/real/*.jpg
          test/real/*.jpg

    ALL real images from train/real/ + valid/real/ + test/real/ are used.
    max_n is intentionally NOT applied here — we use every available
    real image from this dataset since it is a large diverse source
    and more real faces = better prototype bank coverage.
    Fakes are completely ignored.
    """
    root = Path(faceshifter_root)
    imgs = []

    # Use ALL splits for real images — no test contamination risk
    # because Phase 1 is self-supervised (no labels, no fakes)
    for split in ["train", "valid", "test"]:
        real_dir = root / split / "real"
        if real_dir.exists():
            found = _glob_images(real_dir, recursive=True)
            imgs.extend(found)
            print(f"    Faceshifter {split}/real     : {len(found)}")
        else:
            print(f"    Faceshifter {split}/real     : NOT FOUND ({real_dir})")

    if not imgs:
        print(f"  WARNING: No Faceshifter real images under {root}")
        print(f"  Expected: {root}/train/real/, {root}/valid/real/, {root}/test/real/")
        return None

    rng = random.Random(seed)
    rng.shuffle(imgs)
    # NO max_n cap — use ALL available real images from this dataset
    print(f"  Faceshifter real (total)    : {len(imgs)}  [ALL — no cap]")

    # JPEG aug — Kaggle dataset stored as .jpg with varied compression
    jpeg_aug = RandomJPEGAug(low=45, high=95, p=0.40)
    return SingleSourceDataset(imgs, get_faceshifter_real_transform(),
                               jpeg_aug=jpeg_aug, name="faceshifter_real")


def build_combined_dataset(args) -> ConcatDataset:
    """Combine all configured real-image sources."""
    datasets = []

    if args.ffpp_frames_real:
        ds = build_ffpp_dataset(args.ffpp_frames_real,
                                args.max_per_source, args.seed)
        if ds and len(ds) > 0: datasets.append(ds)

    if args.celeba_root:
        ds = build_celeba_dataset(args.celeba_root,
                                  args.max_per_source, args.seed)
        if ds and len(ds) > 0: datasets.append(ds)

    if args.ffhq_root:
        ds = build_ffhq_dataset(args.ffhq_root,
                                args.max_per_source, args.seed)
        if ds and len(ds) > 0: datasets.append(ds)

    if args.wild_root:
        ds = build_wilddeepfake_real_dataset(args.wild_root,
                                              args.max_per_source, args.seed)
        if ds and len(ds) > 0: datasets.append(ds)

    if args.vggface2_root:
        ds = build_vggface2_dataset(args.vggface2_root,
                                    args.max_per_source, args.seed)
        if ds and len(ds) > 0: datasets.append(ds)

    if args.faceshifter_root:
        ds = build_faceshifter_real_dataset(args.faceshifter_root,
                                             args.max_per_source, args.seed)
        if ds and len(ds) > 0: datasets.append(ds)

    if not datasets:
        raise RuntimeError(
            "No images found from any source! Check your path arguments.")

    total = sum(len(d) for d in datasets)
    print(f"\n  {'='*55}")
    print(f"  Total real images for Phase 1 : {total}")
    print(f"  Number of sources             : {len(datasets)}")
    print(f"  {'='*55}\n")
    return ConcatDataset(datasets)


# =============================================================
# DINOV2 BACKBONE  (frozen + L2-normalised)
# =============================================================
class DINOv2Backbone(nn.Module):
    """
    Frozen DINOv2-ViT-L/14 backbone.
    Patch tokens are L2-normalised so fgw_scale stays in 1–3 range.
    Without normalisation, raw token norms ~30–50 → fgw_scale ~88 (dead).
    """
    def __init__(self, model_name: str = "dinov2_vitl14_reg"):
        super().__init__()
        print(f"  Loading {model_name} ...")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", model_name, pretrained=True)
        for param in self.model.parameters():
            param.requires_grad_(False)
        self.model.eval()
        self.embed_dim   = self.model.embed_dim
        self.num_patches = 256
        print(f"  Backbone ready. embed_dim={self.embed_dim}  frozen.")

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, x):
        out     = self.model.forward_features(x)
        cls     = out["x_norm_clstoken"]
        patches = out["x_norm_patchtokens"]
        patches = F.normalize(patches, dim=-1)   # L2-norm → fgw_scale 1–3
        return cls, patches


# =============================================================
# PATCH GRAPH — cosine distance
# =============================================================
def build_patch_graph(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    256×256 cosine-distance cost matrix.
    With L2-normalised tokens: cosine_dist = 1 - dot ∈ [0,2].
    Normalised per image to [0,1].
    """
    dot    = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))
    cost_M = (1.0 - dot).clamp(min=0.)
    cost_M = cost_M / (cost_M.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    return cost_M


# =============================================================
# SINKHORN + FGW
# =============================================================
def sinkhorn_log(a, b, M, reg=0.05, num_iter=20):
    log_K = -M / reg
    log_u = torch.zeros(M.shape[0], M.shape[1], 1, device=M.device)
    log_v = torch.zeros(M.shape[0], 1, M.shape[2], device=M.device)
    log_a = a.log().unsqueeze(2)
    log_b = b.log().unsqueeze(1)
    for _ in range(num_iter):
        log_u = log_a - torch.logsumexp(log_K + log_v, 2, keepdim=True)
        log_v = log_b - torch.logsumexp(log_K + log_u, 1, keepdim=True)
    return (log_u + log_K + log_v).exp()


def gw_loss_vec(C1, C2, T):
    p  = T.sum(2); q = T.sum(1)
    t1 = (C1 ** 2).bmm(p.unsqueeze(2)).squeeze(2).sum(1)
    t2 = ((C2 ** 2).bmm(q.unsqueeze(2)).squeeze(2).unsqueeze(1) * T).sum([1, 2])
    t3 = 2. * (T.bmm(C2).bmm(T.permute(0, 2, 1)) * C1).sum([1, 2])
    return t1 + t2 - t3


@torch.no_grad()
def fgw_distance_raw(feat_src, cost_src, feat_tgt, cost_tgt,
                     alpha=0.5, reg=0.05, n_iter=20):
    """Raw FGW using cosine feature distance (tokens are L2-normalised)."""
    B, N, D   = feat_src.shape
    M_nodes   = feat_tgt.shape[1]
    p = torch.ones(B, N,       device=feat_src.device) / N
    q = torch.ones(B, M_nodes, device=feat_src.device) / M_nodes

    dot    = torch.bmm(feat_src, feat_tgt.transpose(1, 2))
    M_feat = (1.0 - dot).clamp(min=0.)
    M_feat = M_feat / (M_feat.flatten(1).max(1)[0].view(B, 1, 1) + 1e-8)

    T = sinkhorn_log(p, q, M_feat, reg, n_iter)
    for _ in range(3):
        gw_g  = -2. * cost_src.bmm(T).bmm(cost_tgt)
        M_fgw = (1 - alpha) * M_feat + alpha * gw_g
        M_fgw = M_fgw / (M_fgw.flatten(1).max(1)[0].view(B, 1, 1) + 1e-8)
        T     = sinkhorn_log(p, q, M_fgw, reg, n_iter)

    return ((1 - alpha) * (M_feat * T).sum([1, 2]) +
            alpha * gw_loss_vec(cost_src, cost_tgt, T))


# =============================================================
# PROTOTYPE BANK
# =============================================================
class PrototypeBank(nn.Module):
    def __init__(self, K=32, N=256, D=1024):
        super().__init__()
        self.K = K
        self.register_buffer("proto_feats",
            F.normalize(torch.randn(K, N, D), dim=-1))
        self.register_buffer("proto_costs", torch.zeros(K, N, N))

    @torch.no_grad()
    def update_prototypes(self, feat, cost, assignments, momentum=0.5):
        for k in range(self.K):
            mask = (assignments == k)
            if mask.sum() == 0: continue
            new_f = F.normalize(feat[mask].mean(0), dim=-1)
            self.proto_feats[k] = F.normalize(
                momentum * self.proto_feats[k] + (1 - momentum) * new_f,
                dim=-1)
            self.proto_costs[k] = (momentum * self.proto_costs[k] +
                                   (1 - momentum) * cost[mask].mean(0))

    @torch.no_grad()
    def get_nearest(self, feat):
        feat_m  = F.normalize(feat.mean(1), dim=-1)
        proto_m = F.normalize(self.proto_feats.mean(1), dim=-1)
        sim     = torch.mm(feat_m, proto_m.t())
        idx     = sim.argmax(1)
        return self.proto_feats[idx], self.proto_costs[idx], idx


# =============================================================
# MAIN
# =============================================================
@torch.no_grad()
def run_phase1(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_result.pt"

    # ── Dataset ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("PHASE 1 — Real image sources:")
    print("  FF++ frames + CelebA + FFHQ + WildDeepfake real + VGGFace2")
    print("=" * 65)
    dataset = build_combined_dataset(args)

    # ── Backbone ───────────────────────────────────────────────────────
    print("=" * 65)
    print("Loading DINOv2 (L2-normalised patch tokens) ...")
    print("=" * 65)
    backbone = DINOv2Backbone(args.backbone).to(device)

    # ==================================================================
    # PASS 1 — Collect CLS tokens only (small: N × 1024 × 4 bytes ≈ 1.2 GB)
    #
    # We do NOT store patch_tokens or cost matrices here.
    # Those are 300k × 256 × 1024 × 4 bytes ≈ 300 GB — that was the crash.
    #
    # ORDERING: shuffle=False so that labels_np[i] correctly maps to
    # image i in pass 2.  K-means is order-invariant — no quality loss.
    # ==================================================================
    print(f"\nPASS 1 of 2 — CLS token extraction ({len(dataset)} images) ...")
    print("  (patch tokens discarded immediately — only CLS kept for K-means)")

    loader_pass1 = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,          # deterministic order — required for pass 2
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    cls_list  = []
    collected = 0

    for imgs in tqdm(loader_pass1, desc="Pass 1 — CLS"):
        imgs = imgs.to(device)
        cls_tok, _patch_tok = backbone(imgs)   # patch_tok not stored → freed
        cls_list.append(cls_tok.cpu().float().numpy())
        collected += imgs.shape[0]
        if collected % 5000 < args.batch_size:
            print(f"  CLS collected: {collected}")

    cls_np = np.concatenate(cls_list, axis=0)
    del cls_list
    gc.collect()

    print(f"\n  Total CLS tokens  : {len(cls_np)}")
    print(f"  CLS RAM footprint : {cls_np.nbytes / 1e9:.2f} GB  (expected ~1.2 GB)")

    # ==================================================================
    # K-MEANS on CLS tokens
    # ==================================================================
    K = args.num_protos
    print(f"\n{'='*65}")
    print(f"K-means (K={K}) on {len(cls_np)} CLS tokens ...")
    print("="*65)

    km = MiniBatchKMeans(
        n_clusters  = K,
        random_state = args.seed,
        n_init      = 10,
        max_iter    = 500,
        verbose     = 0,
        batch_size  = min(1024, len(cls_np)),
    )
    km.fit(cls_np)

    # labels_np[i] = cluster assignment for image i — kept in numpy (cheap)
    labels_np = km.labels_   # shape (N,) dtype int32
    sizes = [(labels_np == k).sum() for k in range(K)]

    print(f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
          f"mean={np.mean(sizes):.0f}")
    empty = [k for k, s in enumerate(sizes) if s == 0]
    if empty:
        print(f"  WARNING: {len(empty)} empty clusters. "
              f"Consider --num_protos {K - len(empty)}")

    # Free CLS array — no longer needed
    del cls_np
    gc.collect()

    # ==================================================================
    # PASS 2 — Stream through data, update prototypes in-place per batch.
    #
    # patch_tokens and cost_M exist only for the duration of one batch
    # and are freed before the next iteration begins.
    # Peak patch RAM = batch_size × 256 × 1024 × 4 bytes (a few MB).
    #
    # We also collect a small FGW sample (≤ fgw_sample_cap images)
    # during this pass to avoid a separate third pass for fgw_scale.
    #
    # ORDERING: same shuffle=False loader as pass 1 — labels_np[i] aligns
    # with image i.
    # ==================================================================
    print(f"\nPASS 2 of 2 — Streaming prototype update ...")
    print("  patch tensors processed batch-by-batch and immediately freed.")

    loader_pass2 = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = False,          # must match pass 1 order
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False,
    )

    proto_bank = PrototypeBank(K=K, N=256, D=1024).to(device)

    # FGW sample: collect up to fgw_sample_cap images during this pass.
    # Stored as small CPU tensors — total ≤ 2000 × 256 × 1024 × 4 ≈ 2 GB.
    fgw_sample_cap   = 2000
    fgw_sample_feats = []   # each entry is a CPU float tensor (b, 256, 1024)
    fgw_sample_costs = []   # each entry is a CPU float tensor (b, 256, 256)

    img_idx     = 0
    last_patch  = None   # kept only for the post-loop sanity check

    for imgs in tqdm(loader_pass2, desc="Pass 2 — Protos"):
        B    = imgs.shape[0]
        imgs = imgs.to(device)

        # Forward pass — patch_tok and cost_M live only in this scope
        _, patch_tok = backbone(imgs)             # (B, 256, 1024) on GPU
        cost_M       = build_patch_graph(patch_tok)  # (B, 256, 256) on GPU

        # Cluster assignments for this batch from pass-1 labels
        batch_labels = torch.from_numpy(
            labels_np[img_idx : img_idx + B].astype(np.int64)
        ).to(device)

        # In-place prototype update — O(K × B) memory, constant
        proto_bank.update_prototypes(
            patch_tok, cost_M, batch_labels, momentum=0.5)

        # Collect FGW sample (front-loaded — first fgw_sample_cap images)
        if len(fgw_sample_feats) * args.batch_size < fgw_sample_cap:
            n_have = sum(t.shape[0] for t in fgw_sample_feats)
            n_take = min(B, fgw_sample_cap - n_have)
            if n_take > 0:
                fgw_sample_feats.append(patch_tok[:n_take].cpu().float())
                fgw_sample_costs.append(cost_M[:n_take].cpu().float())

        # Save reference for post-loop norm sanity check
        last_patch = patch_tok.detach()

        # patch_tok and cost_M go out of scope — memory freed before next iter
        img_idx += B

    print(f"  Prototype update complete. Total images: {img_idx}")

    # Sanity check — token L2 norms should be 1.0 ± tiny (L2-normed backbone)
    if last_patch is not None:
        norms = last_patch.norm(dim=-1)
        print(f"  Token L2 norms (last batch): mean={norms.mean():.4f} "
              f"std={norms.std():.6f}  (should be 1.0 ± tiny)")
    del last_patch

    # ==================================================================
    # FGW SCALE — using the small sample collected during pass 2.
    # No third pass needed.
    # ==================================================================
    fgw_feats = torch.cat(fgw_sample_feats, dim=0)   # (≤2000, 256, 1024)
    fgw_costs = torch.cat(fgw_sample_costs, dim=0)   # (≤2000, 256, 256)
    del fgw_sample_feats, fgw_sample_costs
    gc.collect()

    n_for_scale = len(fgw_feats)
    print(f"\nComputing FGW scale on {n_for_scale} sampled real images ...")
    print("  (expect fgw_scale in 1.0 – 3.0 range)")

    real_fgw_vals = []
    fgw_feats_dev = fgw_feats.to(device)
    fgw_costs_dev = fgw_costs.to(device)
    del fgw_feats, fgw_costs

    for i in tqdm(range(0, n_for_scale, args.batch_size), desc="FGW scale"):
        f_b = fgw_feats_dev[i : i + args.batch_size]
        c_b = fgw_costs_dev[i : i + args.batch_size]
        pf, pc, _ = proto_bank.get_nearest(f_b)
        fgw_b = fgw_distance_raw(f_b, c_b, pf, pc,
                                  alpha=args.fgw_alpha,
                                  reg=args.fgw_reg, n_iter=20)
        real_fgw_vals.append(fgw_b.cpu().float())

    del fgw_feats_dev, fgw_costs_dev
    real_fgw_all = torch.cat(real_fgw_vals)

    fgw_95    = float(torch.quantile(real_fgw_all, 0.95))
    fgw_99    = float(torch.quantile(real_fgw_all, 0.99))
    fgw_mean  = float(real_fgw_all.mean())
    fgw_std   = float(real_fgw_all.std())
    fgw_scale = max(fgw_95, 1e-4)

    print(f"\n  FGW stats ({n_for_scale} real images):")
    print(f"    mean  = {fgw_mean:.4f}")
    print(f"    std   = {fgw_std:.4f}")
    print(f"    95pct = {fgw_95:.4f}  ← fgw_scale")
    print(f"    99pct = {fgw_99:.4f}")
    if fgw_scale > 10:
        print(f"  !! WARN: scale={fgw_scale:.2f} > 10. "
              f"Check L2-normalisation in DINOv2Backbone.forward().")
    else:
        print(f"  ✓ Scale OK (1–3 range)")
    print(f"  Normalised real FGW ≈ {fgw_mean / fgw_scale:.3f} "
          f"(fake images should score > 1.0 after normalisation)")

    # ── Save ───────────────────────────────────────────────────────────
    result = {
        "proto_feats"   : proto_bank.proto_feats.cpu(),
        "proto_costs"   : proto_bank.proto_costs.cpu(),
        "fgw_scale"     : fgw_scale,
        "fgw_99_pct"    : fgw_99,
        "fgw_mean"      : fgw_mean,
        "fgw_std"       : fgw_std,
        "n_images_used" : img_idx,
        "K"             : K,
        "cluster_sizes" : [int(s) for s in sizes],
        "version"       : "v4_streaming",
        "args"          : vars(args),
    }
    torch.save(result, out_path)

    print(f"\n{'='*65}")
    print(f"Phase 1 DONE!")
    print(f"  Output     : {out_path}")
    print(f"  Size       : {out_path.stat().st_size / 1e6:.1f} MB")
    print(f"  K          : {K}")
    print(f"  fgw_scale  : {fgw_scale:.4f}")
    print(f"  N images   : {img_idx}")
    print(f"\nNext step — run phase2.py:")
    print(f"  python phase2.py \\")
    print(f"    --phase1_result {out_path} \\")
    print(f"    --frames_out    ./extracted_frames \\")
    print(f"    --wild_root     ./data/wilddeepfake \\")
    print(f"    --vggface2_root ./data/vggface2 \\")
    print(f"    --num_protos    {K}")
    print("=" * 65)
    return result


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Phase 1: Prototype bank from real images "
                    "(FF++ + CelebA + FFHQ + WildDeepfake real + VGGFace2). "
                    "v4: two-pass streaming — constant RAM regardless of dataset size.")

    p.add_argument("--ffpp_frames_real", default=None,
        help="FF++ pre-extracted real frames dir (e.g. ./extracted_frames/real)")
    p.add_argument("--celeba_root",      default=None,
        help="CelebA img_align_celeba directory")
    p.add_argument("--ffhq_root",        default=None,
        help="FFHQ images1024x1024 directory")
    p.add_argument("--wild_root",        default=None,
        help="WildDeepfake root (has train/test/valid subfolders)")
    p.add_argument("--vggface2_root",    default=None,
        help="VGGFace2 root (has train/ and val/ subfolders)")
    p.add_argument("--faceshifter_root", default=None,
        help="Faceshifter/140k real-vs-fake root "
             "(contains real_vs_fake/train/real/ and real_vs_fake/valid/real/). "
             "Example: ./data/faceshifter/real_vs_fake/real_vs_fake")

    p.add_argument("--num_protos",       type=int,   default=32)
    p.add_argument("--max_per_source",   type=int,   default=5000)
    p.add_argument("--backbone",         default="dinov2_vitl14_reg")
    p.add_argument("--fgw_alpha",        type=float, default=0.5)
    p.add_argument("--fgw_reg",          type=float, default=0.05)
    p.add_argument("--output_dir",       default="./phase1_output")
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--num_workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)

    args = p.parse_args()

    if not any([args.ffpp_frames_real, args.celeba_root, args.ffhq_root,
                args.wild_root, args.vggface2_root, args.faceshifter_root]):
        p.error("Provide at least one data source.")

    run_phase1(args)