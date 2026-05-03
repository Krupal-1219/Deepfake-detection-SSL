# # """
# # phase2.py  —  v3  (WildDeepfake + FF++ training | VGGFace2 cross-domain eval)
# # ==============================================================================
# # DATASET CHANGES FROM PREVIOUS VERSION:
# #   REMOVED : DFDC  (deleted dataset)

# #   TRAINING (labelled, supervised):
# #     FF++         → real frames from extracted_frames/real/
# #                    fake frames from extracted_frames/fake/
# #     WildDeepfake → real from data/wilddeepfake/{train,valid}/real/
# #                    fake from data/wilddeepfake/{train,valid}/fake/

# #   VALIDATION (in-distribution, from training mix):
# #     10% split of combined FF++ + WildDeepfake training set

# #   CROSS-DOMAIN TEST (replaces DFDC):
# #     VGGFace2 → real from data/vggface2/{train,val}/ (no fakes — identity only)
# #     NOTE: VGGFace2 is REAL ONLY. We use it to verify the model does NOT
# #     misclassify real faces from a completely different domain as fake.
# #     A separate fake set from WildDeepfake test/ is used for the test split.

# #   FULL CROSS-DOMAIN TEST SET:
# #     real  → VGGFace2 val/ (unseen domain real faces)
# #     fake  → WildDeepfake test/fake/ (unseen test split fakes)
# #     This tests: does the model generalise to new real faces AND new fakes?

# # AUGMENTATION PER DATASET:
# #   FF++ real frames (train)    : Light — already 224x224 face-cropped PNGs
# #   FF++ fake frames (train)    : Light + JPEG re-compression simulation
# #   WildDeepfake real (train)   : Strong — in-the-wild, diverse quality
# #   WildDeepfake fake (train)   : Strong + extra JPEG + blur (GAN artifacts vary)
# #   VGGFace2 real (cross-test)  : Val-only (no augment) — clean eval

# # ALL OTHER STEPS PRESERVED FROM v8:
# #   ✔ DINOv2 backbone frozen, L2-normalised patch tokens
# #   ✔ Cosine distance patch graph
# #   ✔ FGW fixed-scale normalisation (loaded from phase1_result.pt)
# #   ✔ Top-K prototype matching (k_match=3)
# #   ✔ SRM frequency branch with Dropout(0.5)
# #   ✔ Feature noise injection (noise_std=0.01)
# #   ✔ Prototype EMA update during training (momentum=0.99)
# #   ✔ FGW margin loss (lambda_fgw=1.0, margin=0.5)
# #   ✔ LR warmup + cosine schedule
# #   ✔ Gradient clipping (max_norm=1.0)

# # ADDED (Objective 4 — PS):
# #   ✔ Heatmap saving: frame-level + region-level overlays saved to
# #     --heatmap_dir (default: ./heatmaps) during final cross-domain evaluation.
# #     For each sample a side-by-side PNG is saved:
# #       [original face | patch heatmap overlay | prediction label]
# #     Controlled by --save_heatmaps (flag) and --heatmap_n (max images to save).

# # HOW TO RUN:
# #   python phase2.py \\
# #     --phase1_result  ./phase1_output/phase1_result.pt \\
# #     --frames_out     ./extracted_frames \\
# #     --wild_root      ./data/wilddeepfake \\
# #     --vggface2_root  ./data/vggface2 \\
# #     --max_per_class  5000 \\
# #     --num_protos     32 \\
# #     --k_match        3 \\
# #     --lambda_fgw     1.0 \\
# #     --fgw_margin     0.5 \\
# #     --epochs         20 \\
# #     --batch_size     4 \\
# #     --num_workers    4 \\
# #     --ckpt_dir       ./checkpoints \\
# #     --save_heatmaps \\
# #     --heatmap_dir    ./heatmaps \\
# #     --heatmap_n      200
# # """

# # import os, cv2, time, argparse, random, io
# # from pathlib import Path
# # import numpy as np
# # import torch
# # import torch.nn as nn
# # import torch.nn.functional as F
# # import torch.optim as optim
# # from torch.utils.data import Dataset, DataLoader, ConcatDataset
# # from torchvision import transforms
# # from PIL import Image
# # from tqdm import tqdm
# # from sklearn.cluster import MiniBatchKMeans
# # from sklearn.metrics import roc_auc_score, accuracy_score


# # # =============================================================
# # # CONSTANTS
# # # =============================================================
# # MEAN = [0.485, 0.456, 0.406]
# # STD  = [0.229, 0.224, 0.225]
# # EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# # # =============================================================
# # # JPEG AUGMENTATION
# # # =============================================================
# # class RandomJPEG:
# #     """Simulate JPEG re-compression at random quality (30–95)."""
# #     def __init__(self, low=30, high=95, p=0.5):
# #         self.low = low; self.high = high; self.p = p

# #     def __call__(self, img: Image.Image) -> Image.Image:
# #         if random.random() > self.p:
# #             return img
# #         buf = io.BytesIO()
# #         img.save(buf, format="JPEG",
# #                  quality=random.randint(self.low, self.high))
# #         buf.seek(0)
# #         return Image.open(buf).copy()


# # # =============================================================
# # # PER-DATASET AUGMENTATION TRANSFORMS
# # # =============================================================

# # def get_ffpp_train_transform():
# #     """
# #     FF++ REAL + FAKE frames, training split.
# #     Already 224×224, face-cropped, H.264 compressed.
# #     Light spatial augmentation (preserve face structure),
# #     JPEG simulation to vary compression artifacts.
# #     """
# #     return transforms.Compose([
# #         transforms.Resize(232),
# #         transforms.RandomCrop(224),
# #         transforms.RandomHorizontalFlip(p=0.5),
# #         transforms.ColorJitter(brightness=0.25, contrast=0.25,
# #                                saturation=0.2, hue=0.06),
# #         transforms.RandomGrayscale(p=0.03),
# #         transforms.RandomApply(
# #             [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
# #         transforms.ToTensor(),
# #         transforms.Normalize(MEAN, STD),
# #     ])


# # def get_wilddeepfake_train_transform():
# #     """
# #     WildDeepfake REAL + FAKE frames, training split.
# #     In-the-wild internet videos: diverse quality, cameras, lighting.
# #     Strong augmentation to cover natural variation range.
# #     JPEG aug at PIL level handles the varied compression.
# #     """
# #     return transforms.Compose([
# #         transforms.Resize(256),
# #         transforms.RandomCrop(224),
# #         transforms.RandomHorizontalFlip(p=0.5),
# #         transforms.ColorJitter(brightness=0.5, contrast=0.5,
# #                                saturation=0.4, hue=0.12),
# #         transforms.RandomGrayscale(p=0.06),
# #         transforms.RandomApply(
# #             [transforms.GaussianBlur(5, sigma=(0.1, 3.5))], p=0.45),
# #         transforms.RandomApply(
# #             [transforms.RandomAffine(degrees=18, translate=(0.12, 0.12),
# #                                      scale=(0.88, 1.12), shear=5)], p=0.4),
# #         transforms.ToTensor(),
# #         transforms.Normalize(MEAN, STD),
# #     ])


# # def get_val_transform():
# #     """
# #     Validation / test transform — no augmentation.
# #     Used for: FF++/WildDeepfake val split + VGGFace2 cross-domain test.
# #     """
# #     return transforms.Compose([
# #         transforms.Resize(224),
# #         transforms.CenterCrop(224),
# #         transforms.ToTensor(),
# #         transforms.Normalize(MEAN, STD),
# #     ])


# # # =============================================================
# # # IMAGE COLLECTION HELPERS
# # # =============================================================
# # def _glob_images(root: Path, recursive: bool = False) -> list:
# #     if not root.exists():
# #         print(f"  WARNING: path does not exist: {root}")
# #         return []
# #     imgs, pattern = [], "**/*" if recursive else "*"
# #     for p in root.glob(pattern):
# #         if p.suffix in EXTS and p.is_file():
# #             imgs.append(str(p))
# #     return imgs


# # # =============================================================
# # # DATASET CLASSES
# # # =============================================================

# # class LabelledDataset(Dataset):
# #     """
# #     Labelled dataset: list of (path, label) pairs.
# #     label: 0=real, 1=fake.
# #     """
# #     def __init__(self, samples: list, transform,
# #                  jpeg_aug=None, name: str = ""):
# #         self.samples   = samples
# #         self.transform = transform
# #         self.jpeg_aug  = jpeg_aug
# #         self.name      = name

# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         path, label = self.samples[idx]
# #         try:
# #             img = Image.open(path).convert("RGB")
# #             if self.jpeg_aug is not None:
# #                 img = self.jpeg_aug(img)
# #             return self.transform(img), torch.tensor(label, dtype=torch.long)
# #         except Exception:
# #             alt_idx = random.randint(0, len(self.samples) - 1)
# #             path, label = self.samples[alt_idx]
# #             img = Image.open(path).convert("RGB")
# #             if self.jpeg_aug is not None:
# #                 img = self.jpeg_aug(img)
# #             return self.transform(img), torch.tensor(label, dtype=torch.long)


# # # =============================================================
# # # DATASET BUILDERS
# # # =============================================================

# # def build_ffpp_samples(frames_out: str, max_per_class=None):
# #     """
# #     Collect FF++ samples from pre-extracted frames.
# #     Expected structure:
# #       extracted_frames/real/*.png  ← real frames (label=0)
# #       extracted_frames/fake/*.png  ← fake frames (label=1)
# #     """
# #     root      = Path(frames_out)
# #     real_dir  = root / "real"
# #     fake_dir  = root / "fake"

# #     def collect(d):
# #         imgs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
# #         return [str(p) for p in (imgs[:max_per_class] if max_per_class else imgs)]

# #     real_imgs = collect(real_dir) if real_dir.exists() else []
# #     fake_imgs = collect(fake_dir) if fake_dir.exists() else []

# #     if not real_imgs:
# #         print(f"  WARNING: No FF++ real frames found in {real_dir}")
# #     if not fake_imgs:
# #         print(f"  WARNING: No FF++ fake frames found in {fake_dir}")

# #     samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
# #     print(f"  FF++ frames: real={len(real_imgs)}  fake={len(fake_imgs)}")
# #     return samples


# # def build_wilddeepfake_samples(wild_root: str, splits: list,
# #                                 max_per_class=None, seed=42):
# #     """
# #     Collect WildDeepfake samples from specified splits.
# #     Structure:
# #       data/wilddeepfake/{split}/real/  ← real face images (label=0)
# #       data/wilddeepfake/{split}/fake/  ← deepfake images  (label=1)
# #     Each real/ and fake/ may have subdirs (one per video source).

# #     splits: list of split names, e.g. ["train","valid"] for training
# #                                        ["test"] for cross-domain test
# #     """
# #     root  = Path(wild_root)
# #     real_imgs, fake_imgs = [], []

# #     for split in splits:
# #         real_dir = root / split / "real"
# #         fake_dir = root / split / "fake"

# #         if real_dir.exists():
# #             found = _glob_images(real_dir, recursive=True)
# #             real_imgs.extend(found)
# #             print(f"    WildDeepfake {split}/real : {len(found)}")
# #         else:
# #             print(f"    WildDeepfake {split}/real : NOT FOUND ({real_dir})")

# #         if fake_dir.exists():
# #             found = _glob_images(fake_dir, recursive=True)
# #             fake_imgs.extend(found)
# #             print(f"    WildDeepfake {split}/fake : {len(found)}")
# #         else:
# #             print(f"    WildDeepfake {split}/fake : NOT FOUND ({fake_dir})")

# #     rng = random.Random(seed)
# #     rng.shuffle(real_imgs); rng.shuffle(fake_imgs)

# #     if max_per_class:
# #         real_imgs = real_imgs[:max_per_class]
# #         fake_imgs = fake_imgs[:max_per_class]

# #     samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
# #     print(f"  WildDeepfake {splits}: real={len(real_imgs)} fake={len(fake_imgs)}")
# #     return samples


# # def build_vggface2_samples(vgg_root: str, max_n=None, seed=42):
# #     """
# #     VGGFace2 REAL-ONLY cross-domain test set.
# #     VGGFace2 has no fakes. We use it to test cross-domain REAL detection
# #     (the model should NOT classify these as fakes).
# #     Structure: data/vggface2/{train,val}/<id>/*.jpg

# #     For cross-domain test we use val/ split only (train/ is larger
# #     and overlaps more with what the model was trained on).
# #     Returns samples with label=0 (all real).
# #     """
# #     root  = Path(vgg_root)
# #     imgs  = []
# #     # Use val/ for cross-domain eval (held out from training)
# #     for split in ["val", "train"]:
# #         split_dir = root / split
# #         if split_dir.exists():
# #             found = _glob_images(split_dir, recursive=True)
# #             imgs.extend(found)
# #             print(f"    VGGFace2 {split}           : {len(found)}")
# #     if not imgs:
# #         print(f"  WARNING: No VGGFace2 images under {root}")
# #         return []
# #     rng = random.Random(seed)
# #     rng.shuffle(imgs)
# #     if max_n: imgs = imgs[:max_n]
# #     print(f"  VGGFace2 real (total)       : {len(imgs)}")
# #     return [(p, 0) for p in imgs]


# # class CombinedDataset(Dataset):
# #     """
# #     Combines FF++ and WildDeepfake into one training dataset.
# #     Each source uses its own per-dataset augmentation.
# #     """
# #     def __init__(self, ffpp_samples, wild_samples,
# #                  ffpp_transform, wild_transform,
# #                  ffpp_jpeg=None, wild_jpeg=None):
# #         self.ffpp_samples  = ffpp_samples
# #         self.wild_samples  = wild_samples
# #         self.ffpp_transform = ffpp_transform
# #         self.wild_transform = wild_transform
# #         self.ffpp_jpeg     = ffpp_jpeg
# #         self.wild_jpeg     = wild_jpeg
# #         # Index: 0..len(ffpp)-1 → ffpp, len(ffpp).. → wild
# #         self.n_ffpp = len(ffpp_samples)

# #     def __len__(self):
# #         return len(self.ffpp_samples) + len(self.wild_samples)

# #     def __getitem__(self, idx):
# #         if idx < self.n_ffpp:
# #             path, label = self.ffpp_samples[idx]
# #             transform   = self.ffpp_transform
# #             jpeg_aug    = self.ffpp_jpeg
# #         else:
# #             path, label = self.wild_samples[idx - self.n_ffpp]
# #             transform   = self.wild_transform
# #             jpeg_aug    = self.wild_jpeg
# #         try:
# #             img = Image.open(path).convert("RGB")
# #             if jpeg_aug: img = jpeg_aug(img)
# #             return transform(img), torch.tensor(label, dtype=torch.long)
# #         except Exception:
# #             # Fallback to random sample
# #             fallback = random.randint(0, len(self) - 1)
# #             return self.__getitem__(fallback)


# # class CrossDomainTestDataset(Dataset):
# #     """
# #     Cross-domain test dataset.
# #     real:  VGGFace2 val images (label=0)
# #     fake:  WildDeepfake test/fake/ images (label=1)

# #     Val-only transform — no augmentation.
# #     Also returns image path for heatmap saving.
# #     """
# #     def __init__(self, vgg_samples, wild_test_fake_samples):
# #         # vgg_samples: list of (path, 0)
# #         # wild_test_fake_samples: list of (path, 1)
# #         self.samples   = vgg_samples + wild_test_fake_samples
# #         self.transform = get_val_transform()
# #         n_r = sum(1 for _, l in self.samples if l == 0)
# #         n_f = sum(1 for _, l in self.samples if l == 1)
# #         print(f"  Cross-domain test: real(VGGFace2)={n_r}  "
# #               f"fake(WildDF test)={n_f}")

# #     def __len__(self):
# #         return len(self.samples)

# #     def __getitem__(self, idx):
# #         path, label = self.samples[idx]
# #         try:
# #             img = Image.open(path).convert("RGB")
# #             return self.transform(img), torch.tensor(label, dtype=torch.long), path
# #         except Exception:
# #             alt = random.randint(0, len(self.samples) - 1)
# #             path, label = self.samples[alt]
# #             img = Image.open(path).convert("RGB")
# #             return self.transform(img), torch.tensor(label, dtype=torch.long), path


# # # =============================================================
# # # DATASET FACTORY — builds train/val/test splits
# # # =============================================================

# # def build_all_datasets(args):
# #     """
# #     Builds and returns train_ds, val_ds, cross_test_ds.

# #     train_ds     : FF++ (train split) + WildDeepfake (train+valid splits)
# #     val_ds       : 10% from combined train set
# #     cross_test_ds: VGGFace2 real (val) + WildDeepfake fake (test)
# #     """
# #     print("\n" + "="*60 + "  DATASETS")

# #     # ── FF++ samples ────────────────────────────────────────────────
# #     print("\nFF++ (pre-extracted frames):")
# #     ffpp_samples = build_ffpp_samples(args.frames_out,
# #                                        max_per_class=args.max_per_class)

# #     # ── WildDeepfake training samples ────────────────────────────────
# #     print("\nWildDeepfake (train + valid splits for training):")
# #     wild_train_samples = build_wilddeepfake_samples(
# #         args.wild_root, splits=["train", "valid"],
# #         max_per_class=args.max_per_class, seed=args.seed)

# #     # ── Combined train+val split ─────────────────────────────────────
# #     # Merge FF++ + WildDeepfake then split 90/10
# #     all_samples = ffpp_samples + wild_train_samples
# #     rng = random.Random(args.seed)
# #     rng.shuffle(all_samples)
# #     n_val   = int(len(all_samples) * 0.10)
# #     val_s   = all_samples[:n_val]
# #     train_s = all_samples[n_val:]

# #     # Separate each split back into FF++ and Wild for per-source transform
# #     def split_samples(samples):
# #         ffpp_p = str(Path(args.frames_out).resolve())
# #         wild_p = str(Path(args.wild_root).resolve())
# #         ffpp_s, wild_s, other_s = [], [], []
# #         for p, l in samples:
# #             ap = str(Path(p).resolve())
# #             if ap.startswith(ffpp_p):
# #                 ffpp_s.append((p, l))
# #             elif ap.startswith(wild_p):
# #                 wild_s.append((p, l))
# #             else:
# #                 other_s.append((p, l))
# #         return ffpp_s, wild_s + other_s

# #     train_ffpp, train_wild = split_samples(train_s)
# #     val_ffpp,   val_wild   = split_samples(val_s)

# #     # Training dataset: per-source augmentation
# #     ffpp_jpeg = RandomJPEG(low=30, high=95, p=0.5)
# #     wild_jpeg = RandomJPEG(low=40, high=95, p=0.45)

# #     train_ds = CombinedDataset(
# #         ffpp_samples  = train_ffpp,
# #         wild_samples  = train_wild,
# #         ffpp_transform= get_ffpp_train_transform(),
# #         wild_transform= get_wilddeepfake_train_transform(),
# #         ffpp_jpeg     = ffpp_jpeg,
# #         wild_jpeg     = wild_jpeg,
# #     )

# #     # Val dataset: val transform, merged (no per-source split needed for eval)
# #     val_ds = LabelledDataset(val_s, get_val_transform(),
# #                               jpeg_aug=None, name="val")

# #     # Stats
# #     n_train_r = sum(1 for _, l in train_s if l == 0)
# #     n_train_f = sum(1 for _, l in train_s if l == 1)
# #     n_val_r   = sum(1 for _, l in val_s   if l == 0)
# #     n_val_f   = sum(1 for _, l in val_s   if l == 1)
# #     print(f"\n  [train] total={len(train_s)}  "
# #           f"real={n_train_r}  fake={n_train_f}  "
# #           f"ratio={n_train_f/max(n_train_r,1):.2f}")
# #     print(f"    └ FF++={len(train_ffpp)}  "
# #           f"WildDF={len(train_wild)}")
# #     print(f"  [val  ] total={len(val_s)}  "
# #           f"real={n_val_r}  fake={n_val_f}")

# #     # ── Cross-domain test set ────────────────────────────────────────
# #     print("\nCross-domain test (VGGFace2 real + WildDeepfake test fake):")

# #     # VGGFace2 real images
# #     vgg_samples = build_vggface2_samples(
# #         args.vggface2_root,
# #         max_n=args.max_per_class, seed=args.seed)

# #     # WildDeepfake test/fake/ images
# #     print("WildDeepfake test/fake (cross-domain fakes):")
# #     wild_test_samples = build_wilddeepfake_samples(
# #         args.wild_root, splits=["test"],
# #         max_per_class=args.max_per_class, seed=args.seed)
# #     # Keep only fakes from test set
# #     wild_test_fakes = [(p, l) for p, l in wild_test_samples if l == 1]
# #     print(f"  WildDeepfake test fakes     : {len(wild_test_fakes)}")

# #     cross_test_ds = CrossDomainTestDataset(vgg_samples, wild_test_fakes)

# #     return train_ds, val_ds, cross_test_ds


# # # =============================================================
# # # FRAME EXTRACTION (FF++ only)
# # # =============================================================
# # def extract_frames_from_video(video_path, out_dir, n_frames=30):
# #     os.makedirs(out_dir, exist_ok=True)
# #     cap = cv2.VideoCapture(video_path)
# #     if not cap.isOpened(): return 0
# #     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #     if total == 0: cap.release(); return 0
# #     positions = np.linspace(0, total-1, min(n_frames, total),
# #                             dtype=int).tolist()
# #     stem = Path(video_path).stem; saved = 0
# #     for pos in positions:
# #         cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
# #         ret, frame = cap.read()
# #         if not ret: continue
# #         frame = cv2.resize(frame, (224, 224))
# #         cv2.imwrite(str(Path(out_dir) / f"{stem}_f{pos:06d}.png"), frame)
# #         saved += 1
# #     cap.release(); return saved


# # def extract_all_ffpp_frames(ffpp_root, frames_out, n_frames=30, max_videos=None):
# #     """Extract frames from FF++ videos into extracted_frames/real/ and /fake/."""
# #     ffpp_root = Path(ffpp_root)
# #     real_out  = Path(frames_out) / "real"
# #     fake_out  = Path(frames_out) / "fake"
# #     real_out.mkdir(parents=True, exist_ok=True)
# #     fake_out.mkdir(parents=True, exist_ok=True)

# #     MANIPS = ["Deepfakes", "Face2Face", "FaceSwap",
# #               "FaceShifter", "NeuralTextures", "DeepFakeDetection"]

# #     print("\n" + "="*60)
# #     print(f"STEP 0 — FF++ Frame extraction (n_frames={n_frames})")
# #     print("="*60)

# #     real_videos = []
# #     for sub in ["youtube", "actors"]:
# #         d = ffpp_root / "original_sequences" / sub / "c23" / "videos"
# #         if d.exists():
# #             real_videos += list(d.glob("*.mp4")) + list(d.glob("*.avi"))
# #     if max_videos: real_videos = real_videos[:max_videos // 2]
# #     print(f"Real videos: {len(real_videos)}")
# #     tr = sum(extract_frames_from_video(str(v), str(real_out), n_frames)
# #              for v in tqdm(real_videos, desc="Real"))
# #     print(f"  → {tr} real frames saved")

# #     fake_videos = []
# #     for manip in MANIPS:
# #         for pat in [ffpp_root / "manipulated_sequences" / manip / "c23" / "videos",
# #                     ffpp_root / "manipulated_sequences" / manip]:
# #             if pat.exists():
# #                 fake_videos += (list(pat.glob("*.mp4")) +
# #                                 list(pat.glob("*.avi")) +
# #                                 list(pat.rglob("*.mp4"))); break
# #     fake_videos = list(set(str(v) for v in fake_videos))
# #     random.shuffle(fake_videos)
# #     if max_videos: fake_videos = fake_videos[:max_videos // 2]
# #     print(f"Fake videos: {len(fake_videos)}")
# #     tf = sum(extract_frames_from_video(v, str(fake_out), n_frames)
# #              for v in tqdm(fake_videos, desc="Fake"))
# #     print(f"  → {tf} fake frames saved")


# # # =============================================================
# # # PART A — DINOv2 BACKBONE
# # # =============================================================
# # class DINOv2Backbone(nn.Module):
# #     """
# #     Frozen DINOv2-ViT-L/14 backbone.
# #     L2-normalises patch tokens → fgw_scale stays in 1–3 range.
# #     """
# #     def __init__(self, model_name: str = "dinov2_vitl14_reg"):
# #         super().__init__()
# #         print(f"  Loading {model_name} ...")
# #         self.model = torch.hub.load(
# #             "facebookresearch/dinov2", model_name, pretrained=True)
# #         for param in self.model.parameters():
# #             param.requires_grad_(False)
# #         self.model.eval()
# #         self.embed_dim   = self.model.embed_dim
# #         self.num_patches = 256
# #         print(f"  Backbone ready. embed_dim={self.embed_dim}  frozen.")

# #     def train(self, mode=True):
# #         super().train(False)
# #         return self

# #     @torch.no_grad()
# #     def forward(self, x):
# #         out     = self.model.forward_features(x)
# #         cls     = out["x_norm_clstoken"]
# #         patches = out["x_norm_patchtokens"]
# #         patches = F.normalize(patches, dim=-1)   # L2-norm
# #         return cls, patches


# # # =============================================================
# # # PART B — SRM FREQUENCY BRANCH
# # # =============================================================
# # class SRMConv(nn.Module):
# #     def __init__(self):
# #         super().__init__()
# #         kernels     = self._build_srm_kernels()
# #         kernels_rgb = kernels.repeat(1, 3, 1, 1) / 3.0
# #         self.register_buffer("weight", kernels_rgb)

# #     def _build_srm_kernels(self):
# #         b1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,2,-1,0],
# #                         [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2.
# #         b2 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],
# #                         [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2.
# #         b3 = np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],
# #                         [0,0,-1,0,0],[0,0,0,0,0]], np.float32) / 4.
# #         b4 = np.array([[0,0,0,0,0],[0,-1,0,0,0],[0,0,2,0,0],
# #                         [0,0,0,-1,0],[0,0,0,0,0]], np.float32) / 2.
# #         b5 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
# #                         [2,-6,8,-6,2],[-1,2,-2,2,-1]], np.float32) / 12.
# #         b6 = -np.ones((5,5), np.float32) / 24.; b6[2,2] = 1.
# #         bases = [b1,b2,b3,b4,b5,b6]
# #         all_k = []
# #         for b in bases:
# #             for r in range(4): all_k.append(np.rot90(b,r).copy())
# #         for b in bases: all_k.append(b.T.copy())
# #         return torch.tensor(
# #             np.stack(all_k[:30])[:,np.newaxis,:,:], dtype=torch.float32)

# #     def forward(self, x):
# #         with torch.no_grad():
# #             return F.conv2d(x, self.weight, padding=2).clamp(-2., 2.)


# # class FrequencyEncoder(nn.Module):
# #     def __init__(self, freq_dim=128):
# #         super().__init__()
# #         self.srm = SRMConv()
# #         self.encoder = nn.Sequential(
# #             nn.Conv2d(30, 64, 3, stride=2, padding=1),
# #             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
# #             nn.Conv2d(64, 128, 3, stride=2, padding=1),
# #             nn.BatchNorm2d(128), nn.ReLU(inplace=True),
# #             nn.AdaptiveAvgPool2d(4),
# #         )
# #         self.proj = nn.Sequential(
# #             nn.Flatten(),
# #             nn.Dropout(0.5),
# #             nn.Linear(128*4*4, freq_dim),
# #             nn.LayerNorm(freq_dim),
# #         )

# #     def forward(self, x):
# #         return self.proj(self.encoder(self.srm(x)))


# # # =============================================================
# # # PART C — PATCH GRAPH
# # # =============================================================
# # def build_patch_graph(patch_tokens: torch.Tensor) -> torch.Tensor:
# #     dot    = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))
# #     cost_M = (1.0 - dot).clamp(min=0.)
# #     cost_M = cost_M / (cost_M.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
# #     return cost_M


# # # =============================================================
# # # PART D — SINKHORN
# # # =============================================================
# # def sinkhorn_log(a, b, M, reg=0.05, num_iter=20):
# #     log_K = -M / reg
# #     log_u = torch.zeros(M.shape[0], M.shape[1], 1, device=M.device)
# #     log_v = torch.zeros(M.shape[0], 1, M.shape[2], device=M.device)
# #     log_a = a.log().unsqueeze(2)
# #     log_b = b.log().unsqueeze(1)
# #     for _ in range(num_iter):
# #         log_u = log_a - torch.logsumexp(log_K + log_v, 2, keepdim=True)
# #         log_v = log_b - torch.logsumexp(log_K + log_u, 1, keepdim=True)
# #     return (log_u + log_K + log_v).exp()


# # # =============================================================
# # # PART E — FGW DISTANCE
# # # =============================================================
# # def gw_loss_vec(C1, C2, T):
# #     p  = T.sum(2); q = T.sum(1)
# #     t1 = (C1**2).bmm(p.unsqueeze(2)).squeeze(2).sum(1)
# #     t2 = ((C2**2).bmm(q.unsqueeze(2)).squeeze(2).unsqueeze(1)*T).sum([1,2])
# #     t3 = 2.*(T.bmm(C2).bmm(T.permute(0,2,1))*C1).sum([1,2])
# #     return t1 + t2 - t3


# # def fgw_distance(feat_src, cost_src, feat_tgt, cost_tgt,
# #                  alpha=0.5, reg=0.05, n_iter=20, scale=None):
# #     B, N, D   = feat_src.shape
# #     M_nodes   = feat_tgt.shape[1]
# #     p = torch.ones(B, N,       device=feat_src.device) / N
# #     q = torch.ones(B, M_nodes, device=feat_src.device) / M_nodes

# #     dot    = torch.bmm(feat_src, feat_tgt.transpose(1, 2))
# #     M_feat = (1.0 - dot).clamp(min=0.)
# #     M_feat = M_feat / (M_feat.flatten(1).max(1)[0].view(B,1,1) + 1e-8)

# #     T = sinkhorn_log(p, q, M_feat, reg, n_iter)
# #     for _ in range(3):
# #         gw_g  = -2. * cost_src.bmm(T).bmm(cost_tgt)
# #         M_fgw = (1-alpha)*M_feat + alpha*gw_g
# #         M_fgw = M_fgw / (M_fgw.flatten(1).max(1)[0].view(B,1,1) + 1e-8)
# #         T     = sinkhorn_log(p, q, M_fgw, reg, n_iter)

# #     raw_dist = ((1-alpha)*(M_feat*T).sum([1,2]) +
# #                 alpha*gw_loss_vec(cost_src, cost_tgt, T))

# #     if scale is not None and scale > 0:
# #         return raw_dist / scale, T
# #     return raw_dist / (raw_dist.max().detach() + 1e-8), T


# # # =============================================================
# # # PART F — PROTOTYPE BANK
# # # =============================================================
# # class PrototypeBank(nn.Module):
# #     def __init__(self, K=32, N=256, D=1024):
# #         super().__init__()
# #         self.K = K
# #         self.register_buffer("proto_feats",
# #             F.normalize(torch.randn(K, N, D), dim=-1))
# #         self.register_buffer("proto_costs", torch.zeros(K, N, N))

# #     @torch.no_grad()
# #     def load_from_phase1(self, result: dict):
# #         pf = result["proto_feats"]
# #         pc = result["proto_costs"]
# #         K_saved = pf.shape[0]
# #         if K_saved != self.K:
# #             print(f"  NOTE: Phase1 K={K_saved} != model K={self.K}. "
# #                   f"Adjusting.")
# #             self.K = K_saved
# #             self.register_buffer("proto_feats", pf.clone())
# #             self.register_buffer("proto_costs", pc.clone())
# #         else:
# #             self.proto_feats.copy_(pf)
# #             self.proto_costs.copy_(pc)
# #         self.proto_feats = F.normalize(self.proto_feats, dim=-1)
# #         print(f"  Prototype bank loaded: K={self.K}  "
# #               f"feats={tuple(self.proto_feats.shape)}")

# #     @torch.no_grad()
# #     def update_prototypes(self, feat, cost, assignments, momentum=0.99):
# #         for k in range(self.K):
# #             mask = (assignments == k)
# #             if mask.sum() == 0: continue
# #             new_f = F.normalize(feat[mask].mean(0), dim=-1)
# #             self.proto_feats[k] = F.normalize(
# #                 momentum * self.proto_feats[k] + (1-momentum)*new_f, dim=-1)
# #             self.proto_costs[k] = (momentum * self.proto_costs[k] +
# #                                    (1-momentum) * cost[mask].mean(0))

# #     def get_nearest(self, feat, k_match=1):
# #         feat_mean  = F.normalize(feat.mean(1), dim=-1)
# #         proto_mean = F.normalize(self.proto_feats.mean(1), dim=-1)
# #         sim = torch.mm(feat_mean, proto_mean.t())

# #         if k_match == 1:
# #             idx = sim.argmax(1)
# #             return self.proto_feats[idx], self.proto_costs[idx], idx
# #         else:
# #             return sim.topk(min(k_match, self.K), dim=1).indices


# # # =============================================================
# # # PART E2 — TOP-K FGW
# # # =============================================================
# # def fgw_topk(feat_src, cost_src, proto_bank: PrototypeBank,
# #              k_match=3, alpha=0.5, reg=0.05, scale=None):
# #     B = feat_src.shape[0]
# #     if k_match <= 1:
# #         pf, pc, idx = proto_bank.get_nearest(feat_src, k_match=1)
# #         dist, T = fgw_distance(feat_src, cost_src, pf, pc,
# #                                alpha, reg, scale=scale)
# #         return dist, T

# #     topk_idx  = proto_bank.get_nearest(feat_src, k_match=k_match)
# #     all_dists = []; last_T = None
# #     for ki in range(k_match):
# #         idx_k  = topk_idx[:, ki]
# #         pf_k   = proto_bank.proto_feats[idx_k]
# #         pc_k   = proto_bank.proto_costs[idx_k]
# #         dist_k, T_k = fgw_distance(feat_src, cost_src, pf_k, pc_k,
# #                                    alpha, reg, scale=scale)
# #         all_dists.append(dist_k)
# #         if ki == 0: last_T = T_k

# #     min_dist = torch.stack(all_dists, dim=1).min(dim=1).values
# #     return min_dist, last_T


# # # =============================================================
# # # PART G — DETECTION HEAD
# # # =============================================================
# # class DetectionHead(nn.Module):
# #     def __init__(self, cls_dim=1024, freq_dim=128):
# #         super().__init__()
# #         fused_dim = cls_dim + freq_dim + 1   # 1153

# #         self.classifier = nn.Sequential(
# #             nn.LayerNorm(fused_dim),
# #             nn.Linear(fused_dim, 512),
# #             nn.GELU(),
# #             nn.Dropout(0.5),
# #             nn.Linear(512, 128),
# #             nn.GELU(),
# #             nn.Dropout(0.3),
# #             nn.Linear(128, 2),
# #         )
# #         self.patch_scorer = nn.Sequential(
# #             nn.Linear(cls_dim, 256),
# #             nn.GELU(),
# #             nn.Linear(256, 1),
# #         )

# #     def forward(self, cls_token, freq_feat, fgw_dist, patch_tokens, T=None):
# #         fused  = torch.cat([cls_token, freq_feat,
# #                             fgw_dist.unsqueeze(1)], dim=1)
# #         logits = self.classifier(fused)
# #         heatmap = self.patch_scorer(patch_tokens).squeeze(-1)
# #         if T is not None:
# #             ent   = -(T*(T+1e-8).log()).sum(-1)
# #             e_min = ent.min(1, keepdim=True)[0]
# #             e_max = ent.max(1, keepdim=True)[0]
# #             heatmap = heatmap + (ent-e_min)/(e_max-e_min+1e-8)
# #         return logits, torch.sigmoid(heatmap)


# # # =============================================================
# # # PART H — FULL DETECTOR
# # # =============================================================
# # class DeepfakeDetector(nn.Module):
# #     def __init__(self, backbone="dinov2_vitl14_reg", num_protos=32,
# #                  fgw_alpha=0.5, fgw_reg=0.05, freq_dim=128,
# #                  k_match=3, noise_std=0.01):
# #         super().__init__()
# #         self.backbone     = DINOv2Backbone(backbone)
# #         D = self.backbone.embed_dim
# #         N = self.backbone.num_patches
# #         self.freq_encoder = FrequencyEncoder(freq_dim=freq_dim)
# #         self.proto_bank   = PrototypeBank(K=num_protos, N=N, D=D)
# #         self.det_head     = DetectionHead(cls_dim=D, freq_dim=freq_dim)
# #         self.fgw_alpha    = fgw_alpha
# #         self.fgw_reg      = fgw_reg
# #         self.fgw_scale    = 1.0
# #         self.k_match      = k_match
# #         self.noise_std    = noise_std

# #     def forward(self, x):
# #         cls_tok, patch_tok = self.backbone(x)
# #         freq_feat          = self.freq_encoder(x)

# #         if self.training and self.noise_std > 0:
# #             patch_tok = patch_tok + torch.randn_like(patch_tok) * self.noise_std
# #             patch_tok = F.normalize(patch_tok, dim=-1)

# #         cost_M = build_patch_graph(patch_tok)
# #         fgw_dist, T = fgw_topk(
# #             patch_tok, cost_M, self.proto_bank,
# #             k_match=self.k_match, alpha=self.fgw_alpha,
# #             reg=self.fgw_reg, scale=self.fgw_scale)

# #         logits, heatmap = self.det_head(
# #             cls_tok, freq_feat, fgw_dist, patch_tok, T)
# #         return logits, heatmap, fgw_dist


# # # =============================================================
# # # PHASE 1 LOAD
# # # =============================================================
# # def load_phase1_result(phase1_path: str, model: DeepfakeDetector,
# #                        device: torch.device):
# #     path = Path(phase1_path)
# #     if not path.exists():
# #         raise FileNotFoundError(
# #             f"Phase 1 result not found: {path}\n"
# #             f"Run phase1_build_graphs.py first.")

# #     print(f"\n{'='*60}\nLOADING PHASE 1 FROM DISK\n{'='*60}")
# #     result = torch.load(path, map_location=device)
# #     model.proto_bank.load_from_phase1(result)

# #     raw_scale = float(result["fgw_scale"])
# #     if raw_scale > 10.0:
# #         corrected = raw_scale / 50.0
# #         print(f"\n  !! fgw_scale={raw_scale:.2f} > 10. "
# #               f"Phase1 built without L2-norm. "
# #               f"Correction: {raw_scale:.2f} → {corrected:.4f}")
# #         print(f"  !! Re-run phase1_build_graphs.py for best results.")
# #         model.fgw_scale = max(corrected, 1e-4)
# #     else:
# #         model.fgw_scale = max(raw_scale, 1e-4)

# #     version = result.get("version", "unknown")
# #     print(f"  fgw_scale = {model.fgw_scale:.4f}  (expected 1–3)")
# #     print(f"  K         = {result['K']}")
# #     print(f"  N images  = {result['n_images_used']}")
# #     print(f"  Version   = {version}")
# #     if version == "v3_wildvgg":
# #         print(f"  ✓ Built with WildDeepfake + VGGFace2 real images")
# #     print()


# # @torch.no_grad()
# # def phase1_ssl_warmup_inline(model: DeepfakeDetector,
# #                              loader: DataLoader,
# #                              device: torch.device,
# #                              max_samples: int = 5000):
# #     """
# #     Inline Phase 1 fallback: uses REAL images from the training loader.
# #     Filters label==0 (real). Builds prototype bank on the fly.
# #     """
# #     print("\n" + "="*60)
# #     print("PHASE 1 — Inline (real images from training loader)")
# #     print("  TIP: Run phase1_build_graphs.py for multi-source prototype bank")
# #     print("="*60)
# #     model.eval()
# #     K = model.proto_bank.K

# #     cls_list, patch_list, cost_list = [], [], []
# #     collected = 0

# #     for imgs, labels in loader:
# #         mask = (labels == 0)
# #         if mask.sum() == 0: continue
# #         imgs_real = imgs[mask].to(device)
# #         cls_tok, patch_tok = model.backbone(imgs_real)
# #         cost_M = build_patch_graph(patch_tok)

# #         cls_list.append(cls_tok.cpu().numpy())
# #         patch_list.append(patch_tok.cpu())
# #         cost_list.append(cost_M.cpu())
# #         collected += mask.sum().item()

# #         if collected % 500 < imgs_real.shape[0]:
# #             print(f"  Collected {collected} real faces...")
# #         if collected >= max_samples: break

# #     if not cls_list:
# #         raise RuntimeError("No real images found in loader.")

# #     cls_np    = np.concatenate(cls_list, axis=0)
# #     patch_all = torch.cat(patch_list, dim=0)
# #     cost_all  = torch.cat(cost_list,  dim=0)

# #     print(f"  K-means K={K} on {len(cls_np)} real faces ...")
# #     km = MiniBatchKMeans(n_clusters=K, random_state=42,
# #                          n_init=10, max_iter=300, verbose=0)
# #     km.fit(cls_np)
# #     asgn  = torch.from_numpy(km.labels_).long().to(device)
# #     sizes = [(asgn==k).sum().item() for k in range(K)]
# #     print(f"  Cluster sizes: min={min(sizes)} max={max(sizes)}")
# #     if min(sizes) == 0:
# #         print("  WARNING: empty cluster. Try --num_protos 16")

# #     model.proto_bank.update_prototypes(
# #         patch_all.to(device), cost_all.to(device), asgn, momentum=0.5)

# #     real_fgw_vals = []
# #     for i in range(0, min(len(patch_all), 500), 8):
# #         f_b = patch_all[i:i+8].to(device)
# #         c_b = cost_all[i:i+8].to(device)
# #         d_b, _ = fgw_topk(f_b, c_b, model.proto_bank,
# #                            k_match=1, alpha=model.fgw_alpha,
# #                            reg=model.fgw_reg, scale=None)
# #         real_fgw_vals.append(d_b.cpu())

# #     real_fgw_all = torch.cat(real_fgw_vals)
# #     model.fgw_scale = max(float(torch.quantile(real_fgw_all, 0.95)), 1e-4)
# #     print(f"  fgw_scale = {model.fgw_scale:.4f}  (expected 1–3)\n")


# # # =============================================================
# # # LOSS
# # # =============================================================

# # # ── 1. Focal Loss — fixes low fake recall by down-weighting easy real examples
# # class FocalLoss(nn.Module):
# #     """
# #     Focal Loss: FL(p) = -alpha * (1-p)^gamma * log(p)
# #     gamma=2 focuses training on hard misclassified fakes.
# #     alpha=0.75 upweights the fake class (minority class).
# #     Directly addresses the real:fake imbalance causing low recall.
# #     """
# #     def __init__(self, gamma=2.0, alpha=0.75):
# #         super().__init__()
# #         self.gamma = gamma
# #         self.alpha = alpha  # weight for fake class (label=1)

# #     def forward(self, logits, labels):
# #         probs   = torch.softmax(logits, dim=1)
# #         # gather prob of the correct class
# #         p_t     = probs[range(len(labels)), labels]
# #         # class weight: alpha for fake(1), 1-alpha for real(0)
# #         alpha_t = torch.where(labels == 1,
# #                               torch.tensor(self.alpha, device=logits.device),
# #                               torch.tensor(1.0 - self.alpha, device=logits.device))
# #         focal_w = alpha_t * (1 - p_t) ** self.gamma
# #         loss    = -(focal_w * torch.log(p_t + 1e-8))
# #         return loss.mean()


# # # ── 2. Supervised Contrastive Loss on CLS tokens
# # class SupConLoss(nn.Module):
# #     """
# #     Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).
# #     Applied on L2-normalised CLS tokens.
# #     Pulls real CLS embeddings together, pushes fake embeddings
# #     away from real clusters in the 1024-d feature space.
# #     temperature=0.07 is standard from the paper.
# #     """
# #     def __init__(self, temperature=0.07):
# #         super().__init__()
# #         self.temp = temperature

# #     def forward(self, features, labels):
# #         # features: (B, D) — CLS tokens, will be L2-normalised here
# #         # labels:   (B,)   — 0=real, 1=fake
# #         features = F.normalize(features, dim=-1)
# #         B = features.shape[0]
# #         if B < 2:
# #             return torch.tensor(0.0, device=features.device)

# #         # similarity matrix (B, B)
# #         sim = torch.mm(features, features.T) / self.temp

# #         # same-class mask, diagonal excluded (no self-contrast)
# #         labels_col = labels.view(-1, 1)
# #         mask = (labels_col == labels_col.T).float()
# #         mask.fill_diagonal_(0)

# #         # if no positive pairs exist in this batch, skip
# #         if mask.sum() == 0:
# #             return torch.tensor(0.0, device=features.device)

# #         # log-sum-exp over all negatives
# #         exp_sim   = torch.exp(sim)
# #         # exclude self from denominator
# #         self_mask = torch.ones_like(exp_sim)
# #         self_mask.fill_diagonal_(0)
# #         log_denom = torch.log((exp_sim * self_mask).sum(1, keepdim=True) + 1e-8)
# #         log_prob  = sim - log_denom

# #         # average loss over positive pairs per anchor
# #         n_pos = mask.sum(1).clamp(min=1)
# #         loss  = -(mask * log_prob).sum(1) / n_pos
# #         return loss.mean()


# # # ── 3. Prototype Contrastive Loss — uses the existing prototype bank
# # class ProtoConLoss(nn.Module):
# #     """
# #     Prototype Contrastive Loss.
# #     Real images: maximise cosine similarity to nearest prototype
# #                  (pull real faces toward real-face prototypes)
# #     Fake images: push similarity below (nearest_sim - margin)
# #                  (push fake faces away from real prototypes)
# #     Directly aligned with the FGW design philosophy.
# #     """
# #     def __init__(self, margin=0.4):
# #         super().__init__()
# #         self.margin = margin

# #     def forward(self, patch_feats, labels, proto_bank):
# #         # patch_feats: (B, 256, 1024) — L2-normalised patch tokens
# #         # labels:      (B,)
# #         feat_mean  = F.normalize(patch_feats.mean(1), dim=-1)   # (B, D)
# #         proto_mean = F.normalize(proto_bank.proto_feats.mean(1), dim=-1)  # (K, D)

# #         sim         = torch.mm(feat_mean, proto_mean.T)   # (B, K)
# #         nearest_sim = sim.max(1).values                   # (B,)

# #         real_mask = (labels == 0).float()
# #         fake_mask = (labels == 1).float()

# #         # real: maximise similarity → loss = 1 - sim
# #         loss_real = real_mask * (1.0 - nearest_sim)
# #         # fake: push similarity below (nearest_sim - margin)
# #         loss_fake = fake_mask * F.relu(nearest_sim + self.margin)

# #         return (loss_real + loss_fake).mean()


# # # ── 4. Combined Detection Loss
# # class DetectionLoss(nn.Module):
# #     """
# #     Combined loss:
# #       L = FocalLoss                     (fixes fake recall — replaces CrossEntropy)
# #         + lambda_fgw  * FGW_margin_loss (structural separation)
# #         + lambda_con  * SupConLoss      (CLS embedding separation)
# #         + lambda_proto* ProtoConLoss    (prototype alignment)

# #     lambda defaults: fgw=1.0, con=0.3, proto=0.2
# #     These are additive — each targets a different aspect of fakeness.
# #     """
# #     def __init__(self, lambda_fgw=1.0, margin=0.5,
# #                  lambda_con=0.3, lambda_proto=0.2,
# #                  focal_gamma=2.0, focal_alpha=0.75):
# #         super().__init__()
# #         self.focal      = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
# #         self.supcon     = SupConLoss(temperature=0.07)
# #         self.proto_con  = ProtoConLoss(margin=0.4)
# #         self.lam_fgw    = lambda_fgw
# #         self.lam_con    = lambda_con
# #         self.lam_proto  = lambda_proto
# #         self.m          = margin

# #     def forward(self, logits, labels, fgw_dist,
# #                 cls_tok=None, patch_feats=None, proto_bank=None):
# #         # 1. Focal loss (replaces CrossEntropy)
# #         l_focal = self.focal(logits, labels)

# #         # 2. FGW margin loss
# #         real   = (labels == 0).float()
# #         fake   = (labels == 1).float()
# #         l_fgw  = (real * fgw_dist +
# #                   fake * F.relu((1.0 + self.m) - fgw_dist)).mean()

# #         # 3. Supervised contrastive on CLS tokens (if provided)
# #         l_con = torch.tensor(0.0, device=logits.device)
# #         if cls_tok is not None:
# #             l_con = self.supcon(cls_tok, labels)

# #         # 4. Prototype contrastive (if provided)
# #         l_proto = torch.tensor(0.0, device=logits.device)
# #         if patch_feats is not None and proto_bank is not None:
# #             l_proto = self.proto_con(patch_feats, labels, proto_bank)

# #         total = (l_focal
# #                  + self.lam_fgw   * l_fgw
# #                  + self.lam_con   * l_con
# #                  + self.lam_proto * l_proto)

# #         return total, l_focal.item(), l_fgw.item()


# # def get_lr_scheduler(optimizer, epochs, warmup_epochs=2):
# #     def lr_lambda(epoch):
# #         if epoch < warmup_epochs:
# #             return float(epoch + 1) / float(warmup_epochs)
# #         progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
# #         return 0.5 * (1.0 + np.cos(np.pi * progress))
# #     return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# # # =============================================================
# # # TRAINING + EVALUATION
# # # =============================================================
# # def train_epoch(model: DeepfakeDetector, loader, optimizer,
# #                 criterion, device, epoch):
# #     model.train()
# #     total_loss, correct, total = 0., 0, 0
# #     fgw_r_sum, fgw_f_sum, n_r, n_f = 0., 0., 0, 0
# #     pbar = tqdm(loader, desc=f"Ep{epoch:02d}", leave=False)

# #     for imgs, labels in pbar:
# #         imgs, labels = imgs.to(device), labels.to(device)

# #         # forward — also get cls_tok and patch_tok for contrastive losses
# #         cls_tok, patch_tok = model.backbone(imgs)
# #         logits, _, fgw     = model(imgs)

# #         # full combined loss: focal + fgw + supcon + proto_con
# #         loss, lce, lfgw = criterion(
# #             logits, labels, fgw,
# #             cls_tok    = cls_tok.detach(),
# #             patch_feats= patch_tok.detach(),
# #             proto_bank = model.proto_bank,
# #         )
# #         optimizer.zero_grad(); loss.backward()
# #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
# #         optimizer.step()

# #         with torch.no_grad():
# #             rm = (labels==0); fm = (labels==1)
# #             if rm.sum()>0:
# #                 fgw_r_sum += fgw[rm].sum().item(); n_r += rm.sum().item()
# #             if fm.sum()>0:
# #                 fgw_f_sum += fgw[fm].sum().item(); n_f += fm.sum().item()

# #         # Prototype EMA update on real images
# #         real_mask = (labels == 0)
# #         if real_mask.sum() > 0:
# #             with torch.no_grad():
# #                 _, pt = model.backbone(imgs[real_mask])
# #                 cm = build_patch_graph(pt)
# #                 _, _, idx = model.proto_bank.get_nearest(pt, k_match=1)
# #                 model.proto_bank.update_prototypes(pt, cm, idx, momentum=0.99)

# #         total_loss += loss.item()
# #         correct    += (logits.argmax(1)==labels).sum().item()
# #         total      += labels.size(0)
# #         fgw_r = fgw_r_sum/max(n_r,1); fgw_f = fgw_f_sum/max(n_f,1)
# #         pbar.set_postfix(loss=f"{loss.item():.3f}",
# #                          acc=f"{correct/total*100:.1f}%",
# #                          sep=f"{fgw_f-fgw_r:+.3f}")

# #     fgw_r = fgw_r_sum/max(n_r,1); fgw_f = fgw_f_sum/max(n_f,1)
# #     print(f"  Train FGW: real={fgw_r:.3f}  fake={fgw_f:.3f}  "
# #           f"sep={fgw_f-fgw_r:+.3f}")
# #     return {"loss": total_loss/len(loader), "acc": correct/total}


# # @torch.no_grad()
# # def evaluate(model: DeepfakeDetector, loader, device, tag=""):
# #     model.eval()
# #     probs_all, labels_all, fgw_all = [], [], []

# #     for batch in loader:
# #         # Support loaders that return (imgs, labels) or (imgs, labels, paths)
# #         imgs, labels = batch[0], batch[1]
# #         logits, _, fgw = model(imgs.to(device))
# #         probs_all.append(torch.softmax(logits,1)[:,1].cpu().numpy())
# #         labels_all.append(labels.numpy())
# #         fgw_all.append(fgw.cpu().numpy())

# #     probs  = np.concatenate(probs_all)
# #     labels = np.concatenate(labels_all)
# #     fgw    = np.concatenate(fgw_all)

# #     if len(np.unique(labels)) < 2:
# #         print(f"  [{tag}] Only one class present — skipping AUC")
# #         return {"auc": 0., "acc": 0., "fgw_real": 0., "fgw_fake": 0.}

# #     auc      = roc_auc_score(labels, probs)
# #     acc      = accuracy_score(labels, probs > 0.5)
# #     fgw_real = fgw[labels==0].mean() if (labels==0).any() else 0.
# #     fgw_fake = fgw[labels==1].mean() if (labels==1).any() else 0.

# #     if auc < 0.5:
# #         print(f"  WARNING AUC<0.5. "
# #               f"Flipped={roc_auc_score(labels,1-probs):.4f}")

# #     print(f"  [{tag:35s}]  AUC={auc:.4f}  ACC={acc*100:.1f}%  "
# #           f"FGW_r={fgw_real:.3f}  FGW_f={fgw_fake:.3f}  "
# #           f"sep={fgw_fake-fgw_real:+.3f}")
# #     return {"auc": auc, "acc": acc,
# #             "fgw_real": fgw_real, "fgw_fake": fgw_fake}


# # # =============================================================
# # # TEST-TIME AUGMENTATION (TTA) — better cross-domain generalisation
# # # =============================================================
# # # TTA augmentations applied at inference time.
# # # Each test image is evaluated N times with different augmentations
# # # and predictions are averaged — improves cross-domain AUC ~1-2%
# # # without any retraining.
# # _TTA_TRANSFORMS = [
# #     # 1. Original (no aug)
# #     transforms.Compose([
# #         transforms.Resize(224), transforms.CenterCrop(224),
# #         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
# #     # 2. Horizontal flip
# #     transforms.Compose([
# #         transforms.Resize(224), transforms.CenterCrop(224),
# #         transforms.RandomHorizontalFlip(p=1.0),
# #         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
# #     # 3. Slight scale up crop
# #     transforms.Compose([
# #         transforms.Resize(256), transforms.CenterCrop(224),
# #         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
# #     # 4. Slight scale down + pad
# #     transforms.Compose([
# #         transforms.Resize(200), transforms.Pad(12),
# #         transforms.CenterCrop(224),
# #         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
# #     # 5. Mild brightness shift (simulate different domain lighting)
# #     transforms.Compose([
# #         transforms.Resize(224), transforms.CenterCrop(224),
# #         transforms.ColorJitter(brightness=0.15, contrast=0.15),
# #         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
# # ]


# # @torch.no_grad()
# # def evaluate_tta(model: DeepfakeDetector, dataset, device,
# #                  tag="", batch_size=4, num_workers=2):
# #     """
# #     TTA evaluation: averages predictions across 5 augmentation views.
# #     Uses the raw dataset (not loader) to apply per-image TTA transforms.
# #     Only used for final cross-domain test — not during training validation
# #     (too slow for per-epoch use).
# #     """
# #     model.eval()
# #     all_probs  = []
# #     all_labels = []

# #     print(f"  Running TTA ({len(_TTA_TRANSFORMS)} views) on {len(dataset)} samples...")

# #     for idx in tqdm(range(len(dataset)), desc=f"TTA {tag}", leave=False):
# #         sample = dataset.samples[idx]
# #         path, label = sample[0], sample[1]

# #         try:
# #             img_pil = Image.open(path).convert("RGB")
# #         except Exception:
# #             continue

# #         view_probs = []
# #         for tfm in _TTA_TRANSFORMS:
# #             tensor = tfm(img_pil).unsqueeze(0).to(device)
# #             logits, _, _ = model(tensor)
# #             p = torch.softmax(logits, dim=1)[0, 1].item()
# #             view_probs.append(p)

# #         all_probs.append(np.mean(view_probs))
# #         all_labels.append(label)

# #     probs  = np.array(all_probs)
# #     labels = np.array(all_labels)

# #     if len(np.unique(labels)) < 2:
# #         print(f"  [{tag}] Only one class — skipping AUC")
# #         return {"auc": 0., "acc": 0., "fgw_real": 0., "fgw_fake": 0.}

# #     auc = roc_auc_score(labels, probs)
# #     acc = accuracy_score(labels, probs > 0.5)
# #     print(f"  [{tag:35s}]  AUC(TTA)={auc:.4f}  ACC={acc*100:.1f}%")
# #     return {"auc": auc, "acc": acc, "fgw_real": 0., "fgw_fake": 0.}


# # # =============================================================
# # # HEATMAP VISUALISATION  (Objective 4 — PS)
# # # =============================================================
# # # DINOv2-L/14 produces 16x16 = 256 patch tokens for 224x224 input.
# # _PATCH_GRID = 16   # sqrt(256)

# # def _denorm(tensor_chw: torch.Tensor) -> np.ndarray:
# #     """Denormalise ImageNet-normalised tensor → uint8 HWC numpy array."""
# #     mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
# #     std  = np.array(STD,  dtype=np.float32).reshape(3, 1, 1)
# #     img  = tensor_chw.cpu().numpy() * std + mean
# #     img  = np.clip(img * 255, 0, 255).astype(np.uint8)
# #     return img.transpose(1, 2, 0)   # HWC


# # def _heatmap_to_color(scores_256: np.ndarray,
# #                       grid: int = _PATCH_GRID) -> np.ndarray:
# #     """
# #     Convert flat patch scores (256,) → jet-coloured 224×224 uint8 RGB overlay.
# #     """
# #     patch_map = scores_256.reshape(grid, grid).astype(np.float32)
# #     patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
# #     patch_map_u8 = (patch_map * 255).astype(np.uint8)
# #     patch_big    = cv2.resize(patch_map_u8, (224, 224),
# #                               interpolation=cv2.INTER_LINEAR)
# #     colored = cv2.applyColorMap(patch_big, cv2.COLORMAP_JET)
# #     colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
# #     return colored


# # def _blend(face_rgb: np.ndarray, heatmap_rgb: np.ndarray,
# #            alpha: float = 0.45) -> np.ndarray:
# #     """Alpha-blend heatmap over face image."""
# #     return np.clip(
# #         (1 - alpha) * face_rgb.astype(np.float32) +
# #         alpha * heatmap_rgb.astype(np.float32), 0, 255
# #     ).astype(np.uint8)


# # def _add_label_bar(canvas: np.ndarray, pred_label: str,
# #                    prob: float, gt_label: str) -> np.ndarray:
# #     """Append a 32-pixel info bar below the 3-panel image."""
# #     bar = np.zeros((32, canvas.shape[1], 3), dtype=np.uint8)
# #     correct = pred_label == gt_label
# #     color   = (80, 200, 80) if correct else (220, 80, 80)
# #     text    = (f"GT:{gt_label}  PRED:{pred_label}  "
# #                f"p={prob:.3f}  {'OK' if correct else 'WRONG'}")
# #     cv2.putText(bar, text, (6, 22),
# #                 cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
# #                 cv2.LINE_AA)
# #     return np.vstack([canvas, bar])


# # @torch.no_grad()
# # def save_heatmaps(model: DeepfakeDetector,
# #                   loader: DataLoader,
# #                   device: torch.device,
# #                   out_dir: str,
# #                   max_n: int = 200):
# #     """
# #     BUG FIX: original used a single counter so real images filled the
# #     quota before any fake images were saved → heatmaps/fake/ was always empty.

# #     FIX: separate counters for real and fake, each gets max_n//2 images.
# #     Both folders will now be populated.
# #     """
# #     model.eval()
# #     out_dir = Path(out_dir)
# #     (out_dir / "real").mkdir(parents=True, exist_ok=True)
# #     (out_dir / "fake").mkdir(parents=True, exist_ok=True)

# #     # ── separate quota per class ──────────────────────────────
# #     max_per_class = max_n // 2
# #     saved_real    = 0
# #     saved_fake    = 0
# #     total_saved   = 0

# #     print(f"\n  Saving heatmaps → {out_dir}  "
# #           f"(max {max_per_class} real + {max_per_class} fake)")

# #     for batch in tqdm(loader, desc="Heatmaps", leave=False):
# #         if saved_real >= max_per_class and saved_fake >= max_per_class:
# #             break

# #         imgs, labels, paths = batch[0], batch[1], batch[2]
# #         imgs_dev = imgs.to(device)

# #         logits, heatmaps, _ = model(imgs_dev)
# #         probs = torch.softmax(logits, dim=1)[:, 1].cpu()

# #         for i in range(imgs.size(0)):
# #             gt_label  = "fake" if int(labels[i]) == 1 else "real"

# #             # ── skip if this class quota is full ────────────
# #             if gt_label == "real"  and saved_real >= max_per_class:
# #                 continue
# #             if gt_label == "fake"  and saved_fake >= max_per_class:
# #                 continue

# #             # ── Face image ──────────────────────────────────
# #             face_rgb   = _denorm(imgs[i])
# #             scores_np  = heatmaps[i].cpu().numpy()
# #             heat_color = _heatmap_to_color(scores_np)
# #             blended    = _blend(face_rgb, heat_color, alpha=0.45)
# #             canvas     = np.concatenate([face_rgb, blended, heat_color], axis=1)

# #             prob_val   = float(probs[i])
# #             pred_label = "fake" if prob_val > 0.5 else "real"
# #             canvas     = _add_label_bar(canvas, pred_label, prob_val, gt_label)

# #             # ── Save ────────────────────────────────────────
# #             counter  = saved_fake if gt_label == "fake" else saved_real
# #             src_stem = Path(paths[i]).stem
# #             fname    = f"{counter:05d}_{src_stem}.png"
# #             out_path = out_dir / gt_label / fname

# #             cv2.imwrite(str(out_path),
# #                         cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

# #             if gt_label == "real":
# #                 saved_real += 1
# #             else:
# #                 saved_fake += 1
# #             total_saved += 1

# #     print(f"  ✓ {total_saved} heatmap images saved to {out_dir}/")
# #     print(f"    └ real/ : {saved_real} images  (should score LOW  — p≈0)")
# #     print(f"    └ fake/ : {saved_fake} images  (should score HIGH — p≈1)")


# # # =============================================================
# # # MAIN
# # # =============================================================
# # def main(args):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"\nDevice: {device}")

# #     # ── STEP 0: Extract FF++ frames if needed ──────────────────────────
# #     if args.ffpp_root:
# #         frames_out = Path(args.frames_out)
# #         n_r = len(list((frames_out/"real").glob("*.png"))) \
# #               if (frames_out/"real").exists() else 0
# #         n_f = len(list((frames_out/"fake").glob("*.png"))) \
# #               if (frames_out/"fake").exists() else 0
# #         if n_r > 50 and n_f > 50:
# #             print(f"\nFF++ frames exist: {n_r} real, {n_f} fake. "
# #                   f"Skipping extraction.")
# #         else:
# #             extract_all_ffpp_frames(args.ffpp_root, args.frames_out,
# #                                     args.frames_per_video, args.max_videos)

# #     # ── STEP 1: Build datasets ─────────────────────────────────────────
# #     train_ds, val_ds, cross_test_ds = build_all_datasets(args)

# #     nw = min(args.num_workers, os.cpu_count() or 2)
# #     train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
# #                               num_workers=nw, pin_memory=True, drop_last=True)
# #     val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
# #                               num_workers=nw, pin_memory=True)
# #     test_loader  = DataLoader(cross_test_ds, args.batch_size, shuffle=False,
# #                               num_workers=nw, pin_memory=True)

# #     # ── Build model ────────────────────────────────────────────────────
# #     print()
# #     model = DeepfakeDetector(
# #         backbone   = args.backbone,
# #         num_protos = args.num_protos,
# #         fgw_alpha  = args.fgw_alpha,
# #         fgw_reg    = args.fgw_reg,
# #         freq_dim   = args.freq_dim,
# #         k_match    = args.k_match,
# #         noise_std  = args.noise_std,
# #     ).to(device)

# #     total     = sum(p.numel() for p in model.parameters())
# #     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
# #     print(f"Params: total={total/1e6:.1f}M  "
# #           f"frozen={((total-trainable)/1e6):.1f}M  "
# #           f"trainable={trainable/1000:.1f}K")
# #     print(f"k_match={args.k_match}  noise_std={args.noise_std}")
# #     print(f"lambda_fgw={args.lambda_fgw}  fgw_margin={args.fgw_margin}\n")

# #     # ── Phase 1: Load prototype bank ──────────────────────────────────
# #     if args.phase1_result:
# #         load_phase1_result(args.phase1_result, model, device)
# #     else:
# #         print("\nWARNING: --phase1_result not set. Running inline Phase 1.\n")
# #         phase1_ssl_warmup_inline(model, train_loader, device,
# #                                  args.warmup_samples)

# #     print(f"  fgw_scale = {model.fgw_scale:.4f}")
# #     if model.fgw_scale > 10:
# #         print("  !! CRITICAL: fgw_scale > 10. FGW signal will be dead.")
# #         print("  !! Re-run phase1_build_graphs.py with L2-normalised tokens.")
# #     else:
# #         print(f"  ✓ fgw_scale OK (1–3 range)\n")

# #     # ── Phase 2: Supervised training ──────────────────────────────────
# #     print("="*60 + "\nPHASE 2 — Supervised training\n" + "="*60)
# #     print("Training on: FF++ + WildDeepfake (train+valid)")
# #     print("Validating on: 10% held-out mix")
# #     print("Cross-domain test: VGGFace2 real + WildDeepfake test fake\n")

# #     criterion = DetectionLoss(lambda_fgw=args.lambda_fgw,
# #                               margin=args.fgw_margin,
# #                               lambda_con=args.lambda_con,
# #                               lambda_proto=args.lambda_proto,
# #                               focal_gamma=args.focal_gamma,
# #                               focal_alpha=args.focal_alpha)
# #     optimizer = optim.AdamW(
# #         filter(lambda p: p.requires_grad, model.parameters()),
# #         lr=args.lr, weight_decay=1e-4)
# #     scheduler = get_lr_scheduler(optimizer, args.epochs, warmup_epochs=2)

# #     ckpt_dir = Path(args.ckpt_dir)
# #     ckpt_dir.mkdir(parents=True, exist_ok=True)
# #     best_auc = 0.

# #     for epoch in range(1, args.epochs + 1):
# #         t0 = time.time()
# #         tr = train_epoch(model, train_loader, optimizer,
# #                          criterion, device, epoch)
# #         vl = evaluate(model, val_loader, device,
# #                       f"Val (FF+++WildDF) ep{epoch:02d}")
# #         scheduler.step()
# #         print(f"  Ep{epoch:02d}  loss={tr['loss']:.4f}  "
# #               f"acc={tr['acc']*100:.1f}%  [{time.time()-t0:.0f}s]\n")

# #         if vl["auc"] > best_auc:
# #             best_auc = vl["auc"]
# #             torch.save({
# #                 "epoch"     : epoch,
# #                 "model"     : model.state_dict(),
# #                 "fgw_scale" : model.fgw_scale,
# #                 "val_auc"   : best_auc,
# #                 "args"      : vars(args),
# #             }, ckpt_dir / "best.pt")
# #             print(f"  ✓ Saved best  AUC={best_auc:.4f}\n")

# #     # ── Final cross-domain evaluation ─────────────────────────────────
# #     print("\n" + "="*60)
# #     print("FINAL — Cross-domain evaluation")
# #     print("  Real: VGGFace2 (unseen domain)")
# #     print("  Fake: WildDeepfake test/fake/ (unseen test split)")
# #     print("="*60)

# #     ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
# #     model.load_state_dict(ckpt["model"])
# #     model.fgw_scale = ckpt.get("fgw_scale", model.fgw_scale)
# #     print(f"Best checkpoint: epoch={ckpt['epoch']}  "
# #           f"val_AUC={ckpt['val_auc']:.4f}  "
# #           f"fgw_scale={model.fgw_scale:.4f}\n")

# #     r_val  = evaluate(model, val_loader,  device,
# #                       "Val (FF+++WildDF) final")
# #     r_cross = evaluate(model, test_loader, device,
# #                        "Cross-domain (VGGFace2+WildDF test)")

# #     # ── TTA evaluation for better cross-domain AUC ────────────────────
# #     if args.use_tta:
# #         print("\n  Running TTA evaluation on cross-domain test set...")
# #         r_cross_tta = evaluate_tta(model, cross_test_ds, device,
# #                                    tag="Cross-domain TTA",
# #                                    batch_size=args.batch_size,
# #                                    num_workers=min(args.num_workers, 2))

# #     # ── Heatmap saving (Objective 4) ───────────────────────────────────
# #     if args.save_heatmaps:
# #         save_heatmaps(model, test_loader, device,
# #                       out_dir=args.heatmap_dir,
# #                       max_n=args.heatmap_n)

# #     print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
# #     print(f"  Val AUC (FF+++WildDF)      : {r_val['auc']:.4f}")
# #     print(f"  Cross-domain AUC           : {r_cross['auc']:.4f}  "
# #           f"← VGGFace2 real + WildDF test fake")
# #     print(f"  FGW sep (val)              : "
# #           f"{r_val['fgw_fake']-r_val['fgw_real']:+.3f}")
# #     print(f"  FGW sep (cross-domain)     : "
# #           f"{r_cross['fgw_fake']-r_cross['fgw_real']:+.3f}")
# #     print(f"  FGW scale                  : {model.fgw_scale:.4f}  "
# #           f"(should be 1–3)")
# #     print(f"  k_match                    : {args.k_match}")
# #     print(f"  lambda_fgw                 : {args.lambda_fgw}")
# #     print(f"  fgw_margin                 : {args.fgw_margin}")
# #     if args.save_heatmaps:
# #         print(f"  Heatmaps saved to          : {args.heatmap_dir}/")
# #     print(f"\n  Ablation row:")
# #     print(f"  | DINOv2-L+SRM+FGW | "
# #           f"FF+++WildDF→VGGFace2 | "
# #           f"val={r_val['auc']:.4f} | "
# #           f"cross={r_cross['auc']:.4f} |")
# #     print("="*60)


# # # =============================================================
# # # CLI
# # # =============================================================
# # if __name__ == "__main__":
# #     p = argparse.ArgumentParser()

# #     # Phase 1
# #     p.add_argument("--phase1_result",    default=None,
# #         help="Path to phase1_result.pt from phase1_build_graphs.py")

# #     # Data
# #     p.add_argument("--frames_out",       default="./extracted_frames",
# #         help="Pre-extracted FF++ frames directory (real/ and fake/ inside)")
# #     p.add_argument("--ffpp_root",        default=None,
# #         help="FF++ root for frame extraction (optional, skip if already extracted)")
# #     p.add_argument("--wild_root",        required=True,
# #         help="WildDeepfake root (contains train/test/valid subfolders)")
# #     p.add_argument("--vggface2_root",    required=True,
# #         help="VGGFace2 root for cross-domain test (contains train/ and val/)")
# #     p.add_argument("--ckpt_dir",         default="./checkpoints")
# #     p.add_argument("--frames_per_video", type=int,   default=30)
# #     p.add_argument("--max_videos",       type=int,   default=None)
# #     p.add_argument("--max_per_class",    type=int,   default=5000)
# #     p.add_argument("--seed",             type=int,   default=42)

# #     # Model
# #     p.add_argument("--backbone",         default="dinov2_vitl14_reg")
# #     p.add_argument("--num_protos",       type=int,   default=32)
# #     p.add_argument("--fgw_alpha",        type=float, default=0.5)
# #     p.add_argument("--fgw_reg",          type=float, default=0.05)
# #     p.add_argument("--freq_dim",         type=int,   default=128)
# #     p.add_argument("--k_match",          type=int,   default=3)
# #     p.add_argument("--noise_std",        type=float, default=0.01)

# #     # Training
# #     p.add_argument("--epochs",           type=int,   default=20)
# #     p.add_argument("--batch_size",       type=int,   default=4)
# #     p.add_argument("--lr",               type=float, default=1e-4)
# #     p.add_argument("--lambda_fgw",       type=float, default=1.0)
# #     p.add_argument("--fgw_margin",       type=float, default=0.5)
# #     p.add_argument("--warmup_samples",   type=int,   default=5000)
# #     p.add_argument("--num_workers",      type=int,   default=4)

# #     # ── NEW: Contrastive + Focal loss weights ──────────────────────────
# #     p.add_argument("--lambda_con",    type=float, default=0.3,
# #         help="Weight for Supervised Contrastive loss on CLS tokens")
# #     p.add_argument("--lambda_proto",  type=float, default=0.2,
# #         help="Weight for Prototype Contrastive loss")
# #     p.add_argument("--focal_gamma",   type=float, default=2.0,
# #         help="Focal loss gamma — higher = more focus on hard fakes")
# #     p.add_argument("--focal_alpha",   type=float, default=0.75,
# #         help="Focal loss alpha — weight for fake class (0.5–0.9)")

# #     # ── NEW: Test-Time Augmentation ────────────────────────────────────
# #     p.add_argument("--use_tta",       action="store_true",
# #         help="Enable Test-Time Augmentation at final cross-domain eval")

# #     # Heatmap saving (Objective 4 — PS)
# #     p.add_argument("--save_heatmaps",    action="store_true",
# #         help="Save region-level heatmap overlays after final evaluation")
# #     p.add_argument("--heatmap_dir",      default="./heatmaps",
# #         help="Directory to write heatmap PNG files")
# #     p.add_argument("--heatmap_n",        type=int,   default=200,
# #         help="Max number of heatmap images to save")

# #     args = p.parse_args()
# #     main(args)

# """
# phase2.py  —  v3  (WildDeepfake + FF++ training | VGGFace2 cross-domain eval)
# ==============================================================================
# DATASET CHANGES FROM PREVIOUS VERSION:
#   REMOVED : DFDC  (deleted dataset)

#   TRAINING (labelled, supervised):
#     FF++         → real frames from extracted_frames/real/
#                    fake frames from extracted_frames/fake/
#     WildDeepfake → real from data/wilddeepfake/{train,valid}/real/
#                    fake from data/wilddeepfake/{train,valid}/fake/

#   VALIDATION (in-distribution, from training mix):
#     10% split of combined FF++ + WildDeepfake training set

#   CROSS-DOMAIN TEST (replaces DFDC):
#     VGGFace2 → real from data/vggface2/{train,val}/ (no fakes — identity only)
#     NOTE: VGGFace2 is REAL ONLY. We use it to verify the model does NOT
#     misclassify real faces from a completely different domain as fake.
#     A separate fake set from WildDeepfake test/ is used for the test split.

#   FULL CROSS-DOMAIN TEST SET:
#     real  → VGGFace2 val/ (unseen domain real faces)
#     fake  → WildDeepfake test/fake/ (unseen test split fakes)
#     This tests: does the model generalise to new real faces AND new fakes?

# AUGMENTATION PER DATASET:
#   FF++ real frames (train)    : Light — already 224x224 face-cropped PNGs
#   FF++ fake frames (train)    : Light + JPEG re-compression simulation
#   WildDeepfake real (train)   : Strong — in-the-wild, diverse quality
#   WildDeepfake fake (train)   : Strong + extra JPEG + blur (GAN artifacts vary)
#   VGGFace2 real (cross-test)  : Val-only (no augment) — clean eval

# ALL OTHER STEPS PRESERVED FROM v8:
#   ✔ DINOv2 backbone frozen, L2-normalised patch tokens
#   ✔ Cosine distance patch graph
#   ✔ FGW fixed-scale normalisation (loaded from phase1_result.pt)
#   ✔ Top-K prototype matching (k_match=3)
#   ✔ SRM frequency branch with Dropout(0.5)
#   ✔ Feature noise injection (noise_std=0.01)
#   ✔ Prototype EMA update during training (momentum=0.99)
#   ✔ FGW margin loss (lambda_fgw=1.0, margin=0.5)
#   ✔ LR warmup + cosine schedule
#   ✔ Gradient clipping (max_norm=1.0)

# ADDED (Objective 4 — PS):
#   ✔ Heatmap saving: frame-level + region-level overlays saved to
#     --heatmap_dir (default: ./heatmaps) during final cross-domain evaluation.
#     For each sample a side-by-side PNG is saved:
#       [original face | patch heatmap overlay | prediction label]
#     Controlled by --save_heatmaps (flag) and --heatmap_n (max images to save).

# HOW TO RUN:
#   python phase2.py \\
#     --phase1_result  ./phase1_output/phase1_result.pt \\
#     --frames_out     ./extracted_frames \\
#     --wild_root      ./data/wilddeepfake \\
#     --vggface2_root  ./data/vggface2 \\
#     --max_per_class  5000 \\
#     --num_protos     32 \\
#     --k_match        3 \\
#     --lambda_fgw     1.0 \\
#     --fgw_margin     0.5 \\
#     --epochs         20 \\
#     --batch_size     4 \\
#     --num_workers    4 \\
#     --ckpt_dir       ./checkpoints \\
#     --save_heatmaps \\
#     --heatmap_dir    ./heatmaps \\
#     --heatmap_n      200
# """

# import os, cv2, time, argparse, random, io
# from pathlib import Path
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader, ConcatDataset
# from torchvision import transforms
# from PIL import Image
# from tqdm import tqdm
# from sklearn.cluster import MiniBatchKMeans
# from sklearn.metrics import roc_auc_score, accuracy_score


# # =============================================================
# # CONSTANTS
# # =============================================================
# MEAN = [0.485, 0.456, 0.406]
# STD  = [0.229, 0.224, 0.225]
# EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# # =============================================================
# # JPEG AUGMENTATION
# # =============================================================
# class RandomJPEG:
#     """Simulate JPEG re-compression at random quality (30–95)."""
#     def __init__(self, low=30, high=95, p=0.5):
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
# # PER-DATASET AUGMENTATION TRANSFORMS
# # =============================================================

# def get_ffpp_train_transform():
#     """
#     FF++ REAL + FAKE frames, training split.
#     Already 224×224, face-cropped, H.264 compressed.
#     Light spatial augmentation (preserve face structure),
#     JPEG simulation to vary compression artifacts.
#     """
#     return transforms.Compose([
#         transforms.Resize(232),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.25, contrast=0.25,
#                                saturation=0.2, hue=0.06),
#         transforms.RandomGrayscale(p=0.03),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


# def get_wilddeepfake_train_transform():
#     """
#     WildDeepfake REAL + FAKE frames, training split.
#     In-the-wild internet videos: diverse quality, cameras, lighting.
#     Strong augmentation to cover natural variation range.
#     JPEG aug at PIL level handles the varied compression.
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


# def get_faceshifter_train_transform():
#     """
#     Faceshifter / 140k Real-vs-Fake dataset training split.
#     Real images are FFHQ-sourced (high quality).
#     Fake images are StyleGAN-generated — clean, no compression artifacts.
#     Strong augmentation needed since source is very clean.
#     JPEG simulation critical — fakes have no real compression artifacts.
#     """
#     return transforms.Compose([
#         transforms.Resize(288),
#         transforms.RandomCrop(224),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.45, contrast=0.45,
#                                saturation=0.38, hue=0.10),
#         transforms.RandomGrayscale(p=0.05),
#         transforms.RandomApply(
#             [transforms.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.40),
#         transforms.RandomApply(
#             [transforms.RandomAffine(degrees=15, translate=(0.10, 0.10),
#                                      scale=(0.88, 1.12))], p=0.35),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


# def get_val_transform():
#     """
#     Validation / test transform — no augmentation.
#     Used for: FF++/WildDeepfake val split + VGGFace2 cross-domain test.
#     """
#     return transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN, STD),
#     ])


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
# # DATASET CLASSES
# # =============================================================

# class LabelledDataset(Dataset):
#     """
#     Labelled dataset: list of (path, label) pairs.
#     label: 0=real, 1=fake.
#     """
#     def __init__(self, samples: list, transform,
#                  jpeg_aug=None, name: str = ""):
#         self.samples   = samples
#         self.transform = transform
#         self.jpeg_aug  = jpeg_aug
#         self.name      = name

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         try:
#             img = Image.open(path).convert("RGB")
#             if self.jpeg_aug is not None:
#                 img = self.jpeg_aug(img)
#             return self.transform(img), torch.tensor(label, dtype=torch.long)
#         except Exception:
#             alt_idx = random.randint(0, len(self.samples) - 1)
#             path, label = self.samples[alt_idx]
#             img = Image.open(path).convert("RGB")
#             if self.jpeg_aug is not None:
#                 img = self.jpeg_aug(img)
#             return self.transform(img), torch.tensor(label, dtype=torch.long)


# # =============================================================
# # DATASET BUILDERS
# # =============================================================

# def build_ffpp_samples(frames_out: str, max_per_class=None):
#     """
#     Collect FF++ samples from pre-extracted frames.
#     Expected structure:
#       extracted_frames/real/*.png  ← real frames (label=0)
#       extracted_frames/fake/*.png  ← fake frames (label=1)
#     """
#     root      = Path(frames_out)
#     real_dir  = root / "real"
#     fake_dir  = root / "fake"

#     def collect(d):
#         imgs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
#         return [str(p) for p in imgs]

#     real_imgs = collect(real_dir) if real_dir.exists() else []
#     fake_imgs = collect(fake_dir) if fake_dir.exists() else []

#     if not real_imgs:
#         print(f"  WARNING: No FF++ real frames found in {real_dir}")
#     if not fake_imgs:
#         print(f"  WARNING: No FF++ fake frames found in {fake_dir}")

#     samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
#     print(f"  FF++ frames: real={len(real_imgs)}  fake={len(fake_imgs)}")
#     return samples


# def build_wilddeepfake_samples(wild_root: str, splits: list,
#                                 max_per_class=None, seed=42):
#     """
#     Collect WildDeepfake samples from specified splits.
#     Structure:
#       data/wilddeepfake/{split}/real/  ← real face images (label=0)
#       data/wilddeepfake/{split}/fake/  ← deepfake images  (label=1)
#     Each real/ and fake/ may have subdirs (one per video source).

#     splits: list of split names, e.g. ["train","valid"] for training
#                                        ["test"] for cross-domain test
#     """
#     root  = Path(wild_root)
#     real_imgs, fake_imgs = [], []

#     for split in splits:
#         real_dir = root / split / "real"
#         fake_dir = root / split / "fake"

#         if real_dir.exists():
#             found = _glob_images(real_dir, recursive=True)
#             real_imgs.extend(found)
#             print(f"    WildDeepfake {split}/real : {len(found)}")
#         else:
#             print(f"    WildDeepfake {split}/real : NOT FOUND ({real_dir})")

#         if fake_dir.exists():
#             found = _glob_images(fake_dir, recursive=True)
#             fake_imgs.extend(found)
#             print(f"    WildDeepfake {split}/fake : {len(found)}")
#         else:
#             print(f"    WildDeepfake {split}/fake : NOT FOUND ({fake_dir})")

#     rng = random.Random(seed)
#     rng.shuffle(real_imgs); rng.shuffle(fake_imgs)


#     samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
#     print(f"  WildDeepfake {splits}: real={len(real_imgs)} fake={len(fake_imgs)}")
#     return samples


# def build_vggface2_samples(vgg_root: str, max_n=None, seed=42):
#     """
#     VGGFace2 REAL-ONLY cross-domain test set.
#     VGGFace2 has no fakes. We use it to test cross-domain REAL detection
#     (the model should NOT classify these as fakes).
#     Structure: data/vggface2/{train,val}/<id>/*.jpg

#     For cross-domain test we use val/ split only (train/ is larger
#     and overlaps more with what the model was trained on).
#     Returns samples with label=0 (all real).
#     """
#     root  = Path(vgg_root)
#     imgs  = []
#     # Use val/ for cross-domain eval (held out from training)
#     for split in ["val", "train"]:
#         split_dir = root / split
#         if split_dir.exists():
#             found = _glob_images(split_dir, recursive=True)
#             imgs.extend(found)
#             print(f"    VGGFace2 {split}           : {len(found)}")
#     if not imgs:
#         print(f"  WARNING: No VGGFace2 images under {root}")
#         return []
#     rng = random.Random(seed)
#     rng.shuffle(imgs)
    
#     print(f"  VGGFace2 real (total)       : {len(imgs)}")
#     return [(p, 0) for p in imgs]


# def build_faceshifter_samples(faceshifter_root: str, splits: list, seed=42):
#     """
#     Faceshifter / 140k Real-vs-Fake Kaggle dataset.
#     Structure:
#       data/faceshifter/real_vs_fake/real_vs_fake/
#           {train,valid,test}/real/*.jpg   label=0
#           {train,valid,test}/fake/*.jpg   label=1

#     ALL images used — no max_per_class cap.
#     splits: ["train","valid"] for training, ["test"] for evaluation.
#     """
#     root = Path(faceshifter_root)
#     real_imgs, fake_imgs = [], []

#     for split in splits:
#         real_dir = root / split / "real"
#         fake_dir = root / split / "fake"

#         if real_dir.exists():
#             found = _glob_images(real_dir, recursive=True)
#             real_imgs.extend(found)
#             print(f"    Faceshifter {split}/real  : {len(found)}")
#         else:
#             print(f"    Faceshifter {split}/real  : NOT FOUND ({real_dir})")

#         if fake_dir.exists():
#             found = _glob_images(fake_dir, recursive=True)
#             fake_imgs.extend(found)
#             print(f"    Faceshifter {split}/fake  : {len(found)}")
#         else:
#             print(f"    Faceshifter {split}/fake  : NOT FOUND ({fake_dir})")

#     rng = random.Random(seed)
#     rng.shuffle(real_imgs)
#     rng.shuffle(fake_imgs)
#     # NO cap — use all available images
#     samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
#     print(f"  Faceshifter {splits}: real={len(real_imgs)} fake={len(fake_imgs)}  [ALL]")
#     return samples


# class CombinedDataset(Dataset):
#     """
#     Combines FF++, WildDeepfake, and Faceshifter into one training dataset.
#     Each source uses its own per-dataset augmentation.
#     Faceshifter is added as a third source (StyleGAN fakes + FFHQ reals).
#     """
#     def __init__(self, ffpp_samples, wild_samples, face_samples,
#                  ffpp_transform, wild_transform, face_transform,
#                  ffpp_jpeg=None, wild_jpeg=None, face_jpeg=None):
#         self.ffpp_samples   = ffpp_samples
#         self.wild_samples   = wild_samples
#         self.face_samples   = face_samples
#         self.ffpp_transform = ffpp_transform
#         self.wild_transform = wild_transform
#         self.face_transform = face_transform
#         self.ffpp_jpeg      = ffpp_jpeg
#         self.wild_jpeg      = wild_jpeg
#         self.face_jpeg      = face_jpeg
#         self.n_ffpp         = len(ffpp_samples)
#         self.n_wild         = len(wild_samples)

#     def __len__(self):
#         return self.n_ffpp + self.n_wild + len(self.face_samples)

#     def __getitem__(self, idx):
#         if idx < self.n_ffpp:
#             path, label = self.ffpp_samples[idx]
#             transform   = self.ffpp_transform
#             jpeg_aug    = self.ffpp_jpeg
#         elif idx < self.n_ffpp + self.n_wild:
#             path, label = self.wild_samples[idx - self.n_ffpp]
#             transform   = self.wild_transform
#             jpeg_aug    = self.wild_jpeg
#         else:
#             path, label = self.face_samples[idx - self.n_ffpp - self.n_wild]
#             transform   = self.face_transform
#             jpeg_aug    = self.face_jpeg
#         try:
#             img = Image.open(path).convert("RGB")
#             if jpeg_aug: img = jpeg_aug(img)
#             return transform(img), torch.tensor(label, dtype=torch.long)
#         except Exception:
#             fallback = random.randint(0, len(self) - 1)
#             return self.__getitem__(fallback)


# class CrossDomainTestDataset(Dataset):
#     """
#     Cross-domain test dataset — combines THREE sources:
#       real : VGGFace2 val images          (label=0) — unseen real domain
#       fake : WildDeepfake test/fake/       (label=1) — in-the-wild fakes
#       fake : Faceshifter test/fake/        (label=1) — StyleGAN fakes

#     This tests generalisation across two completely different fake types:
#     video-based face swap (WildDeepfake) AND GAN synthesis (Faceshifter).
#     Val-only transform — no augmentation.
#     Returns image path for heatmap saving.
#     """
#     def __init__(self, vgg_samples, wild_test_fake_samples,
#                  face_test_fake_samples=None):
#         face_fakes = face_test_fake_samples or []
#         self.samples   = vgg_samples + wild_test_fake_samples + face_fakes
#         self.transform = get_val_transform()
#         n_r  = sum(1 for _, l in self.samples if l == 0)
#         n_f  = sum(1 for _, l in self.samples if l == 1)
#         n_wf = len(wild_test_fake_samples)
#         n_ff = len(face_fakes)
#         print(f"  Cross-domain test: real(VGGFace2)={n_r}  "
#               f"fake(WildDF)={n_wf}  fake(Faceshifter)={n_ff}  "
#               f"total_fake={n_f}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, label = self.samples[idx]
#         try:
#             img = Image.open(path).convert("RGB")
#             return self.transform(img), torch.tensor(label, dtype=torch.long), path
#         except Exception:
#             alt = random.randint(0, len(self.samples) - 1)
#             path, label = self.samples[alt]
#             img = Image.open(path).convert("RGB")
#             return self.transform(img), torch.tensor(label, dtype=torch.long), path


# # =============================================================
# # DATASET FACTORY — builds train/val/test splits
# # =============================================================

# def build_all_datasets(args):
#     """
#     Builds and returns train_ds, val_ds, cross_test_ds.

#     train_ds     : FF++ + WildDeepfake (train+valid) + Faceshifter (train+valid)
#     val_ds       : 10% from combined training set
#     cross_test_ds: VGGFace2 real + WildDeepfake test fakes + Faceshifter test fakes

#     ALL images used — no max_per_class cap on Faceshifter.
#     FF++ and WildDeepfake respect max_per_class.
#     """
#     print("\n" + "="*60 + "  DATASETS")

#     # ── FF++ samples ─────────────────────────────────────────────
#     print("\nFF++ (pre-extracted frames):")
#     ffpp_samples = build_ffpp_samples(args.frames_out)

#     # ── WildDeepfake training samples ────────────────────────────
#     print("\nWildDeepfake (train + valid splits for training):")
#     wild_train_samples = build_wilddeepfake_samples(
#         args.wild_root, splits=["train", "valid"],
#         seed=args.seed)

#     # ── Faceshifter training samples — ALL images, no cap ─────────
#     face_train_samples = []
#     if args.faceshifter_root:
#         print("\nFaceshifter (train + valid splits for training) [ALL]:")
#         face_train_samples = build_faceshifter_samples(
#             args.faceshifter_root, splits=["train", "valid"],
#             seed=args.seed)

#     # ── Combined train+val split ──────────────────────────────────
#     # Merge all three sources then split 90/10
#     all_samples = ffpp_samples + wild_train_samples + face_train_samples
#     rng = random.Random(args.seed)
#     rng.shuffle(all_samples)
#     n_val   = int(len(all_samples) * 0.10)
#     val_s   = all_samples[:n_val]
#     train_s = all_samples[n_val:]

#     # Separate each split back into sources for per-source augmentation
#     def split_samples(samples):
#         ffpp_p = str(Path(args.frames_out).resolve())
#         wild_p = str(Path(args.wild_root).resolve())
#         face_p = str(Path(args.faceshifter_root).resolve()) \
#                  if args.faceshifter_root else ""
#         ffpp_s, wild_s, face_s = [], [], []
#         for path, label in samples:
#             ap = str(Path(path).resolve())
#             if ap.startswith(ffpp_p):
#                 ffpp_s.append((path, label))
#             elif face_p and ap.startswith(face_p):
#                 face_s.append((path, label))
#             else:
#                 wild_s.append((path, label))
#         return ffpp_s, wild_s, face_s

#     train_ffpp, train_wild, train_face = split_samples(train_s)
#     val_ffpp,   val_wild,   val_face   = split_samples(val_s)

#     # Per-source JPEG augmentation
#     ffpp_jpeg = RandomJPEG(low=30, high=95, p=0.5)
#     wild_jpeg = RandomJPEG(low=40, high=95, p=0.45)
#     face_jpeg = RandomJPEG(low=45, high=95, p=0.40)

#     train_ds = CombinedDataset(
#         ffpp_samples   = train_ffpp,
#         wild_samples   = train_wild,
#         face_samples   = train_face,
#         ffpp_transform = get_ffpp_train_transform(),
#         wild_transform = get_wilddeepfake_train_transform(),
#         face_transform = get_faceshifter_train_transform(),
#         ffpp_jpeg      = ffpp_jpeg,
#         wild_jpeg      = wild_jpeg,
#         face_jpeg      = face_jpeg,
#     )

#     val_ds = LabelledDataset(val_s, get_val_transform(),
#                              jpeg_aug=None, name="val")

#     # Stats
#     n_train_r = sum(1 for _, l in train_s if l == 0)
#     n_train_f = sum(1 for _, l in train_s if l == 1)
#     n_val_r   = sum(1 for _, l in val_s   if l == 0)
#     n_val_f   = sum(1 for _, l in val_s   if l == 1)
#     print(f"\n  [train] total={len(train_s)}  "
#           f"real={n_train_r}  fake={n_train_f}  "
#           f"ratio={n_train_f/max(n_train_r,1):.2f}")
#     print(f"    └ FF++={len(train_ffpp)}  "
#           f"WildDF={len(train_wild)}  "
#           f"Faceshifter={len(train_face)}")
#     print(f"  [val  ] total={len(val_s)}  "
#           f"real={n_val_r}  fake={n_val_f}")

#     # ── Cross-domain test set ─────────────────────────────────────
#     print("\nCross-domain test:")
#     print("  Real  : VGGFace2 (unseen domain)")
#     print("  Fake1 : WildDeepfake test/fake/ (video face-swap)")
#     print("  Fake2 : Faceshifter test/fake/  (StyleGAN synthesis)")

#     vgg_samples = build_vggface2_samples(
#         args.vggface2_root,
#         seed=args.seed)

#     print("\nWildDeepfake test/fake:")
#     wild_test_samples = build_wilddeepfake_samples(
#         args.wild_root, splits=["test"],
#         seed=args.seed)
#     wild_test_fakes = [(p, l) for p, l in wild_test_samples if l == 1]
#     print(f"  WildDeepfake test fakes     : {len(wild_test_fakes)}")

#     # Faceshifter test fakes — ALL images
#     face_test_fakes = []
#     if args.faceshifter_root:
#         print("\nFaceshifter test/fake [ALL]:")
#         face_test_samples = build_faceshifter_samples(
#             args.faceshifter_root, splits=["test"],
#             seed=args.seed)
#         face_test_fakes = [(p, l) for p, l in face_test_samples if l == 1]
#         print(f"  Faceshifter test fakes      : {len(face_test_fakes)}")

#     cross_test_ds = CrossDomainTestDataset(
#         vgg_samples, wild_test_fakes, face_test_fakes)

#     return train_ds, val_ds, cross_test_ds


# # =============================================================
# # FRAME EXTRACTION (FF++ only)
# # =============================================================
# def extract_frames_from_video(video_path, out_dir, n_frames=30):
#     os.makedirs(out_dir, exist_ok=True)
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened(): return 0
#     total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     if total == 0: cap.release(); return 0
#     positions = np.linspace(0, total-1, min(n_frames, total),
#                             dtype=int).tolist()
#     stem = Path(video_path).stem; saved = 0
#     for pos in positions:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
#         ret, frame = cap.read()
#         if not ret: continue
#         frame = cv2.resize(frame, (224, 224))
#         cv2.imwrite(str(Path(out_dir) / f"{stem}_f{pos:06d}.png"), frame)
#         saved += 1
#     cap.release(); return saved


# def extract_all_ffpp_frames(ffpp_root, frames_out, n_frames=30, max_videos=None):
#     """Extract frames from FF++ videos into extracted_frames/real/ and /fake/."""
#     ffpp_root = Path(ffpp_root)
#     real_out  = Path(frames_out) / "real"
#     fake_out  = Path(frames_out) / "fake"
#     real_out.mkdir(parents=True, exist_ok=True)
#     fake_out.mkdir(parents=True, exist_ok=True)

#     MANIPS = ["Deepfakes", "Face2Face", "FaceSwap",
#               "FaceShifter", "NeuralTextures", "DeepFakeDetection"]

#     print("\n" + "="*60)
#     print(f"STEP 0 — FF++ Frame extraction (n_frames={n_frames})")
#     print("="*60)

#     real_videos = []
#     for sub in ["youtube", "actors"]:
#         d = ffpp_root / "original_sequences" / sub / "c23" / "videos"
#         if d.exists():
#             real_videos += list(d.glob("*.mp4")) + list(d.glob("*.avi"))
#     if max_videos: real_videos = real_videos[:max_videos // 2]
#     print(f"Real videos: {len(real_videos)}")
#     tr = sum(extract_frames_from_video(str(v), str(real_out), n_frames)
#              for v in tqdm(real_videos, desc="Real"))
#     print(f"  → {tr} real frames saved")

#     fake_videos = []
#     for manip in MANIPS:
#         for pat in [ffpp_root / "manipulated_sequences" / manip / "c23" / "videos",
#                     ffpp_root / "manipulated_sequences" / manip]:
#             if pat.exists():
#                 fake_videos += (list(pat.glob("*.mp4")) +
#                                 list(pat.glob("*.avi")) +
#                                 list(pat.rglob("*.mp4"))); break
#     fake_videos = list(set(str(v) for v in fake_videos))
#     random.shuffle(fake_videos)
#     if max_videos: fake_videos = fake_videos[:max_videos // 2]
#     print(f"Fake videos: {len(fake_videos)}")
#     tf = sum(extract_frames_from_video(v, str(fake_out), n_frames)
#              for v in tqdm(fake_videos, desc="Fake"))
#     print(f"  → {tf} fake frames saved")


# # =============================================================
# # PART A — DINOv2 BACKBONE
# # =============================================================
# class DINOv2Backbone(nn.Module):
#     """
#     Frozen DINOv2-ViT-L/14 backbone.
#     L2-normalises patch tokens → fgw_scale stays in 1–3 range.
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
#         patches = F.normalize(patches, dim=-1)   # L2-norm
#         return cls, patches


# # =============================================================
# # PART B — SRM FREQUENCY BRANCH
# # =============================================================
# class SRMConv(nn.Module):
#     def __init__(self):
#         super().__init__()
#         kernels     = self._build_srm_kernels()
#         kernels_rgb = kernels.repeat(1, 3, 1, 1) / 3.0
#         self.register_buffer("weight", kernels_rgb)

#     def _build_srm_kernels(self):
#         b1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,2,-1,0],
#                         [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2.
#         b2 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],
#                         [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2.
#         b3 = np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],
#                         [0,0,-1,0,0],[0,0,0,0,0]], np.float32) / 4.
#         b4 = np.array([[0,0,0,0,0],[0,-1,0,0,0],[0,0,2,0,0],
#                         [0,0,0,-1,0],[0,0,0,0,0]], np.float32) / 2.
#         b5 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
#                         [2,-6,8,-6,2],[-1,2,-2,2,-1]], np.float32) / 12.
#         b6 = -np.ones((5,5), np.float32) / 24.; b6[2,2] = 1.
#         bases = [b1,b2,b3,b4,b5,b6]
#         all_k = []
#         for b in bases:
#             for r in range(4): all_k.append(np.rot90(b,r).copy())
#         for b in bases: all_k.append(b.T.copy())
#         return torch.tensor(
#             np.stack(all_k[:30])[:,np.newaxis,:,:], dtype=torch.float32)

#     def forward(self, x):
#         with torch.no_grad():
#             return F.conv2d(x, self.weight, padding=2).clamp(-2., 2.)


# class FrequencyEncoder(nn.Module):
#     def __init__(self, freq_dim=128):
#         super().__init__()
#         self.srm = SRMConv()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(30, 64, 3, stride=2, padding=1),
#             nn.BatchNorm2d(64), nn.ReLU(inplace=True),
#             nn.Conv2d(64, 128, 3, stride=2, padding=1),
#             nn.BatchNorm2d(128), nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d(4),
#         )
#         self.proj = nn.Sequential(
#             nn.Flatten(),
#             nn.Dropout(0.5),
#             nn.Linear(128*4*4, freq_dim),
#             nn.LayerNorm(freq_dim),
#         )

#     def forward(self, x):
#         return self.proj(self.encoder(self.srm(x)))


# # =============================================================
# # PART C — PATCH GRAPH
# # =============================================================
# def build_patch_graph(patch_tokens: torch.Tensor) -> torch.Tensor:
#     dot    = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))
#     cost_M = (1.0 - dot).clamp(min=0.)
#     cost_M = cost_M / (cost_M.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
#     return cost_M


# # =============================================================
# # PART D — SINKHORN
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


# # =============================================================
# # PART E — FGW DISTANCE
# # =============================================================
# def gw_loss_vec(C1, C2, T):
#     p  = T.sum(2); q = T.sum(1)
#     t1 = (C1**2).bmm(p.unsqueeze(2)).squeeze(2).sum(1)
#     t2 = ((C2**2).bmm(q.unsqueeze(2)).squeeze(2).unsqueeze(1)*T).sum([1,2])
#     t3 = 2.*(T.bmm(C2).bmm(T.permute(0,2,1))*C1).sum([1,2])
#     return t1 + t2 - t3


# def fgw_distance(feat_src, cost_src, feat_tgt, cost_tgt,
#                  alpha=0.5, reg=0.05, n_iter=20, scale=None):
#     B, N, D   = feat_src.shape
#     M_nodes   = feat_tgt.shape[1]
#     p = torch.ones(B, N,       device=feat_src.device) / N
#     q = torch.ones(B, M_nodes, device=feat_src.device) / M_nodes

#     dot    = torch.bmm(feat_src, feat_tgt.transpose(1, 2))
#     M_feat = (1.0 - dot).clamp(min=0.)
#     M_feat = M_feat / (M_feat.flatten(1).max(1)[0].view(B,1,1) + 1e-8)

#     T = sinkhorn_log(p, q, M_feat, reg, n_iter)
#     for _ in range(3):
#         gw_g  = -2. * cost_src.bmm(T).bmm(cost_tgt)
#         M_fgw = (1-alpha)*M_feat + alpha*gw_g
#         M_fgw = M_fgw / (M_fgw.flatten(1).max(1)[0].view(B,1,1) + 1e-8)
#         T     = sinkhorn_log(p, q, M_fgw, reg, n_iter)

#     raw_dist = ((1-alpha)*(M_feat*T).sum([1,2]) +
#                 alpha*gw_loss_vec(cost_src, cost_tgt, T))

#     if scale is not None and scale > 0:
#         return raw_dist / scale, T
#     return raw_dist / (raw_dist.max().detach() + 1e-8), T


# # =============================================================
# # PART F — PROTOTYPE BANK
# # =============================================================
# class PrototypeBank(nn.Module):
#     def __init__(self, K=32, N=256, D=1024):
#         super().__init__()
#         self.K = K
#         self.register_buffer("proto_feats",
#             F.normalize(torch.randn(K, N, D), dim=-1))
#         self.register_buffer("proto_costs", torch.zeros(K, N, N))

#     @torch.no_grad()
#     def load_from_phase1(self, result: dict):
#         pf = result["proto_feats"]
#         pc = result["proto_costs"]
#         K_saved = pf.shape[0]
#         if K_saved != self.K:
#             print(f"  NOTE: Phase1 K={K_saved} != model K={self.K}. "
#                   f"Adjusting.")
#             self.K = K_saved
#             self.register_buffer("proto_feats", pf.clone())
#             self.register_buffer("proto_costs", pc.clone())
#         else:
#             self.proto_feats.copy_(pf)
#             self.proto_costs.copy_(pc)
#         self.proto_feats = F.normalize(self.proto_feats, dim=-1)
#         print(f"  Prototype bank loaded: K={self.K}  "
#               f"feats={tuple(self.proto_feats.shape)}")

#     @torch.no_grad()
#     def update_prototypes(self, feat, cost, assignments, momentum=0.99):
#         for k in range(self.K):
#             mask = (assignments == k)
#             if mask.sum() == 0: continue
#             new_f = F.normalize(feat[mask].mean(0), dim=-1)
#             self.proto_feats[k] = F.normalize(
#                 momentum * self.proto_feats[k] + (1-momentum)*new_f, dim=-1)
#             self.proto_costs[k] = (momentum * self.proto_costs[k] +
#                                    (1-momentum) * cost[mask].mean(0))

#     def get_nearest(self, feat, k_match=1):
#         feat_mean  = F.normalize(feat.mean(1), dim=-1)
#         proto_mean = F.normalize(self.proto_feats.mean(1), dim=-1)
#         sim = torch.mm(feat_mean, proto_mean.t())

#         if k_match == 1:
#             idx = sim.argmax(1)
#             return self.proto_feats[idx], self.proto_costs[idx], idx
#         else:
#             return sim.topk(min(k_match, self.K), dim=1).indices


# # =============================================================
# # PART E2 — TOP-K FGW
# # =============================================================
# def fgw_topk(feat_src, cost_src, proto_bank: PrototypeBank,
#              k_match=3, alpha=0.5, reg=0.05, scale=None):
#     B = feat_src.shape[0]
#     if k_match <= 1:
#         pf, pc, idx = proto_bank.get_nearest(feat_src, k_match=1)
#         dist, T = fgw_distance(feat_src, cost_src, pf, pc,
#                                alpha, reg, scale=scale)
#         return dist, T

#     topk_idx  = proto_bank.get_nearest(feat_src, k_match=k_match)
#     all_dists = []; last_T = None
#     for ki in range(k_match):
#         idx_k  = topk_idx[:, ki]
#         pf_k   = proto_bank.proto_feats[idx_k]
#         pc_k   = proto_bank.proto_costs[idx_k]
#         dist_k, T_k = fgw_distance(feat_src, cost_src, pf_k, pc_k,
#                                    alpha, reg, scale=scale)
#         all_dists.append(dist_k)
#         if ki == 0: last_T = T_k

#     min_dist = torch.stack(all_dists, dim=1).min(dim=1).values
#     return min_dist, last_T


# # =============================================================
# # PART G — DETECTION HEAD
# # =============================================================
# class DetectionHead(nn.Module):
#     def __init__(self, cls_dim=1024, freq_dim=128):
#         super().__init__()
#         fused_dim = cls_dim + freq_dim + 1   # 1153

#         self.classifier = nn.Sequential(
#             nn.LayerNorm(fused_dim),
#             nn.Linear(fused_dim, 512),
#             nn.GELU(),
#             nn.Dropout(0.5),
#             nn.Linear(512, 128),
#             nn.GELU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 2),
#         )
#         self.patch_scorer = nn.Sequential(
#             nn.Linear(cls_dim, 256),
#             nn.GELU(),
#             nn.Linear(256, 1),
#         )

#     def forward(self, cls_token, freq_feat, fgw_dist, patch_tokens, T=None):
#         fused  = torch.cat([cls_token, freq_feat,
#                             fgw_dist.unsqueeze(1)], dim=1)
#         logits = self.classifier(fused)
#         heatmap = self.patch_scorer(patch_tokens).squeeze(-1)
#         if T is not None:
#             ent   = -(T*(T+1e-8).log()).sum(-1)
#             e_min = ent.min(1, keepdim=True)[0]
#             e_max = ent.max(1, keepdim=True)[0]
#             heatmap = heatmap + (ent-e_min)/(e_max-e_min+1e-8)
#         return logits, torch.sigmoid(heatmap)


# # =============================================================
# # PART H — FULL DETECTOR
# # =============================================================
# class DeepfakeDetector(nn.Module):
#     def __init__(self, backbone="dinov2_vitl14_reg", num_protos=32,
#                  fgw_alpha=0.5, fgw_reg=0.05, freq_dim=128,
#                  k_match=3, noise_std=0.01):
#         super().__init__()
#         self.backbone     = DINOv2Backbone(backbone)
#         D = self.backbone.embed_dim
#         N = self.backbone.num_patches
#         self.freq_encoder = FrequencyEncoder(freq_dim=freq_dim)
#         self.proto_bank   = PrototypeBank(K=num_protos, N=N, D=D)
#         self.det_head     = DetectionHead(cls_dim=D, freq_dim=freq_dim)
#         self.fgw_alpha    = fgw_alpha
#         self.fgw_reg      = fgw_reg
#         self.fgw_scale    = 1.0
#         self.k_match      = k_match
#         self.noise_std    = noise_std

#     def forward(self, x):
#         cls_tok, patch_tok = self.backbone(x)
#         freq_feat          = self.freq_encoder(x)

#         if self.training and self.noise_std > 0:
#             patch_tok = patch_tok + torch.randn_like(patch_tok) * self.noise_std
#             patch_tok = F.normalize(patch_tok, dim=-1)

#         cost_M = build_patch_graph(patch_tok)
#         fgw_dist, T = fgw_topk(
#             patch_tok, cost_M, self.proto_bank,
#             k_match=self.k_match, alpha=self.fgw_alpha,
#             reg=self.fgw_reg, scale=self.fgw_scale)

#         logits, heatmap = self.det_head(
#             cls_tok, freq_feat, fgw_dist, patch_tok, T)
#         return logits, heatmap, fgw_dist


# # =============================================================
# # PHASE 1 LOAD
# # =============================================================
# def load_phase1_result(phase1_path: str, model: DeepfakeDetector,
#                        device: torch.device):
#     path = Path(phase1_path)
#     if not path.exists():
#         raise FileNotFoundError(
#             f"Phase 1 result not found: {path}\n"
#             f"Run phase1_build_graphs.py first.")

#     print(f"\n{'='*60}\nLOADING PHASE 1 FROM DISK\n{'='*60}")
#     result = torch.load(path, map_location=device)
#     model.proto_bank.load_from_phase1(result)

#     raw_scale = float(result["fgw_scale"])
#     if raw_scale > 10.0:
#         corrected = raw_scale / 50.0
#         print(f"\n  !! fgw_scale={raw_scale:.2f} > 10. "
#               f"Phase1 built without L2-norm. "
#               f"Correction: {raw_scale:.2f} → {corrected:.4f}")
#         print(f"  !! Re-run phase1_build_graphs.py for best results.")
#         model.fgw_scale = max(corrected, 1e-4)
#     else:
#         model.fgw_scale = max(raw_scale, 1e-4)

#     version = result.get("version", "unknown")
#     print(f"  fgw_scale = {model.fgw_scale:.4f}  (expected 1–3)")
#     print(f"  K         = {result['K']}")
#     print(f"  N images  = {result['n_images_used']}")
#     print(f"  Version   = {version}")
#     if version == "v3_wildvgg":
#         print(f"  ✓ Built with WildDeepfake + VGGFace2 real images")
#     print()


# @torch.no_grad()
# def phase1_ssl_warmup_inline(model: DeepfakeDetector,
#                              loader: DataLoader,
#                              device: torch.device,
#                              max_samples: int = 5000):
#     """
#     Inline Phase 1 fallback: uses REAL images from the training loader.
#     Filters label==0 (real). Builds prototype bank on the fly.
#     """
#     print("\n" + "="*60)
#     print("PHASE 1 — Inline (real images from training loader)")
#     print("  TIP: Run phase1_build_graphs.py for multi-source prototype bank")
#     print("="*60)
#     model.eval()
#     K = model.proto_bank.K

#     cls_list, patch_list, cost_list = [], [], []
#     collected = 0

#     for imgs, labels in loader:
#         mask = (labels == 0)
#         if mask.sum() == 0: continue
#         imgs_real = imgs[mask].to(device)
#         cls_tok, patch_tok = model.backbone(imgs_real)
#         cost_M = build_patch_graph(patch_tok)

#         cls_list.append(cls_tok.cpu().numpy())
#         patch_list.append(patch_tok.cpu())
#         cost_list.append(cost_M.cpu())
#         collected += mask.sum().item()

#         if collected % 500 < imgs_real.shape[0]:
#             print(f"  Collected {collected} real faces...")
#         if collected >= max_samples: break

#     if not cls_list:
#         raise RuntimeError("No real images found in loader.")

#     cls_np    = np.concatenate(cls_list, axis=0)
#     patch_all = torch.cat(patch_list, dim=0)
#     cost_all  = torch.cat(cost_list,  dim=0)

#     print(f"  K-means K={K} on {len(cls_np)} real faces ...")
#     km = MiniBatchKMeans(n_clusters=K, random_state=42,
#                          n_init=10, max_iter=300, verbose=0)
#     km.fit(cls_np)
#     asgn  = torch.from_numpy(km.labels_).long().to(device)
#     sizes = [(asgn==k).sum().item() for k in range(K)]
#     print(f"  Cluster sizes: min={min(sizes)} max={max(sizes)}")
#     if min(sizes) == 0:
#         print("  WARNING: empty cluster. Try --num_protos 16")

#     model.proto_bank.update_prototypes(
#         patch_all.to(device), cost_all.to(device), asgn, momentum=0.5)

#     real_fgw_vals = []
#     for i in range(0, min(len(patch_all), 500), 8):
#         f_b = patch_all[i:i+8].to(device)
#         c_b = cost_all[i:i+8].to(device)
#         d_b, _ = fgw_topk(f_b, c_b, model.proto_bank,
#                            k_match=1, alpha=model.fgw_alpha,
#                            reg=model.fgw_reg, scale=None)
#         real_fgw_vals.append(d_b.cpu())

#     real_fgw_all = torch.cat(real_fgw_vals)
#     model.fgw_scale = max(float(torch.quantile(real_fgw_all, 0.95)), 1e-4)
#     print(f"  fgw_scale = {model.fgw_scale:.4f}  (expected 1–3)\n")


# # =============================================================
# # LOSS
# # =============================================================

# # ── 1. Focal Loss — fixes low fake recall by down-weighting easy real examples
# class FocalLoss(nn.Module):
#     """
#     Focal Loss: FL(p) = -alpha * (1-p)^gamma * log(p)
#     gamma=2 focuses training on hard misclassified fakes.
#     alpha=0.75 upweights the fake class (minority class).
#     Directly addresses the real:fake imbalance causing low recall.
#     """
#     def __init__(self, gamma=2.0, alpha=0.75):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha  # weight for fake class (label=1)

#     def forward(self, logits, labels):
#         probs   = torch.softmax(logits, dim=1)
#         # gather prob of the correct class
#         p_t     = probs[range(len(labels)), labels]
#         # class weight: alpha for fake(1), 1-alpha for real(0)
#         alpha_t = torch.where(labels == 1,
#                               torch.tensor(self.alpha, device=logits.device),
#                               torch.tensor(1.0 - self.alpha, device=logits.device))
#         focal_w = alpha_t * (1 - p_t) ** self.gamma
#         loss    = -(focal_w * torch.log(p_t + 1e-8))
#         return loss.mean()


# # ── 2. Supervised Contrastive Loss on CLS tokens
# class SupConLoss(nn.Module):
#     """
#     Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).
#     Applied on L2-normalised CLS tokens.
#     Pulls real CLS embeddings together, pushes fake embeddings
#     away from real clusters in the 1024-d feature space.
#     temperature=0.07 is standard from the paper.
#     """
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temp = temperature

#     def forward(self, features, labels):
#         # features: (B, D) — CLS tokens, will be L2-normalised here
#         # labels:   (B,)   — 0=real, 1=fake
#         features = F.normalize(features, dim=-1)
#         B = features.shape[0]
#         if B < 2:
#             return torch.tensor(0.0, device=features.device)

#         # similarity matrix (B, B)
#         sim = torch.mm(features, features.T) / self.temp

#         # same-class mask, diagonal excluded (no self-contrast)
#         labels_col = labels.view(-1, 1)
#         mask = (labels_col == labels_col.T).float()
#         mask.fill_diagonal_(0)

#         # if no positive pairs exist in this batch, skip
#         if mask.sum() == 0:
#             return torch.tensor(0.0, device=features.device)

#         # log-sum-exp over all negatives
#         exp_sim   = torch.exp(sim)
#         # exclude self from denominator
#         self_mask = torch.ones_like(exp_sim)
#         self_mask.fill_diagonal_(0)
#         log_denom = torch.log((exp_sim * self_mask).sum(1, keepdim=True) + 1e-8)
#         log_prob  = sim - log_denom

#         # average loss over positive pairs per anchor
#         n_pos = mask.sum(1).clamp(min=1)
#         loss  = -(mask * log_prob).sum(1) / n_pos
#         return loss.mean()


# # ── 3. Prototype Contrastive Loss — uses the existing prototype bank
# class ProtoConLoss(nn.Module):
#     """
#     Prototype Contrastive Loss.
#     Real images: maximise cosine similarity to nearest prototype
#                  (pull real faces toward real-face prototypes)
#     Fake images: push similarity below (nearest_sim - margin)
#                  (push fake faces away from real prototypes)
#     Directly aligned with the FGW design philosophy.
#     """
#     def __init__(self, margin=0.4):
#         super().__init__()
#         self.margin = margin

#     def forward(self, patch_feats, labels, proto_bank):
#         # patch_feats: (B, 256, 1024) — L2-normalised patch tokens
#         # labels:      (B,)
#         feat_mean  = F.normalize(patch_feats.mean(1), dim=-1)   # (B, D)
#         proto_mean = F.normalize(proto_bank.proto_feats.mean(1), dim=-1)  # (K, D)

#         sim         = torch.mm(feat_mean, proto_mean.T)   # (B, K)
#         nearest_sim = sim.max(1).values                   # (B,)

#         real_mask = (labels == 0).float()
#         fake_mask = (labels == 1).float()

#         # real: maximise similarity → loss = 1 - sim
#         loss_real = real_mask * (1.0 - nearest_sim)
#         # fake: push similarity below (nearest_sim - margin)
#         loss_fake = fake_mask * F.relu(nearest_sim + self.margin)

#         return (loss_real + loss_fake).mean()


# # ── 4. Combined Detection Loss
# class DetectionLoss(nn.Module):
#     """
#     Combined loss:
#       L = FocalLoss                     (fixes fake recall — replaces CrossEntropy)
#         + lambda_fgw  * FGW_margin_loss (structural separation)
#         + lambda_con  * SupConLoss      (CLS embedding separation)
#         + lambda_proto* ProtoConLoss    (prototype alignment)

#     lambda defaults: fgw=1.0, con=0.3, proto=0.2
#     These are additive — each targets a different aspect of fakeness.
#     """
#     def __init__(self, lambda_fgw=1.0, margin=0.5,
#                  lambda_con=0.3, lambda_proto=0.2,
#                  focal_gamma=2.0, focal_alpha=0.75):
#         super().__init__()
#         self.focal      = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
#         self.supcon     = SupConLoss(temperature=0.07)
#         self.proto_con  = ProtoConLoss(margin=0.4)
#         self.lam_fgw    = lambda_fgw
#         self.lam_con    = lambda_con
#         self.lam_proto  = lambda_proto
#         self.m          = margin

#     def forward(self, logits, labels, fgw_dist,
#                 cls_tok=None, patch_feats=None, proto_bank=None):
#         # 1. Focal loss (replaces CrossEntropy)
#         l_focal = self.focal(logits, labels)

#         # 2. FGW margin loss
#         real   = (labels == 0).float()
#         fake   = (labels == 1).float()
#         l_fgw  = (real * fgw_dist +
#                   fake * F.relu((1.0 + self.m) - fgw_dist)).mean()

#         # 3. Supervised contrastive on CLS tokens (if provided)
#         l_con = torch.tensor(0.0, device=logits.device)
#         if cls_tok is not None:
#             l_con = self.supcon(cls_tok, labels)

#         # 4. Prototype contrastive (if provided)
#         l_proto = torch.tensor(0.0, device=logits.device)
#         if patch_feats is not None and proto_bank is not None:
#             l_proto = self.proto_con(patch_feats, labels, proto_bank)

#         total = (l_focal
#                  + self.lam_fgw   * l_fgw
#                  + self.lam_con   * l_con
#                  + self.lam_proto * l_proto)

#         return total, l_focal.item(), l_fgw.item()


# def get_lr_scheduler(optimizer, epochs, warmup_epochs=2):
#     def lr_lambda(epoch):
#         if epoch < warmup_epochs:
#             return float(epoch + 1) / float(warmup_epochs)
#         progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
#         return 0.5 * (1.0 + np.cos(np.pi * progress))
#     return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# # =============================================================
# # TRAINING + EVALUATION
# # =============================================================
# def train_epoch(model: DeepfakeDetector, loader, optimizer,
#                 criterion, device, epoch):
#     model.train()
#     total_loss, correct, total = 0., 0, 0
#     fgw_r_sum, fgw_f_sum, n_r, n_f = 0., 0., 0, 0
#     pbar = tqdm(loader, desc=f"Ep{epoch:02d}", leave=False)

#     for imgs, labels in pbar:
#         imgs, labels = imgs.to(device), labels.to(device)

#         # forward — also get cls_tok and patch_tok for contrastive losses
#         cls_tok, patch_tok = model.backbone(imgs)
#         logits, _, fgw     = model(imgs)

#         # full combined loss: focal + fgw + supcon + proto_con
#         loss, lce, lfgw = criterion(
#             logits, labels, fgw,
#             cls_tok    = cls_tok.detach(),
#             patch_feats= patch_tok.detach(),
#             proto_bank = model.proto_bank,
#         )
#         optimizer.zero_grad(); loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         with torch.no_grad():
#             rm = (labels==0); fm = (labels==1)
#             if rm.sum()>0:
#                 fgw_r_sum += fgw[rm].sum().item(); n_r += rm.sum().item()
#             if fm.sum()>0:
#                 fgw_f_sum += fgw[fm].sum().item(); n_f += fm.sum().item()

#         # Prototype EMA update on real images
#         real_mask = (labels == 0)
#         if real_mask.sum() > 0:
#             with torch.no_grad():
#                 _, pt = model.backbone(imgs[real_mask])
#                 cm = build_patch_graph(pt)
#                 _, _, idx = model.proto_bank.get_nearest(pt, k_match=1)
#                 model.proto_bank.update_prototypes(pt, cm, idx, momentum=0.99)

#         total_loss += loss.item()
#         correct    += (logits.argmax(1)==labels).sum().item()
#         total      += labels.size(0)
#         fgw_r = fgw_r_sum/max(n_r,1); fgw_f = fgw_f_sum/max(n_f,1)
#         pbar.set_postfix(loss=f"{loss.item():.3f}",
#                          acc=f"{correct/total*100:.1f}%",
#                          sep=f"{fgw_f-fgw_r:+.3f}")

#     fgw_r = fgw_r_sum/max(n_r,1); fgw_f = fgw_f_sum/max(n_f,1)
#     print(f"  Train FGW: real={fgw_r:.3f}  fake={fgw_f:.3f}  "
#           f"sep={fgw_f-fgw_r:+.3f}")
#     return {"loss": total_loss/len(loader), "acc": correct/total}


# @torch.no_grad()
# def evaluate(model: DeepfakeDetector, loader, device, tag=""):
#     model.eval()
#     probs_all, labels_all, fgw_all = [], [], []

#     for batch in loader:
#         # Support loaders that return (imgs, labels) or (imgs, labels, paths)
#         imgs, labels = batch[0], batch[1]
#         logits, _, fgw = model(imgs.to(device))
#         probs_all.append(torch.softmax(logits,1)[:,1].cpu().numpy())
#         labels_all.append(labels.numpy())
#         fgw_all.append(fgw.cpu().numpy())

#     probs  = np.concatenate(probs_all)
#     labels = np.concatenate(labels_all)
#     fgw    = np.concatenate(fgw_all)

#     if len(np.unique(labels)) < 2:
#         print(f"  [{tag}] Only one class present — skipping AUC")
#         return {"auc": 0., "acc": 0., "fgw_real": 0., "fgw_fake": 0.}

#     auc      = roc_auc_score(labels, probs)
#     acc      = accuracy_score(labels, probs > 0.5)
#     fgw_real = fgw[labels==0].mean() if (labels==0).any() else 0.
#     fgw_fake = fgw[labels==1].mean() if (labels==1).any() else 0.

#     if auc < 0.5:
#         print(f"  WARNING AUC<0.5. "
#               f"Flipped={roc_auc_score(labels,1-probs):.4f}")

#     print(f"  [{tag:35s}]  AUC={auc:.4f}  ACC={acc*100:.1f}%  "
#           f"FGW_r={fgw_real:.3f}  FGW_f={fgw_fake:.3f}  "
#           f"sep={fgw_fake-fgw_real:+.3f}")
#     return {"auc": auc, "acc": acc,
#             "fgw_real": fgw_real, "fgw_fake": fgw_fake}


# # =============================================================
# # TEST-TIME AUGMENTATION (TTA) — better cross-domain generalisation
# # =============================================================
# # TTA augmentations applied at inference time.
# # Each test image is evaluated N times with different augmentations
# # and predictions are averaged — improves cross-domain AUC ~1-2%
# # without any retraining.
# _TTA_TRANSFORMS = [
#     # 1. Original (no aug)
#     transforms.Compose([
#         transforms.Resize(224), transforms.CenterCrop(224),
#         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
#     # 2. Horizontal flip
#     transforms.Compose([
#         transforms.Resize(224), transforms.CenterCrop(224),
#         transforms.RandomHorizontalFlip(p=1.0),
#         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
#     # 3. Slight scale up crop
#     transforms.Compose([
#         transforms.Resize(256), transforms.CenterCrop(224),
#         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
#     # 4. Slight scale down + pad
#     transforms.Compose([
#         transforms.Resize(200), transforms.Pad(12),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
#     # 5. Mild brightness shift (simulate different domain lighting)
#     transforms.Compose([
#         transforms.Resize(224), transforms.CenterCrop(224),
#         transforms.ColorJitter(brightness=0.15, contrast=0.15),
#         transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
# ]


# @torch.no_grad()
# def evaluate_tta(model: DeepfakeDetector, dataset, device,
#                  tag="", batch_size=4, num_workers=2):
#     """
#     TTA evaluation: averages predictions across 5 augmentation views.
#     Uses the raw dataset (not loader) to apply per-image TTA transforms.
#     Only used for final cross-domain test — not during training validation
#     (too slow for per-epoch use).
#     """
#     model.eval()
#     all_probs  = []
#     all_labels = []

#     print(f"  Running TTA ({len(_TTA_TRANSFORMS)} views) on {len(dataset)} samples...")

#     for idx in tqdm(range(len(dataset)), desc=f"TTA {tag}", leave=False):
#         sample = dataset.samples[idx]
#         path, label = sample[0], sample[1]

#         try:
#             img_pil = Image.open(path).convert("RGB")
#         except Exception:
#             continue

#         view_probs = []
#         for tfm in _TTA_TRANSFORMS:
#             tensor = tfm(img_pil).unsqueeze(0).to(device)
#             logits, _, _ = model(tensor)
#             p = torch.softmax(logits, dim=1)[0, 1].item()
#             view_probs.append(p)

#         all_probs.append(np.mean(view_probs))
#         all_labels.append(label)

#     probs  = np.array(all_probs)
#     labels = np.array(all_labels)

#     if len(np.unique(labels)) < 2:
#         print(f"  [{tag}] Only one class — skipping AUC")
#         return {"auc": 0., "acc": 0., "fgw_real": 0., "fgw_fake": 0.}

#     auc = roc_auc_score(labels, probs)
#     acc = accuracy_score(labels, probs > 0.5)
#     print(f"  [{tag:35s}]  AUC(TTA)={auc:.4f}  ACC={acc*100:.1f}%")
#     return {"auc": auc, "acc": acc, "fgw_real": 0., "fgw_fake": 0.}


# # =============================================================
# # HEATMAP VISUALISATION  (Objective 4 — PS)
# # =============================================================
# # DINOv2-L/14 produces 16x16 = 256 patch tokens for 224x224 input.
# _PATCH_GRID = 16   # sqrt(256)

# def _denorm(tensor_chw: torch.Tensor) -> np.ndarray:
#     """Denormalise ImageNet-normalised tensor → uint8 HWC numpy array."""
#     mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
#     std  = np.array(STD,  dtype=np.float32).reshape(3, 1, 1)
#     img  = tensor_chw.cpu().numpy() * std + mean
#     img  = np.clip(img * 255, 0, 255).astype(np.uint8)
#     return img.transpose(1, 2, 0)   # HWC


# def _heatmap_to_color(scores_256: np.ndarray,
#                       grid: int = _PATCH_GRID) -> np.ndarray:
#     """
#     Convert flat patch scores (256,) → jet-coloured 224×224 uint8 RGB overlay.
#     """
#     patch_map = scores_256.reshape(grid, grid).astype(np.float32)
#     patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
#     patch_map_u8 = (patch_map * 255).astype(np.uint8)
#     patch_big    = cv2.resize(patch_map_u8, (224, 224),
#                               interpolation=cv2.INTER_LINEAR)
#     colored = cv2.applyColorMap(patch_big, cv2.COLORMAP_JET)
#     colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
#     return colored


# def _blend(face_rgb: np.ndarray, heatmap_rgb: np.ndarray,
#            alpha: float = 0.45) -> np.ndarray:
#     """Alpha-blend heatmap over face image."""
#     return np.clip(
#         (1 - alpha) * face_rgb.astype(np.float32) +
#         alpha * heatmap_rgb.astype(np.float32), 0, 255
#     ).astype(np.uint8)


# def _add_label_bar(canvas: np.ndarray, pred_label: str,
#                    prob: float, gt_label: str) -> np.ndarray:
#     """Append a 32-pixel info bar below the 3-panel image."""
#     bar = np.zeros((32, canvas.shape[1], 3), dtype=np.uint8)
#     correct = pred_label == gt_label
#     color   = (80, 200, 80) if correct else (220, 80, 80)
#     text    = (f"GT:{gt_label}  PRED:{pred_label}  "
#                f"p={prob:.3f}  {'OK' if correct else 'WRONG'}")
#     cv2.putText(bar, text, (6, 22),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
#                 cv2.LINE_AA)
#     return np.vstack([canvas, bar])


# @torch.no_grad()
# def save_heatmaps(model: DeepfakeDetector,
#                   loader: DataLoader,
#                   device: torch.device,
#                   out_dir: str,
#                   max_n: int = 200):
#     """
#     BUG FIX: original used a single counter so real images filled the
#     quota before any fake images were saved → heatmaps/fake/ was always empty.

#     FIX: separate counters for real and fake, each gets max_n//2 images.
#     Both folders will now be populated.
#     """
#     model.eval()
#     out_dir = Path(out_dir)
#     (out_dir / "real").mkdir(parents=True, exist_ok=True)
#     (out_dir / "fake").mkdir(parents=True, exist_ok=True)

#     # ── separate quota per class ──────────────────────────────
#     max_per_class = max_n // 2
#     saved_real    = 0
#     saved_fake    = 0
#     total_saved   = 0

#     print(f"\n  Saving heatmaps → {out_dir}  "
#           f"(max {max_per_class} real + {max_per_class} fake)")

#     for batch in tqdm(loader, desc="Heatmaps", leave=False):
#         if saved_real >= max_per_class and saved_fake >= max_per_class:
#             break

#         imgs, labels, paths = batch[0], batch[1], batch[2]
#         imgs_dev = imgs.to(device)

#         logits, heatmaps, _ = model(imgs_dev)
#         probs = torch.softmax(logits, dim=1)[:, 1].cpu()

#         for i in range(imgs.size(0)):
#             gt_label  = "fake" if int(labels[i]) == 1 else "real"

#             # ── skip if this class quota is full ────────────
#             if gt_label == "real"  and saved_real >= max_per_class:
#                 continue
#             if gt_label == "fake"  and saved_fake >= max_per_class:
#                 continue

#             # ── Face image ──────────────────────────────────
#             face_rgb   = _denorm(imgs[i])
#             scores_np  = heatmaps[i].cpu().numpy()
#             heat_color = _heatmap_to_color(scores_np)
#             blended    = _blend(face_rgb, heat_color, alpha=0.45)
#             canvas     = np.concatenate([face_rgb, blended, heat_color], axis=1)

#             prob_val   = float(probs[i])
#             pred_label = "fake" if prob_val > 0.5 else "real"
#             canvas     = _add_label_bar(canvas, pred_label, prob_val, gt_label)

#             # ── Save ────────────────────────────────────────
#             counter  = saved_fake if gt_label == "fake" else saved_real
#             src_stem = Path(paths[i]).stem
#             fname    = f"{counter:05d}_{src_stem}.png"
#             out_path = out_dir / gt_label / fname

#             cv2.imwrite(str(out_path),
#                         cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

#             if gt_label == "real":
#                 saved_real += 1
#             else:
#                 saved_fake += 1
#             total_saved += 1

#     print(f"  ✓ {total_saved} heatmap images saved to {out_dir}/")
#     print(f"    └ real/ : {saved_real} images  (should score LOW  — p≈0)")
#     print(f"    └ fake/ : {saved_fake} images  (should score HIGH — p≈1)")


# # =============================================================
# # MAIN
# # =============================================================
# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nDevice: {device}")

#     # ── STEP 0: Extract FF++ frames if needed ──────────────────────────
#     if args.ffpp_root:
#         frames_out = Path(args.frames_out)
#         n_r = len(list((frames_out/"real").glob("*.png"))) \
#               if (frames_out/"real").exists() else 0
#         n_f = len(list((frames_out/"fake").glob("*.png"))) \
#               if (frames_out/"fake").exists() else 0
#         if n_r > 50 and n_f > 50:
#             print(f"\nFF++ frames exist: {n_r} real, {n_f} fake. "
#                   f"Skipping extraction.")
#         else:
#             extract_all_ffpp_frames(args.ffpp_root, args.frames_out,
#                                     args.frames_per_video, args.max_videos)

#     # ── STEP 1: Build datasets ─────────────────────────────────────────
#     train_ds, val_ds, cross_test_ds = build_all_datasets(args)

#     nw = min(args.num_workers, os.cpu_count() or 2)
#     train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
#                               num_workers=nw, pin_memory=True, drop_last=True)
#     val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
#                               num_workers=nw, pin_memory=True)
#     test_loader  = DataLoader(cross_test_ds, args.batch_size, shuffle=False,
#                               num_workers=nw, pin_memory=True)

#     # ── Build model ────────────────────────────────────────────────────
#     print()
#     model = DeepfakeDetector(
#         backbone   = args.backbone,
#         num_protos = args.num_protos,
#         fgw_alpha  = args.fgw_alpha,
#         fgw_reg    = args.fgw_reg,
#         freq_dim   = args.freq_dim,
#         k_match    = args.k_match,
#         noise_std  = args.noise_std,
#     ).to(device)

#     total     = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Params: total={total/1e6:.1f}M  "
#           f"frozen={((total-trainable)/1e6):.1f}M  "
#           f"trainable={trainable/1000:.1f}K")
#     print(f"k_match={args.k_match}  noise_std={args.noise_std}")
#     print(f"lambda_fgw={args.lambda_fgw}  fgw_margin={args.fgw_margin}\n")

#     # ── Phase 1: Load prototype bank ──────────────────────────────────
#     if args.phase1_result:
#         load_phase1_result(args.phase1_result, model, device)
#     else:
#         print("\nWARNING: --phase1_result not set. Running inline Phase 1.\n")
#         phase1_ssl_warmup_inline(model, train_loader, device,
#                                  args.warmup_samples)

#     print(f"  fgw_scale = {model.fgw_scale:.4f}")
#     if model.fgw_scale > 10:
#         print("  !! CRITICAL: fgw_scale > 10. FGW signal will be dead.")
#         print("  !! Re-run phase1_build_graphs.py with L2-normalised tokens.")
#     else:
#         print(f"  ✓ fgw_scale OK (1–3 range)\n")

#     # ── Phase 2: Supervised training ──────────────────────────────────
#     print("="*60 + "\nPHASE 2 — Supervised training\n" + "="*60)
#     print("Training on: FF++ + WildDeepfake (train+valid)")
#     print("Validating on: 10% held-out mix")
#     print("Cross-domain test: VGGFace2 real + WildDeepfake test fake\n")

#     criterion = DetectionLoss(lambda_fgw=args.lambda_fgw,
#                               margin=args.fgw_margin,
#                               lambda_con=args.lambda_con,
#                               lambda_proto=args.lambda_proto,
#                               focal_gamma=args.focal_gamma,
#                               focal_alpha=args.focal_alpha)
#     optimizer = optim.AdamW(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.lr, weight_decay=1e-4)
#     scheduler = get_lr_scheduler(optimizer, args.epochs, warmup_epochs=2)

#     ckpt_dir = Path(args.ckpt_dir)
#     ckpt_dir.mkdir(parents=True, exist_ok=True)
#     best_auc = 0.

#     for epoch in range(1, args.epochs + 1):
#         t0 = time.time()
#         tr = train_epoch(model, train_loader, optimizer,
#                          criterion, device, epoch)
#         vl = evaluate(model, val_loader, device,
#                       f"Val (FF+++WildDF) ep{epoch:02d}")
#         scheduler.step()
#         print(f"  Ep{epoch:02d}  loss={tr['loss']:.4f}  "
#               f"acc={tr['acc']*100:.1f}%  [{time.time()-t0:.0f}s]\n")

#         if vl["auc"] > best_auc:
#             best_auc = vl["auc"]
#             torch.save({
#                 "epoch"     : epoch,
#                 "model"     : model.state_dict(),
#                 "fgw_scale" : model.fgw_scale,
#                 "val_auc"   : best_auc,
#                 "args"      : vars(args),
#             }, ckpt_dir / "best.pt")
#             print(f"  ✓ Saved best  AUC={best_auc:.4f}\n")

#     # ── Final cross-domain evaluation ─────────────────────────────────
#     print("\n" + "="*60)
#     print("FINAL — Cross-domain evaluation")
#     print("  Real: VGGFace2 (unseen domain)")
#     print("  Fake: WildDeepfake test/fake/ (unseen test split)")
#     print("="*60)

#     ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
#     model.load_state_dict(ckpt["model"])
#     model.fgw_scale = ckpt.get("fgw_scale", model.fgw_scale)
#     print(f"Best checkpoint: epoch={ckpt['epoch']}  "
#           f"val_AUC={ckpt['val_auc']:.4f}  "
#           f"fgw_scale={model.fgw_scale:.4f}\n")

#     r_val  = evaluate(model, val_loader,  device,
#                       "Val (FF+++WildDF) final")
#     r_cross = evaluate(model, test_loader, device,
#                        "Cross-domain (VGGFace2+WildDF test)")

#     # ── TTA evaluation for better cross-domain AUC ────────────────────
#     if args.use_tta:
#         print("\n  Running TTA evaluation on cross-domain test set...")
#         r_cross_tta = evaluate_tta(model, cross_test_ds, device,
#                                    tag="Cross-domain TTA",
#                                    batch_size=args.batch_size,
#                                    num_workers=min(args.num_workers, 2))

#     # ── Heatmap saving (Objective 4) ───────────────────────────────────
#     if args.save_heatmaps:
#         save_heatmaps(model, test_loader, device,
#                       out_dir=args.heatmap_dir,
#                       max_n=args.heatmap_n)

#     print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
#     print(f"  Val AUC (FF+++WildDF)      : {r_val['auc']:.4f}")
#     print(f"  Cross-domain AUC           : {r_cross['auc']:.4f}  "
#           f"← VGGFace2 real + WildDF test fake")
#     print(f"  FGW sep (val)              : "
#           f"{r_val['fgw_fake']-r_val['fgw_real']:+.3f}")
#     print(f"  FGW sep (cross-domain)     : "
#           f"{r_cross['fgw_fake']-r_cross['fgw_real']:+.3f}")
#     print(f"  FGW scale                  : {model.fgw_scale:.4f}  "
#           f"(should be 1–3)")
#     print(f"  k_match                    : {args.k_match}")
#     print(f"  lambda_fgw                 : {args.lambda_fgw}")
#     print(f"  fgw_margin                 : {args.fgw_margin}")
#     if args.save_heatmaps:
#         print(f"  Heatmaps saved to          : {args.heatmap_dir}/")
#     print(f"\n  Ablation row:")
#     print(f"  | DINOv2-L+SRM+FGW | "
#           f"FF+++WildDF→VGGFace2 | "
#           f"val={r_val['auc']:.4f} | "
#           f"cross={r_cross['auc']:.4f} |")
#     print("="*60)


# # =============================================================
# # CLI
# # =============================================================
# if __name__ == "__main__":
#     p = argparse.ArgumentParser()

#     # Phase 1
#     p.add_argument("--phase1_result",    default=None,
#         help="Path to phase1_result.pt from phase1_build_graphs.py")

#     # Data
#     p.add_argument("--frames_out",       default="./extracted_frames",
#         help="Pre-extracted FF++ frames directory (real/ and fake/ inside)")
#     p.add_argument("--ffpp_root",        default=None,
#         help="FF++ root for frame extraction (optional, skip if already extracted)")
#     p.add_argument("--wild_root",        required=True,
#         help="WildDeepfake root (contains train/test/valid subfolders)")
#     p.add_argument("--vggface2_root",    required=True,
#         help="VGGFace2 root for cross-domain test (contains train/ and val/)")
#     p.add_argument("--faceshifter_root", default=None,
#         help="Faceshifter/140k real-vs-fake root "
#              "(e.g. ./data/faceshifter/real_vs_fake/real_vs_fake). "
#              "Used for training (train+valid) and testing (test). ALL images used.")
#     p.add_argument("--ckpt_dir",         default="./checkpoints")
#     p.add_argument("--frames_per_video", type=int,   default=30)
#     p.add_argument("--max_videos",       type=int,   default=None)
#     p.add_argument("--max_per_class",    type=int,   default=5000)
#     p.add_argument("--seed",             type=int,   default=42)

#     # Model
#     p.add_argument("--backbone",         default="dinov2_vitl14_reg")
#     p.add_argument("--num_protos",       type=int,   default=32)
#     p.add_argument("--fgw_alpha",        type=float, default=0.5)
#     p.add_argument("--fgw_reg",          type=float, default=0.05)
#     p.add_argument("--freq_dim",         type=int,   default=128)
#     p.add_argument("--k_match",          type=int,   default=3)
#     p.add_argument("--noise_std",        type=float, default=0.01)

#     # Training
#     p.add_argument("--epochs",           type=int,   default=20)
#     p.add_argument("--batch_size",       type=int,   default=4)
#     p.add_argument("--lr",               type=float, default=1e-4)
#     p.add_argument("--lambda_fgw",       type=float, default=1.0)
#     p.add_argument("--fgw_margin",       type=float, default=0.5)
#     p.add_argument("--warmup_samples",   type=int,   default=5000)
#     p.add_argument("--num_workers",      type=int,   default=4)

#     # ── NEW: Contrastive + Focal loss weights ──────────────────────────
#     p.add_argument("--lambda_con",    type=float, default=0.3,
#         help="Weight for Supervised Contrastive loss on CLS tokens")
#     p.add_argument("--lambda_proto",  type=float, default=0.2,
#         help="Weight for Prototype Contrastive loss")
#     p.add_argument("--focal_gamma",   type=float, default=2.0,
#         help="Focal loss gamma — higher = more focus on hard fakes")
#     p.add_argument("--focal_alpha",   type=float, default=0.75,
#         help="Focal loss alpha — weight for fake class (0.5–0.9)")

#     # ── NEW: Test-Time Augmentation ────────────────────────────────────
#     p.add_argument("--use_tta",       action="store_true",
#         help="Enable Test-Time Augmentation at final cross-domain eval")

#     # Heatmap saving (Objective 4 — PS)
#     p.add_argument("--save_heatmaps",    action="store_true",
#         help="Save region-level heatmap overlays after final evaluation")
#     p.add_argument("--heatmap_dir",      default="./heatmaps",
#         help="Directory to write heatmap PNG files")
#     p.add_argument("--heatmap_n",        type=int,   default=200,
#         help="Max number of heatmap images to save")

#     args = p.parse_args()
#     main(args)













"""
phase2.py  —  v3-fixed  (WildDeepfake + FF++ training | VGGFace2 cross-domain eval)
==============================================================================
FIXES vs original:
  FIX 1 [PERF/CORRECTNESS] — train_epoch: backbone ran 3x per batch.
    Now DeepfakeDetector.forward() returns (logits, heatmap, fgw_dist,
    cls_tok, patch_tok, cost_M) so a SINGLE forward pass provides
    everything needed for all losses + EMA update.

  FIX 2 [RAM OOM] — build_vggface2_samples: loaded ALL paths from
    VGGFace2 train+val (up to 3.3M strings ~660 MB RAM).
    Now capped at --vgg_max_n (default 10 000) sampled from val/ only.

  FIX 3 [CORRECTNESS] — EMA update used a redundant 3rd backbone call
    on imgs[real_mask].  Now reuses patch_tok/cost_M from the single
    forward pass already computed.

  FIX 4 [CORRECTNESS] — SupConLoss received cls_tok from a separate
    backbone call (different stochastic dropout / noise state than the
    main forward).  Now uses cls_tok from the same single forward.

ALL OTHER BEHAVIOUR PRESERVED FROM v3:
  ✔ DINOv2 backbone frozen, L2-normalised patch tokens
  ✔ Cosine distance patch graph
  ✔ FGW fixed-scale normalisation (loaded from phase1_result.pt)
  ✔ Top-K prototype matching (k_match=3)
  ✔ SRM frequency branch with Dropout(0.5)
  ✔ Feature noise injection (noise_std=0.01)
  ✔ Prototype EMA update during training (momentum=0.99)
  ✔ FGW margin loss (lambda_fgw=1.0, margin=0.5)
  ✔ Focal + SupCon + ProtoCon losses
  ✔ LR warmup + cosine schedule
  ✔ Gradient clipping (max_norm=1.0)
  ✔ TTA evaluation
  ✔ Heatmap saving (Objective 4 — PS)

HOW TO RUN:
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
    --fgw_margin       0.5 \
    --epochs           30 \
    --batch_size       4 \
    --num_workers      4 \
    --ckpt_dir         ./checkpoints_v5 \
    --save_heatmaps \
    --heatmap_dir      ./heatmaps_v5 \
    --heatmap_n        300 \
    --use_tta
"""

import os, cv2, time, argparse, random, io
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, accuracy_score


# =============================================================
# CONSTANTS
# =============================================================
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
EXTS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# =============================================================
# JPEG AUGMENTATION
# =============================================================
class RandomJPEG:
    """Simulate JPEG re-compression at random quality (30–95)."""
    def __init__(self, low=30, high=95, p=0.5):
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
# PER-DATASET AUGMENTATION TRANSFORMS
# =============================================================

def get_ffpp_train_transform():
    """
    FF++ REAL + FAKE frames, training split.
    Already 224×224, face-cropped, H.264 compressed.
    Light spatial augmentation (preserve face structure),
    JPEG simulation to vary compression artifacts.
    """
    return transforms.Compose([
        transforms.Resize(232),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25,
                               saturation=0.2, hue=0.06),
        transforms.RandomGrayscale(p=0.03),
        transforms.RandomApply(
            [transforms.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_wilddeepfake_train_transform():
    """
    WildDeepfake REAL + FAKE frames, training split.
    In-the-wild internet videos: diverse quality, cameras, lighting.
    Strong augmentation to cover natural variation range.
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


def get_faceshifter_train_transform():
    """
    Faceshifter / 140k Real-vs-Fake dataset training split.
    Real images are FFHQ-sourced (high quality).
    Fake images are StyleGAN-generated — clean, no compression artifacts.
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


def get_val_transform():
    """
    Validation / test transform — no augmentation.
    Used for: FF++/WildDeepfake val split + VGGFace2 cross-domain test.
    """
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


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
# DATASET CLASSES
# =============================================================

class LabelledDataset(Dataset):
    """
    Labelled dataset: list of (path, label) pairs.
    label: 0=real, 1=fake.
    """
    def __init__(self, samples: list, transform,
                 jpeg_aug=None, name: str = ""):
        self.samples   = samples
        self.transform = transform
        self.jpeg_aug  = jpeg_aug
        self.name      = name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.jpeg_aug is not None:
                img = self.jpeg_aug(img)
            return self.transform(img), torch.tensor(label, dtype=torch.long)
        except Exception:
            alt_idx = random.randint(0, len(self.samples) - 1)
            path, label = self.samples[alt_idx]
            img = Image.open(path).convert("RGB")
            if self.jpeg_aug is not None:
                img = self.jpeg_aug(img)
            return self.transform(img), torch.tensor(label, dtype=torch.long)


# =============================================================
# DATASET BUILDERS
# =============================================================

def build_ffpp_samples(frames_out: str, max_per_class=None):
    """
    Collect FF++ samples from pre-extracted frames.
    Expected structure:
      extracted_frames/real/*.png  ← real frames (label=0)
      extracted_frames/fake/*.png  ← fake frames (label=1)
    """
    root      = Path(frames_out)
    real_dir  = root / "real"
    fake_dir  = root / "fake"

    def collect(d):
        imgs = sorted(d.glob("*.png")) + sorted(d.glob("*.jpg"))
        return [str(p) for p in imgs]

    real_imgs = collect(real_dir) if real_dir.exists() else []
    fake_imgs = collect(fake_dir) if fake_dir.exists() else []

    if not real_imgs:
        print(f"  WARNING: No FF++ real frames found in {real_dir}")
    if not fake_imgs:
        print(f"  WARNING: No FF++ fake frames found in {fake_dir}")

    samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
    print(f"  FF++ frames: real={len(real_imgs)}  fake={len(fake_imgs)}")
    return samples


def build_wilddeepfake_samples(wild_root: str, splits: list,
                                max_per_class=None, seed=42):
    """
    Collect WildDeepfake samples from specified splits.
    Structure:
      data/wilddeepfake/{split}/real/  ← real face images (label=0)
      data/wilddeepfake/{split}/fake/  ← deepfake images  (label=1)
    Each real/ and fake/ may have subdirs (one per video source).
    """
    root  = Path(wild_root)
    real_imgs, fake_imgs = [], []

    for split in splits:
        real_dir = root / split / "real"
        fake_dir = root / split / "fake"

        if real_dir.exists():
            found = _glob_images(real_dir, recursive=True)
            real_imgs.extend(found)
            print(f"    WildDeepfake {split}/real : {len(found)}")
        else:
            print(f"    WildDeepfake {split}/real : NOT FOUND ({real_dir})")

        if fake_dir.exists():
            found = _glob_images(fake_dir, recursive=True)
            fake_imgs.extend(found)
            print(f"    WildDeepfake {split}/fake : {len(found)}")
        else:
            print(f"    WildDeepfake {split}/fake : NOT FOUND ({fake_dir})")

    rng = random.Random(seed)
    rng.shuffle(real_imgs); rng.shuffle(fake_imgs)

    samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
    print(f"  WildDeepfake {splits}: real={len(real_imgs)} fake={len(fake_imgs)}")
    return samples


def build_vggface2_samples(vgg_root: str, max_n=10000, seed=42):
    """
    VGGFace2 REAL-ONLY cross-domain test set.
    VGGFace2 has no fakes. We use it to test cross-domain REAL detection.
    Structure: data/vggface2/{train,val}/<id>/*.jpg

    FIX (BUG 11): Original loaded ALL paths from train+val (up to 3.3M
    strings = ~660 MB RAM, plus massive DataLoader init time).
    Now capped at max_n=10000 sampled from val/ only, which is the
    correct held-out split for cross-domain evaluation.
    Returns samples with label=0 (all real).
    """
    root = Path(vgg_root)
    imgs = []

    # Prefer val/ for cross-domain eval (held-out from training).
    # Fall back to train/ only if val/ is empty/missing.
    for split in ["val", "train"]:
        split_dir = root / split
        if split_dir.exists():
            found = _glob_images(split_dir, recursive=True)
            imgs.extend(found)
            print(f"    VGGFace2 {split}           : {len(found)} found")
            # Stop after val/ if we already have enough
            if split == "val" and len(found) >= max_n:
                break

    if not imgs:
        print(f"  WARNING: No VGGFace2 images under {root}")
        return []

    rng = random.Random(seed)
    rng.shuffle(imgs)

    # FIX: cap to max_n to avoid RAM OOM with 3.3M-image datasets
    if max_n and len(imgs) > max_n:
        imgs = imgs[:max_n]
        print(f"  VGGFace2 real (capped)      : {len(imgs)} / original larger")
    else:
        print(f"  VGGFace2 real (total)       : {len(imgs)}")

    return [(p, 0) for p in imgs]


def build_faceshifter_samples(faceshifter_root: str, splits: list, seed=42):
    """
    Faceshifter / 140k Real-vs-Fake Kaggle dataset.
    Structure:
      data/faceshifter/real_vs_fake/real_vs_fake/
          {train,valid,test}/real/*.jpg   label=0
          {train,valid,test}/fake/*.jpg   label=1
    ALL images used — no max_per_class cap.
    """
    root = Path(faceshifter_root)
    real_imgs, fake_imgs = [], []

    for split in splits:
        real_dir = root / split / "real"
        fake_dir = root / split / "fake"

        if real_dir.exists():
            found = _glob_images(real_dir, recursive=True)
            real_imgs.extend(found)
            print(f"    Faceshifter {split}/real  : {len(found)}")
        else:
            print(f"    Faceshifter {split}/real  : NOT FOUND ({real_dir})")

        if fake_dir.exists():
            found = _glob_images(fake_dir, recursive=True)
            fake_imgs.extend(found)
            print(f"    Faceshifter {split}/fake  : {len(found)}")
        else:
            print(f"    Faceshifter {split}/fake  : NOT FOUND ({fake_dir})")

    rng = random.Random(seed)
    rng.shuffle(real_imgs)
    rng.shuffle(fake_imgs)
    samples = [(p, 0) for p in real_imgs] + [(p, 1) for p in fake_imgs]
    print(f"  Faceshifter {splits}: real={len(real_imgs)} fake={len(fake_imgs)}  [ALL]")
    return samples


class CombinedDataset(Dataset):
    """
    Combines FF++, WildDeepfake, and Faceshifter into one training dataset.
    Each source uses its own per-dataset augmentation.
    """
    def __init__(self, ffpp_samples, wild_samples, face_samples,
                 ffpp_transform, wild_transform, face_transform,
                 ffpp_jpeg=None, wild_jpeg=None, face_jpeg=None):
        self.ffpp_samples   = ffpp_samples
        self.wild_samples   = wild_samples
        self.face_samples   = face_samples
        self.ffpp_transform = ffpp_transform
        self.wild_transform = wild_transform
        self.face_transform = face_transform
        self.ffpp_jpeg      = ffpp_jpeg
        self.wild_jpeg      = wild_jpeg
        self.face_jpeg      = face_jpeg
        self.n_ffpp         = len(ffpp_samples)
        self.n_wild         = len(wild_samples)

    def __len__(self):
        return self.n_ffpp + self.n_wild + len(self.face_samples)

    def __getitem__(self, idx):
        if idx < self.n_ffpp:
            path, label = self.ffpp_samples[idx]
            transform   = self.ffpp_transform
            jpeg_aug    = self.ffpp_jpeg
        elif idx < self.n_ffpp + self.n_wild:
            path, label = self.wild_samples[idx - self.n_ffpp]
            transform   = self.wild_transform
            jpeg_aug    = self.wild_jpeg
        else:
            path, label = self.face_samples[idx - self.n_ffpp - self.n_wild]
            transform   = self.face_transform
            jpeg_aug    = self.face_jpeg
        try:
            img = Image.open(path).convert("RGB")
            if jpeg_aug: img = jpeg_aug(img)
            return transform(img), torch.tensor(label, dtype=torch.long)
        except Exception:
            fallback = random.randint(0, len(self) - 1)
            return self.__getitem__(fallback)


class CrossDomainTestDataset(Dataset):
    """
    Cross-domain test dataset — combines THREE sources:
      real : VGGFace2 val images          (label=0) — unseen real domain
      fake : WildDeepfake test/fake/       (label=1) — in-the-wild fakes
      fake : Faceshifter test/fake/        (label=1) — StyleGAN fakes

    Returns (image_tensor, label, path) for heatmap saving.
    """
    def __init__(self, vgg_samples, wild_test_fake_samples,
                 face_test_fake_samples=None):
        face_fakes = face_test_fake_samples or []
        self.samples   = vgg_samples + wild_test_fake_samples + face_fakes
        self.transform = get_val_transform()
        n_r  = sum(1 for _, l in self.samples if l == 0)
        n_f  = sum(1 for _, l in self.samples if l == 1)
        n_wf = len(wild_test_fake_samples)
        n_ff = len(face_fakes)
        print(f"  Cross-domain test: real(VGGFace2)={n_r}  "
              f"fake(WildDF)={n_wf}  fake(Faceshifter)={n_ff}  "
              f"total_fake={n_f}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), torch.tensor(label, dtype=torch.long), path
        except Exception:
            alt = random.randint(0, len(self.samples) - 1)
            path, label = self.samples[alt]
            img = Image.open(path).convert("RGB")
            return self.transform(img), torch.tensor(label, dtype=torch.long), path


# =============================================================
# DATASET FACTORY — builds train/val/test splits
# =============================================================

def build_all_datasets(args):
    """
    Builds and returns train_ds, val_ds, cross_test_ds.

    train_ds     : FF++ + WildDeepfake (train+valid) + Faceshifter (train+valid)
    val_ds       : 10% from combined training set
    cross_test_ds: VGGFace2 real + WildDeepfake test fakes + Faceshifter test fakes
    """
    print("\n" + "="*60 + "  DATASETS")

    # ── FF++ samples ─────────────────────────────────────────────
    print("\nFF++ (pre-extracted frames):")
    ffpp_samples = build_ffpp_samples(args.frames_out)

    # ── WildDeepfake training samples ────────────────────────────
    print("\nWildDeepfake (train + valid splits for training):")
    wild_train_samples = build_wilddeepfake_samples(
        args.wild_root, splits=["train", "valid"],
        seed=args.seed)

    # ── Faceshifter training samples — ALL images, no cap ─────────
    face_train_samples = []
    if args.faceshifter_root:
        print("\nFaceshifter (train + valid splits for training) [ALL]:")
        face_train_samples = build_faceshifter_samples(
            args.faceshifter_root, splits=["train", "valid"],
            seed=args.seed)

    # ── Combined train+val split ──────────────────────────────────
    all_samples = ffpp_samples + wild_train_samples + face_train_samples
    rng = random.Random(args.seed)
    rng.shuffle(all_samples)
    n_val   = int(len(all_samples) * 0.10)
    val_s   = all_samples[:n_val]
    train_s = all_samples[n_val:]

    # Separate each split back into sources for per-source augmentation
    def split_samples(samples):
        ffpp_p = str(Path(args.frames_out).resolve())
        wild_p = str(Path(args.wild_root).resolve())
        face_p = str(Path(args.faceshifter_root).resolve()) \
                 if args.faceshifter_root else ""
        ffpp_s, wild_s, face_s = [], [], []
        for path, label in samples:
            ap = str(Path(path).resolve())
            if ap.startswith(ffpp_p):
                ffpp_s.append((path, label))
            elif face_p and ap.startswith(face_p):
                face_s.append((path, label))
            else:
                wild_s.append((path, label))
        return ffpp_s, wild_s, face_s

    train_ffpp, train_wild, train_face = split_samples(train_s)
    val_ffpp,   val_wild,   val_face   = split_samples(val_s)

    # Per-source JPEG augmentation
    ffpp_jpeg = RandomJPEG(low=30, high=95, p=0.5)
    wild_jpeg = RandomJPEG(low=40, high=95, p=0.45)
    face_jpeg = RandomJPEG(low=45, high=95, p=0.40)

    train_ds = CombinedDataset(
        ffpp_samples   = train_ffpp,
        wild_samples   = train_wild,
        face_samples   = train_face,
        ffpp_transform = get_ffpp_train_transform(),
        wild_transform = get_wilddeepfake_train_transform(),
        face_transform = get_faceshifter_train_transform(),
        ffpp_jpeg      = ffpp_jpeg,
        wild_jpeg      = wild_jpeg,
        face_jpeg      = face_jpeg,
    )

    val_ds = LabelledDataset(val_s, get_val_transform(),
                             jpeg_aug=None, name="val")

    # Stats
    n_train_r = sum(1 for _, l in train_s if l == 0)
    n_train_f = sum(1 for _, l in train_s if l == 1)
    n_val_r   = sum(1 for _, l in val_s   if l == 0)
    n_val_f   = sum(1 for _, l in val_s   if l == 1)
    print(f"\n  [train] total={len(train_s)}  "
          f"real={n_train_r}  fake={n_train_f}  "
          f"ratio={n_train_f/max(n_train_r,1):.2f}")
    print(f"    └ FF++={len(train_ffpp)}  "
          f"WildDF={len(train_wild)}  "
          f"Faceshifter={len(train_face)}")
    print(f"  [val  ] total={len(val_s)}  "
          f"real={n_val_r}  fake={n_val_f}")

    # ── Cross-domain test set ─────────────────────────────────────
    print("\nCross-domain test:")
    print("  Real  : VGGFace2 (unseen domain)")
    print("  Fake1 : WildDeepfake test/fake/ (video face-swap)")
    print("  Fake2 : Faceshifter test/fake/  (StyleGAN synthesis)")

    # FIX (BUG 11): capped at args.vgg_max_n (default 10000) from val/ only
    vgg_samples = build_vggface2_samples(
        args.vggface2_root,
        max_n=args.vgg_max_n,
        seed=args.seed)

    print("\nWildDeepfake test/fake:")
    wild_test_samples = build_wilddeepfake_samples(
        args.wild_root, splits=["test"],
        seed=args.seed)
    wild_test_fakes = [(p, l) for p, l in wild_test_samples if l == 1]
    print(f"  WildDeepfake test fakes     : {len(wild_test_fakes)}")

    face_test_fakes = []
    if args.faceshifter_root:
        print("\nFaceshifter test/fake [ALL]:")
        face_test_samples = build_faceshifter_samples(
            args.faceshifter_root, splits=["test"],
            seed=args.seed)
        face_test_fakes = [(p, l) for p, l in face_test_samples if l == 1]
        print(f"  Faceshifter test fakes      : {len(face_test_fakes)}")

    cross_test_ds = CrossDomainTestDataset(
        vgg_samples, wild_test_fakes, face_test_fakes)

    return train_ds, val_ds, cross_test_ds


# =============================================================
# FRAME EXTRACTION (FF++ only)
# =============================================================
def extract_frames_from_video(video_path, out_dir, n_frames=30):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0: cap.release(); return 0
    positions = np.linspace(0, total-1, min(n_frames, total),
                            dtype=int).tolist()
    stem = Path(video_path).stem; saved = 0
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.resize(frame, (224, 224))
        cv2.imwrite(str(Path(out_dir) / f"{stem}_f{pos:06d}.png"), frame)
        saved += 1
    cap.release(); return saved


def extract_all_ffpp_frames(ffpp_root, frames_out, n_frames=30, max_videos=None):
    """Extract frames from FF++ videos into extracted_frames/real/ and /fake/."""
    ffpp_root = Path(ffpp_root)
    real_out  = Path(frames_out) / "real"
    fake_out  = Path(frames_out) / "fake"
    real_out.mkdir(parents=True, exist_ok=True)
    fake_out.mkdir(parents=True, exist_ok=True)

    MANIPS = ["Deepfakes", "Face2Face", "FaceSwap",
              "FaceShifter", "NeuralTextures", "DeepFakeDetection"]

    print("\n" + "="*60)
    print(f"STEP 0 — FF++ Frame extraction (n_frames={n_frames})")
    print("="*60)

    real_videos = []
    for sub in ["youtube", "actors"]:
        d = ffpp_root / "original_sequences" / sub / "c23" / "videos"
        if d.exists():
            real_videos += list(d.glob("*.mp4")) + list(d.glob("*.avi"))
    if max_videos: real_videos = real_videos[:max_videos // 2]
    print(f"Real videos: {len(real_videos)}")
    tr = sum(extract_frames_from_video(str(v), str(real_out), n_frames)
             for v in tqdm(real_videos, desc="Real"))
    print(f"  → {tr} real frames saved")

    fake_videos = []
    for manip in MANIPS:
        for pat in [ffpp_root / "manipulated_sequences" / manip / "c23" / "videos",
                    ffpp_root / "manipulated_sequences" / manip]:
            if pat.exists():
                fake_videos += (list(pat.glob("*.mp4")) +
                                list(pat.glob("*.avi")) +
                                list(pat.rglob("*.mp4"))); break
    fake_videos = list(set(str(v) for v in fake_videos))
    random.shuffle(fake_videos)
    if max_videos: fake_videos = fake_videos[:max_videos // 2]
    print(f"Fake videos: {len(fake_videos)}")
    tf = sum(extract_frames_from_video(v, str(fake_out), n_frames)
             for v in tqdm(fake_videos, desc="Fake"))
    print(f"  → {tf} fake frames saved")


# =============================================================
# PART A — DINOv2 BACKBONE
# =============================================================
class DINOv2Backbone(nn.Module):
    """
    Frozen DINOv2-ViT-L/14 backbone.
    L2-normalises patch tokens → fgw_scale stays in 1–3 range.
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
        patches = F.normalize(patches, dim=-1)   # L2-norm
        return cls, patches


# =============================================================
# PART B — SRM FREQUENCY BRANCH
# =============================================================
class SRMConv(nn.Module):
    def __init__(self):
        super().__init__()
        kernels     = self._build_srm_kernels()
        kernels_rgb = kernels.repeat(1, 3, 1, 1) / 3.0
        self.register_buffer("weight", kernels_rgb)

    def _build_srm_kernels(self):
        b1 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,-1,2,-1,0],
                        [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2.
        b2 = np.array([[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],
                        [0,0,0,0,0],[0,0,0,0,0]], np.float32) / 2.
        b3 = np.array([[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],
                        [0,0,-1,0,0],[0,0,0,0,0]], np.float32) / 4.
        b4 = np.array([[0,0,0,0,0],[0,-1,0,0,0],[0,0,2,0,0],
                        [0,0,0,-1,0],[0,0,0,0,0]], np.float32) / 2.
        b5 = np.array([[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],
                        [2,-6,8,-6,2],[-1,2,-2,2,-1]], np.float32) / 12.
        b6 = -np.ones((5,5), np.float32) / 24.; b6[2,2] = 1.
        bases = [b1,b2,b3,b4,b5,b6]
        all_k = []
        for b in bases:
            for r in range(4): all_k.append(np.rot90(b,r).copy())
        for b in bases: all_k.append(b.T.copy())
        return torch.tensor(
            np.stack(all_k[:30])[:,np.newaxis,:,:], dtype=torch.float32)

    def forward(self, x):
        with torch.no_grad():
            return F.conv2d(x, self.weight, padding=2).clamp(-2., 2.)


class FrequencyEncoder(nn.Module):
    def __init__(self, freq_dim=128):
        super().__init__()
        self.srm = SRMConv()
        self.encoder = nn.Sequential(
            nn.Conv2d(30, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128*4*4, freq_dim),
            nn.LayerNorm(freq_dim),
        )

    def forward(self, x):
        return self.proj(self.encoder(self.srm(x)))


# =============================================================
# PART C — PATCH GRAPH
# =============================================================
def build_patch_graph(patch_tokens: torch.Tensor) -> torch.Tensor:
    dot    = torch.bmm(patch_tokens, patch_tokens.transpose(1, 2))
    cost_M = (1.0 - dot).clamp(min=0.)
    cost_M = cost_M / (cost_M.flatten(1).max(1)[0].view(-1, 1, 1) + 1e-8)
    return cost_M


# =============================================================
# PART D — SINKHORN
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


# =============================================================
# PART E — FGW DISTANCE
# =============================================================
def gw_loss_vec(C1, C2, T):
    p  = T.sum(2); q = T.sum(1)
    t1 = (C1**2).bmm(p.unsqueeze(2)).squeeze(2).sum(1)
    t2 = ((C2**2).bmm(q.unsqueeze(2)).squeeze(2).unsqueeze(1)*T).sum([1,2])
    t3 = 2.*(T.bmm(C2).bmm(T.permute(0,2,1))*C1).sum([1,2])
    return t1 + t2 - t3


def fgw_distance(feat_src, cost_src, feat_tgt, cost_tgt,
                 alpha=0.5, reg=0.05, n_iter=20, scale=None):
    B, N, D   = feat_src.shape
    M_nodes   = feat_tgt.shape[1]
    p = torch.ones(B, N,       device=feat_src.device) / N
    q = torch.ones(B, M_nodes, device=feat_src.device) / M_nodes

    dot    = torch.bmm(feat_src, feat_tgt.transpose(1, 2))
    M_feat = (1.0 - dot).clamp(min=0.)
    M_feat = M_feat / (M_feat.flatten(1).max(1)[0].view(B,1,1) + 1e-8)

    T = sinkhorn_log(p, q, M_feat, reg, n_iter)
    for _ in range(3):
        gw_g  = -2. * cost_src.bmm(T).bmm(cost_tgt)
        M_fgw = (1-alpha)*M_feat + alpha*gw_g
        M_fgw = M_fgw / (M_fgw.flatten(1).max(1)[0].view(B,1,1) + 1e-8)
        T     = sinkhorn_log(p, q, M_fgw, reg, n_iter)

    raw_dist = ((1-alpha)*(M_feat*T).sum([1,2]) +
                alpha*gw_loss_vec(cost_src, cost_tgt, T))

    if scale is not None and scale > 0:
        return raw_dist / scale, T
    return raw_dist / (raw_dist.max().detach() + 1e-8), T


# =============================================================
# PART F — PROTOTYPE BANK
# =============================================================
class PrototypeBank(nn.Module):
    def __init__(self, K=32, N=256, D=1024):
        super().__init__()
        self.K = K
        self.register_buffer("proto_feats",
            F.normalize(torch.randn(K, N, D), dim=-1))
        self.register_buffer("proto_costs", torch.zeros(K, N, N))

    @torch.no_grad()
    def load_from_phase1(self, result: dict):
        pf = result["proto_feats"]
        pc = result["proto_costs"]
        K_saved = pf.shape[0]
        if K_saved != self.K:
            print(f"  NOTE: Phase1 K={K_saved} != model K={self.K}. "
                  f"Adjusting.")
            self.K = K_saved
            self.register_buffer("proto_feats", pf.clone())
            self.register_buffer("proto_costs", pc.clone())
        else:
            self.proto_feats.copy_(pf)
            self.proto_costs.copy_(pc)
        self.proto_feats = F.normalize(self.proto_feats, dim=-1)
        print(f"  Prototype bank loaded: K={self.K}  "
              f"feats={tuple(self.proto_feats.shape)}")

    @torch.no_grad()
    def update_prototypes(self, feat, cost, assignments, momentum=0.99):
        for k in range(self.K):
            mask = (assignments == k)
            if mask.sum() == 0: continue
            new_f = F.normalize(feat[mask].mean(0), dim=-1)
            self.proto_feats[k] = F.normalize(
                momentum * self.proto_feats[k] + (1-momentum)*new_f, dim=-1)
            self.proto_costs[k] = (momentum * self.proto_costs[k] +
                                   (1-momentum) * cost[mask].mean(0))

    def get_nearest(self, feat, k_match=1):
        feat_mean  = F.normalize(feat.mean(1), dim=-1)
        proto_mean = F.normalize(self.proto_feats.mean(1), dim=-1)
        sim = torch.mm(feat_mean, proto_mean.t())

        if k_match == 1:
            idx = sim.argmax(1)
            return self.proto_feats[idx], self.proto_costs[idx], idx
        else:
            return sim.topk(min(k_match, self.K), dim=1).indices


# =============================================================
# PART E2 — TOP-K FGW
# =============================================================
def fgw_topk(feat_src, cost_src, proto_bank: PrototypeBank,
             k_match=3, alpha=0.5, reg=0.05, scale=None):
    B = feat_src.shape[0]
    if k_match <= 1:
        pf, pc, idx = proto_bank.get_nearest(feat_src, k_match=1)
        dist, T = fgw_distance(feat_src, cost_src, pf, pc,
                               alpha, reg, scale=scale)
        return dist, T

    topk_idx  = proto_bank.get_nearest(feat_src, k_match=k_match)
    all_dists = []; last_T = None
    for ki in range(k_match):
        idx_k  = topk_idx[:, ki]
        pf_k   = proto_bank.proto_feats[idx_k]
        pc_k   = proto_bank.proto_costs[idx_k]
        dist_k, T_k = fgw_distance(feat_src, cost_src, pf_k, pc_k,
                                   alpha, reg, scale=scale)
        all_dists.append(dist_k)
        if ki == 0: last_T = T_k

    min_dist = torch.stack(all_dists, dim=1).min(dim=1).values
    return min_dist, last_T


# =============================================================
# PART G — DETECTION HEAD
# =============================================================
class DetectionHead(nn.Module):
    def __init__(self, cls_dim=1024, freq_dim=128):
        super().__init__()
        fused_dim = cls_dim + freq_dim + 1   # 1153

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        self.patch_scorer = nn.Sequential(
            nn.Linear(cls_dim, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, cls_token, freq_feat, fgw_dist, patch_tokens, T=None):
        fused  = torch.cat([cls_token, freq_feat,
                            fgw_dist.unsqueeze(1)], dim=1)
        logits = self.classifier(fused)
        heatmap = self.patch_scorer(patch_tokens).squeeze(-1)
        if T is not None:
            ent   = -(T*(T+1e-8).log()).sum(-1)
            e_min = ent.min(1, keepdim=True)[0]
            e_max = ent.max(1, keepdim=True)[0]
            heatmap = heatmap + (ent-e_min)/(e_max-e_min+1e-8)
        return logits, torch.sigmoid(heatmap)


# =============================================================
# PART H — FULL DETECTOR
# FIX (BUG 1, 2, 3, 4): forward() now returns cls_tok and patch_tok
# so train_epoch can use them directly without a second backbone call.
# =============================================================
class DeepfakeDetector(nn.Module):
    def __init__(self, backbone="dinov2_vitl14_reg", num_protos=32,
                 fgw_alpha=0.5, fgw_reg=0.05, freq_dim=128,
                 k_match=3, noise_std=0.01):
        super().__init__()
        self.backbone     = DINOv2Backbone(backbone)
        D = self.backbone.embed_dim
        N = self.backbone.num_patches
        self.freq_encoder = FrequencyEncoder(freq_dim=freq_dim)
        self.proto_bank   = PrototypeBank(K=num_protos, N=N, D=D)
        self.det_head     = DetectionHead(cls_dim=D, freq_dim=freq_dim)
        self.fgw_alpha    = fgw_alpha
        self.fgw_reg      = fgw_reg
        self.fgw_scale    = 1.0
        self.k_match      = k_match
        self.noise_std    = noise_std

    def forward(self, x):
        """
        Returns: logits, heatmap, fgw_dist, cls_tok, patch_tok, cost_M

        FIX: previously returned only (logits, heatmap, fgw_dist).
        train_epoch then called model.backbone(imgs) separately to get
        cls_tok and patch_tok → backbone ran TWICE (or 3x with EMA).
        Now a single forward() call provides everything.
        """
        cls_tok, patch_tok = self.backbone(x)
        freq_feat          = self.freq_encoder(x)

        if self.training and self.noise_std > 0:
            patch_tok = patch_tok + torch.randn_like(patch_tok) * self.noise_std
            patch_tok = F.normalize(patch_tok, dim=-1)

        cost_M = build_patch_graph(patch_tok)
        fgw_dist, T = fgw_topk(
            patch_tok, cost_M, self.proto_bank,
            k_match=self.k_match, alpha=self.fgw_alpha,
            reg=self.fgw_reg, scale=self.fgw_scale)

        logits, heatmap = self.det_head(
            cls_tok, freq_feat, fgw_dist, patch_tok, T)

        return logits, heatmap, fgw_dist, cls_tok, patch_tok, cost_M


# =============================================================
# PHASE 1 LOAD
# =============================================================
def load_phase1_result(phase1_path: str, model: DeepfakeDetector,
                       device: torch.device):
    path = Path(phase1_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Phase 1 result not found: {path}\n"
            f"Run phase1_build_graphs.py first.")

    print(f"\n{'='*60}\nLOADING PHASE 1 FROM DISK\n{'='*60}")
    result = torch.load(path, map_location=device)
    model.proto_bank.load_from_phase1(result)

    raw_scale = float(result["fgw_scale"])
    if raw_scale > 10.0:
        corrected = raw_scale / 16.0
        print(f"\n  !! fgw_scale={raw_scale:.2f} > 10. "
              f"Phase1 built without L2-norm. "
              f"Correction: {raw_scale:.2f} → {corrected:.4f}")
        print(f"  !! Re-run phase1_build_graphs.py for best results.")
        model.fgw_scale = max(corrected, 1e-4)
    else:
        model.fgw_scale = max(raw_scale, 1e-4)

    version = result.get("version", "unknown")
    print(f"  fgw_scale = {model.fgw_scale:.4f}  (expected 1–3)")
    print(f"  K         = {result['K']}")
    print(f"  N images  = {result['n_images_used']}")
    print(f"  Version   = {version}")
    if version == "v3_wildvgg":
        print(f"  ✓ Built with WildDeepfake + VGGFace2 real images")
    print()


@torch.no_grad()
def phase1_ssl_warmup_inline(model: DeepfakeDetector,
                             loader: DataLoader,
                             device: torch.device,
                             max_samples: int = 5000):
    """
    Inline Phase 1 fallback: uses REAL images from the training loader.
    Filters label==0 (real). Builds prototype bank on the fly.
    FIX: uses the 6-return forward() signature.
    """
    print("\n" + "="*60)
    print("PHASE 1 — Inline (real images from training loader)")
    print("  TIP: Run phase1_build_graphs.py for multi-source prototype bank")
    print("="*60)
    model.eval()
    K = model.proto_bank.K

    cls_list, patch_list, cost_list = [], [], []
    collected = 0

    for imgs, labels in loader:
        mask = (labels == 0)
        if mask.sum() == 0: continue
        imgs_real = imgs[mask].to(device)
        # FIX: use backbone directly (no grad needed, eval mode)
        cls_tok, patch_tok = model.backbone(imgs_real)
        cost_M = build_patch_graph(patch_tok)

        cls_list.append(cls_tok.cpu().numpy())
        patch_list.append(patch_tok.cpu())
        cost_list.append(cost_M.cpu())
        collected += mask.sum().item()

        if collected % 500 < imgs_real.shape[0]:
            print(f"  Collected {collected} real faces...")
        if collected >= max_samples: break

    if not cls_list:
        raise RuntimeError("No real images found in loader.")

    cls_np    = np.concatenate(cls_list, axis=0)
    patch_all = torch.cat(patch_list, dim=0)
    cost_all  = torch.cat(cost_list,  dim=0)

    print(f"  K-means K={K} on {len(cls_np)} real faces ...")
    km = MiniBatchKMeans(n_clusters=K, random_state=42,
                         n_init=10, max_iter=300, verbose=0)
    km.fit(cls_np)
    asgn  = torch.from_numpy(km.labels_).long().to(device)
    sizes = [(asgn==k).sum().item() for k in range(K)]
    print(f"  Cluster sizes: min={min(sizes)} max={max(sizes)}")
    if min(sizes) == 0:
        print("  WARNING: empty cluster. Try --num_protos 16")

    model.proto_bank.update_prototypes(
        patch_all.to(device), cost_all.to(device), asgn, momentum=0.5)

    real_fgw_vals = []
    for i in range(0, min(len(patch_all), 500), 8):
        f_b = patch_all[i:i+8].to(device)
        c_b = cost_all[i:i+8].to(device)
        d_b, _ = fgw_topk(f_b, c_b, model.proto_bank,
                           k_match=1, alpha=model.fgw_alpha,
                           reg=model.fgw_reg, scale=None)
        real_fgw_vals.append(d_b.cpu())

    real_fgw_all = torch.cat(real_fgw_vals)
    model.fgw_scale = max(float(torch.quantile(real_fgw_all, 0.95)), 1e-4)
    print(f"  fgw_scale = {model.fgw_scale:.4f}  (expected 1–3)\n")


# =============================================================
# LOSS
# =============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p) = -alpha * (1-p)^gamma * log(p)
    gamma=2 focuses training on hard misclassified fakes.
    alpha=0.75 upweights the fake class (minority class).
    """
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        probs   = torch.softmax(logits, dim=1)
        p_t     = probs[range(len(labels)), labels]
        alpha_t = torch.where(labels == 1,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1.0 - self.alpha, device=logits.device))
        focal_w = alpha_t * (1 - p_t) ** self.gamma
        loss    = -(focal_w * torch.log(p_t + 1e-8))
        return loss.mean()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).
    Applied on L2-normalised CLS tokens.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, features, labels):
        features = F.normalize(features, dim=-1)
        B = features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=features.device)

        sim = torch.mm(features, features.T) / self.temp

        labels_col = labels.view(-1, 1)
        mask = (labels_col == labels_col.T).float()
        mask.fill_diagonal_(0)

        if mask.sum() == 0:
            return torch.tensor(0.0, device=features.device)

        exp_sim   = torch.exp(sim)
        self_mask = torch.ones_like(exp_sim)
        self_mask.fill_diagonal_(0)
        log_denom = torch.log((exp_sim * self_mask).sum(1, keepdim=True) + 1e-8)
        log_prob  = sim - log_denom

        n_pos = mask.sum(1).clamp(min=1)
        loss  = -(mask * log_prob).sum(1) / n_pos
        return loss.mean()


class ProtoConLoss(nn.Module):
    """
    Prototype Contrastive Loss.
    Real images: maximise cosine similarity to nearest prototype.
    Fake images: push similarity below (nearest_sim - margin).
    """
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, patch_feats, labels, proto_bank):
        feat_mean  = F.normalize(patch_feats.mean(1), dim=-1)
        proto_mean = F.normalize(proto_bank.proto_feats.mean(1), dim=-1)

        sim         = torch.mm(feat_mean, proto_mean.T)
        nearest_sim = sim.max(1).values

        real_mask = (labels == 0).float()
        fake_mask = (labels == 1).float()

        loss_real = real_mask * (1.0 - nearest_sim)
        loss_fake = fake_mask * F.relu(nearest_sim + self.margin)

        return (loss_real + loss_fake).mean()


class DetectionLoss(nn.Module):
    """
    Combined loss:
      L = FocalLoss
        + lambda_fgw  * FGW_margin_loss
        + lambda_con  * SupConLoss
        + lambda_proto* ProtoConLoss
    """
    def __init__(self, lambda_fgw=1.0, margin=0.5,
                 lambda_con=0.3, lambda_proto=0.2,
                 focal_gamma=2.0, focal_alpha=0.75):
        super().__init__()
        self.focal      = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        self.supcon     = SupConLoss(temperature=0.07)
        self.proto_con  = ProtoConLoss(margin=0.4)
        self.lam_fgw    = lambda_fgw
        self.lam_con    = lambda_con
        self.lam_proto  = lambda_proto
        self.m          = margin

    def forward(self, logits, labels, fgw_dist,
                cls_tok=None, patch_feats=None, proto_bank=None):
        l_focal = self.focal(logits, labels)

        real   = (labels == 0).float()
        fake   = (labels == 1).float()
        l_fgw  = (real * fgw_dist +
                  fake * F.relu((1.0 + self.m) - fgw_dist)).mean()

        l_con = torch.tensor(0.0, device=logits.device)
        if cls_tok is not None:
            l_con = self.supcon(cls_tok, labels)

        l_proto = torch.tensor(0.0, device=logits.device)
        if patch_feats is not None and proto_bank is not None:
            l_proto = self.proto_con(patch_feats, labels, proto_bank)

        total = (l_focal
                 + self.lam_fgw   * l_fgw
                 + self.lam_con   * l_con
                 + self.lam_proto * l_proto)

        return total, l_focal.item(), l_fgw.item()


def get_lr_scheduler(optimizer, epochs, warmup_epochs=2):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================
# TRAINING + EVALUATION
# =============================================================
def train_epoch(model: DeepfakeDetector, loader, optimizer,
                criterion, device, epoch):
    """
    FIX (BUG 1, 2, 3, 4):
    Previously this function called model.backbone(imgs) separately to get
    cls_tok and patch_tok for contrastive losses, then called model(imgs)
    again for logits/fgw → backbone ran TWICE per batch.
    Then for EMA it called model.backbone(imgs[real_mask]) → THREE times.

    Now model.forward() returns all six values in one pass:
      logits, heatmap, fgw, cls_tok, patch_tok, cost_M
    The EMA update reuses patch_tok and cost_M from that same pass.
    Backbone runs exactly ONCE per batch.
    """
    model.train()
    total_loss, correct, total = 0., 0, 0
    fgw_r_sum, fgw_f_sum, n_r, n_f = 0., 0., 0, 0
    pbar = tqdm(loader, desc=f"Ep{epoch:02d}", leave=False)

    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)

        # ── Single forward pass — provides everything ──────────────
        # FIX: was backbone(imgs) + model(imgs) + backbone(imgs[real]) = 3x
        logits, _, fgw, cls_tok, patch_tok, cost_M = model(imgs)

        # ── Loss — uses cls_tok and patch_tok from the SAME forward ─
        # FIX: previously cls_tok came from a separate backbone call with
        # a different noise state → inconsistent with fgw computation.
        loss, lce, lfgw = criterion(
            logits, labels, fgw,
            cls_tok    = cls_tok.detach(),
            patch_feats= patch_tok.detach(),
            proto_bank = model.proto_bank,
        )
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            rm = (labels==0); fm = (labels==1)
            if rm.sum()>0:
                fgw_r_sum += fgw[rm].sum().item(); n_r += rm.sum().item()
            if fm.sum()>0:
                fgw_f_sum += fgw[fm].sum().item(); n_f += fm.sum().item()

        # ── Prototype EMA update — reuses patch_tok / cost_M ───────
        # FIX: was model.backbone(imgs[real_mask]) — third backbone call.
        # Now we slice the already-computed patch_tok and cost_M tensors.
        real_mask = (labels == 0)
        if real_mask.sum() > 0:
            with torch.no_grad():
                pt_real = patch_tok[real_mask].detach()
                cm_real = cost_M[real_mask].detach()
                _, _, idx = model.proto_bank.get_nearest(pt_real, k_match=1)
                model.proto_bank.update_prototypes(
                    pt_real, cm_real, idx, momentum=0.99)

        total_loss += loss.item()
        correct    += (logits.argmax(1)==labels).sum().item()
        total      += labels.size(0)
        fgw_r = fgw_r_sum/max(n_r,1); fgw_f = fgw_f_sum/max(n_f,1)
        pbar.set_postfix(loss=f"{loss.item():.3f}",
                         acc=f"{correct/total*100:.1f}%",
                         sep=f"{fgw_f-fgw_r:+.3f}")

    fgw_r = fgw_r_sum/max(n_r,1); fgw_f = fgw_f_sum/max(n_f,1)
    print(f"  Train FGW: real={fgw_r:.3f}  fake={fgw_f:.3f}  "
          f"sep={fgw_f-fgw_r:+.3f}")
    return {"loss": total_loss/len(loader), "acc": correct/total}


@torch.no_grad()
def evaluate(model: DeepfakeDetector, loader, device, tag=""):
    model.eval()
    probs_all, labels_all, fgw_all = [], [], []

    for batch in loader:
        imgs, labels = batch[0], batch[1]
        # FIX: unpack 6-tuple from forward(); ignore extra values
        logits, _, fgw, *_ = model(imgs.to(device))
        probs_all.append(torch.softmax(logits,1)[:,1].cpu().numpy())
        labels_all.append(labels.numpy())
        fgw_all.append(fgw.cpu().numpy())

    probs  = np.concatenate(probs_all)
    labels = np.concatenate(labels_all)
    fgw    = np.concatenate(fgw_all)

    if len(np.unique(labels)) < 2:
        print(f"  [{tag}] Only one class present — skipping AUC")
        return {"auc": 0., "acc": 0., "fgw_real": 0., "fgw_fake": 0.}

    auc      = roc_auc_score(labels, probs)
    acc      = accuracy_score(labels, probs > 0.5)
    fgw_real = fgw[labels==0].mean() if (labels==0).any() else 0.
    fgw_fake = fgw[labels==1].mean() if (labels==1).any() else 0.

    if auc < 0.5:
        print(f"  WARNING AUC<0.5. "
              f"Flipped={roc_auc_score(labels,1-probs):.4f}")

    print(f"  [{tag:35s}]  AUC={auc:.4f}  ACC={acc*100:.1f}%  "
          f"FGW_r={fgw_real:.3f}  FGW_f={fgw_fake:.3f}  "
          f"sep={fgw_fake-fgw_real:+.3f}")
    return {"auc": auc, "acc": acc,
            "fgw_real": fgw_real, "fgw_fake": fgw_fake}


# =============================================================
# TEST-TIME AUGMENTATION (TTA)
# =============================================================
_TTA_TRANSFORMS = [
    # 1. Original (no aug)
    transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    # 3. Slight scale up crop
    transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    # 4. Slight scale down + pad
    transforms.Compose([
        transforms.Resize(200), transforms.Pad(12),
        transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
    # 5. Mild brightness shift
    transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)]),
]


@torch.no_grad()
def evaluate_tta(model: DeepfakeDetector, dataset, device,
                 tag="", batch_size=4, num_workers=2):
    """
    TTA evaluation: averages predictions across 5 augmentation views.
    FIX: unpack 6-tuple from forward().
    """
    model.eval()
    all_probs  = []
    all_labels = []

    print(f"  Running TTA ({len(_TTA_TRANSFORMS)} views) on {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset)), desc=f"TTA {tag}", leave=False):
        sample = dataset.samples[idx]
        path, label = sample[0], sample[1]

        try:
            img_pil = Image.open(path).convert("RGB")
        except Exception:
            continue

        view_probs = []
        for tfm in _TTA_TRANSFORMS:
            tensor = tfm(img_pil).unsqueeze(0).to(device)
            # FIX: unpack 6-tuple
            logits, _, _, *_ = model(tensor)
            p = torch.softmax(logits, dim=1)[0, 1].item()
            view_probs.append(p)

        all_probs.append(np.mean(view_probs))
        all_labels.append(label)

    probs  = np.array(all_probs)
    labels = np.array(all_labels)

    if len(np.unique(labels)) < 2:
        print(f"  [{tag}] Only one class — skipping AUC")
        return {"auc": 0., "acc": 0., "fgw_real": 0., "fgw_fake": 0.}

    auc = roc_auc_score(labels, probs)
    acc = accuracy_score(labels, probs > 0.5)
    print(f"  [{tag:35s}]  AUC(TTA)={auc:.4f}  ACC={acc*100:.1f}%")
    return {"auc": auc, "acc": acc, "fgw_real": 0., "fgw_fake": 0.}


# =============================================================
# HEATMAP VISUALISATION  (Objective 4 — PS)
# =============================================================
_PATCH_GRID = 16   # sqrt(256) for DINOv2-L/14

def _denorm(tensor_chw: torch.Tensor) -> np.ndarray:
    """Denormalise ImageNet-normalised tensor → uint8 HWC numpy array."""
    mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(STD,  dtype=np.float32).reshape(3, 1, 1)
    img  = tensor_chw.cpu().numpy() * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img.transpose(1, 2, 0)


def _heatmap_to_color(scores_256: np.ndarray,
                      grid: int = _PATCH_GRID) -> np.ndarray:
    """Convert flat patch scores (256,) → jet-coloured 224×224 uint8 RGB overlay."""
    patch_map = scores_256.reshape(grid, grid).astype(np.float32)
    patch_map = (patch_map - patch_map.min()) / (patch_map.max() - patch_map.min() + 1e-8)
    patch_map_u8 = (patch_map * 255).astype(np.uint8)
    patch_big    = cv2.resize(patch_map_u8, (224, 224),
                              interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(patch_big, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return colored


def _blend(face_rgb: np.ndarray, heatmap_rgb: np.ndarray,
           alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend heatmap over face image."""
    return np.clip(
        (1 - alpha) * face_rgb.astype(np.float32) +
        alpha * heatmap_rgb.astype(np.float32), 0, 255
    ).astype(np.uint8)


def _add_label_bar(canvas: np.ndarray, pred_label: str,
                   prob: float, gt_label: str) -> np.ndarray:
    """Append a 32-pixel info bar below the 3-panel image."""
    bar = np.zeros((32, canvas.shape[1], 3), dtype=np.uint8)
    correct = pred_label == gt_label
    color   = (80, 200, 80) if correct else (220, 80, 80)
    text    = (f"GT:{gt_label}  PRED:{pred_label}  "
               f"p={prob:.3f}  {'OK' if correct else 'WRONG'}")
    cv2.putText(bar, text, (6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1,
                cv2.LINE_AA)
    return np.vstack([canvas, bar])


@torch.no_grad()
def save_heatmaps(model: DeepfakeDetector,
                  loader: DataLoader,
                  device: torch.device,
                  out_dir: str,
                  max_n: int = 200):
    """
    Save frame-level + region-level heatmap overlays.
    FIX: unpack 6-tuple from model.forward().
    Separate quotas for real and fake so both folders are populated.
    """
    model.eval()
    out_dir = Path(out_dir)
    (out_dir / "real").mkdir(parents=True, exist_ok=True)
    (out_dir / "fake").mkdir(parents=True, exist_ok=True)

    max_per_class = max_n // 2
    saved_real    = 0
    saved_fake    = 0
    total_saved   = 0

    print(f"\n  Saving heatmaps → {out_dir}  "
          f"(max {max_per_class} real + {max_per_class} fake)")

    for batch in tqdm(loader, desc="Heatmaps", leave=False):
        if saved_real >= max_per_class and saved_fake >= max_per_class:
            break

        imgs, labels, paths = batch[0], batch[1], batch[2]
        imgs_dev = imgs.to(device)

        # FIX: unpack 6-tuple from forward()
        logits, heatmaps, _, *_ = model(imgs_dev)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu()

        for i in range(imgs.size(0)):
            gt_label  = "fake" if int(labels[i]) == 1 else "real"

            if gt_label == "real"  and saved_real >= max_per_class:
                continue
            if gt_label == "fake"  and saved_fake >= max_per_class:
                continue

            face_rgb   = _denorm(imgs[i])
            scores_np  = heatmaps[i].cpu().numpy()
            heat_color = _heatmap_to_color(scores_np)
            blended    = _blend(face_rgb, heat_color, alpha=0.45)
            canvas     = np.concatenate([face_rgb, blended, heat_color], axis=1)

            prob_val   = float(probs[i])
            pred_label = "fake" if prob_val > 0.5 else "real"
            canvas     = _add_label_bar(canvas, pred_label, prob_val, gt_label)

            counter  = saved_fake if gt_label == "fake" else saved_real
            src_stem = Path(paths[i]).stem
            fname    = f"{counter:05d}_{src_stem}.png"
            out_path = out_dir / gt_label / fname

            cv2.imwrite(str(out_path),
                        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            if gt_label == "real":
                saved_real += 1
            else:
                saved_fake += 1
            total_saved += 1

    print(f"  ✓ {total_saved} heatmap images saved to {out_dir}/")
    print(f"    └ real/ : {saved_real} images  (should score LOW  — p≈0)")
    print(f"    └ fake/ : {saved_fake} images  (should score HIGH — p≈1)")


# =============================================================
# MAIN
# =============================================================
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── STEP 0: Extract FF++ frames if needed ──────────────────────────
    if args.ffpp_root:
        frames_out = Path(args.frames_out)
        n_r = len(list((frames_out/"real").glob("*.png"))) \
              if (frames_out/"real").exists() else 0
        n_f = len(list((frames_out/"fake").glob("*.png"))) \
              if (frames_out/"fake").exists() else 0
        if n_r > 50 and n_f > 50:
            print(f"\nFF++ frames exist: {n_r} real, {n_f} fake. "
                  f"Skipping extraction.")
        else:
            extract_all_ffpp_frames(args.ffpp_root, args.frames_out,
                                    args.frames_per_video, args.max_videos)

    # ── STEP 1: Build datasets ─────────────────────────────────────────
    train_ds, val_ds, cross_test_ds = build_all_datasets(args)

    nw = min(args.num_workers, os.cpu_count() or 2)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)
    test_loader  = DataLoader(cross_test_ds, args.batch_size, shuffle=False,
                              num_workers=nw, pin_memory=True)

    # ── Build model ────────────────────────────────────────────────────
    print()
    model = DeepfakeDetector(
        backbone   = args.backbone,
        num_protos = args.num_protos,
        fgw_alpha  = args.fgw_alpha,
        fgw_reg    = args.fgw_reg,
        freq_dim   = args.freq_dim,
        k_match    = args.k_match,
        noise_std  = args.noise_std,
    ).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total/1e6:.1f}M  "
          f"frozen={((total-trainable)/1e6):.1f}M  "
          f"trainable={trainable/1000:.1f}K")
    print(f"k_match={args.k_match}  noise_std={args.noise_std}")
    print(f"lambda_fgw={args.lambda_fgw}  fgw_margin={args.fgw_margin}\n")

    # ── Phase 1: Load prototype bank ──────────────────────────────────
    if args.phase1_result:
        load_phase1_result(args.phase1_result, model, device)
    else:
        print("\nWARNING: --phase1_result not set. Running inline Phase 1.\n")
        phase1_ssl_warmup_inline(model, train_loader, device,
                                 args.warmup_samples)

    print(f"  fgw_scale = {model.fgw_scale:.4f}")
    if model.fgw_scale > 10:
        print("  !! CRITICAL: fgw_scale > 10. FGW signal will be dead.")
        print("  !! Re-run phase1_build_graphs.py with L2-normalised tokens.")
    else:
        print(f"  ✓ fgw_scale OK (1–3 range)\n")

    # ── Phase 2: Supervised training ──────────────────────────────────
    print("="*60 + "\nPHASE 2 — Supervised training\n" + "="*60)
    print("Training on: FF++ + WildDeepfake (train+valid)")
    print("Validating on: 10% held-out mix")
    print("Cross-domain test: VGGFace2 real + WildDeepfake test fake\n")

    criterion = DetectionLoss(lambda_fgw=args.lambda_fgw,
                              margin=args.fgw_margin,
                              lambda_con=args.lambda_con,
                              lambda_proto=args.lambda_proto,
                              focal_gamma=args.focal_gamma,
                              focal_alpha=args.focal_alpha)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, args.epochs, warmup_epochs=2)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer,
                         criterion, device, epoch)
        vl = evaluate(model, val_loader, device,
                      f"Val (FF+++WildDF) ep{epoch:02d}")
        scheduler.step()
        print(f"  Ep{epoch:02d}  loss={tr['loss']:.4f}  "
              f"acc={tr['acc']*100:.1f}%  [{time.time()-t0:.0f}s]\n")

        if vl["auc"] > best_auc:
            best_auc = vl["auc"]
            torch.save({
                "epoch"     : epoch,
                "model"     : model.state_dict(),
                "fgw_scale" : model.fgw_scale,
                "val_auc"   : best_auc,
                "args"      : vars(args),
            }, ckpt_dir / "best.pt")
            print(f"  ✓ Saved best  AUC={best_auc:.4f}\n")

    # ── Final cross-domain evaluation ─────────────────────────────────
    print("\n" + "="*60)
    print("FINAL — Cross-domain evaluation")
    print("  Real: VGGFace2 (unseen domain)")
    print("  Fake: WildDeepfake test/fake/ (unseen test split)")
    print("="*60)

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    model.fgw_scale = ckpt.get("fgw_scale", model.fgw_scale)
    print(f"Best checkpoint: epoch={ckpt['epoch']}  "
          f"val_AUC={ckpt['val_auc']:.4f}  "
          f"fgw_scale={model.fgw_scale:.4f}\n")

    r_val   = evaluate(model, val_loader,  device,
                       "Val (FF+++WildDF) final")
    r_cross = evaluate(model, test_loader, device,
                       "Cross-domain (VGGFace2+WildDF test)")

    # ── TTA evaluation ─────────────────────────────────────────────────
    if args.use_tta:
        print("\n  Running TTA evaluation on cross-domain test set...")
        r_cross_tta = evaluate_tta(model, cross_test_ds, device,
                                   tag="Cross-domain TTA",
                                   batch_size=args.batch_size,
                                   num_workers=min(args.num_workers, 2))

    # ── Heatmap saving (Objective 4) ───────────────────────────────────
    if args.save_heatmaps:
        save_heatmaps(model, test_loader, device,
                      out_dir=args.heatmap_dir,
                      max_n=args.heatmap_n)

    print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
    print(f"  Val AUC (FF+++WildDF)      : {r_val['auc']:.4f}")
    print(f"  Cross-domain AUC           : {r_cross['auc']:.4f}  "
          f"← VGGFace2 real + WildDF test fake")
    print(f"  FGW sep (val)              : "
          f"{r_val['fgw_fake']-r_val['fgw_real']:+.3f}")
    print(f"  FGW sep (cross-domain)     : "
          f"{r_cross['fgw_fake']-r_cross['fgw_real']:+.3f}")
    print(f"  FGW scale                  : {model.fgw_scale:.4f}  "
          f"(should be 1–3)")
    print(f"  k_match                    : {args.k_match}")
    print(f"  lambda_fgw                 : {args.lambda_fgw}")
    print(f"  fgw_margin                 : {args.fgw_margin}")
    if args.save_heatmaps:
        print(f"  Heatmaps saved to          : {args.heatmap_dir}/")
    print(f"\n  Ablation row:")
    print(f"  | DINOv2-L+SRM+FGW | "
          f"FF+++WildDF→VGGFace2 | "
          f"val={r_val['auc']:.4f} | "
          f"cross={r_cross['auc']:.4f} |")
    print("="*60)


# =============================================================
# CLI
# =============================================================
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Phase 1
    p.add_argument("--phase1_result",    default=None)

    # Data
    p.add_argument("--frames_out",       default="./extracted_frames")
    p.add_argument("--ffpp_root",        default=None)
    p.add_argument("--wild_root",        required=True)
    p.add_argument("--vggface2_root",    required=True)
    p.add_argument("--faceshifter_root", default=None)
    p.add_argument("--ckpt_dir",         default="./checkpoints")
    p.add_argument("--frames_per_video", type=int,   default=30)
    p.add_argument("--max_videos",       type=int,   default=None)
    p.add_argument("--max_per_class",    type=int,   default=5000)
    p.add_argument("--seed",             type=int,   default=42)

    # FIX (BUG 11): new argument to cap VGGFace2 path loading
    p.add_argument("--vgg_max_n",        type=int,   default=10000,
        help="Max VGGFace2 images for cross-domain test. "
             "Default 10000 avoids loading 3.3M paths (~660MB RAM).")

    # Model
    p.add_argument("--backbone",         default="dinov2_vitl14_reg")
    p.add_argument("--num_protos",       type=int,   default=32)
    p.add_argument("--fgw_alpha",        type=float, default=0.5)
    p.add_argument("--fgw_reg",          type=float, default=0.05)
    p.add_argument("--freq_dim",         type=int,   default=128)
    p.add_argument("--k_match",          type=int,   default=3)
    p.add_argument("--noise_std",        type=float, default=0.01)

    # Training
    p.add_argument("--epochs",           type=int,   default=20)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--lr",               type=float, default=1e-4)
    p.add_argument("--lambda_fgw",       type=float, default=1.0)
    p.add_argument("--fgw_margin",       type=float, default=0.5)
    p.add_argument("--warmup_samples",   type=int,   default=5000)
    p.add_argument("--num_workers",      type=int,   default=4)

    # Contrastive + Focal loss weights
    p.add_argument("--lambda_con",    type=float, default=0.3)
    p.add_argument("--lambda_proto",  type=float, default=0.2)
    p.add_argument("--focal_gamma",   type=float, default=2.0)
    p.add_argument("--focal_alpha",   type=float, default=0.75)

    # TTA
    p.add_argument("--use_tta",       action="store_true")

    # Heatmap saving
    p.add_argument("--save_heatmaps",    action="store_true")
    p.add_argument("--heatmap_dir",      default="./heatmaps")
    p.add_argument("--heatmap_n",        type=int,   default=200)

    args = p.parse_args()
    main(args)