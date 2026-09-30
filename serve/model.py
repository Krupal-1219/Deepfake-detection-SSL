"""
Model definition for serving, copied verbatim from newphase2.py (PART A-H)
so checkpoint keys match exactly. Keep in sync with the training script.
"""
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


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
# SERVING HELPERS
# =============================================================
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "checkpoints_v5" / "best.pt"


def load_detector(weights_path=None, device=None):
    """Build DeepfakeDetector and load a phase-2 checkpoint (best.pt).

    best.pt already contains the frozen DINOv2 weights, so the hub model is
    built without its pretrained download and everything comes from the file.
    """
    weights_path = Path(weights_path or os.environ.get("DEEPFAKE_WEIGHTS", DEFAULT_WEIGHTS))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    hub_load = torch.hub.load
    torch.hub.load = lambda repo, name, pretrained=True, **kw: hub_load(
        repo, name, pretrained=False, **kw)
    try:
        model = DeepfakeDetector(
            backbone=args["backbone"], num_protos=args["num_protos"],
            fgw_alpha=args["fgw_alpha"], fgw_reg=args["fgw_reg"],
            freq_dim=args["freq_dim"], k_match=args["k_match"],
            noise_std=args["noise_std"])
    finally:
        torch.hub.load = hub_load

    model.load_state_dict(ckpt["model"], strict=True)
    model.fgw_scale = float(ckpt["fgw_scale"])
    model.to(device).eval()
    info = {"epoch": int(ckpt["epoch"]), "val_auc": float(ckpt["val_auc"]),
            "fgw_scale": model.fgw_scale, "backbone": args["backbone"],
            "num_protos": args["num_protos"], "k_match": args["k_match"],
            "device": str(device)}
    return model, info
