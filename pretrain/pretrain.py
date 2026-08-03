'''
Apache-2.0.

BridgeVLA pretrain on the RoboPoint detection corpus. PaliGemma image
tokens feed a ConvexUpSample head that predicts a grounding heatmap;
rows are grounding-only, so `up_grounding` is the head that trains and
its weights are mirrored into `up_action` at save time (finetune warm-
starts its single `up0` from that bucket).
``use_modified_focal_loss`` (default False) selects between the
original BridgeVLA softmax+CE supervision and the CenterNet-style
modified focal loss. ``use_lm_aux_loss`` (default False) gates the
language auxiliary loss — inert here, since RoboPoint rows carry no LM
suffix.
Manual DDP loop with two-stage freeze/unfreeze.
'''
import os
import ast
import json
import random
import sys
import yaml
import argparse
import datetime
import subprocess
import textwrap
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import cycle
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from tqdm import tqdm
from einops import rearrange

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from bridgevla.mvt.raft_utils import ConvexUpSample
from bridgevla.mvt.heads_focal import build_focal_dual_head
from bridgevla.mvt.memory import MemoryBlock
import bridgevla.mvt.utils as mvt_utils


USE_SWANLAB = False


# CenterNet-style heatmap / focal-loss constants: a per-class binary heatmap with Gaussian-blurred GT
# (center = 1), sigmoid + modified focal loss, head bias pre-initialized to a pi=0.1 prior.
HM_MIN_OVERLAP = 0.7   # gaussian_radius IoU tolerance
HM_MIN_RADIUS  = 6     # grounding floor (σ=(2·6+1)/6≈2.17 ~ old fixed sigma=2)
HM_VLA_MIN_RADIUS = 15 # VLA floor (σ=(2·15+1)/6≈5.17) — wider plateau around
                       # next_kf TCP: richer gradient for the action heads without blurring the peak.
HM_PRIOR_LOGIT = -2.19 # log((1-π)/π) with π=0.1 — matches RetinaNet/CenterNet


# ---- Utilities ----
class _TeeStream:
    # Mirror Python-level stdout/stderr into a log file so a run folder always contains its own log.
    # C-library writes through fd 1/2 still need pretrain.sh's shell-level tee.
    def __init__(self, primary, log_file):
        self._primary = primary
        self._log = log_file

    def write(self, data):
        self._primary.write(data)
        try:
            self._log.write(data)
        except Exception:
            pass
        return len(data) if isinstance(data, (str, bytes)) else 0

    def flush(self):
        self._primary.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self):
        return self._primary.fileno()

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _install_tee_logging(run_dir: str, rank: int) -> None:
    # Per-rank log avoids concurrent writes on a shared FS; line-buffered so `tail -f` keeps up.
    log_path = os.path.join(run_dir, f"train_rank{rank}.log")
    log_fp = open(log_path, "a", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)
    print(f"[Pretrain] rank {rank} logging stdout/stderr to {log_path}", flush=True)


def is_list_string(s):
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']') and len(s) >= 2):
        return False
    try:
        parsed = ast.literal_eval(s)
        return isinstance(parsed, list)
    except (SyntaxError, ValueError):
        return False


# ---- CenterNet-style GT heatmap + modified focal loss ----
# GT is a per-pixel soft label in [0, 1]: 1.0 at each object center with a 2D Gaussian falling off by an
# IoU-derived radius. Overlapping bboxes merge elementwise-max so every center stays at 1.0.
def _centernet_gaussian_radius(h: float, w: float,
                               min_overlap: float = HM_MIN_OVERLAP) -> float:
    """Min Gaussian radius such that a prediction anywhere inside the circle
    still has IoU >= min_overlap with the GT bbox (CornerNet formula).
    """
    a1 = 1.0
    b1 = h + w
    c1 = w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + (b1 * b1 - 4 * a1 * c1) ** 0.5) / 2

    a2 = 4.0
    b2 = 2 * (h + w)
    c2 = (1 - min_overlap) * w * h
    r2 = (b2 + (b2 * b2 - 4 * a2 * c2) ** 0.5) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (h + w)
    c3 = (min_overlap - 1) * w * h
    r3 = (b3 + (b3 * b3 - 4 * a3 * c3) ** 0.5) / 2
    return min(r1, r2, r3)


def _gaussian2d_kernel(radius: int, sigma: float,
                       dtype=torch.float32) -> torch.Tensor:
    """2D isotropic Gaussian of shape (2r+1, 2r+1) with peak 1.0 at the
    center. Tiny values near the float epsilon are zeroed for cleanliness.
    """
    diameter = 2 * radius + 1
    m = (diameter - 1) / 2.0
    ys = torch.arange(diameter, dtype=dtype) - m
    xs = torch.arange(diameter, dtype=dtype) - m
    yy = ys.view(-1, 1).expand(diameter, diameter)
    xx = xs.view(1, -1).expand(diameter, diameter)
    g = torch.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    eps = torch.finfo(g.dtype).eps * g.max()
    g = torch.where(g < eps, torch.zeros_like(g), g)
    return g


def _draw_umich_gaussian_(hm: torch.Tensor, cx: int, cy: int,
                          radius: int) -> None:
    """In-place paste of a 2D Gaussian onto `hm` (shape [H, W]) centered at
    (cx, cy). Overlap is resolved by elementwise max so each GT center
    pixel stays at exactly 1.0 even when multiple objects' Gaussians
    intersect. Handles image-boundary clipping like CenterNet's version.
    """
    sigma = (2 * radius + 1) / 6.0
    gauss = _gaussian2d_kernel(radius, sigma, dtype=hm.dtype).to(hm.device)
    H, W = hm.shape
    left,  right  = min(cx, radius), min(W - cx, radius + 1)
    top,   bottom = min(cy, radius), min(H - cy, radius + 1)
    if left + right <= 0 or top + bottom <= 0:
        return
    hm_slice   = hm[cy - top : cy + bottom, cx - left : cx + right]
    gauss_slice = gauss[radius - top : radius + bottom,
                        radius - left : radius + right]
    torch.maximum(hm_slice, gauss_slice, out=hm_slice)


def _build_centernet_gt_heatmap(
    bboxes_batch, H: int, W: int, device=None, dtype=torch.float32,
    is_action_batch=None,
) -> torch.Tensor:
    """Build CenterNet-style GT heatmaps from per-sample normalized
    (cx, cy, bw, bh) bboxes.

    Returns (bs, 1, H, W) with:
      * each bbox center pixel = 1.0,
      * 2D Gaussian soft labels decaying around each center,
      * radius = max(gaussian_radius(bh_px, bw_px, HM_MIN_OVERLAP),
                     floor) where floor is HM_VLA_MIN_RADIUS for rows
        flagged in `is_action_batch` (default: all grounding) and
        HM_MIN_RADIUS otherwise. Lets VLA rows carry a wider Gaussian
        around next_kf TCP without blowing out grounding supervision.
      * multi-bbox overlap resolved by max (never softmax-normalized).
    """
    bs = len(bboxes_batch)
    hm = torch.zeros((bs, 1, H, W), dtype=dtype)
    for b_idx, bboxes in enumerate(bboxes_batch):
        assert len(bboxes) >= 1, "empty bbox list in batch"
        floor = (HM_VLA_MIN_RADIUS
                 if (is_action_batch is not None and bool(is_action_batch[b_idx]))
                 else HM_MIN_RADIUS)
        for (cx, cy, bw, bh) in bboxes:
            cx_px = float(cx) * W
            cy_px = float(cy) * H
            bw_px = max(float(bw) * W, 1.0)
            bh_px = max(float(bh) * H, 1.0)
            r_cn = _centernet_gaussian_radius(bh_px, bw_px, HM_MIN_OVERLAP)
            radius = max(int(r_cn), floor)
            cx_i = int(round(cx_px))
            cy_i = int(round(cy_px))
            cx_i = min(max(cx_i, 0), W - 1)
            cy_i = min(max(cy_i, 0), H - 1)
            _draw_umich_gaussian_(hm[b_idx, 0], cx_i, cy_i, radius)
    if device is not None:
        hm = hm.to(device)
    return hm


def _modified_focal_loss_per_sample(
    logits: torch.Tensor, gt: torch.Tensor,
) -> torch.Tensor:
    """CornerNet/CenterNet modified focal loss, computed per sample.

    logits : (bs, 1, H, W) raw (no activation applied yet).
    gt     : (bs, 1, H, W) soft labels in [0, 1], center pixels == 1.
    Returns: (bs,) per-sample loss. Rows with no positive (should not
             happen in practice) fall back to -neg_sum, matching
             CenterNet _neg_loss's num_pos==0 branch.
    """
    p = torch.clamp(logits.sigmoid(), min=1e-6, max=1 - 1e-6)
    pos_inds = gt.eq(1).to(p.dtype)
    neg_inds = gt.lt(1).to(p.dtype)
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(p)       * torch.pow(1 - p, 2) * pos_inds
    neg_loss = torch.log(1 - p)   * torch.pow(p, 2)     * neg_weights * neg_inds

    dims = (1, 2, 3)
    num_pos = pos_inds.sum(dim=dims)             # (bs,)
    pos_sum = pos_loss.sum(dim=dims)
    neg_sum = neg_loss.sum(dim=dims)
    denom   = num_pos.clamp(min=1.0)
    loss    = -(pos_sum + neg_sum) / denom
    # A row with no positives uses -neg_sum undivided, matching CenterNet _neg_loss.
    return torch.where(num_pos > 0, loss, -neg_sum)


def visualize_bboxes_and_heatmap(image, bboxes_norm, heatmap_tensor, save_path,
                                 caption=None, logits_tensor=None,
                                 bbox_colors=('red', 'lime', 'cyan', 'yellow'),
                                 bbox_width=2,
                                 extra_points=None,
                                 image_title=None):
    """Render bboxes + GT heatmap (and optional point overlays) to disk.

    `extra_points`: list of {xy_norm, color, filled, radius, label} dicts —
    marker overlays for targets too small to read off the bbox alone.
    """
    resized_img = image.resize((224, 224))
    draw = ImageDraw.Draw(resized_img)
    color_cycle = cycle(bbox_colors)
    for bbox in bboxes_norm:
        cx, cy, w, h = bbox
        x0 = max(0, int((cx - w / 2) * 224))
        y0 = max(0, int((cy - h / 2) * 224))
        x1 = min(223, int((cx + w / 2) * 224))
        y1 = min(223, int((cy + h / 2) * 224))
        draw.rectangle([x0, y0, x1, y1], outline=next(color_cycle), width=bbox_width)

    legend_items: List[Tuple[str, str]] = []  # (label, color)
    if extra_points:
        for pt in extra_points:
            x, y = pt["xy_norm"]
            px = int(round(x * 224))
            py = int(round(y * 224))
            r = int(pt.get("radius", 7))
            color = pt.get("color", "cyan")
            if pt.get("filled"):
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
                draw.ellipse([px - r - 2, py - r - 2, px + r + 2, py + r + 2],
                             outline="white", width=2)
            else:
                draw.ellipse([px - r, py - r, px + r, py + r],
                             outline=color, width=2)
            if pt.get("label"):
                legend_items.append((pt["label"], color))

    heatmap = heatmap_tensor.squeeze().cpu().numpy()
    ncols = 3 if logits_tensor is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    axes[0].imshow(resized_img)
    axes[0].set_title(image_title or f'Image with {len(bboxes_norm)} BBoxes')
    axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='viridis', alpha=0.95)
    axes[1].imshow(resized_img, alpha=0.05)
    axes[1].set_title('Heatmap'); axes[1].axis('off')
    if logits_tensor is not None:
        logits = logits_tensor.squeeze().cpu().numpy()
        im = axes[2].imshow(logits, cmap='viridis')
        axes[2].set_title('Raw Logits'); axes[2].axis('off')
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Split the caption on '\n' so each logical line is textwrap.fill'd independently.
    bottom_lines = []
    if legend_items:
        bottom_lines.append("  |  ".join(f"● {lbl}" for lbl, _ in legend_items))
    if caption is not None:
        for line in caption.split("\n"):
            bottom_lines.append(line)
    if bottom_lines:
        fig.text(0.5, 0.02,
                 "\n".join(textwrap.fill(line, width=140) for line in bottom_lines),
                 ha='center', va='bottom', fontsize=9)
        # Reserve more vertical space when the caption spans multiple lines.
        fig.subplots_adjust(bottom=min(0.35, 0.10 + 0.035 * len(bottom_lines)))
    plt.savefig(save_path)
    plt.close(fig)


# ---- RoboPoint grounding dataset ----
# Pretrain consumes exactly one corpus: the RoboPoint detection JSON (image + referring phrase + boxes).
# Images are center-cropped to a square and boxes re-normalized to [0, 1] of the crop, so everything
# downstream sees one schema: {image, text, bboxes_cxcywh, dataset, category}.

_BBOX_FORMATS = ("pixel_xywh", "norm_xyxy", "norm_cxcywh")


# --- Prompt templates ---
# Graspable samples pick a grasp template with prob GRASPABLE_PROMPT_PROB else a detect one; uniform within a pool.
GRASPABLE_PROMPTS = (
    "Grasp the {target}.",
    "Pick up the {target}.",
    "Lift the {target}.",
    "Hold the {target}.",
    "Grab the {target}.",
    "Fetch the {target}.",
    "Retrieve the {target}.",
    "Take the {target}.",
)
NON_GRASPABLE_PROMPTS = (
    "Detect the {target}.",
    "Search for the {target}.",
    "Identify the {target}.",
    "Spot the {target}.",
    "Observe the {target}.",
    "Focus on the {target}.",
    "Approach the {target}.",
    "Find the {target}.",
    "Locate the {target}.",
    "Point to the {target}.",
    "Track the {target}.",
)
GRASPABLE_PROMPT_PROB = 0.5

# A graspable-tagged sample whose largest bbox covers >10% of the crop is demoted to detect prompts.
LARGE_BBOX_AREA_THRESHOLD = 0.1

PRETRAIN_GLOBAL_SEED = 42


def _wrap_with_prompt(text: str, category: str, rng: random.Random) -> str:
    """Wrap a noun phrase with a random grasp/detect template."""
    if category == "graspable" and rng.random() < GRASPABLE_PROMPT_PROB:
        template = rng.choice(GRASPABLE_PROMPTS)
    else:
        template = rng.choice(NON_GRASPABLE_PROMPTS)
    return template.format(target=text)


@dataclass
class GroundingSample:
    """Per-sample record produced by the RoboPoint parser."""
    image_path: str
    text: str
    bboxes: Tuple[Tuple[float, float, float, float], ...]
    bbox_format: str  # one of _BBOX_FORMATS
    dataset: str
    category: str = "non_graspable"


def _to_pixel_xyxy(bboxes, bbox_format, W, H):
    """Convert parser-provided bboxes to pixel xyxy in the original frame."""
    out = []
    if bbox_format == "pixel_xywh":
        for x, y, w, h in bboxes:
            out.append((float(x), float(y), float(x + w), float(y + h)))
    elif bbox_format == "norm_xyxy":
        for x1, y1, x2, y2 in bboxes:
            out.append((float(x1) * W, float(y1) * H,
                        float(x2) * W, float(y2) * H))
    elif bbox_format == "norm_cxcywh":
        for cx, cy, w, h in bboxes:
            out.append(((float(cx) - float(w) / 2.0) * W,
                        (float(cy) - float(h) / 2.0) * H,
                        (float(cx) + float(w) / 2.0) * W,
                        (float(cy) + float(h) / 2.0) * H))
    else:
        raise ValueError(f"Unknown bbox_format: {bbox_format}")
    return out


def _center_crop_params(W: int, H: int) -> Tuple[int, int, int]:
    """Return (side, left, top) for the largest inscribed-square center crop."""
    side = min(W, H)
    left = (W - side) // 2
    top = (H - side) // 2
    return side, left, top


def _center_crop_resize_with_bboxes(img, bboxes_xyxy_px, res):
    """Center-crop to a square, resize to `res`, and map pixel-xyxy bboxes
    to normalized-cxcywh of the cropped frame. Out-of-crop bboxes drop."""
    W, H = img.size
    side, left, top = _center_crop_params(W, H)
    if (W, H) != (res, res):
        img = img.crop((left, top, left + side, top + side))
        if side != res:
            img = img.resize((res, res), Image.BILINEAR)

    side_f = float(side)
    kept = []
    for x1, y1, x2, y2 in bboxes_xyxy_px:
        x1c = max(0.0, x1 - left)
        y1c = max(0.0, y1 - top)
        x2c = min(side_f, x2 - left)
        y2c = min(side_f, y2 - top)
        if x2c <= x1c or y2c <= y1c:
            continue
        cx = (x1c + x2c) * 0.5 / side_f
        cy = (y1c + y2c) * 0.5 / side_f
        bw = (x2c - x1c) / side_f
        bh = (y2c - y1c) / side_f
        kept.append((cx, cy, bw, bh))
    return img, kept


class GroundingDataset(Dataset):
    """Grounding dataset over pre-parsed GroundingSample records."""

    def __init__(self, samples: Sequence[GroundingSample], res: int = 224):
        self.samples = list(samples)
        self.res = res

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        W_orig, H_orig = img.size
        bboxes_xyxy = _to_pixel_xyxy(s.bboxes, s.bbox_format, W_orig, H_orig)
        img, cxcywh = _center_crop_resize_with_bboxes(img, bboxes_xyxy, self.res)
        if not cxcywh:
            # Rare at 224 center-crop. Tiny fallback keeps batch shape valid.
            cxcywh = [(0.5, 0.5, 0.02, 0.02)]

        category = s.category
        if category == "graspable":
            max_area = max(bw * bh for (_, _, bw, bh) in cxcywh)
            if max_area > LARGE_BBOX_AREA_THRESHOLD:
                category = "non_graspable"

        return {
            "image": img,
            "text": _wrap_with_prompt(s.text, category, rng=random),
            "bboxes_cxcywh": cxcywh,
            "dataset": s.dataset,
            "category": category,
        }


# --- Parser ---------------------------------------------------------------

_ROBOPOINT_PROMPT_A = ("<image>\nPlease provide the bounding box coordinate "
                       "of the region this sentence describes: ")
_ROBOPOINT_PROMPT_A_NOIMG = ("Please provide the bounding box coordinate "
                             "of the region this sentence describes: ")
_ROBOPOINT_PROMPT_B = (" Format the result as a list of tuples, i.e. "
                       "[(x1, y1, w1, h1), (x2, y2, w2, h2), ...], where x "
                       "and y are the normalized pixel locations of the "
                       "object centers, and w and h are the normalized "
                       "object widths and heights. All values of x, y, w, "
                       "and h should be between 0 and 1.")


def parse_coco_robopoint(image_folder: str, json_path: str) -> List[GroundingSample]:
    """Parse the RoboPoint-detection JSON (COCO-image backed)."""
    with open(json_path, "r") as f:
        data = json.load(f)
    samples: List[GroundingSample] = []
    for src in tqdm(data, desc="Parsing COCO RoboPoint"):
        conv = src["conversations"]
        assert len(conv) % 2 == 0
        img_path = os.path.join(image_folder, src["image"])
        for i in range(1, len(conv), 2):
            ans_str = conv[i]["value"]
            if not is_list_string(ans_str):
                continue
            query = conv[i - 1]["value"]
            try:
                ans = ast.literal_eval(ans_str)
            except (SyntaxError, ValueError):
                continue

            if _ROBOPOINT_PROMPT_A in query or _ROBOPOINT_PROMPT_A_NOIMG in query:
                phrase = (query.replace(_ROBOPOINT_PROMPT_A, "")
                              .replace(_ROBOPOINT_PROMPT_A_NOIMG, ""))
                if not (isinstance(ans, list) and len(ans) == 4
                        and all(isinstance(v, (int, float)) for v in ans)):
                    continue
                bboxes = (tuple(float(v) for v in ans),)
                fmt = "norm_xyxy"
            elif _ROBOPOINT_PROMPT_B in query:
                phrase = query.replace(_ROBOPOINT_PROMPT_B, "")
                if not (isinstance(ans, list) and len(ans) >= 1
                        and all(isinstance(v, tuple) and len(v) == 4 for v in ans)):
                    continue
                bboxes = tuple(tuple(float(x) for x in v) for v in ans)
                fmt = "norm_cxcywh"
            else:
                continue

            phrase = phrase.replace("<image>\n", "").strip()
            if not phrase:
                continue
            samples.append(GroundingSample(
                image_path=img_path,
                text=phrase,
                bboxes=bboxes,
                bbox_format=fmt,
                dataset="coco",
            ))
    return samples


# --- Collator -------------------------------------------------------------

@dataclass
class GroundingCollator(object):
    """Build a batch from GroundingDataset items.

    RoboPoint rows are heatmap-only: no LM suffix, no action routing. The
    batch therefore carries just the PaliGemma tensors plus the per-row
    bboxes / tags that the loss and the visualizers read. forward()
    defaults `hm_weights` to all-ones and `is_action` to all-False, which
    is exactly the grounding-only regime this data implies.
    """
    processor: AutoProcessor

    def __call__(self, data):
        texts = [ex["text"] for ex in data]
        images = [ex["image"] for ex in data]
        prompts = ["<image>" + t for t in texts]
        tokens = self.processor(
            text=prompts, images=images,
            return_tensors="pt", padding="longest",
        )
        tokens["raw_text"] = texts                             # prompt text (diagnostics)
        tokens["bboxes"] = [ex["bboxes_cxcywh"] for ex in data]  # list[list[(cx,cy,w,h)]]
        tokens["dataset_tag"] = [ex["dataset"] for ex in data]
        tokens["category_tag"] = [ex["category"] for ex in data]
        return tokens


# --- Dataset assembly -----------------------------------------------------

def build_pretrain_datasets(cfg: dict, processor: AutoProcessor, res: int):
    """Parse the RoboPoint corpus and split off a deterministic val holdout.

    Paths come from `image_folder` / `json_detection_path` (yaml, overridable
    on the CLI). `val.holdout_samples` rows are held out of training and
    served as the validation split; 0 disables validation entirely.

    The shuffle is seeded from `cfg.seed` with a local RNG, so every rank
    derives the identical split without communicating.

    Returns (train_ds, val_ds, collator); `val_ds` is None when no holdout
    is configured.
    """
    image_folder = cfg.get("image_folder")
    json_detection_path = cfg.get("json_detection_path")
    assert image_folder and json_detection_path, (
        "pretrain data paths missing — set `image_folder` / "
        "`json_detection_path` in the config (or pass --image_folder / "
        "--json_detection_path)."
    )
    samples = parse_coco_robopoint(image_folder, json_detection_path)
    if not samples:
        raise RuntimeError(
            f"parse_coco_robopoint produced zero samples from "
            f"{json_detection_path}"
        )

    rng = random.Random(int(cfg.get("seed", PRETRAIN_GLOBAL_SEED)))
    rng.shuffle(samples)

    n_holdout = int((cfg.get("val") or {}).get("holdout_samples", 0))
    # Cap the holdout at half the corpus (guards tiny smoke-test JSONs against an empty train split).
    n_holdout = max(0, min(n_holdout, len(samples) // 2))
    val_samples = samples[:n_holdout]
    train_samples = samples[n_holdout:]

    smoke_n = int(os.environ.get("SMOKE_MAX_SAMPLES", "0"))
    if smoke_n > 0:
        train_samples = train_samples[:smoke_n]
        print(f"[SMOKE] truncated train set to {len(train_samples)} samples",
              flush=True)

    print(f"[Pretrain] RoboPoint samples: {len(train_samples)} train"
          f" + {len(val_samples)} val holdout", flush=True)

    collator = GroundingCollator(processor=processor)
    train_ds = GroundingDataset(train_samples, res=res)
    val_ds = GroundingDataset(val_samples, res=res) if val_samples else None
    return train_ds, val_ds, collator

# ---- Model - pretrain counterpart of finetune/mvt_single.py ----
class BridgeVLAPretrainModel(nn.Module):
    """PaliGemma image tokens ─► up_grounding ─► grounding rows
                                ─► up_action    ─► action rows

    Single 2D image (num_img=1) — no renderer. Two structurally identical
    ConvexUpSample heads are dispatched per-row by `is_action`:
      * up_grounding — "where is X *now*" heatmaps for image grounding.
      * up_action    — "where should the next keyframe TCP go" heatmaps,
                       whose GT sits at a future TCP pixel rather than on
                       the object (different spatial distribution from
                       grounding GT — keeping heads separate avoids
                       negative transfer).
    The shipped pretrain corpus (RoboPoint) is grounding-only, so
    `is_action` is all-False and only up_grounding trains; save_checkpoint
    mirrors it into the `up_action.*` keys. The head stays in the model so
    a checkpoint keeps the state-dict layout mvt_single.MVT expects (its
    loader warm-starts finetune's single `up0` from up_action).
    """

    def __init__(self, model_id,
                 img_size=224, img_patch_size=14,
                 use_modified_focal_loss: bool = False,
                 use_lm_aux_loss: bool = False,
                 memory_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        # PaliGemma image tokens feed the heatmap head(s) directly at vlm_dim=2048. ``kwargs`` swallows
        # legacy config keys from removed branches so old configs still load.
        self.img_size = img_size
        self.img_patch_size = img_patch_size
        self.num_pat_img = img_size // img_patch_size  # 16
        self.num_img = 1
        self.use_modified_focal_loss = bool(use_modified_focal_loss)
        self.use_lm_aux_loss = bool(use_lm_aux_loss)

        # PaliGemma: bf16 (matches mvt_single).
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self.vlm_dim = self.model.config.hidden_size  # 2048

        # Processor is kept as a non-submodule attribute (no state saved).
        self._processor = AutoProcessor.from_pretrained(model_id)

        # Two task-specific upsample heads on the raw 2048-dim PaliGemma features; ``forward`` dispatches
        # per row so each head specializes. The bias prior applies only when use_modified_focal_loss=True.
        self.up_grounding = ConvexUpSample(
            in_dim=self.vlm_dim, out_dim=1, up_ratio=self.img_patch_size,
        )
        self.up_action = ConvexUpSample(
            in_dim=self.vlm_dim, out_dim=1, up_ratio=self.img_patch_size,
        )
        if self.use_modified_focal_loss:
            with torch.no_grad():
                self.up_grounding.net_out[-1].bias.fill_(HM_PRIOR_LOGIT)
            # up_action warm-starts from up_grounding so both heads share one convex-upsample prior.
            self.up_action.load_state_dict(self.up_grounding.state_dict())
        # Soft-label CE for the default (original BridgeVLA) path; reduction "none" for per-row weighting.
        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")

        # ---- Episodic memory (spatial anchor + temporal episodic) ----
        # Mirrors finetune/bridgevla/mvt/mvt_single.py. Pretrain has no zoom stage 2, so only the stage-1
        # blocks exist; finetune cross-loads mem_spatial_s2 from mem_spatial_s1. Anchor / hist PaliGemma
        # forwards run under no_grad and are detached, and residual exits are zero-init (identity at t=0).
        self.memory_cfg = dict(memory_cfg) if memory_cfg else {}
        self.memory_enabled = bool(self.memory_cfg.get("enabled", False))
        if self.memory_enabled:
            mem_kw = dict(
                dim=self.vlm_dim,
                heads=int(self.memory_cfg.get("heads", 8)),
                dim_head=int(self.memory_cfg.get("dim_head", 128)),
                num_layers=int(self.memory_cfg.get("num_layers", 2)),
                ffn_mult=int(self.memory_cfg.get("ffn_mult", 2)),
                use_fast=bool(self.memory_cfg.get("use_fast_attn", False)),
            )
            self.K_temporal = int(self.memory_cfg.get("k_temporal", 4))
            if self.memory_cfg.get("spatial_at_mvt1", True):
                self.mem_spatial_s1 = MemoryBlock(kind="spatial", **mem_kw)
            if self.memory_cfg.get("temporal_at_mvt1", True):
                # The temporal MemoryBlock is purely visual + per-slot index PE - do NOT pass action_dim
                # (removed upstream; it raises TypeError). K_temporal must match GemBench's memory.k_temporal.
                self.mem_temporal_s1 = MemoryBlock(
                    kind="temporal", K_temporal=self.K_temporal, **mem_kw,
                )
        else:
            self.K_temporal = 0

    def _build_gt_heatmap(self, bboxes_batch, h, w, device, is_action_batch=None):
        """CenterNet-style GT heatmap (center=1, max-merged multi-bbox).
        Used when ``use_modified_focal_loss=True``.
        """
        return _build_centernet_gt_heatmap(
            bboxes_batch, h, w, device=device, is_action_batch=is_action_batch,
        )

    def _build_softmax_gt_heatmap(self, bboxes_batch, h, w, device,
                                  sigma: float = 1.5, thres_sigma_times: int = 3):
        """Original BridgeVLA softmax-normalized Gaussian GT.

        Returns (bs, h*w, 1) with each (h*w) slice summing to 1. Multi-bbox
        rows use the bbox center as the supervision point and average their
        per-bbox softmax heatmaps. Matches ``mvt_utils.generate_hm_from_pt``
        semantics (sigma + threshold + softmax).

        NOTE (multi-target semantics): this is the DEFAULT pretrain GT
        path (use_modified_focal_loss=False). The whole heatmap is one
        probability distribution that sums to 1 — N non-overlapping
        targets each get a peak of ~1/N (NOT 1.0 each). Loss is soft-label
        CE on raw logits (see forward() ~line 2128). The peak-1 +
        per-pixel sigmoid + focal-loss formulation lives only in the
        focal branch (_build_centernet_gt_heatmap).
        """
        bs = len(bboxes_batch)
        out = torch.zeros((bs, h * w, 1), dtype=torch.float32, device=device)
        for b_idx, bboxes in enumerate(bboxes_batch):
            assert len(bboxes) >= 1
            pts = []
            for (cx, cy, _bw, _bh) in bboxes:
                pts.append([float(cx) * w, float(cy) * h])
            pt = torch.tensor(pts, dtype=torch.float32, device=device)
            hm = mvt_utils.generate_hm_from_pt(
                pt, (h, w), sigma=sigma, thres_sigma_times=thres_sigma_times,
            )                                                # (num_pt, h, w)
            hm_avg = hm.mean(dim=0).reshape(h * w, 1)         # (h*w, 1)
            # Re-normalize after averaging so the final heatmap still sums to 1.
            hm_avg = hm_avg / (hm_avg.sum() + 1e-6)
            out[b_idx] = hm_avg
        return out

    def _extract_image_tokens(self, hidden_states, attention_mask):
        """Pull the (bs, 256, vlm_dim) image-token grid out of a PaliGemma
        last-hidden-state. Per-sample loop because attention_mask varies
        across rows (LM suffix tokens make sequence lengths differ after
        padding); image tokens always sit at the START of the non-padded
        sequence.
        """
        bs = hidden_states.shape[0]
        image_tokens = []
        for i in range(bs):
            nz = torch.nonzero(attention_mask[i] != 0, as_tuple=True)[0]
            non_zero_output = hidden_states[i][nz]
            assert non_zero_output.shape[0] > 256 * self.num_img
            image_tokens.append(non_zero_output[: 256 * self.num_img])
        return torch.stack(image_tokens)  # (bs, 256, vlm_dim)

    def _action_features_with_memory(self, *, image_tokens, x_feat,
                                      idx_action, input_ids, attention_mask,
                                      anchor_pixel_values, hist_pixel_values,
                                      anchor_mask, hist_mask, hist_action):
        """Build per-row action-head input for the rows in ``idx_action``,
        applying spatial-anchor + temporal-episodic memory cross-attention
        when memory is enabled. Mirrors finetune mvt_single._apply_memory
        with V=1 (single view) and stage=1 semantics (both spatial s1 +
        temporal s1, no s2). Returns (n_a, 2048, 16, 16) fp32.

        When memory is disabled OR the caller didn't supply anchor/hist
        tensors, falls back to selecting raw cur-frame features —
        functionally identical to ``x_feat.index_select(0, idx_action)``.
        """
        n_a = idx_action.numel()
        if n_a == 0:
            return x_feat[0:0]                       # empty (0, 2048, 16, 16)

        memory_active = (
            self.memory_enabled
            and anchor_pixel_values is not None
            and hist_pixel_values is not None
            and hist_action is not None
        )
        if not memory_active:
            return x_feat.index_select(0, idx_action)

        # Action-row current-frame query in token layout (n_a, 256, vlm_dim), taken from the pre-rearrange
        # `image_tokens` so patch ordering matches the (h1, h2) split used downstream.
        cur_q = image_tokens.index_select(0, idx_action).to(torch.float32)

        H_p = W_p = self.num_pat_img
        N_q = H_p * W_p

        # Anchor PaliGemma forward (no_grad); reuses the action rows' input_ids since anchor and cur share the goal.
        sub_ids = input_ids.index_select(0, idx_action)
        sub_mask = attention_mask.index_select(0, idx_action)
        anchor_pix_a = anchor_pixel_values.index_select(0, idx_action)
        anchor_tok = self._paligemma_image_tokens_nograd(
            sub_ids, anchor_pix_a, sub_mask,
        ).detach().to(torch.float32)                 # (n_a, 256, vlm_dim)

        # Temporal PaliGemma forward, chunked by K so peak memory stays at the per-slot shape. Slots are
        # independent under no_grad, so batched-K and per-K outputs agree to float noise.
        K = hist_pixel_values.shape[1]
        hist_pix_a = hist_pixel_values.index_select(0, idx_action)  # (n_a, K, 3, H, W)
        hist_tok_list = []
        for k in range(K):
            pix_k = hist_pix_a[:, k].contiguous()                   # (n_a, 3, H, W)
            tok_k = self._paligemma_image_tokens_nograd(
                sub_ids, pix_k, sub_mask,
            ).detach().to(torch.float32)                            # (n_a, N_q, vlm_dim)
            hist_tok_list.append(tok_k)
        hist_tok = torch.stack(hist_tok_list, dim=1)                # (n_a, K, N_q, vlm_dim)

        h_mask_a = (
            hist_mask.index_select(0, idx_action).bool()
            if hist_mask is not None
            else torch.ones(n_a, K, dtype=torch.bool, device=cur_q.device)
        )
        a_mask_a = (
            anchor_mask.index_select(0, idx_action).bool()
            if anchor_mask is not None
            else torch.ones(n_a, dtype=torch.bool, device=cur_q.device)
        )

        # Order: temporal then spatial, matching mvt_single._apply_memory. num_views=1 (pretrain is single-image).
        if hasattr(self, "mem_temporal_s1"):
            cur_q = self.mem_temporal_s1(
                cur_q, hist_tok,
                kv_mask=h_mask_a, num_views=1,
            )
        if hasattr(self, "mem_spatial_s1"):
            a_kv_mask = a_mask_a.view(n_a, 1).expand(n_a, N_q)
            cur_q = self.mem_spatial_s1(cur_q, anchor_tok, kv_mask=a_kv_mask)

        # (n_a, 256, vlm_dim) -> (n_a, vlm_dim, 16, 16), matching x_pali's einops rule with num_img=1.
        cur_feat = rearrange(
            cur_q, 'b (h1 h2) d -> b d h1 h2', h1=H_p, h2=W_p,
        )
        return cur_feat.to(x_feat.dtype)

    def _paligemma_image_tokens_nograd(self, input_ids, pixel_values,
                                        attention_mask):
        """No-grad PaliGemma forward returning (bs, 256, vlm_dim) image tokens.

        Used for memory anchor + history forwards. We deliberately omit
        ``labels`` / ``token_type_ids``: there is no LM target on memory
        frames, and the image-token outputs at positions [0, 256) only see
        prefix attention (bidirectional in both modes), so dropping the
        prefix-LM gating doesn't affect them. Inputs are cast to bf16 to
        match self.model.dtype.

        Memory-path savings vs the main forward:
          * ``del outputs`` after extracting image tokens releases the 17
            unused hidden layers + the lm_head logits before returning.
          * ``use_cache=False`` — explicit (gradient_checkpointing already
            forces this; setting it suppresses the warning).

        Why no ``logits_to_keep=1``: it triggers a deterministic SIGFPE
        inside cuBLAS on H20 (sm_90) + torch 2.5.1+cu121 + bf16/fp16,
        independent of batch size. The (bs, 1, vlm_dim) → (bs, 1, 257152)
        lm_head GEMM picks a buggy kernel; the full (bs, 286, 257152)
        shape selects a stable one. We pay ~280 MB of temporary logits
        per call, but ``del outputs`` releases it immediately.
        """
        pali_pixel_values = pixel_values.to(self.model.dtype)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pali_pixel_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            image_tokens = self._extract_image_tokens(
                outputs.hidden_states[-1], attention_mask,
            )
            del outputs
        return image_tokens

    def forward(self, input_ids, pixel_values, attention_mask,
                bboxes, raw_text=None, labels=None, hm_weights=None,
                is_action=None, token_type_ids=None,
                anchor_pixel_values=None, hist_pixel_values=None,
                anchor_mask=None, hist_mask=None, hist_action=None):
        """
        pixel_values: (bs, 3, 224, 224) in [-1, 1] (PaliGemma processor range).
        bboxes:       list[list[(cx, cy, w, h)]], normalized to [0, 1].
        raw_text:     list[str] — raw prompt text (kept for diagnostics).
        labels:       (bs, seq) prefix-LM labels (-100 elsewhere) or None.
        hm_weights:   (bs,) per-row heatmap weight, or None => all-ones
                      (what the grounding-only RoboPoint collator emits).
        is_action:    (bs,) bool/0-1 tensor — True routes to up_action
                      (next-keyframe TCP prediction), False routes to
                      up_grounding (image-grounding rows). None defaults
                      to all-grounding, which is what the shipped
                      RoboPoint pipeline produces.
        token_type_ids: (bs, seq) int tensor (0=prefix, 1=suffix) from the
                      PaliGemma processor. MUST be passed together with
                      `labels` on LM-supervised rows — transformers gates
                      prefix-LM attention on `is_training = token_type_ids
                      is not None and labels is not None` (see
                      PaliGemmaForConditionalGeneration._update_causal_mask).
                      Omitting either silently degenerates to FULLY
                      bidirectional attention across the entire sequence,
                      which leaks the target suffix into the logit at the
                      `\\n` boundary and drives loss_lm to ~0 without the
                      model actually learning the box→text mapping.
        anchor_pixel_values: (bs, 3, H, W) PaliGemma-processed RGB of the
                      first keyframe of each VLA row's episode. Non-VLA
                      rows carry placeholder (cur-frame) values; the
                      ``anchor_mask`` zeros the spatial-memory contribution
                      for non-action rows. None = memory disabled.
        hist_pixel_values: (bs, K, 3, H, W) the K most-recent prior keyframes
                      per VLA row (slot 0 = most recent). Placeholder for
                      non-VLA rows.
        anchor_mask:  (bs,) bool — True where the spatial anchor is valid
                      (cur kf is NOT episode-start AND row is a VLA row).
        hist_mask:    (bs, K) bool — per-slot validity (False where step_idx
                      - k - 1 < 0 or row is non-VLA).
        hist_action:  (bs, K, 9) float — historical action placeholder. Now
                      VESTIGIAL: historical-action PE was removed from
                      MemoryBlock, so the temporal block no longer projects
                      it. It is retained only as part of the ``memory_active``
                      gate (its presence signals the caller supplied the
                      action-memory tensors); the tensor contents are unused.
        """
        bs = input_ids.shape[0]
        h = w = self.img_size
        assert pixel_values.shape[-1] == h and pixel_values.shape[-2] == w

        # PaliGemma runs in bf16.
        pali_pixel_values = pixel_values.to(self.model.dtype)

        # --- PaliGemma --- token_type_ids + labels select is_training=True: prefix positions bidirectional,
        # suffix causal. Its own outputs.loss is ignored; loss_lm is re-derived below to stay consistent with hm_weights.
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pali_pixel_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            output_hidden_states=True,
        )
        x = outputs.hidden_states[-1]  # (bs, seq, vlm_dim)

        # Extract the 256 image tokens (same strategy as mvt_single).
        image_tokens = self._extract_image_tokens(x, attention_mask)
        x_pali = rearrange(
            image_tokens,
            'b (c h1 h2) d -> (b c) d h1 h2',
            c=self.num_img, h1=self.num_pat_img, h2=self.num_pat_img,
        )  # (bs, vlm_dim, 16, 16)

        # PaliGemma image tokens are fed directly to the heads.
        x_feat = x_pali.to(torch.float32)  # (bs, 2048, 16, 16)

        # Per-row dispatch to the two task-specific upsample heads.
        if is_action is None:
            is_action_bool = torch.zeros(bs, dtype=torch.bool, device=x_feat.device)
        else:
            is_action_bool = is_action.to(device=x_feat.device, dtype=torch.bool)
        idx_ground = (~is_action_bool).nonzero(as_tuple=False).flatten()
        idx_action = is_action_bool.nonzero(as_tuple=False).flatten()

        trans = x_feat.new_zeros(bs, 1, h, w)
        if idx_ground.numel() > 0:
            out_g = self.up_grounding(x_feat.index_select(0, idx_ground))
            trans = trans.index_copy(0, idx_ground, out_g)
        if idx_action.numel() > 0:
            cur_feat_a = self._action_features_with_memory(
                image_tokens=image_tokens,
                x_feat=x_feat,
                idx_action=idx_action,
                input_ids=input_ids,
                attention_mask=attention_mask,
                anchor_pixel_values=anchor_pixel_values,
                hist_pixel_values=hist_pixel_values,
                anchor_mask=anchor_mask,
                hist_mask=hist_mask,
                hist_action=hist_action,
            )                                      # (n_a, 2048, 16, 16) fp32
            out_a = self.up_action(cur_feat_a)
            trans = trans.index_copy(0, idx_action, out_a)

        # q_trans is kept as flattened raw logits for downstream argmax metrics / viz.
        q_trans = trans.view(bs, self.num_img, h * w).transpose(1, 2)  # (bs, h*w, 1)

        # ---- Heatmap loss ----
        #   use_modified_focal_loss=False (default, original BridgeVLA): softmax-normalized Gaussian GT + soft-label CE.
        #   use_modified_focal_loss=True: peak-1 CenterNet Gaussian GT + per-pixel sigmoid focal loss.
        w_hm = (
            hm_weights.to(trans.device).to(trans.dtype)
            if hm_weights is not None
            else trans.new_ones(bs)
        )
        is_action_f = is_action_bool.to(trans.dtype)
        gnd_w = (1.0 - is_action_f) * w_hm
        act_w = is_action_f * w_hm
        hm_denom = w_hm.sum().clamp(min=1.0)
        if self.use_modified_focal_loss:
            gt_hm = self._build_gt_heatmap(
                bboxes, h, w, trans.device,
                is_action_batch=is_action_bool.tolist(),
            )                                                              # (bs,1,H,W)
            per_sample_hm = _modified_focal_loss_per_sample(trans, gt_hm)  # (bs,)
        else:
            # Soft-label CE: q_trans (bs, h*w, 1) logits against the softmax-normalized Gaussian action_t.
            action_trans_2d = self._build_softmax_gt_heatmap(
                bboxes, h, w, trans.device,
            )                                                            # (bs, h*w, 1)
            per_sample_hm = self._cross_entropy_loss(
                q_trans, action_trans_2d
            ).mean(dim=1)                                                  # (bs,)
        loss_hm = (per_sample_hm * w_hm).sum() / hm_denom
        hm_n_valid = int((w_hm > 0).sum().item())

        # Per-head contribution to loss_hm (up_grounding vs up_action).
        loss_hm_ground = (per_sample_hm * gnd_w).sum() / hm_denom
        loss_hm_action = (per_sample_hm * act_w).sum() / hm_denom
        hm_n_ground = int((gnd_w > 0).sum().item())
        hm_n_action = int((act_w > 0).sum().item())

        # Manual LM CE so the denominator is the real count of supervised positions (nn.CrossEntropyLoss
        # would NaN on all-ignore batches). Gated by use_lm_aux_loss, off by default.
        if labels is not None and self.use_lm_aux_loss:
            logits = outputs.logits  # (bs, seq, vocab)
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            valid = (shift_labels != -100).sum()
            if valid.item() > 0:
                ce_sum = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100, reduction="sum",
                )
                loss_lm = ce_sum / valid.to(ce_sum.dtype)
            else:
                loss_lm = torch.zeros((), device=loss_hm.device, dtype=loss_hm.dtype)
            lm_n_valid = int(valid.item())
        else:
            loss_lm = None
            lm_n_valid = 0

        return {
            "loss": loss_hm,            # legacy alias
            "loss_hm": loss_hm,
            "loss_hm_ground": loss_hm_ground,   # contribution; sums to loss_hm
            "loss_hm_action": loss_hm_action,   # contribution; sums to loss_hm
            "loss_lm": loss_lm,         # None when batch has no LM rows
            "lm_n_valid": lm_n_valid,
            "hm_n_valid": hm_n_valid,
            "hm_n_ground": hm_n_ground,         # num rows routed to up_grounding
            "hm_n_action": hm_n_action,         # num rows routed to up_action
            "q_trans": q_trans,
        }


def setup_distributed(backend="nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif os.getenv("DEBUG", "false").lower() == "true":
        print("Cannot find RANK and WORLD_SIZE — entering single-GPU debug mode")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "9001")
        os.environ.setdefault("LOCAL_RANK", "0")
    else:
        raise RuntimeError(
            "Distributed env vars not found. "
            "Launch with torchrun / srun, or set DEBUG=true for single-GPU mode."
        )

    dist.init_process_group(
        backend=backend,
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
    )


# ---- Training utilities ----
def freeze_params(model, freeze_paligemma):
    """Apply the pretrain freeze policy.

    PaliGemma, priority order (first match wins):
      1. `vision_tower.*` — always frozen (SigLIP fixed pretrained vision).
      2. `lm_head` / `embed_tokens` — **always frozen**. Tied weights in HF
         PaliGemma (`tie_word_embeddings=True`) → freezing either name
         freezes the shared Parameter. LM aux loss still works (when
         enabled): gradient flows THROUGH lm_head into the Gemma body;
         only lm_head itself stays pristine.
      3. Everything else under `model.*` follows `freeze_paligemma`.
    Heads (`up_grounding` / `up_action`) are always trainable.
    """
    always_frozen_substrings = ("vision_tower", "lm_head", "embed_tokens")
    for name, param in model.named_parameters():
        if name.startswith("model."):
            if any(s in name for s in always_frozen_substrings):
                param.requires_grad = False
            else:
                param.requires_grad = not freeze_paligemma
        else:
            # Heads (up_grounding / up_action): always trainable.
            param.requires_grad = True

    # Catch silent regressions if HF renames or substring lists change.
    try:
        vision_tower_mod = model.model.vision_tower
        lm_head_mod = model.model.language_model.lm_head
        emb_mod = model.model.language_model.model.embed_tokens
    except AttributeError:
        return

    # All three must stay frozen. lm_head / embed_tokens share a tied Parameter, but assert separately in case tying is dropped upstream.
    for mod_name, mod in (("vision_tower", vision_tower_mod),
                          ("lm_head", lm_head_mod),
                          ("embed_tokens", emb_mod)):
        for pname, pparam in mod.named_parameters():
            if pparam.requires_grad:
                raise RuntimeError(
                    f"freeze_params: {mod_name}.{pname} is trainable — "
                    f"expected frozen."
                )


def build_optimizer(model, lr, weight_decay=0.0, betas=(0.9, 0.95),
                    lr_scales=None):
    """AdamW with optional per-name-prefix LR scaling.

    `lr_scales`: dict[prefix -> multiplier]. First matching prefix wins per
    parameter; unmatched trainable params land in the default group with
    scale 1.0. Each group stores `lr_scale` so the warmup updater can
    preserve it.
    """
    inner = model.module if isinstance(model, DDP) else model
    lr_scales = dict(lr_scales or {})
    default_params: list = []
    scaled_params: dict = {prefix: [] for prefix in lr_scales}
    for name, p in inner.named_parameters():
        if not p.requires_grad:
            continue
        hit = next((pref for pref in lr_scales if name.startswith(pref)), None)
        if hit is None:
            default_params.append(p)
        else:
            scaled_params[hit].append(p)
    groups = [{"params": default_params, "lr": lr, "lr_scale": 1.0,
               "name": "default"}]
    for prefix, scale in lr_scales.items():
        if scaled_params[prefix]:
            groups.append({
                "params": scaled_params[prefix],
                "lr": lr * float(scale),
                "lr_scale": float(scale),
                "name": prefix,
            })
    return torch.optim.AdamW(groups, weight_decay=weight_decay, betas=betas)


def save_checkpoint(model, path, epoch, extra=None):
    """Save the full model state-dict.

    The pretrain corpus is grounding-only, so only `up_grounding` ever
    receives gradient — `up_action` would be saved at its random init. We
    mirror the trained head into the `up_action.*` keys so finetune's
    loader (which prefers `up_action` when warm-starting its single `up0`,
    see mvt_single._load) picks up trained weights instead of noise. This
    reproduces original BridgeVLA semantics, where the one pretrain head
    is loaded into finetune's `up0`.
    """
    model_to_save = model.module if isinstance(model, DDP) else model
    sd = model_to_save.state_dict()
    ground_keys = [k for k in sd if k.startswith("up_grounding.")]
    if ground_keys:
        sd = dict(sd)
        for k in ground_keys:
            sd["up_action." + k[len("up_grounding."):]] = sd[k]
    payload = {"epoch": epoch, "model_state": sd}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def run_epoch(ddp_model, loader, sampler, state, *,
              epoch, base_lr, lr_warmup_steps, weight_decay,
              freeze_threshold_step, iters_per_epoch,
              device, rank, logging_steps,
              lm_lambda_max: float = 0.0,
              lm_warmup_steps: int = 0,
              lr_scales: dict = None,
              grad_accum_steps: int = 1):
    """Run one epoch with mid-epoch Stage1→Stage2 transition support and
    gradient accumulation.

    `state`: mutable dict with
        stage            (1|2),
        cumulative_step  (= MACRO / optimizer-step count, never resets),
        optimizer        (replaced on transition).

    Gradient accumulation semantics:
      * Effective batch    = bs × world_size × grad_accum_steps.
      * MICRO step         = one DataLoader iteration / forward+backward.
      * MACRO step         = one optimizer.step (every grad_accum_steps
                             micros).
      * `state["cumulative_step"]` counts MACRO steps. All schedules below
        are interpreted in MACRO-step units.
      * Per micro: loss is scaled by 1/grad_accum_steps before backward,
        so the accumulated grad equals the mean grad over the macro window
        (matches the gradient magnitude of a single step at the effective
        batch size).
      * DDP all-reduce: skipped on non-final micros via ddp_model.no_sync()
        so a macro produces one all-reduce instead of grad_accum_steps.
      * Stage 1→2 transition is gated to macro boundaries (clean state:
        zero gradients, no in-flight backward).
      * Trailing partial macro at end of epoch is dropped (no
        optimizer.step on a half-filled accumulator) so gradient scale
        stays consistent across all macros.

    Warmup schedules (BOTH macro-cumulative, neither resets at Stage2):
      * `lr_warmup_steps`   — linear LR warmup 0 → base_lr (macro steps).
      * `lm_warmup_steps`   — linear LM-loss-weight ramp 0 → lm_lambda_max
                              (macro steps).

    Logging:
      * Per-micro losses are accumulated; on macro boundaries an averaged
        total is written to tqdm.set_postfix_str + swanlab.log, keyed by
        macro `state["cumulative_step"]`. logging_steps is in macros.
    """
    assert grad_accum_steps >= 1, grad_accum_steps
    ddp_model.train()
    inner = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model

    sampler.set_epoch(epoch)

    # Drop trailing micros that don't complete a macro window; last active micro = macro_iters * accum - 1.
    macro_iters_per_epoch = iters_per_epoch // grad_accum_steps
    last_active_micro = macro_iters_per_epoch * grad_accum_steps - 1

    log_prefix = f"[S{state['stage']}] "
    iterator = tqdm(loader, disable=(rank != 0), desc=f"{log_prefix}epoch {epoch}")

    # Per-macro accumulators: floats summed across micros (divided by grad_accum_steps post-step), ints counted.
    def _new_accum():
        return {
            "loss": 0.0, "loss_hm": 0.0, "loss_hm_g": 0.0, "loss_hm_a": 0.0,
            "loss_lm": 0.0,
            "hm_n": 0, "hm_n_g": 0, "hm_n_a": 0, "lm_n": 0,
            "lm_weight": 0.0,
        }
    accum = _new_accum()

    for it, batch in enumerate(iterator):
        # Drop trailing partial macro window (keep gradient scale clean).
        if it > last_active_micro:
            break

        micro_idx = it % grad_accum_steps
        is_first_micro = (micro_idx == 0)
        is_last_micro = (micro_idx == grad_accum_steps - 1)

        # ---- Mid-epoch stage-2 transition ----
        # Rebuild DDP (its reducer is pinned to the requires_grad set at ctor, so params flipped False->True
        # later never all-reduce) and the optimizer (so newly-unfrozen params join the groups). Gated to
        # is_first_micro: flipping requires_grad mid-accumulation would leave stale partial gradients behind.
        if (is_first_micro
                and state["stage"] == 1
                and state["cumulative_step"] >= freeze_threshold_step):
            freeze_params(inner, freeze_paligemma=False)
            if "local_rank" in state:
                ddp_model = DDP(
                    inner, device_ids=[state["local_rank"]],
                    find_unused_parameters=True,
                )
                state["ddp_model"] = ddp_model
            state["optimizer"] = build_optimizer(
                ddp_model, lr=base_lr, weight_decay=weight_decay,
                lr_scales=lr_scales,
            )
            state["stage"] = 2
            log_prefix = f"[S{state['stage']}] "
            iterator.set_description(f"{log_prefix}epoch {epoch}")
            if rank == 0:
                n_train = sum(
                    p.numel() for p in ddp_model.parameters() if p.requires_grad
                ) / 1e9
                print(
                    f"[Stage 2] Unfroze PaliGemma at epoch={epoch} iter={it} "
                    f"(cumulative_step={state['cumulative_step']}). "
                    f"Trainable params: {n_train:.3f}B",
                    flush=True,
                )

        optimizer = state["optimizer"]

        # zero_grad ONCE per macro window (start of accumulation cycle).
        if is_first_micro:
            optimizer.zero_grad(set_to_none=True)
            accum = _new_accum()

        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        hm_weights = batch.get("hm_weights")
        if hm_weights is not None:
            hm_weights = hm_weights.to(device)
        is_action = batch.get("is_action")
        if is_action is not None:
            is_action = is_action.to(device)
        # token_type_ids is only emitted when suffix= is passed; thread it through so PaliGemma uses its prefix-LM mask.
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Memory tensors: absent for the grounding-only RoboPoint collator, hosted here for collators that emit them.
        def _maybe_to(t):
            return t.to(device) if t is not None else None
        anchor_pixel_values = _maybe_to(batch.get("anchor_pixel_values"))
        hist_pixel_values = _maybe_to(batch.get("hist_pixel_values"))
        anchor_mask = _maybe_to(batch.get("anchor_mask"))
        hist_mask = _maybe_to(batch.get("hist_mask"))
        hist_action = _maybe_to(batch.get("hist_action"))

        # Skip the DDP all-reduce on every micro except the last of the macro window (a no-op when accum == 1).
        sync_ctx = (
            ddp_model.no_sync()
            if (grad_accum_steps > 1 and not is_last_micro)
            else nullcontext()
        )
        with sync_ctx:
            out = ddp_model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                bboxes=batch["bboxes"],
                raw_text=batch["raw_text"],
                labels=labels,
                hm_weights=hm_weights,
                is_action=is_action,
                token_type_ids=token_type_ids,
                anchor_pixel_values=anchor_pixel_values,
                hist_pixel_values=hist_pixel_values,
                anchor_mask=anchor_mask,
                hist_mask=hist_mask,
                hist_action=hist_action,
            )
            loss_hm = out["loss_hm"]
            loss_hm_ground = out["loss_hm_ground"]
            loss_hm_action = out["loss_hm_action"]
            loss_lm = out["loss_lm"]          # None when batch has no LM rows
            lm_n_valid = out["lm_n_valid"]
            hm_n_valid = out["hm_n_valid"]
            hm_n_ground = out["hm_n_ground"]
            hm_n_action = out["hm_n_action"]

            # LM weight uses the current macro's cumulative_step, frozen across the window so scaling is consistent.
            if lm_lambda_max > 0.0 and loss_lm is not None:
                if lm_warmup_steps <= 0:
                    lm_weight = lm_lambda_max
                else:
                    lm_weight = lm_lambda_max * min(
                        1.0, state["cumulative_step"] / lm_warmup_steps
                    )
                total_loss = loss_hm + lm_weight * loss_lm
            else:
                lm_weight = 0.0
                total_loss = loss_hm

            # Scale by 1/accum so the accumulated gradient equals the mean gradient at the effective batch.
            (total_loss / float(grad_accum_steps)).backward()

        # Accumulate stats for macro-level logging; sums become means after optimizer.step below.
        accum["loss"] += float(total_loss.item())
        accum["loss_hm"] += float(loss_hm.item())
        accum["loss_hm_g"] += float(loss_hm_ground.item())
        accum["loss_hm_a"] += float(loss_hm_action.item())
        if loss_lm is not None:
            accum["loss_lm"] += float((lm_weight * loss_lm).item())
        accum["hm_n"] += int(hm_n_valid)
        accum["hm_n_g"] += int(hm_n_ground)
        accum["hm_n_a"] += int(hm_n_action)
        accum["lm_n"] += int(lm_n_valid)
        accum["lm_weight"] = float(lm_weight)  # constant within a macro

        if not is_last_micro:
            continue

        # ---- Macro-step boundary: lr update + optimizer.step. Warmup is macro-cumulative across both stages. ----
        warm = (
            1.0 if lr_warmup_steps <= 0
            else min(1.0, (state["cumulative_step"] + 1) / lr_warmup_steps)
        )
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr * pg.get("lr_scale", 1.0) * warm
        optimizer.step()

        state["cumulative_step"] += 1
        dist.barrier()

        # ---- Log on macro-step boundaries (logging_steps is in macro units) ----
        if rank == 0 and (state["cumulative_step"] % logging_steps == 0):
            cur_lr = optimizer.param_groups[0]["lr"]
            inv = 1.0 / float(grad_accum_steps)
            avg_loss = accum["loss"] * inv
            avg_hm = accum["loss_hm"] * inv
            avg_hm_g = accum["loss_hm_g"] * inv
            avg_hm_a = accum["loss_hm_a"] * inv
            avg_lm = accum["loss_lm"] * inv
            iterator.set_postfix_str(
                f"epoch={epoch} step={state['cumulative_step']} "
                f"loss={avg_loss:.4f} "
                f"hm={avg_hm:.4f}(n={accum['hm_n']}) "
                f"[g={avg_hm_g:.4f}(n={accum['hm_n_g']}) "
                f"a={avg_hm_a:.4f}(n={accum['hm_n_a']})] "
                f"lm={avg_lm:.4f}(n={accum['lm_n']},"
                f"w={accum['lm_weight']:.3f}) "
                f"lr={cur_lr:.3e}"
            )
            if USE_SWANLAB:
                import swanlab
                swanlab.log(
                    {
                        "loss": avg_loss,
                        "loss_hm": avg_hm,
                        "loss_hm_ground": avg_hm_g,
                        "loss_hm_action": avg_hm_a,
                        "loss_lm": avg_lm,
                        "hm_n_valid": accum["hm_n"],
                        "hm_n_ground": accum["hm_n_g"],
                        "hm_n_action": accum["hm_n_a"],
                        "lm_n_valid": accum["lm_n"],
                        "lm_weight": accum["lm_weight"],
                        "lr": cur_lr,
                        "stage": state["stage"],
                    },
                    step=state["cumulative_step"],
                )


# ---- Validation ----
def _pointhit_and_center_l1(q_trans: torch.Tensor,
                            bboxes_batch: Sequence[Sequence[Tuple[float, float, float, float]]],
                            hm_size: int):
    """Per-sample PointHit@bbox (0/1) and center L1 distance (pixels).

    q_trans   : (bs, h*w, 1) raw heatmap logits from the model.
    bboxes_batch: list of per-sample list[(cx, cy, w, h)] normalized to [0, 1]
                  of the cropped frame. `hm_size` is the side of the heatmap
                  (= img_size; 224 here).
    Returns two (bs,) float tensors on q_trans.device.
    """
    bs = q_trans.shape[0]
    # argmax flat index -> (py, px); CE reads q_trans as (bs, h*w, 1) row-major, i = y*W + x.
    amax_idx = q_trans.squeeze(-1).argmax(dim=1)  # (bs,) long
    py = (amax_idx // hm_size).float()
    px = (amax_idx % hm_size).float()
    hits = torch.zeros(bs, device=q_trans.device)
    dists = torch.zeros(bs, device=q_trans.device)
    S = float(hm_size)
    for i in range(bs):
        bbs = bboxes_batch[i]
        if not bbs:
            dists[i] = float("nan")
            continue
        # Per-bbox hit test (any containing box counts) + min-L1 to the nearest bbox center.
        hit_i = 0.0
        best_dist = None
        xi, yi = float(px[i].item()), float(py[i].item())
        for cx, cy, bw, bh in bbs:
            x0 = (cx - bw * 0.5) * S
            y0 = (cy - bh * 0.5) * S
            x1 = (cx + bw * 0.5) * S
            y1 = (cy + bh * 0.5) * S
            if x0 <= xi <= x1 and y0 <= yi <= y1:
                hit_i = 1.0
            d = abs(xi - cx * S) + abs(yi - cy * S)
            if best_dist is None or d < best_dist:
                best_dist = d
        hits[i] = hit_i
        dists[i] = best_dist if best_dist is not None else float("nan")
    return hits, dists


def _viz_gt_and_pred_hm(model, q_trans_flat, bboxes_i, is_action_natural,
                        h, w, device):
    """Build (gt_hm_2d, pred_hm_2d) matching the model's training objective.

    ``use_modified_focal_loss=True``:  GT = peak-1 CenterNet Gaussian (radius
    depends on ``is_action``). Pred = per-pixel sigmoid of raw logits.

    ``use_modified_focal_loss=False`` (default = original BridgeVLA):
        GT = softmax-normalized Gaussian over (h*w) (sums to 1).
        Pred = softmax over (h*w) of raw logits → reshape to (h, w).
        At convergence the overlay should peak at one pixel — using sigmoid
        here would spread across the image and look wrong even when training
        is healthy.

    q_trans_flat : (1, h*w, 1) raw logits.
    bboxes_i     : list[(cx, cy, w, h)] for this sample (normalized).
    Returns two (h, w) torch.Tensor on ``device``.
    """
    use_focal = bool(getattr(model, "use_modified_focal_loss", False))
    if use_focal:
        gt = model._build_gt_heatmap(
            [bboxes_i], h, w, device,
            is_action_batch=[bool(is_action_natural)],
        )[0, 0]                                      # (h, w)
        pred = q_trans_flat.view(h, w).sigmoid()
    else:
        gt = model._build_softmax_gt_heatmap(
            [bboxes_i], h, w, device,
        )[0, :, 0].view(h, w)                        # (h, w)
        pred = torch.softmax(q_trans_flat.view(-1), dim=0).view(h, w)
    return gt, pred


def _render_val_figure(image, gt_heatmap, pred_logits, pred_heatmap,
                       bboxes_cxcywh, prompt, save_path):
    """4-panel val-viz: image+GT bbox / image+GT hm / raw pred logits /
    image+pred hm. The VLM prompt (`<image>` + text) goes at the bottom.
    """
    img224 = image.resize((224, 224))

    # Panel 1: image with GT bbox in red.
    img_with_bbox = img224.copy()
    draw = ImageDraw.Draw(img_with_bbox)
    for cx, cy, bw, bh in bboxes_cxcywh:
        x0 = max(0, int((cx - bw / 2) * 224))
        y0 = max(0, int((cy - bh / 2) * 224))
        x1 = min(223, int((cx + bw / 2) * 224))
        y1 = min(223, int((cy + bh / 2) * 224))
        draw.rectangle([x0, y0, x1, y1], outline="red", width=2)

    gt_np = gt_heatmap.detach().cpu().float().numpy().squeeze()
    logits_np = pred_logits.detach().cpu().float().numpy().squeeze()
    pred_np = pred_heatmap.detach().cpu().float().numpy().squeeze()

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(img_with_bbox)
    axes[0].set_title("Image + GT bbox"); axes[0].axis("off")

    axes[1].imshow(img224)
    axes[1].imshow(gt_np, cmap="viridis", alpha=0.55)
    axes[1].set_title("Image + GT heatmap"); axes[1].axis("off")

    im = axes[2].imshow(logits_np, cmap="viridis")
    axes[2].set_title("Pred logits (raw)"); axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(img224)
    axes[3].imshow(pred_np, cmap="viridis", alpha=0.6)
    axes[3].set_title("Image + pred heatmap"); axes[3].axis("off")

    # `prompt` may already be a multi-line composed caption; split and reflow each line.
    lines = [textwrap.fill(ln, width=160) for ln in str(prompt).split("\n")]
    fig.text(0.5, 0.02, "\n".join(lines), ha="center", va="bottom", fontsize=10)
    fig.subplots_adjust(bottom=min(0.30, 0.10 + 0.04 * len(lines)))
    plt.savefig(save_path, dpi=90)
    plt.close(fig)


@torch.no_grad()
def _save_val_viz_samples(inner_model, val_ds, val_collate, *,
                          device, out_dir, n_viz, img_size, epoch):
    """Render `n_viz` qualitative val samples (rank-0 only).

    Indices are evenly spaced over the holdout so the same `n_viz` samples
    are drawn every epoch -> easy A/B comparison across epochs.

    Output layout: <out_dir>/sample_00.png ...
    """
    if n_viz <= 0 or val_ds is None or len(val_ds) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    n_viz = min(n_viz, len(val_ds))
    step = max(1, len(val_ds) // n_viz)
    picks = list(range(0, len(val_ds), step))[:n_viz]

    for nn, idx in enumerate(picks):
        sample = val_ds[idx]
        batch = val_collate([sample])
        tokens = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        out = inner_model(
            input_ids=tokens["input_ids"],
            pixel_values=tokens["pixel_values"],
            attention_mask=tokens["attention_mask"],
            bboxes=batch["bboxes"],
            raw_text=batch["raw_text"],
            token_type_ids=tokens.get("token_type_ids"),
        )
        q_trans_flat = out["q_trans"].detach()           # (1, H*W, 1)
        q_trans = q_trans_flat.view(img_size, img_size)  # raw logits 2D

        # GT + pred heatmaps follow the loss objective, so the overlay tracks what training optimizes.
        bboxes_i = batch["bboxes"][0]
        gt_hm, pred_hm = _viz_gt_and_pred_hm(
            inner_model, q_trans_flat, bboxes_i,
            is_action_natural=False,
            h=img_size, w=img_size, device=device,
        )

        _render_val_figure(
            image=sample["image"],
            gt_heatmap=gt_hm,
            pred_logits=q_trans,
            pred_heatmap=pred_hm,
            bboxes_cxcywh=bboxes_i,
            prompt=f"prompt:  {_format_vlm_prompt(sample['text'])}",
            save_path=os.path.join(out_dir, f"sample_{nn:02d}.png"),
        )
    print(f"[val-viz] epoch {epoch}: saved {len(picks)} samples -> {out_dir}",
          flush=True)


@torch.no_grad()
def run_validation(ddp_model, val_loader, *, device, rank, world_size,
                   img_size: int, cumulative_step: int, epoch: int,
                   n_viz: int = 0, viz_out_dir: Optional[str] = None):
    """Evaluate PointHit@bbox + center_l1 + loss_hm + loss_lm on the val loader.

    Metrics are averaged over samples (hits / l1) or over supervised positions
    (loss_lm) with DDP all_reduce. Rank-0 returns a populated dict; other
    ranks return the same dict so callers can log uniformly.

    If `n_viz > 0` and `viz_out_dir` is set, rank 0 ALSO renders that many
    qualitative 4-panel figures after the DDP metric pass.
    """
    if val_loader is None:
        return {}
    ddp_model.eval()
    tot_hit = torch.zeros((), device=device)
    tot_l1 = torch.zeros((), device=device)
    tot_l1_n = torch.zeros((), device=device)
    tot_hm = torch.zeros((), device=device)
    tot_hm_n = torch.zeros((), device=device)
    tot_lm = torch.zeros((), device=device)
    tot_lm_n = torch.zeros((), device=device)
    tot_n = torch.zeros((), device=device)

    iterator = tqdm(val_loader, disable=(rank != 0), desc=f"[val] epoch {epoch}")
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(device)
        hm_weights = batch.get("hm_weights")
        if hm_weights is not None:
            hm_weights = hm_weights.to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        is_action = batch.get("is_action")
        if is_action is not None:
            is_action = is_action.to(device)

        def _maybe_to(t):
            return t.to(device) if t is not None else None
        anchor_pixel_values = _maybe_to(batch.get("anchor_pixel_values"))
        hist_pixel_values = _maybe_to(batch.get("hist_pixel_values"))
        anchor_mask = _maybe_to(batch.get("anchor_mask"))
        hist_mask = _maybe_to(batch.get("hist_mask"))
        hist_action = _maybe_to(batch.get("hist_action"))

        out = ddp_model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            bboxes=batch["bboxes"],
            raw_text=batch["raw_text"],
            labels=labels,
            hm_weights=hm_weights,
            is_action=is_action,
            token_type_ids=token_type_ids,
            anchor_pixel_values=anchor_pixel_values,
            hist_pixel_values=hist_pixel_values,
            anchor_mask=anchor_mask,
            hist_mask=hist_mask,
            hist_action=hist_action,
        )
        bs = input_ids.shape[0]
        hits, dists = _pointhit_and_center_l1(
            out["q_trans"], batch["bboxes"], img_size,
        )
        valid = torch.isfinite(dists)
        tot_hit += hits[valid].sum()
        tot_l1 += dists[valid].sum()
        tot_l1_n += valid.sum().float()
        tot_n += float(bs)

        # Heatmap CE — weighted sum over rec rows (hm_weight=1).
        loss_hm = out["loss_hm"]
        n_hm = float(out["hm_n_valid"])
        if n_hm > 0:
            tot_hm += loss_hm.detach().float() * n_hm
            tot_hm_n += n_hm

        # LM CE — weighted sum over supervised positions.
        loss_lm = out["loss_lm"]
        n_lm = float(out["lm_n_valid"])
        if loss_lm is not None and n_lm > 0:
            tot_lm += loss_lm.detach().float() * n_lm
            tot_lm_n += n_lm

    if world_size > 1:
        for t in (tot_hit, tot_l1, tot_l1_n, tot_hm, tot_hm_n,
                  tot_lm, tot_lm_n, tot_n):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    point_hit = (tot_hit / tot_l1_n.clamp(min=1.0)).item()
    center_l1 = (tot_l1 / tot_l1_n.clamp(min=1.0)).item()
    loss_hm = (tot_hm / tot_hm_n.clamp(min=1.0)).item()
    loss_lm = (tot_lm / tot_lm_n.clamp(min=1.0)).item() if tot_lm_n.item() > 0 else 0.0
    n_samples = int(tot_n.item())

    metrics = {
        "val/point_hit": point_hit,
        "val/center_l1": center_l1,
        "val/loss_hm": loss_hm,
        "val/loss_lm": loss_lm,
        "val/n_samples": n_samples,
    }
    if rank == 0:
        print(
            f"[val] epoch={epoch} n={n_samples} "
            f"point_hit={point_hit:.4f} center_l1={center_l1:.2f}px "
            f"loss_hm={loss_hm:.4f} loss_lm={loss_lm:.4f}",
            flush=True,
        )
        if USE_SWANLAB:
            import swanlab
            swanlab.log(metrics, step=cumulative_step)
        # Qualitative panel (rank 0) against the un-DDP inner module: single-sample DDP forwards add needless sync.
        if n_viz > 0 and viz_out_dir is not None:
            inner = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
            _save_val_viz_samples(
                inner, val_loader.dataset, val_loader.collate_fn,
                device=device, out_dir=viz_out_dir, n_viz=n_viz,
                img_size=img_size, epoch=epoch,
            )
    if world_size > 1:
        dist.barrier()
    ddp_model.train()
    return metrics


# ---- Checkpoint loader (used by finetune; also handy for validation here) ----
def split_pretrain_state(state_dict):
    """Partition a pretrain state_dict by submodule prefix. Returns a dict of
    {submodule_name: inner_state_dict} plus 'model' for PaliGemma.

    Post-head-split (May 2026): `up_grounding` / `up_action` replace the
    legacy single `up0` bucket. Old checkpoints with `up0.*` keys still
    populate the `up0` bucket so downstream finetune loaders that handle
    the legacy key keep working.

    Memory (May 2026 +): `mem_spatial_s1` / `mem_temporal_s1` carry the
    pretrain stage-1 memory weights. Pretrain produces no `mem_spatial_s2`;
    finetune cross-loads it from `mem_spatial_s1` at load time.
    """
    buckets = {
        "model": {},
        "up_grounding": {}, "up_action": {},
        "up0": {},  # legacy — pre-split pretrain ckpts only
        "mem_spatial_s1": {}, "mem_temporal_s1": {},
    }
    for k, v in state_dict.items():
        for prefix in buckets:
            p = prefix + "."
            if k.startswith(p):
                buckets[prefix][k[len(p):]] = v
                break
    return buckets


def load_pretrain_checkpoint(path):
    """Load a pretrain checkpoint.

    Accepts both this script's own output (a `torch.save` dict) and the
    original BridgeVLA release (a HF-Trainer directory of sharded
    safetensors); the latter's top-level PaliGemma keys are re-prefixed with
    `model.` by `mvt_utils.normalize_pretrain_state_dict`.

    Legacy single-head ckpts (pre-head-split, keys under `up0.*` — this is
    what the original BridgeVLA pretrain carries) are migrated on load: the
    old `up0.*` weights are tiled into both `up_grounding.*` and
    `up_action.*` so the new two-head model receives a warm start in each
    head. Post-split ckpts already have the new keys and pass through
    unchanged.
    """
    if os.path.isdir(path):
        from safetensors import safe_open
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            index = json.load(f)
        sd = {}
        for shard_file in sorted(set(index["weight_map"].values())):
            with safe_open(os.path.join(path, shard_file), framework="pt") as f:
                for k in f.keys():
                    sd[k.replace("module.", "")] = f.get_tensor(k)
    else:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            sd = ckpt["model_state"]
        else:
            sd = ckpt
        sd = {k.replace("module.", ""): v for k, v in sd.items()}

    sd, n_rewritten = mvt_utils.normalize_pretrain_state_dict(sd)
    if n_rewritten:
        print(f"[load_pretrain_checkpoint] original-BridgeVLA key layout "
              f"detected: re-prefixed {n_rewritten} backbone keys with "
              f"`model.`.")

    legacy_keys = [k for k in sd if k.startswith("up0.")]
    has_new_heads = any(
        k.startswith("up_grounding.") or k.startswith("up_action.") for k in sd
    )
    if legacy_keys and not has_new_heads:
        print(f"[load_pretrain_checkpoint] migrating {len(legacy_keys)} legacy "
              f"`up0.*` keys -> `up_grounding.*` + `up_action.*` (warm-start "
              f"both heads from the single pretrain head).")
        for k in legacy_keys:
            tail = k[len("up0."):]
            sd["up_grounding." + tail] = sd[k]
            sd["up_action." + tail] = sd[k].clone()
            del sd[k]
    return sd


# ---- Experiment entry point ----
def experiment(cmd_args):
    with open(cmd_args.config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Rank-offset so each rank draws distinct prompt templates.
    seed = int(cfg.get("seed", PRETRAIN_GLOBAL_SEED))
    seed_per_rank = seed + rank
    random.seed(seed_per_rank)
    torch.manual_seed(seed_per_rank)
    torch.cuda.manual_seed_all(seed_per_rank)
    try:
        import numpy as _np
        _np.random.seed(seed_per_rank)
    except ImportError:
        pass
    if rank == 0:
        print(f"[Pretrain] global seed: {seed} (per-rank {seed_per_rank})")

    # Paths / model IDs
    model_id = os.environ.get("PALIGEMMA_PATH", "google/paligemma-3b-pt-224")

    # CLI overrides for the RoboPoint corpus paths (yaml holds the defaults).
    if cmd_args.image_folder is not None:
        cfg["image_folder"] = cmd_args.image_folder
    if cmd_args.json_detection_path is not None:
        cfg["json_detection_path"] = cmd_args.json_detection_path

    # Config knobs (defaults suited to 2×H100 runs).
    bs = int(cfg["bs"])
    lr = float(cfg["lr"])
    num_epochs = int(cfg["num_train_epochs"])
    # Fractional allowed (e.g. 0.5 transitions half-way through epoch 0).
    freeze_epochs = float(cfg.get("freeze_epochs", 2))
    # Top-level LR warmup (per-stage). Prefer `lr_warmup_steps`; legacy `warmup_steps` still works.
    lr_warmup_steps = int(cfg.get("lr_warmup_steps",
                                  cfg.get("warmup_steps", 400)))
    logging_steps = int(cfg.get("logging_steps", 10))
    save_total_limit = int(cfg.get("save_total_limit", 30))
    num_workers = int(cfg.get("dataloader_num_workers", 8))
    # Gradient accumulation: micro-iterations per optimizer step; effective batch = bs x world_size x accum.
    # Every step-counted schedule below is in MACRO (optimizer-step) units so changing accum never rescales them.
    grad_accum_steps = int(cfg.get("grad_accumulation_steps", 1))
    assert grad_accum_steps >= 1, grad_accum_steps
    # weight_decay is derived as weight_decay_lr_ratio * lr so both scale together under decoupled AdamW;
    # an explicit yaml weight_decay still wins. Default ratio 0.1 => wd = 5e-6 at lr = 5e-5.
    if "weight_decay" in cfg:
        weight_decay = float(cfg["weight_decay"])
    else:
        weight_decay = float(cfg.get("weight_decay_lr_ratio", 0.1)) * lr

    # Aux LM loss weight ramps 0 -> lambda_max over loc_lm.warmup_steps CUMULATIVE steps (never resets on
    # Stage 2). Needs use_lm_aux_loss AND a source emitting LM suffixes - RoboPoint emits none, so it stays off.
    lm_cfg = cfg.get("loc_lm", {}) or {}
    lm_enabled = bool(cfg.get("use_lm_aux_loss", False)) and bool(
        lm_cfg.get("enabled", True))
    lm_lambda_max = float(lm_cfg.get("lambda_max", 0.1)) if lm_enabled else 0.0
    lm_warmup_steps = int(lm_cfg.get("warmup_steps", 1000))

    # PaliGemma Gemma-2B body + multi_modal_projector on a reduced-LR group; lm_head / embed_tokens stay at 1.0x.
    paligemma_lr_scale = float(cfg.get("paligemma_lr_scale", 1.0))

    # Output dir (rank 0 builds it, then broadcasts). PRETRAIN_RUN_STAMP lets pretrain.sh pre-compute the
    # stamp so its tee lands in the same run folder as the checkpoints.
    exp_name = cfg.get("exp_name", "pretrain")
    if rank == 0:
        stamp = (
            os.environ.get("PRETRAIN_RUN_STAMP")
            or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        output_path = os.path.join(cfg["output_dir"], exp_name, stamp)
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.yaml"), "w") as fp:
            yaml.safe_dump(cfg, fp)
        out_list = [output_path]
    else:
        out_list = [None]
    dist.broadcast_object_list(out_list, src=0)
    output_path = out_list[0]

    # Capture this rank's stdout/stderr into the run folder, alongside pretrain.sh's shell-level tee.
    _install_tee_logging(output_path, rank)

    if rank == 0:
        print(f"[Pretrain] world_size={world_size}, output_dir={output_path}")

    # ---- Model ----
    use_modified_focal_loss = bool(cfg.get("use_modified_focal_loss", False))
    use_lm_aux_loss = bool(cfg.get("use_lm_aux_loss", False))
    # Episodic memory mirroring the GemBench finetune config, so a pretrain checkpoint can populate
    # finetune's mem_spatial_s1 / mem_temporal_s1; finetune cross-loads mem_spatial_s2 from mem_spatial_s1.
    memory_cfg = dict(cfg.get("memory") or {})
    memory_enabled = bool(memory_cfg.get("enabled", False))
    if rank == 0:
        if memory_enabled:
            print(f"[Pretrain] memory: enabled, "
                  f"k_temporal={memory_cfg.get('k_temporal', 4)}, "
                  f"layers={memory_cfg.get('num_layers', 2)}, "
                  f"heads={memory_cfg.get('heads', 8)}")
        else:
            print("[Pretrain] memory: DISABLED")
    model = BridgeVLAPretrainModel(
        model_id=model_id,
        img_size=cfg.get("img_size", 224),
        img_patch_size=cfg.get("img_patch_size", 14),
        use_modified_focal_loss=use_modified_focal_loss,
        use_lm_aux_loss=use_lm_aux_loss,
        memory_cfg=memory_cfg,
    )
    processor = model._processor
    # Non-reentrant checkpointing is required for DDP + find_unused_parameters=True (reentrant trips the reducer).
    model.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = model.to(device_id)

    # Apply the STARTING stage's freeze BEFORE the DDP ctor: DDP's reducer snapshots requires_grad at
    # construction, so a param flipped False->True later is updated locally but never all-reduced.
    # freeze_epochs == 0 starts directly in Stage 2; > 0 starts in Stage 1 and rebuilds DDP at the transition.
    initial_stage = 1 if freeze_epochs > 0 else 2
    freeze_params(model, freeze_paligemma=(initial_stage == 1))
    if rank == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9
        stage_desc = ("frozen (Stage 1)" if initial_stage == 1
                      else "unfrozen from start (freeze_epochs=0, skipping Stage 1)")
        print(f"[Stage {initial_stage}] PaliGemma {stage_desc}. "
              f"Trainable params: {n_train:.3f}B")

    ddp_model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    # ---- Dataset (RoboPoint corpus + deterministic val holdout) ----
    train_ds, val_ds, collate_fn = build_pretrain_datasets(
        cfg, processor, res=cfg.get("img_size", 224),
    )
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, drop_last=True, seed=seed)
    # torch.* is auto-seeded per worker but random / numpy are not - seed them so prompt draws reproduce.
    def _worker_init_fn(worker_id: int):
        worker_seed = torch.initial_seed() % (2 ** 32)
        random.seed(worker_seed)
        try:
            import numpy as _np
            _np.random.seed(worker_seed)
        except ImportError:
            pass
    loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn, persistent_workers=(num_workers > 0),
        worker_init_fn=_worker_init_fn,
    )

    if rank == 0:
        print(f"[Pretrain] dataset size: {len(train_ds)}, iters/epoch: {len(loader)}")

    # ---- Val loader (built from the holdout carved off above) ----
    if val_ds is not None:
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=rank,
            shuffle=False, drop_last=False, seed=seed,
        )
        # Smaller batch than train to stay in memory; no persistent workers so the eval shards release cleanly.
        val_loader = DataLoader(
            val_ds, batch_size=bs, sampler=val_sampler,
            num_workers=min(num_workers, 4), pin_memory=True,
            collate_fn=collate_fn, persistent_workers=False,
            worker_init_fn=_worker_init_fn,
        )
        if rank == 0:
            print(f"[Val] dataset size: {len(val_ds)}, iters/epoch: {len(val_loader)}")
    else:
        val_loader = None

    # Per-prefix LR scaling: the Stage-2 unfrozen PaliGemma Gemma trunk stays conservative to protect the
    # tied lm_head / embed_tokens path. lm_head / embed_tokens / vision_tower get no scale.
    lr_scales = {}
    if paligemma_lr_scale != 1.0:
        # Gemma decoder body only - the broader `model.language_model.model.` prefix would capture embed_tokens.
        lr_scales["model.language_model.model.layers."] = paligemma_lr_scale
        lr_scales["model.language_model.model.norm."] = paligemma_lr_scale
        lr_scales["model.multi_modal_projector."] = paligemma_lr_scale
    optimizer = build_optimizer(
        ddp_model, lr=lr, weight_decay=weight_decay, lr_scales=lr_scales,
    )
    if rank == 0:
        print(f"[Pretrain] paligemma_lr_scale={paligemma_lr_scale}, "
              f"optimizer has {len(optimizer.param_groups)} param group(s)")
        wd_ratio = weight_decay / lr if lr > 0 else 0.0
        print(f"[Pretrain] lr={lr:.2e}, weight_decay={weight_decay:.2e} "
              f"(= {wd_ratio:.3f} × lr)")

    # ---- SwanLab (driven by $SWANLAB_MODE, set by pretrain.sh's SWANLAB_UPLOAD) ----
    #   "cloud" uploads (needs $SWANLAB_API_KEY); "offline" writes <output_path>/swanlog/ only; "local"
    #   connects to a self-hosted swanboard. View offline logs with: `swanlab watch -l <output_path>/swanlog`
    global USE_SWANLAB
    if rank == 0 and not cmd_args.debug:
        swanlab_mode = os.environ.get("SWANLAB_MODE", "offline")
        try:
            import swanlab
            swanlab_logdir = os.path.join(output_path, "swanlog")
            os.makedirs(swanlab_logdir, exist_ok=True)
            swanlab.init(
                project=cfg.get("swanlab_project", "bridgevla_pretrain"),
                experiment_name=exp_name + "_" + os.path.basename(output_path),
                config=cfg,
                mode=swanlab_mode,
                logdir=swanlab_logdir,
            )
            USE_SWANLAB = True
            print(f"[Info] SwanLab enabled ({swanlab_mode} mode — logs at {swanlab_logdir}).")
        except Exception as e:
            print(f"[Info] SwanLab init failed ({e}); continuing without SwanLab.")

    # ---- Training loop with two-stage freeze (fractional freeze_epochs) ----
    iters_per_epoch = len(loader)                       # micro-iters / epoch
    macro_iters_per_epoch = iters_per_epoch // grad_accum_steps
    # freeze_threshold_step is in MACRO steps, so it stays meaningful whatever grad_accum_steps is.
    freeze_threshold_step = int(round(freeze_epochs * macro_iters_per_epoch))
    if rank == 0:
        eff_batch = bs * world_size * grad_accum_steps
        print(
            f"[Pretrain] grad_accumulation_steps={grad_accum_steps} ⇒ "
            f"effective batch = bs({bs}) × world({world_size}) × "
            f"accum({grad_accum_steps}) = {eff_batch}"
        )
        print(
            f"[Pretrain] iters_per_epoch={iters_per_epoch} (micro) → "
            f"{macro_iters_per_epoch} macro / epoch "
            f"(trailing {iters_per_epoch - macro_iters_per_epoch * grad_accum_steps} "
            f"micro(s) dropped per epoch)"
        )
        print(
            f"[Pretrain] freeze_epochs={freeze_epochs} ⇒ "
            f"transition at cumulative MACRO step {freeze_threshold_step}"
        )
        if lm_lambda_max > 0.0:
            print(
                f"[Pretrain] loc_lm: enabled, lambda_max={lm_lambda_max}, "
                f"lm_warmup_steps={lm_warmup_steps} (macro, cumulative)"
            )
        else:
            print("[Pretrain] loc_lm: DISABLED (no LM aux loss)")

    state = {
        "stage": initial_stage,
        "cumulative_step": 0,
        "optimizer": optimizer,
        # Exposed so the mid-epoch Stage 1->2 transition can rebuild DDP in place.
        "ddp_model": ddp_model,
        "local_rank": local_rank,
    }
    saved_ckpts = []
    for epoch in range(num_epochs):
        if rank == 0:
            print(
                f"=== Stage {state['stage']} | epoch {epoch}/{num_epochs - 1} ===",
                flush=True,
            )

        # Validation at the START of each epoch, so epoch 0 measures the freshly-initialized model and the
        # viz shows what the model predicted BEFORE the next epoch trains. Metrics land under val/*.
        if val_loader is not None:
            val_sampler.set_epoch(epoch)
            val_cfg = cfg.get("val") or {}
            n_viz = int(val_cfg.get("n_viz_samples", 0))
            viz_out_dir = (
                os.path.join(output_path, "val_viz", f"epoch_{epoch:03d}")
                if n_viz > 0 else None
            )
            run_validation(
                state["ddp_model"], val_loader,
                device=device_id, rank=rank, world_size=world_size,
                img_size=cfg.get("img_size", 224),
                cumulative_step=state["cumulative_step"],
                epoch=epoch,
                n_viz=n_viz,
                viz_out_dir=viz_out_dir,
            )

        run_epoch(
            state["ddp_model"], loader, sampler, state,
            epoch=epoch, base_lr=lr, lr_warmup_steps=lr_warmup_steps,
            weight_decay=weight_decay,
            freeze_threshold_step=freeze_threshold_step,
            iters_per_epoch=iters_per_epoch,
            device=device_id, rank=rank, logging_steps=logging_steps,
            lm_lambda_max=lm_lambda_max,
            lm_warmup_steps=lm_warmup_steps,
            lr_scales=lr_scales,
            grad_accum_steps=grad_accum_steps,
        )

        # Save every epoch; pull ddp_model from state because the Stage-2 transition may have rebuilt it.
        if rank == 0:
            ddp = state["ddp_model"]
            ck_path = os.path.join(output_path, f"pretrain_epoch_{epoch}.pth")
            save_checkpoint(ddp, ck_path, epoch, extra={"stage": state["stage"]})
            saved_ckpts.append(ck_path)
            while save_total_limit > 0 and len(saved_ckpts) > save_total_limit:
                old = saved_ckpts.pop(0)
                if os.path.exists(old):
                    os.remove(old)
            last_path = os.path.join(output_path, "pretrain_last.pth")
            save_checkpoint(ddp, last_path, epoch, extra={"stage": state["stage"]})
            print(f"[Save] {ck_path}")

        dist.barrier()

    if rank == 0:
        print("[Pretrain] Finished.")
    dist.destroy_process_group()


# ---- Stand-alone debug helpers ----


def _format_vlm_prompt(text: str) -> str:
    """Mirror GroundingCollator: prompt sent to PaliGemma = '<image>' + text."""
    return f"<image>{text}"


def visualise_dataset(cfg_path, save_dir="./debug_samples", n_batches=2):
    """Render `n_batches` mock batches (bs from config) to `save_dir`.

    Each batch is a random draw of `bs` samples from the training stream,
    rendered exactly as the DataLoader would deliver them (prompt template
    included). Output per batch goes to `<save_dir>/batch{i}/`.
    """
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds, _, _ = build_pretrain_datasets(
        cfg, processor=None, res=cfg.get("img_size", 224),
    )
    print(f"Total samples in stream: {len(ds)}")

    bs = int(cfg.get("bs", 64))
    rng = random.Random(int(cfg.get("seed", PRETRAIN_GLOBAL_SEED)))

    os.makedirs(save_dir, exist_ok=True)
    for b in range(n_batches):
        batch_dir = os.path.join(save_dir, f"batch{b}")
        os.makedirs(batch_dir, exist_ok=True)
        batch_idxs = rng.sample(range(len(ds)), min(bs, len(ds)))

        print(f"\n=== batch {b} (size={len(batch_idxs)}) -> {batch_dir} ===")
        for row, idx in enumerate(batch_idxs):
            data = ds[idx]
            image = data["image"]
            text = data["text"]
            bboxes = data["bboxes_cxcywh"]
            tag = data["dataset"]
            category = data.get("category", "non_graspable")

            hm = _build_centernet_gt_heatmap([bboxes], 224, 224)[0, 0]
            out_path = os.path.join(batch_dir, f"row{row:02d}_{tag}.png")
            visualize_bboxes_and_heatmap(
                image, bboxes, hm, out_path,
                caption=f"prompt:  {_format_vlm_prompt(text)}",
                image_title=f"{tag}[{category}]: {len(bboxes)} bbox(es)",
            )
            print(f"  row{row:02d} {tag}  n_bbox={len(bboxes)}  text={text!r}")


def test_inference(cmd_args):
    """Load a pretrain checkpoint and render qualitative eval panels.

    Mirrors the validation-viz pattern (_save_val_viz_samples / _render_val_figure):
      * Source dataset: the val holdout carved off the RoboPoint corpus
        (`val.holdout_samples`); falls back to the training split when no
        holdout is configured.
      * `n_samples_per_task` deterministically-spaced picks (same indices
        across runs on the same config → easy A/B between checkpoints).
      * 4-panel figure per sample (image+GT bbox / image+GT hm / raw pred
        logits / image+pred hm) with dataset / prompt / per-sample metrics
        (PointHit@bbox, center L1, heatmap CE) in the caption.
      * Aggregate metrics are printed and dumped to `eval_metrics.json`
        alongside the panels.
    """
    with open(cmd_args.config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    device = "cuda"
    model_id = os.environ.get("PALIGEMMA_PATH", "google/paligemma-3b-pt-224")

    img_size = int(cfg.get("img_size", 224))
    model = BridgeVLAPretrainModel(
        model_id=model_id,
        img_size=img_size,
        img_patch_size=int(cfg.get("img_patch_size", 14)),
        use_modified_focal_loss=bool(cfg.get("use_modified_focal_loss", False)),
        use_lm_aux_loss=bool(cfg.get("use_lm_aux_loss", False)),
        memory_cfg=cfg.get("memory") or {},
    )
    ckpt_path = cmd_args.checkpoint_path
    assert ckpt_path is not None, "Must provide --checkpoint_path for test_inference"
    sd = load_pretrain_checkpoint(ckpt_path)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[eval] Missing keys:    {len(missing)} (first 10: {missing[:10]})")
    print(f"[eval] Unexpected keys: {len(unexpected)} (first 10: {unexpected[:10]})")
    model = model.to(device).eval()
    # Training enables gradient_checkpointing (forcing use_cache=False); neither suits eval and both have caused SIGFPE.
    if hasattr(model.model, "gradient_checkpointing_disable"):
        model.model.gradient_checkpointing_disable()
    if hasattr(model.model, "config"):
        model.model.config.use_cache = True
    processor = model._processor

    if cmd_args.image_folder is not None:
        cfg["image_folder"] = cmd_args.image_folder
    if cmd_args.json_detection_path is not None:
        cfg["json_detection_path"] = cmd_args.json_detection_path

    train_ds, val_ds, collator = build_pretrain_datasets(
        cfg, processor, res=img_size,
    )
    if val_ds is not None and len(val_ds) > 0:
        ds, source_tag = val_ds, "val"
    else:
        ds, source_tag = train_ds, "train"
        print("[eval] no val holdout configured — evaluating on the train "
              "split instead.", flush=True)

    n_picks = int(getattr(cmd_args, "n_samples_per_task", 20) or 20)
    n_picks = min(n_picks, len(ds))
    step = max(1, len(ds) // n_picks)
    picks = list(range(0, len(ds), step))[:n_picks]

    checkpoint_dir = os.path.dirname(ckpt_path) or "."
    checkpoint_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(checkpoint_dir, f"eval_{checkpoint_name}_{stamp}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[eval] source={source_tag} ({len(ds)} samples), rendering "
          f"{len(picks)} panels -> {save_dir}", flush=True)

    stats = {"n_samples": 0, "n_hit": 0, "sum_l1": 0.0, "n_l1": 0,
             "sum_hm": 0.0, "n_hm": 0}

    with torch.no_grad():
        for sub_idx, idx in enumerate(tqdm(picks, desc="[eval]")):
            sample = ds[idx]
            batch = collator([sample])
            tokens = {
                k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)
            }
            out = model(
                input_ids=tokens["input_ids"],
                pixel_values=tokens["pixel_values"],
                attention_mask=tokens["attention_mask"],
                bboxes=batch["bboxes"],
                raw_text=batch["raw_text"],
                token_type_ids=tokens.get("token_type_ids"),
            )

            q_trans_flat = out["q_trans"].detach()          # (1, H*W, 1)
            q_trans = q_trans_flat.view(img_size, img_size)
            bboxes_i = batch["bboxes"][0]
            # GT + pred heatmap match the model's training objective.
            gt_hm, pred_hm = _viz_gt_and_pred_hm(
                model, q_trans_flat, bboxes_i,
                is_action_natural=False,
                h=img_size, w=img_size, device=device,
            )

            hits, dists = _pointhit_and_center_l1(
                q_trans_flat, batch["bboxes"], img_size,
            )
            hit_i = float(hits[0].item())
            d0 = dists[0]
            l1_i = float(d0.item()) if torch.isfinite(d0).item() else float("nan")
            if torch.isfinite(d0).item():
                stats["n_l1"] += 1
                stats["sum_l1"] += l1_i
                stats["n_hit"] += int(hit_i)
            if int(out["hm_n_valid"]) > 0:
                stats["sum_hm"] += float(out["loss_hm"].item())
                stats["n_hm"] += 1
            stats["n_samples"] += 1

            metric_bits = [
                f"PointHit={hit_i:.0f}",
                (f"center_L1={l1_i:.2f}px" if l1_i == l1_i else "center_L1=NaN"),
                f"loss_hm={float(out['loss_hm'].item()):.4f}",
            ]
            caption_lines = [
                f"dataset: {sample['dataset']}  source: {source_tag}",
                f"prompt:  {_format_vlm_prompt(sample['text'])}",
                "  ".join(metric_bits),
            ]
            _render_val_figure(
                image=sample["image"],
                gt_heatmap=gt_hm,
                pred_logits=q_trans,
                pred_heatmap=pred_hm,
                bboxes_cxcywh=bboxes_i,
                prompt="\n".join(caption_lines),
                save_path=os.path.join(save_dir,
                                       f"{source_tag}_{sub_idx:03d}.png"),
            )

    point_hit = (stats["n_hit"] / stats["n_l1"]) if stats["n_l1"] > 0 else None
    center_l1 = (stats["sum_l1"] / stats["n_l1"]) if stats["n_l1"] > 0 else None
    loss_hm = (stats["sum_hm"] / stats["n_hm"]) if stats["n_hm"] > 0 else None

    print("\n" + "=" * 72)
    print(f"[eval] Summary ({ckpt_path})")
    print("=" * 72)
    print(f"  source={source_tag}  n={stats['n_samples']}  "
          f"n_available={len(ds)}")
    if point_hit is not None:
        print(f"  PointHit@bbox : {point_hit:.4f}")
        print(f"  center_l1_avg : {center_l1:.2f} px")
    if loss_hm is not None:
        print(f"  loss_hm_avg   : {loss_hm:.4f}  "
              f"(n_supervised={stats['n_hm']})")
    print("=" * 72)

    metrics_path = os.path.join(save_dir, "eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "checkpoint":    ckpt_path,
            "config":        cmd_args.config_path,
            "img_size":      img_size,
            "source":        source_tag,
            "n_samples":     stats["n_samples"],
            "n_available":   len(ds),
            "point_hit":     point_hit,
            "center_l1_avg": center_l1,
            "loss_hm_avg":   loss_hm,
        }, f, indent=2)
    print(f"[eval] metrics JSON: {metrics_path}")


# ---- CLI entry ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches", type=int, default=2,
                        help="1: visualise data; 2: pretrain; 3: test inference")
    parser.add_argument("--config_path", type=str, default="pretrain_config.yaml")
    parser.add_argument("--json_detection_path", type=str, default=None)
    parser.add_argument("--image_folder", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Override checkpoint_dir in config for test_inference")
    parser.add_argument("--n_samples_per_task", type=int, default=20,
                        help="(branches=3) Number of eval panels to render.")
    parser.add_argument("--debug", action="store_true")
    cmd_args = parser.parse_args()

    if cmd_args.branches == 1:
        # Stamp each run so back-to-back branch=1 invocations don't overwrite prior visualizations.
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "debug_samples", stamp)
        print(f"[branch=1] visualising into: {save_dir}")
        visualise_dataset(cmd_args.config_path, save_dir=save_dir)
    elif cmd_args.branches == 2:
        experiment(cmd_args)
    elif cmd_args.branches == 3:
        test_inference(cmd_args)
    else:
        raise ValueError(f"Unknown branch: {cmd_args.branches}")
