"""CenterNet-style GT heatmap helpers + modified focal loss + dual-head
ConvexUpSample factory.

This module is the home of the "modified focal loss + companion
patch-token post-processing network" branch. It is only used when
``cfg.use_modified_focal_loss=True``. The default BridgeVLA path uses a
single ``up0`` head with softmax + cross-entropy loss (see
``bridgevla.mvt.heads_original``).

Kept as an explicit, isolated file so the focal-loss branch can be
re-enabled cleanly for ablation. Heatmap semantics:

* GT is a per-pixel soft label in [0, 1] with peak 1.0 at the target pixel
  and a 2D Gaussian fall-off (NOT softmax-normalized).
* Loss is the CornerNet/CenterNet modified focal loss on raw logits.
* Final-conv bias prior ``HM_PRIOR_LOGIT`` gives sigmoid ≈ 0.1 at step 0.
"""

import torch
from torch import nn

from bridgevla.mvt.raft_utils import ConvexUpSample


HM_MIN_OVERLAP    = 0.7
HM_MIN_RADIUS     = 6       # grounding floor (σ=(2·6+1)/6≈2.17)
HM_VLA_MIN_RADIUS = 8       # VLA (TCP) radius floor — σ=(2·8+1)/6≈2.83
HM_PRIOR_LOGIT    = -2.19   # log((1-π)/π) with π=0.1


def _centernet_gaussian_radius(h, w, min_overlap=HM_MIN_OVERLAP):
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


def _gaussian2d_kernel(radius, sigma, dtype=torch.float32):
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


def _draw_umich_gaussian_(hm, cx, cy, radius):
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


def generate_centernet_hm_from_bbox(bboxes_cxcywh, res, floor_radius=HM_MIN_RADIUS):
    """CenterNet-style GT heatmap from normalized (cx,cy,bw,bh) bboxes."""
    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res
    bs = bboxes_cxcywh.shape[0]
    hm_cpu = torch.zeros((bs, resy, resx), dtype=torch.float32)
    cpu_boxes = bboxes_cxcywh.detach().cpu()
    for i in range(bs):
        cx, cy, bw, bh = [float(v) for v in cpu_boxes[i].tolist()]
        cx_px = cx * resx
        cy_px = cy * resy
        bw_px = max(bw * resx, 1.0)
        bh_px = max(bh * resy, 1.0)
        r_cn = _centernet_gaussian_radius(bh_px, bw_px, HM_MIN_OVERLAP)
        radius = max(int(r_cn), int(floor_radius))
        cx_i = min(max(int(round(cx_px)), 0), resx - 1)
        cy_i = min(max(int(round(cy_px)), 0), resy - 1)
        _draw_umich_gaussian_(hm_cpu[i], cx_i, cy_i, radius)
    return hm_cpu.to(bboxes_cxcywh.device)


def generate_centernet_hm_from_pt(pt, res, radius=HM_VLA_MIN_RADIUS):
    """CenterNet-style GT heatmap from (x, y) pixel points. Peak=1, no softmax."""
    num_pt, x = pt.shape
    assert x == 2

    if isinstance(res, int):
        resx = resy = res
    else:
        resx, resy = res

    hm_cpu = torch.zeros((num_pt, resy, resx), dtype=torch.float32)
    pt_cpu = pt.detach().cpu()
    for i in range(num_pt):
        cx_i = int(round(float(pt_cpu[i, 0])))
        cy_i = int(round(float(pt_cpu[i, 1])))
        cx_i = min(max(cx_i, 0), resx - 1)
        cy_i = min(max(cy_i, 0), resy - 1)
        _draw_umich_gaussian_(hm_cpu[i], cx_i, cy_i, int(radius))
    return hm_cpu.to(pt.device)


def modified_focal_loss_per_heatmap(logits, gt):
    """CornerNet/CenterNet modified focal loss, returned per-heatmap (N, C)."""
    if logits.dim() == 3:
        logits = logits.unsqueeze(1)
        gt = gt.unsqueeze(1)
    assert logits.shape == gt.shape, (logits.shape, gt.shape)

    p = torch.clamp(logits.sigmoid(), min=1e-6, max=1 - 1e-6)
    pos_inds = gt.eq(1).to(p.dtype)
    neg_inds = gt.lt(1).to(p.dtype)
    neg_weights = torch.pow(1 - gt, 4)

    pos_loss = torch.log(p)       * torch.pow(1 - p, 2) * pos_inds
    neg_loss = torch.log(1 - p)   * torch.pow(p, 2)     * neg_weights * neg_inds

    dims = (2, 3)
    num_pos = pos_inds.sum(dim=dims)             # (N, C)
    pos_sum = pos_loss.sum(dim=dims)
    neg_sum = neg_loss.sum(dim=dims)
    denom   = num_pos.clamp(min=1.0)
    per_map = -(pos_sum + neg_sum) / denom
    per_map = torch.where(num_pos > 0, per_map, -neg_sum)
    return per_map


def modified_focal_loss(logits, gt):
    """Scalar variant of `modified_focal_loss_per_heatmap` (mean over N, C)."""
    return modified_focal_loss_per_heatmap(logits, gt).mean()


def build_focal_dual_head(in_dim: int, up_ratio: int):
    """Build the dual heatmap heads (up_action + up_grounding).

    Both heads take ``in_dim``-channel feature maps (default: PaliGemma 2048)
    and emit a 1-channel heatmap. Final-conv bias is pre-set to
    ``HM_PRIOR_LOGIT`` so sigmoid output starts at ≈0.1 (RetinaNet/CenterNet
    focal-loss prior).

    Returns ``(up_action, up_grounding)``.
    """
    up_action = ConvexUpSample(in_dim=in_dim, out_dim=1, up_ratio=up_ratio)
    up_grounding = ConvexUpSample(in_dim=in_dim, out_dim=1, up_ratio=up_ratio)
    with torch.no_grad():
        up_action.net_out[-1].bias.fill_(HM_PRIOR_LOGIT)
        up_grounding.net_out[-1].bias.fill_(HM_PRIOR_LOGIT)
    return up_action, up_grounding
