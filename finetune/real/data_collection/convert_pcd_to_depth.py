#!/usr/bin/env python
"""Convert the real-robot ``zed_pcd`` XYZ dumps to depth + intrinsics.

Why
---
``zed_pcd/{i}.pkl`` stores the ZED ``MEASURE.XYZRGBA`` buffer as a
(1080,1920,3) float32 array = 24.9 MB per frame. The X and Y channels are
*exactly* a pinhole deprojection of the Z channel::

    X = (u - cx) / fx * Z        Y = (v - cy) / fy * Z

(measured residual over sampled frames: max |dX| = 8.2e-07 m, i.e. float32
rounding noise). So two thirds of every file is recomputable from Z plus four
numbers. Storing Z alone as uint16 shrinks the dataset ~6x on its own, and
~26x once it is PNG-compressed.

What this writes (in place, non-destructive)
--------------------------------------------
    {episode}/zed_depth/{i}.png     uint16 depth, DEPTH_SCALE units per metre
    {episode}/intrinsic.pkl         per-frame fitted {fx,fy,cx,cy} + metadata
    {episode}/zed_rgb/{i}.jpg       RGB re-encoded (optional, --rgb jpeg)

Originals are **never deleted**. Run ``--delete-originals`` as a separate,
explicit second pass once you have read the verification report.

Verification
------------
Every frame is checked twice, so "is the model right" is never confused with
"what did quantisation cost":

  * ``err_model`` — reconstruct X,Y from the *unquantised* float Z and the
    fitted K, compare against the stored X,Y. This proves the pinhole model
    captures the data exactly. Must be < ``--tol-model`` (default 1e-5 m).
  * ``err_total`` — reconstruct from the *quantised* uint16 depth, i.e. the
    real end-to-end error a trainer would see. Must be < ``--tol-total``
    (default 5e-4 m). At DEPTH_SCALE=4000 the predicted worst case is
    0.5/4000 * max|(u-cx)/fx| ~= 1.1e-4 m.

An episode that fails either check is left completely untouched and reported.

Usage
-----
    python finetune/real/data_collection/convert_pcd_to_depth.py \
        --data-root /path/to/your/real_dataset --jobs 8

    # after reviewing the report:
    python .../convert_pcd_to_depth.py --data-root ... --delete-originals
"""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# uint16 depth quantisation: units per metre.
#   4000 -> 0.25 mm resolution, 16.38 m range.
# The observed data maxes out around 5.4 m, so 16 m leaves generous headroom
# for new scenes while keeping the quantisation error (0.125 mm) far below
# anything the policy can act on.
DEPTH_SCALE = 4000.0
DEPTH_MAX_U16 = 65535

# Pixels with Z below this are treated as invalid (ZED writes 0 for "no
# measurement"; the collection script also zeroes NaNs).
MIN_VALID_Z = 0.05


# depth image io (cv2 preferred: unambiguous uint16 PNG; PIL as fallback)
try:
    import cv2

    def _write_png16(path, arr):
        if not cv2.imwrite(path, arr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
            raise RuntimeError(f"cv2.imwrite failed: {path}")

    def _read_png16(path):
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise RuntimeError(f"cv2.imread failed: {path}")
        return a

    def _write_jpeg(path, rgb):
        # rgb is stored channel-order-as-is; cv2 writes the array verbatim.
        if not cv2.imwrite(path, rgb, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"cv2.imwrite failed: {path}")

    def _read_jpeg(path):
        a = cv2.imread(path, cv2.IMREAD_COLOR)
        if a is None:
            raise RuntimeError(f"cv2.imread failed: {path}")
        return a

except ImportError:  # pragma: no cover - cv2 is present in all project envs
    from PIL import Image

    def _write_png16(path, arr):
        Image.fromarray(arr).save(path, compress_level=3)

    def _read_png16(path):
        return np.asarray(Image.open(path)).astype(np.uint16)

    def _write_jpeg(path, rgb):
        Image.fromarray(rgb).save(path, quality=95)

    def _read_jpeg(path):
        return np.asarray(Image.open(path))


def fit_intrinsics(xyz: np.ndarray, subsample: int = 7):
    """Least-squares fit of (fx, fy, cx, cy) from a stored XYZ image.

    Fits on a strided subsample for speed; the caller verifies on every pixel.
    Returns None when the frame has too few valid points to fit.
    """
    h, w, _ = xyz.shape
    sub = xyz[::subsample, ::subsample]
    z = sub[..., 2]
    valid = np.isfinite(z) & (z > MIN_VALID_Z) & np.isfinite(sub[..., 0]) & np.isfinite(sub[..., 1])
    if valid.sum() < 1000:
        return None

    u = np.arange(0, w, subsample, dtype=np.float64)[None, :]
    v = np.arange(0, h, subsample, dtype=np.float64)[:, None]
    u = np.broadcast_to(u, z.shape)[valid]
    v = np.broadcast_to(v, z.shape)[valid]
    zv = z[valid].astype(np.float64)

    # X/Z is linear in u with slope 1/fx and intercept -cx/fx.
    au = np.polyfit(u, sub[..., 0][valid].astype(np.float64) / zv, 1)
    av = np.polyfit(v, sub[..., 1][valid].astype(np.float64) / zv, 1)
    if au[0] == 0 or av[0] == 0:
        return None
    fx = 1.0 / au[0]
    fy = 1.0 / av[0]
    cx = -au[1] * fx
    cy = -av[1] * fy
    return float(fx), float(fy), float(cx), float(cy)


def deproject(depth_m: np.ndarray, K, u_coords=None, v_coords=None):
    """Rebuild an (H,W,3) XYZ image from a metric depth image and K."""
    fx, fy, cx, cy = K
    h, w = depth_m.shape
    u = np.arange(w, dtype=np.float64) if u_coords is None else np.asarray(u_coords, dtype=np.float64)
    v = np.arange(h, dtype=np.float64) if v_coords is None else np.asarray(v_coords, dtype=np.float64)
    x = ((u[None, :] - cx) / fx) * depth_m
    y = ((v[:, None] - cy) / fy) * depth_m
    return np.stack([x, y, depth_m], axis=-1)


def _frame_indices(ep: str):
    pcd_dir = os.path.join(ep, "zed_pcd")
    if not os.path.isdir(pcd_dir):
        return []
    idx = []
    for f in os.listdir(pcd_dir):
        if f.endswith(".pkl"):
            try:
                idx.append(int(os.path.splitext(f)[0]))
            except ValueError:
                pass
    return sorted(idx)


def convert_episode(ep: str, args) -> dict:
    """Convert one episode. Writes nothing until every frame has verified."""
    rel = os.path.relpath(ep, args.data_root)
    res = {
        "episode": rel, "ok": False, "n": 0, "reason": "",
        "err_model": 0.0, "err_total": 0.0,
        "src_bytes": 0, "dst_bytes": 0,
        "n_clipped": 0, "n_nonfinite": 0, "max_z": 0.0,
    }

    frames = _frame_indices(ep)
    if not frames:
        res["reason"] = "no zed_pcd frames"
        return res
    if frames != list(range(len(frames))):
        res["reason"] = f"non-contiguous frame indices: {frames}"
        return res

    depth_dir = os.path.join(ep, "zed_depth")
    if os.path.isdir(depth_dir) and not args.force:
        existing = [f for f in os.listdir(depth_dir) if f.endswith(".png")]
        if len(existing) == len(frames) and os.path.exists(os.path.join(ep, "intrinsic.pkl")):
            res["ok"] = True
            res["n"] = len(frames)
            res["reason"] = "already converted (skipped)"
            return res

    # ---- pass 1: convert + verify entirely in memory -------------------
    staged = []   # (frame_idx, uint16 depth)
    staged_rgb = []
    Ks = []
    try:
        for i in frames:
            pcd_path = os.path.join(ep, "zed_pcd", f"{i}.pkl")
            with open(pcd_path, "rb") as f:
                xyz = np.asarray(pickle.load(f))[:, :, :3].astype(np.float32)
            res["src_bytes"] += os.path.getsize(pcd_path)

            finite = np.isfinite(xyz).all(axis=-1)
            res["n_nonfinite"] += int((~finite).sum())

            K = fit_intrinsics(xyz)
            if K is None:
                res["reason"] = f"frame {i}: too few valid points to fit intrinsics"
                return res
            Ks.append(K)

            z = np.where(finite, xyz[..., 2], 0.0).astype(np.float64)
            z = np.where(z > MIN_VALID_Z, z, 0.0)
            res["max_z"] = max(res["max_z"], float(z.max()))

            # --- check A: pinhole model exactness (unquantised Z) ---
            valid = z > MIN_VALID_Z
            rec_f = deproject(z, K)
            err_model = float(np.abs(rec_f - xyz)[valid].max()) if valid.any() else 0.0
            res["err_model"] = max(res["err_model"], err_model)
            if err_model > args.tol_model:
                res["reason"] = (f"frame {i}: pinhole model residual {err_model:.3e} m "
                                 f"> tol_model {args.tol_model:.1e}")
                return res

            # --- quantise ---
            q = np.rint(z * DEPTH_SCALE)
            res["n_clipped"] += int((q > DEPTH_MAX_U16).sum())
            d16 = np.clip(q, 0, DEPTH_MAX_U16).astype(np.uint16)

            # --- check B: end-to-end error through the quantised depth ---
            rec_q = deproject(d16.astype(np.float64) / DEPTH_SCALE, K)
            err_total = float(np.abs(rec_q - xyz)[valid].max()) if valid.any() else 0.0
            res["err_total"] = max(res["err_total"], err_total)
            if err_total > args.tol_total:
                res["reason"] = (f"frame {i}: end-to-end residual {err_total:.3e} m "
                                 f"> tol_total {args.tol_total:.1e}")
                return res

            staged.append((i, d16))

            if args.rgb == "jpeg":
                rgb_pkl = os.path.join(ep, "zed_rgb", f"{i}.pkl")
                if os.path.exists(rgb_pkl):
                    with open(rgb_pkl, "rb") as f:
                        rgb = np.asarray(pickle.load(f))[:, :, :3]
                    res["src_bytes"] += os.path.getsize(rgb_pkl)
                    staged_rgb.append((i, np.ascontiguousarray(rgb)))
    except Exception:
        res["reason"] = "exception: " + traceback.format_exc(limit=3)
        return res

    # ---- pass 2: everything verified, now write -----------------------
    os.makedirs(depth_dir, exist_ok=True)
    for i, d16 in staged:
        p = os.path.join(depth_dir, f"{i}.png")
        _write_png16(p, d16)
        # Read back and confirm the file on disk decodes bit-identically.
        if not np.array_equal(_read_png16(p), d16):
            res["reason"] = f"frame {i}: PNG round-trip mismatch on disk"
            return res
        res["dst_bytes"] += os.path.getsize(p)

    for i, rgb in staged_rgb:
        p = os.path.join(ep, "zed_rgb", f"{i}.jpg")
        _write_jpeg(p, rgb)
        res["dst_bytes"] += os.path.getsize(p)

    meta = {
        "fx": np.array([k[0] for k in Ks], dtype=np.float64),
        "fy": np.array([k[1] for k in Ks], dtype=np.float64),
        "cx": np.array([k[2] for k in Ks], dtype=np.float64),
        "cy": np.array([k[3] for k in Ks], dtype=np.float64),
        "depth_scale": DEPTH_SCALE,
        "shape": tuple(staged[0][1].shape),
        "num_frames": len(staged),
        "source": "fitted from zed_pcd by convert_pcd_to_depth.py",
        "err_model_max": res["err_model"],
        "err_total_max": res["err_total"],
    }
    with open(os.path.join(ep, "intrinsic.pkl"), "wb") as f:
        pickle.dump(meta, f)

    res["ok"] = True
    res["n"] = len(staged)
    return res


def _worker(ep, args):
    try:
        return convert_episode(ep, args)
    except Exception:
        return {"episode": ep, "ok": False, "n": 0,
                "reason": "worker crash: " + traceback.format_exc(limit=3),
                "err_model": 0.0, "err_total": 0.0, "src_bytes": 0, "dst_bytes": 0,
                "n_clipped": 0, "n_nonfinite": 0, "max_z": 0.0}


def find_episodes(root):
    eps = []
    for task_group in sorted(os.listdir(root)):
        tg = os.path.join(root, task_group)
        if not os.path.isdir(tg):
            continue
        for name in sorted(os.listdir(tg)):
            ep = os.path.join(tg, name)
            if os.path.isdir(os.path.join(ep, "zed_pcd")):
                eps.append(ep)
    return eps


def delete_originals(root, args):
    """Second pass: remove zed_pcd/ and zed_rgb/*.pkl for verified episodes."""
    eps = find_episodes(root)
    freed = 0
    removed = 0
    for ep in eps:
        rel = os.path.relpath(ep, root)
        depth_dir = os.path.join(ep, "zed_depth")
        intr = os.path.join(ep, "intrinsic.pkl")
        frames = _frame_indices(ep)
        if not (os.path.isdir(depth_dir) and os.path.exists(intr)):
            print(f"  SKIP {rel}: not converted")
            continue
        n_png = len([f for f in os.listdir(depth_dir) if f.endswith(".png")])
        if n_png != len(frames):
            print(f"  SKIP {rel}: {n_png} png vs {len(frames)} pcd frames")
            continue
        if args.rgb == "jpeg":
            n_jpg = len([f for f in os.listdir(os.path.join(ep, "zed_rgb"))
                         if f.endswith(".jpg")])
            if n_jpg != len(frames):
                print(f"  SKIP {rel}: {n_jpg} jpg vs {len(frames)} frames")
                continue
        pcd_dir = os.path.join(ep, "zed_pcd")
        freed += sum(os.path.getsize(os.path.join(pcd_dir, f))
                     for f in os.listdir(pcd_dir))
        shutil.rmtree(pcd_dir)
        if args.rgb == "jpeg":
            rgb_dir = os.path.join(ep, "zed_rgb")
            for f in os.listdir(rgb_dir):
                if f.endswith(".pkl"):
                    p = os.path.join(rgb_dir, f)
                    freed += os.path.getsize(p)
                    os.remove(p)
        removed += 1
        print(f"  removed originals: {rel}")
    print(f"\n{removed} episodes cleaned, {freed/1e9:.2f} GB freed")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--rgb", choices=["jpeg", "none"], default="jpeg",
                    help="also re-encode zed_rgb/*.pkl to JPEG q95 (default: jpeg)")
    ap.add_argument("--tol-model", type=float, default=1e-5,
                    help="max allowed pinhole-model residual, metres")
    ap.add_argument("--tol-total", type=float, default=5e-4,
                    help="max allowed end-to-end residual through quantised depth, metres")
    ap.add_argument("--force", action="store_true", help="reconvert already-converted episodes")
    ap.add_argument("--delete-originals", action="store_true",
                    help="second pass: delete zed_pcd/ (and zed_rgb/*.pkl) for verified episodes")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    if not os.path.isdir(root):
        sys.exit(f"data-root not found: {root}")

    if args.delete_originals:
        print(f"Deleting originals under {root}\n")
        delete_originals(root, args)
        return

    eps = find_episodes(root)
    print(f"Found {len(eps)} episodes under {root}")
    print(f"depth: uint16 PNG @ {DEPTH_SCALE:.0f} units/m "
          f"({1000/DEPTH_SCALE:.3f} mm, {DEPTH_MAX_U16/DEPTH_SCALE:.2f} m range); rgb: {args.rgb}")
    print(f"tolerances: model {args.tol_model:.1e} m, total {args.tol_total:.1e} m\n")

    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futs = {pool.submit(_worker, ep, args): ep for ep in eps}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            tag = "ok  " if r["ok"] else "FAIL"
            extra = f" ({r['reason']})" if r["reason"] else ""
            print(f"[{k}/{len(eps)}] {tag} {r['episode']}  n={r['n']}  "
                  f"err_model={r['err_model']:.2e}  err_total={r['err_total']:.2e}{extra}")

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    src = sum(r["src_bytes"] for r in ok)
    dst = sum(r["dst_bytes"] for r in ok)

    print("\n" + "=" * 72)
    print(f"converted   : {len(ok)}/{len(results)} episodes, {sum(r['n'] for r in ok)} frames")
    if src and dst:
        print(f"size        : {src/1e9:.2f} GB -> {dst/1e9:.3f} GB  ({src/dst:.1f}x smaller)")
    if ok:
        print(f"err_model   : max {max(r['err_model'] for r in ok):.3e} m  (pinhole exactness)")
        print(f"err_total   : max {max(r['err_total'] for r in ok):.3e} m  (through quantised depth)")
        print(f"max depth   : {max(r['max_z'] for r in ok):.3f} m "
              f"(uint16 range {DEPTH_MAX_U16/DEPTH_SCALE:.2f} m)")
        print(f"clipped px  : {sum(r['n_clipped'] for r in ok)}")
        print(f"non-finite  : {sum(r['n_nonfinite'] for r in ok)} px in source")
    if bad:
        print(f"\nFAILED ({len(bad)}) — these episodes were left untouched:")
        for r in bad:
            print(f"  {r['episode']}: {r['reason']}")
    print("=" * 72)
    print("\nOriginals were NOT deleted. After reviewing the numbers above, run the")
    print("same command with --delete-originals to reclaim the space.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
