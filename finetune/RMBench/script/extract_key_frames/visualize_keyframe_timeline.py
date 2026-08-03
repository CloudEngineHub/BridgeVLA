"""
Timeline overview: all keyframes of an episode in one grid (head camera RGB).
Easier to review extraction quality at a glance.

Usage:
    python data/visualize_keyframe_timeline.py
    python data/visualize_keyframe_timeline.py --task battery_try --n_episodes 3
"""

import argparse
import io
import json
import math
import os

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def decode_rgb(raw_bytes):
    if isinstance(raw_bytes, bytes):
        return np.array(Image.open(io.BytesIO(raw_bytes)))
    if isinstance(raw_bytes, np.ndarray) and raw_bytes.dtype.kind == "S":
        return np.array(Image.open(io.BytesIO(raw_bytes.tobytes())))
    if isinstance(raw_bytes, np.void):
        return np.array(Image.open(io.BytesIO(bytes(raw_bytes))))
    return np.array(raw_bytes)


def get_font(size=11):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def build_timeline(hdf5_path, keyframes, n_frames, cols=5, thumb_w=320, thumb_h=240):
    n = len(keyframes)
    rows = math.ceil(n / cols)
    header_h = 36
    label_h = 52
    canvas = Image.new("RGB", (cols * thumb_w, header_h + rows * (thumb_h + label_h)), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = get_font(12)
    font_sm = get_font(10)

    draw.text((8, 8), f"{os.path.basename(hdf5_path)} | {n_frames} frames | {n} keyframes", fill=(255, 255, 0), font=font)

    with h5py.File(hdf5_path, "r") as f:
        head_rgb = f["observation/head_camera/rgb"]
        for i, kf in enumerate(keyframes):
            r, c = divmod(i, cols)
            x0 = c * thumb_w
            y0 = header_h + r * (thumb_h + label_h)

            frame_idx = kf["frame_idx"]
            img = decode_rgb(head_rgb[frame_idx])
            img = Image.fromarray(img).resize((thumb_w, thumb_h), Image.BILINEAR)
            canvas.paste(img, (x0, y0))

            reason = kf.get("reason", "")
            if len(reason) > 42:
                reason = reason[:39] + "..."
            line1 = f"KF{i:02d}  f={frame_idx}/{n_frames-1}  L={kf['left_gripper']:.1f} R={kf['right_gripper']:.1f}"
            draw.rectangle([x0, y0 + thumb_h, x0 + thumb_w - 1, y0 + thumb_h + label_h], fill=(16, 16, 16))
            draw.text((x0 + 4, y0 + thumb_h + 2), line1, fill=(220, 220, 220), font=font_sm)
            draw.text((x0 + 4, y0 + thumb_h + 18), reason, fill=(180, 220, 255), font=font_sm)

    return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/data")
    parser.add_argument("--keyframe_root", default="data/keyframes")
    parser.add_argument("--output_root", default="data/keyframe_vis_timeline")
    parser.add_argument("--task", default=None)
    parser.add_argument("--n_episodes", type=int, default=3)
    parser.add_argument("--cols", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)

    if args.task:
        tasks = [args.task]
    else:
        tasks = sorted(
            f.replace(".json", "")
            for f in os.listdir(args.keyframe_root)
            if f.endswith(".json") and not f.startswith("_")
        )

    for task in tasks:
        kf_path = os.path.join(args.keyframe_root, f"{task}.json")
        if not os.path.exists(kf_path):
            print(f"[SKIP] {task}: no keyframes json")
            continue

        with open(kf_path) as f:
            kf_data = json.load(f)

        ep_names = sorted(kf_data.keys(), key=lambda x: int(x.replace("episode", "")))[: args.n_episodes]
        print(f"Processing {task} ({len(ep_names)} episodes)...")

        for ep_name in ep_names:
            ep_info = kf_data[ep_name]
            hdf5_path = os.path.join(args.data_root, task, "demo_clean", "data", f"{ep_name}.hdf5")
            if not os.path.exists(hdf5_path):
                print(f"  [SKIP] {hdf5_path}")
                continue

            out_dir = os.path.join(args.output_root, task)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{ep_name}_timeline.png")

            canvas = build_timeline(
                hdf5_path, ep_info["keyframes"], ep_info["n_frames"], cols=args.cols
            )
            canvas.save(out_path)
            print(f"  {ep_name}: {len(ep_info['keyframes'])} kfs -> {out_path}")

    print(f"\nDone. Output: {args.output_root}/")


if __name__ == "__main__":
    main()
