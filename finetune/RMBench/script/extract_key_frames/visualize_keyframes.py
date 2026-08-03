"""
Visualize keyframes by saving the corresponding RGB images from HDF5 data.
Saves 3 episodes per task, with keyframe images annotated with frame index and reason.

This reads from the keyframe-only HDF5 layout produced under
``keyframe_data/<task>/keyframe_depth/data/<episode>.hdf5``. In that
layout each HDF5 stores ONLY the keyframes (not the full trajectory), and the
``keyframe_indices`` dataset maps each stored frame back to its original
trajectory frame index. We therefore look up each keyframe's HDF5 row via
``keyframe_indices`` rather than indexing by the original ``frame_idx``.
"""

import json
import os
import io
from glob import glob

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def decode_rgb(raw_bytes):
    """Decode RGB image from HDF5 stored bytes."""
    if isinstance(raw_bytes, bytes):
        return np.array(Image.open(io.BytesIO(raw_bytes)))
    elif isinstance(raw_bytes, np.ndarray) and raw_bytes.dtype.kind == 'S':
        return np.array(Image.open(io.BytesIO(raw_bytes.tobytes())))
    elif isinstance(raw_bytes, np.void):
        return np.array(Image.open(io.BytesIO(bytes(raw_bytes))))
    else:
        return np.array(raw_bytes)


def draw_text_on_image(img_array, text, position=(5, 5), font_size=14):
    """Draw text with background on image."""
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    bbox = draw.textbbox(position, text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0, 180))
    draw.text(position, text, fill=(255, 255, 0), font=font)
    return np.array(img)


def main():
    # New keyframe-only data layout: <data_root>/<task>/keyframe_depth/data/<episode>.hdf5
    data_root = "data/keyframe_data"
    keyframe_root = "data/keyframes"
    output_root = "data/keyframe_vis"
    n_episodes = 3

    os.makedirs(output_root, exist_ok=True)

    tasks = sorted([
        f.replace(".json", "")
        for f in os.listdir(keyframe_root)
        if f.endswith(".json") and not f.startswith("_")
    ])

    for task in tasks:
        print(f"Processing {task}...")
        kf_path = os.path.join(keyframe_root, f"{task}.json")
        with open(kf_path) as f:
            kf_data = json.load(f)

        def _hdf5_path(ep):
            return os.path.join(data_root, task, "keyframe_depth", "data", f"{ep}.hdf5")

        # Pick first n_episodes episodes (sorted by index) that actually exist on disk.
        sorted_eps = sorted(kf_data.keys(), key=lambda x: int(x.replace("episode", "")))
        ep_names = [ep for ep in sorted_eps if os.path.exists(_hdf5_path(ep))][:n_episodes]

        if not ep_names:
            print(f"  [SKIP] no hdf5 found under {os.path.join(data_root, task, 'keyframe_depth', 'data')}")
            continue

        for ep_name in ep_names:
            ep_info = kf_data[ep_name]
            hdf5_path = _hdf5_path(ep_name)

            ep_out_dir = os.path.join(output_root, task, ep_name)
            os.makedirs(ep_out_dir, exist_ok=True)

            f = h5py.File(hdf5_path, "r")
            head_rgb = f["observation/head_camera/rgb"]
            n_stored = head_rgb.shape[0]
            # Map original trajectory frame index -> stored HDF5 row.
            kf_indices = list(f["keyframe_indices"][:]) if "keyframe_indices" in f else None

            n_frames = ep_info["n_frames"]
            keyframes = ep_info["keyframes"]
            print(f"  {ep_name}: {n_stored} stored keyframes, {len(keyframes)} json keyframes")

            for kf_idx, kf in enumerate(keyframes):
                frame_idx = kf["frame_idx"]
                reason = kf["reason"]
                lg = kf["left_gripper"]
                rg = kf["right_gripper"]
                lang = kf.get("language_annotation", "")

                # Resolve the row inside the keyframe-only HDF5.
                if kf_indices is not None:
                    if frame_idx in kf_indices:
                        row = kf_indices.index(frame_idx)
                    elif kf_idx < n_stored:
                        row = kf_idx
                    else:
                        print(f"    [ERR] frame {frame_idx} not in keyframe_indices")
                        continue
                else:
                    row = kf_idx

                # Decode image
                raw = head_rgb[row]
                try:
                    img = decode_rgb(raw)
                except Exception as e:
                    print(f"    [ERR] frame {frame_idx} (row {row}): {e}")
                    continue

                # Annotate
                line1 = f"KF{kf_idx} | frame={frame_idx}/{n_frames} | L_grip={lg:.2f} R_grip={rg:.2f}"
                line2 = f"{reason}"
                line3 = f"{lang[:80]}" if lang else ""

                img = draw_text_on_image(img, line1, position=(5, 5), font_size=13)
                img = draw_text_on_image(img, line2, position=(5, 22), font_size=12)
                if line3:
                    img = draw_text_on_image(img, line3, position=(5, 38), font_size=11)

                # Save
                out_path = os.path.join(ep_out_dir, f"kf{kf_idx:02d}_frame{frame_idx:04d}.png")
                Image.fromarray(img).save(out_path)

            f.close()

        print(f"  -> Saved to {os.path.join(output_root, task)}/")

    print(f"\nDone. All visualizations saved to {output_root}/")


if __name__ == "__main__":
    main()
