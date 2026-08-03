"""
Visualize keyframes with all 4 camera views side by side,
plus TCP positions projected onto each view.

Output: For each keyframe, a single image with 4 panels:
  [head_camera] [front_camera]
  [left_camera] [right_camera]
Each panel shows the RGB image with left TCP (red) and right TCP (blue) projected.

Saves 3 episodes per task.
"""

import io
import json
import os
from glob import glob

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


CAMERAS = ["head_camera", "front_camera", "left_camera", "right_camera"]
TCP_RADIUS = 5
LEFT_TCP_COLOR = (255, 50, 50)    # red
RIGHT_TCP_COLOR = (50, 50, 255)   # blue
# Offset from EE link (wrist) to gripper fingertip center, along local x-axis.
# Stored endpose uses get_left_ee_pose() which has offset=0; the true TCP
# (get_left_tcp_pose) applies gripper_bias=0.12 along local x of the
# composite rotation = quat2mat(ee.q) @ global_trans_matrix @ delta_matrix.
# The stored quaternion already encodes this composite rotation (wxyz convention).
GRIPPER_OFFSET = 0.12


def _endpose_to_gripper_center(endpose_7d):
    """
    Convert stored endpose (wrist/EE link) to gripper fingertip center.
    
    Stored endpose = [x, y, z, w, qx, qy, qz]  (transforms3d wxyz convention).
    Gripper center = pos + R_composite @ [GRIPPER_OFFSET, 0, 0]
    where R_composite is recovered from the stored quaternion.
    """
    pos = endpose_7d[:3]
    quat_wxyz = endpose_7d[3:]  # [w, x, y, z] from transforms3d
    # Convert to scipy convention [x, y, z, w]
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    gripper_center = pos + R @ np.array([GRIPPER_OFFSET, 0, 0])
    return gripper_center


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


def project_tcp_to_pixel(tcp_xyz, intrinsic_cv, extrinsic_cv):
    """
    Project a 3D world point to 2D pixel coordinates.
    
    Args:
        tcp_xyz: (3,) world coordinates [x, y, z]
        intrinsic_cv: (3, 3) camera intrinsic matrix
        extrinsic_cv: (3, 4) camera extrinsic [R|t]
    
    Returns:
        (u, v) pixel coordinates, or None if behind camera
    """
    point_h = np.append(tcp_xyz, 1.0)  # [x, y, z, 1]
    p_cam = extrinsic_cv @ point_h     # (3,) in camera coords
    if p_cam[2] <= 0:
        return None  # behind camera
    p_pixel = intrinsic_cv @ p_cam
    u = p_pixel[0] / p_pixel[2]
    v = p_pixel[1] / p_pixel[2]
    return (u, v)


def draw_tcp_on_image(img_pil, left_tcp_uv, right_tcp_uv, cam_name):
    """Draw TCP markers and camera label on image."""
    draw = ImageDraw.Draw(img_pil)
    w, h = img_pil.size

    # Draw TCP points
    if left_tcp_uv is not None:
        u, v = int(left_tcp_uv[0]), int(left_tcp_uv[1])
        if 0 <= u < w and 0 <= v < h:
            draw.ellipse(
                [u - TCP_RADIUS, v - TCP_RADIUS, u + TCP_RADIUS, v + TCP_RADIUS],
                fill=LEFT_TCP_COLOR, outline=(255, 255, 255), width=1
            )
            draw.text((u + TCP_RADIUS + 2, v - 6), "L", fill=LEFT_TCP_COLOR)

    if right_tcp_uv is not None:
        u, v = int(right_tcp_uv[0]), int(right_tcp_uv[1])
        if 0 <= u < w and 0 <= v < h:
            draw.ellipse(
                [u - TCP_RADIUS, v - TCP_RADIUS, u + TCP_RADIUS, v + TCP_RADIUS],
                fill=RIGHT_TCP_COLOR, outline=(255, 255, 255), width=1
            )
            draw.text((u + TCP_RADIUS + 2, v - 6), "R", fill=RIGHT_TCP_COLOR)

    # Camera label
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except:
        font = ImageFont.load_default()
    draw.text((3, 3), cam_name, fill=(255, 255, 0), font=font)

    return img_pil


def create_multiview_image(hdf5_file, frame_idx, kf_info):
    """
    Create a 2x2 grid image showing all 4 camera views at a given frame,
    with TCP positions projected.
    """
    # Get TCP positions (correct wrist -> gripper center)
    left_ep = hdf5_file["endpose/left_endpose"][frame_idx]
    right_ep = hdf5_file["endpose/right_endpose"][frame_idx]
    left_tcp = _endpose_to_gripper_center(left_ep)
    right_tcp = _endpose_to_gripper_center(right_ep)

    panels = []
    for cam in CAMERAS:
        # Decode RGB
        raw = hdf5_file[f"observation/{cam}/rgb"][frame_idx]
        try:
            img = decode_rgb(raw)
        except:
            img = np.zeros((240, 320, 3), dtype=np.uint8)

        # Get camera parameters for this frame
        intrinsic = hdf5_file[f"observation/{cam}/intrinsic_cv"][frame_idx]  # (3,3)
        extrinsic = hdf5_file[f"observation/{cam}/extrinsic_cv"][frame_idx]  # (3,4)

        left_uv = project_tcp_to_pixel(left_tcp, intrinsic, extrinsic)
        right_uv = project_tcp_to_pixel(right_tcp, intrinsic, extrinsic)

        # Draw
        img_pil = Image.fromarray(img)
        img_pil = draw_tcp_on_image(img_pil, left_uv, right_uv, cam)
        panels.append(img_pil)

    # Compose 2x2 grid
    w, h = panels[0].size
    grid = Image.new("RGB", (w * 2, h * 2))
    grid.paste(panels[0], (0, 0))       # head_camera top-left
    grid.paste(panels[1], (w, 0))       # front_camera top-right
    grid.paste(panels[2], (0, h))       # left_camera bottom-left
    grid.paste(panels[3], (w, h))       # right_camera bottom-right

    # Add keyframe info bar at bottom
    info_h = 36
    final = Image.new("RGB", (w * 2, h * 2 + info_h), (30, 30, 30))
    final.paste(grid, (0, 0))

    draw = ImageDraw.Draw(final)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except:
        font = ImageFont.load_default()

    reason = kf_info.get("reason", "")
    lg = kf_info.get("left_gripper", -1)
    rg = kf_info.get("right_gripper", -1)
    lang = kf_info.get("language_annotation", "")
    n_frames = kf_info.get("n_frames", "?")

    line1 = f"frame={frame_idx}/{n_frames} | L_grip={lg:.2f} R_grip={rg:.2f} | {reason}"
    line2 = f"{lang[:100]}" if lang else ""
    draw.text((5, h * 2 + 2), line1, fill=(255, 255, 0), font=font)
    if line2:
        draw.text((5, h * 2 + 17), line2, fill=(200, 200, 200), font=font)

    return final


def main():
    data_root = "data/data"
    keyframe_root = "data/keyframes"
    output_root = "data/keyframe_vis_multiview"
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

        ep_names = sorted(kf_data.keys(), key=lambda x: int(x.replace("episode", "")))[:n_episodes]

        for ep_name in ep_names:
            ep_info = kf_data[ep_name]
            hdf5_path = os.path.join(data_root, task, "demo_clean", "data", f"{ep_name}.hdf5")
            if not os.path.exists(hdf5_path):
                print(f"  [SKIP] {hdf5_path}")
                continue

            ep_out_dir = os.path.join(output_root, task, ep_name)
            os.makedirs(ep_out_dir, exist_ok=True)

            hf = h5py.File(hdf5_path, "r")
            n_frames = ep_info["n_frames"]
            keyframes = ep_info["keyframes"]
            print(f"  {ep_name}: {len(keyframes)} keyframes")

            for kf_idx, kf in enumerate(keyframes):
                frame_idx = kf["frame_idx"]
                kf["n_frames"] = n_frames

                img = create_multiview_image(hf, frame_idx, kf)
                out_path = os.path.join(ep_out_dir, f"kf{kf_idx:02d}_frame{frame_idx:04d}.png")
                img.save(out_path)

            hf.close()

        print(f"  -> Saved to {os.path.join(output_root, task)}/")

    print(f"\nDone. Output: {output_root}/")


if __name__ == "__main__":
    main()
