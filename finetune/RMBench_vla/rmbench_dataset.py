"""
Dual-arm keyframe dataset for RMBench (RoboTwin 2.0 / SAPIEN aloha-agilex).

The keyframes are ALREADY extracted: each
``keyframe_data/<task>/keyframe_depth/data/episode<N>.hdf5`` contains ONLY the
keyframes (one slot per keyframe, slot 0 = frame 0), with 4 cameras — 2 static
world cameras (head, front) + the 2 per-arm wrist cameras (left, right); see
``utils/peract_utils_rmbench.CAMERAS`` — carrying JPEG-encoded RGB + millimeter
depth + per-frame OpenCV intrinsics/extrinsics (the wrist extrinsics change
every keyframe), and per-arm EE-link endposes + gripper values.

So unlike memoryBench (which runs keypoint_discovery + builds an npz cache),
this dataset reads the HDF5 directly. We keep only a lightweight JSON index of
``(task, ep, step_idx)`` transitions; per-item we decode RGB, unproject depth
to a world-frame point cloud, and compute the dual-arm GT (TCP-center target
per arm). The emitted sample mirrors the GemBench/memoryBench layout so the
dual-arm agent (``update_rmbench``) and the memory-bundle machinery
(``_build_memory_inputs_from_replay``) can be reused unchanged.

Sample layout (per camera ``c`` in CAMERAS):
    sample[c]            = {"rgb": (3,H,W) uint8->float later, "pcd": (3,H,W)}
    sample["left_action"]  = (8,) [tcp_xyz(3), quat_xyzw(4), grip(1)]
    sample["right_action"] = (8,)
    sample["lang_goal"]    = str
    sample["tasks"]        = str
  memory (when enabled):
    sample["anchor_<c>"]   = {"rgb","pcd"} for frame 0
    sample["anchor_mask"]  = bool (step_idx > 0)
    sample["hist{k}_<c>"]  = {"rgb","pcd"} for K temporal slots. With
                             select=keyframe_gt: slot 0/1 = the two most-
                             recent executed keyframes (t-1, t-2), slots
                             2..K-1 = subtask-boundary memory keyframes
                             (<= t-3, newest first). With select=sliding:
                             slot k = k-th most recent keyframe.
    sample["hist_mask"]    = (K,) bool
    sample["step_idx"]     = int64

"""

import hashlib
import io
import json
import os
import random
import time
from collections import OrderedDict
from typing import List, Optional, Sequence

import fcntl
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils.peract_utils_rmbench import (
    CAMERAS,
    CAMERA_HDF5_NAMES,
    IMAGE_SIZE,
    depth_to_world_pcd,
    endpose_to_tcp_center,
    world_to_render_pos,
)


CACHE_VERSION = 1

# Per-task minimum subtask length (in KEYFRAMES) for a subtask to contribute a
# memory keyframe. A subtask whose keyframe count is < this threshold is
# SKIPPED: none of its keyframes get mem_label=1, so it is neither a
# discriminator positive nor admitted into temporal memory. Tasks absent from
# this dict use 0 (every subtask's last keyframe counts, original behavior).
#
# blocks_ranking_try is the long-horizon outlier: its subtasks are either large
# (~18 keyframes) or tiny "nudge" segments (2-3 keyframes). Dropping the tiny
# ones (<5) cuts per-episode memory keyframes from <=11 down to <=5, which
# bounds the temporal-memory width (and the PaliGemma activation peak) without
# losing the semantically meaningful subtask boundaries.
SUBTASK_MIN_KEYFRAMES = {
    "blocks_ranking_try": 5,
}

# Tasks where CONSECUTIVE keyframes carrying the SAME language_annotation text
# are nonetheless DISTINCT subtasks. The default subtask grouping merges
# contiguous same-text keyframes into one subtask -- correct for tasks where a
# repeated sentence is just multiple motion phases of ONE action (e.g.
# put_back_block's "Pick up the block..." split across grasp/lift/place moves).
# But it is WRONG for counting tasks like press_button, where "Press the middle
# button one time." repeated N times means N DISTINCT presses and the COUNT is
# the whole point. For tasks listed here we group by the ORIGINAL language
# segment index (``subtask_idx`` baked into the keyframes json) instead of text
# identity, so each repeated action becomes its own subtask boundary. All other
# tasks keep text-identity grouping unchanged. Missing subtask_idx raises at
# dataset init — there is no silent text-based fallback.
SUBTASK_BY_SEGMENT_TASKS = {"press_button"}


def _decode_rgb(raw) -> np.ndarray:
    """Decode a single HDF5 RGB entry (JPEG bytes) to (H, W, 3) uint8."""
    if isinstance(raw, bytes):
        return np.array(Image.open(io.BytesIO(raw)))
    if isinstance(raw, np.ndarray) and raw.dtype.kind == "S":
        return np.array(Image.open(io.BytesIO(raw.tobytes())))
    if isinstance(raw, np.void):
        return np.array(Image.open(io.BytesIO(bytes(raw))))
    return np.array(raw)


def _resize_rgb(img: np.ndarray, target_hw: int) -> np.ndarray:
    if img.shape[0] == target_hw and img.shape[1] == target_hw:
        return img
    pil = Image.fromarray(img).resize((target_hw, target_hw), resample=Image.BILINEAR)
    return np.asarray(pil)


def _resize_pcd(pcd: np.ndarray, target_hw: int) -> np.ndarray:
    """(H, W, 3) world XYZ -> (target_hw, target_hw, 3) via NEAREST.

    Nearest avoids fabricating phantom intermediate-depth points across depth
    discontinuities (bilinear would blend a near and a far surface). The
    splat renderer projects to mvt.img_size regardless, so the only effect of
    downsampling here is point density.
    """
    if pcd.shape[0] == target_hw and pcd.shape[1] == target_hw:
        return pcd.astype(np.float32, copy=False)
    t = torch.from_numpy(np.ascontiguousarray(pcd)).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(target_hw, target_hw), mode="nearest")
    return t.squeeze(0).permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def _episode_sort_key(name: str) -> int:
    try:
        return int(name.replace("episode", "").replace(".hdf5", ""))
    except ValueError:
        return 1 << 30


class RMBench_Dataset(Dataset):
    """Direct-from-HDF5 dual-arm keyframe transition dataset.

    Args:
        data_root: ``.../keyframe_data`` (each subdir is a task with
            ``keyframe_depth/data/episode<N>.hdf5``).
        cameras: world camera names (default CAMERAS = head/front/left/right).
        tasks: task names; None -> all subdirs that contain keyframe_depth.
        ep_per_task: cap episodes per task.
        image_size: resize target for RGB + projected PCD grid.
        memory_enabled / memory_k_temporal: anchor (frame 0) + K most-recent
            history slots, mirroring memoryBench semantics.
        instructions_dir: ``description/task_instruction`` — per-task JSON with
            a ``seen`` list; a random entry is the language goal.
        index_cache_dir: where to drop the JSON transition index (DDP-safe).
    """

    def __init__(
        self,
        data_root: str,
        cameras: Sequence[str] = tuple(CAMERAS),
        tasks: Optional[Sequence[str]] = None,
        ep_per_task: int = 100,
        image_size: int = IMAGE_SIZE,
        memory_enabled: bool = False,
        memory_k_temporal: int = 4,
        memory_select: str = "keyframe_gt",
        keyframes_dir: Optional[str] = None,
        instructions_dir: Optional[str] = None,
        index_cache_dir: Optional[str] = None,
        episode_cache_size: int = 4,
        verbose: bool = True,
    ):
        self.data_root = data_root
        self.cameras = list(cameras)
        self.ep_per_task = int(ep_per_task)
        self.image_size = int(image_size)
        self.memory_enabled = bool(memory_enabled)
        self.memory_k_temporal = int(memory_k_temporal)
        # Memory frame-selection policy:
        #   "keyframe_gt" -> NEW: slot 0/1 pinned to the two most-recent
        #       EXECUTED keyframes (t-1, t-2); slots 2..K-1 hold the GT
        #       "last keyframe of each completed subtask" (label==1)
        #       seen STRICTLY before the current step (slot 0 = most recent,
        #       capped at memory_k_temporal = M). A discriminator (trained on
        #       mem_label) predicts these at inference. No sliding window.
        #   "sliding"     -> LEGACY: K most-recent keyframes (original behavior).
        self.memory_select = str(memory_select)
        self.episode_cache_size = int(episode_cache_size)
        self.verbose = verbose

        if tasks is None:
            tasks = []
            for name in sorted(os.listdir(data_root)):
                if os.path.isdir(os.path.join(data_root, name, "keyframe_depth", "data")):
                    tasks.append(name)
        self.tasks = list(tasks)

        # Per-task language goals. REQUIRED: language conditioning is core —
        # a missing instructions dir / task json / empty "seen" list must fail
        # loudly instead of silently degrading lang_goal to the bare task name.
        self.instructions_dict = {}
        if instructions_dir is None or not os.path.isdir(instructions_dir):
            raise FileNotFoundError(
                f"[RMBench] instructions_dir missing or not a directory: "
                f"{instructions_dir!r}. Expected the per-task instruction jsons "
                f"shipped in-repo at finetune/RMBench/description/task_instruction/."
            )
        _no_instr = []
        for t in self.tasks:
            ti = os.path.join(instructions_dir, f"{t}.json")
            seen = []
            if os.path.exists(ti):
                with open(ti) as f:
                    d = json.load(f)
                seen = d.get("seen", []) or []
            if seen:
                self.instructions_dict[t] = list(seen)
            else:
                _no_instr.append(ti)
        if _no_instr:
            raise FileNotFoundError(
                f"[RMBench] no usable 'seen' instructions for {len(_no_instr)} "
                f"task(s): " + ", ".join(_no_instr) + ". Language conditioning "
                f"would silently degrade to bare task names — refusing to run."
            )

        # Per-task keyframe metadata (frame_idx + language_annotation per
        # keyframe). Used to derive ``mem_label`` -- the GT "last keyframe of
        # each subtask" -- for the discriminator + keyframe_gt memory policy.
        # Each <keyframes_dir>/<task>.json maps "episode<N>" -> {keyframes: [...]}.
        # REQUIRED for every task when memory is on: a missing json would
        # silently zero mem_label (memory supervision off) — hard error instead.
        self._keyframes = {}
        if self.memory_enabled:
            if keyframes_dir is None:
                keyframes_dir = os.path.join(
                    os.path.dirname(os.path.normpath(self.data_root)), "keyframes"
                )
            self.keyframes_dir = keyframes_dir
            for t in self.tasks:
                kf_path = os.path.join(keyframes_dir, f"{t}.json")
                if not os.path.exists(kf_path):
                    raise FileNotFoundError(
                        f"[RMBench] memory_enabled but keyframes metadata is "
                        f"missing for task '{t}': {kf_path}. Without it "
                        f"mem_label would silently become all-zero and the "
                        f"memory discriminator would train on no positives. "
                        f"Fetch the released copy (scripts/download_checkpoints_hf.sh "
                        f"rmbench_data -> RMBench/data/keyframes/) or regenerate "
                        f"with finetune/RMBench/script/extract_key_frames/"
                        f"extract_keyframes.py."
                    )
                with open(kf_path) as f:
                    self._keyframes[t] = json.load(f)
            for t in self.tasks:
                if t in SUBTASK_BY_SEGMENT_TASKS:
                    self._validate_segment_keyframes(t)

        if index_cache_dir is None:
            index_cache_dir = os.path.join(self.data_root, "_vla_index")
        self.index_cache_dir = index_cache_dir
        os.makedirs(self.index_cache_dir, exist_ok=True)

        self._episode_cache: "OrderedDict[tuple, dict]" = OrderedDict()

        t0 = time.time()
        self.index = self._build_or_load_index()
        if self.verbose:
            print(f"[RMBench] Indexed {len(self.index)} keyframe transitions across "
                  f"{len(self.tasks)} tasks in {time.time()-t0:.1f}s")

        self.num_tasks = len(self.tasks)
        self.num_task_paths = sum(1 for e in self.index if e["step_idx"] == 0)

    # ---- episode hdf5 path ------------------------------------------------

    def _ep_path(self, task: str, ep_idx: int) -> str:
        return os.path.join(
            self.data_root, task, "keyframe_depth", "data", f"episode{ep_idx}.hdf5"
        )

    # ---- index build (DDP-safe; md5, never builtin hash) ------------------

    def _index_cache_path(self) -> str:
        tag = (",".join(sorted(self.tasks))
               + f"|ep{self.ep_per_task}|sz{self.image_size}|cv{CACHE_VERSION}")
        digest = hashlib.md5(tag.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.index_cache_dir, f"_index_{digest}.json")

    def _build_or_load_index(self) -> List[dict]:
        index_path = self._index_cache_path()
        try:
            import torch.distributed as dist
            ddp_active = dist.is_available() and dist.is_initialized()
        except Exception:
            ddp_active = False

        if ddp_active:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                self._build_index_locked(index_path)
            else:
                deadline = time.time() + 60 * 60 * 2
                while not os.path.exists(index_path):
                    if time.time() > deadline:
                        raise RuntimeError(
                            f"[RMBench] rank {dist.get_rank()} timed out waiting "
                            f"for rank 0 to publish {index_path}"
                        )
                    time.sleep(3.0)
            with open(index_path, "r") as f:
                return json.load(f)

        return self._build_index_locked(index_path)

    def _build_index_locked(self, index_path: str) -> List[dict]:
        if os.path.exists(index_path):
            with open(index_path, "r") as f:
                return json.load(f)

        lock_path = index_path + ".lock"
        with open(lock_path, "w") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    return json.load(f)

            index = []
            for task in self.tasks:
                ddir = os.path.join(self.data_root, task, "keyframe_depth", "data")
                if not os.path.isdir(ddir):
                    raise FileNotFoundError(
                        f"[RMBench] missing keyframe data dir for task "
                        f"'{task}': {ddir}. Silently skipping a task would "
                        f"shrink the dataset — fetch the released keyframe "
                        f"data (scripts/download_checkpoints_hf.sh rmbench_data) "
                        f"or drop the task from `tasks`."
                    )
                ep_names = sorted(
                    [f for f in os.listdir(ddir) if f.endswith(".hdf5")],
                    key=_episode_sort_key,
                )[: self.ep_per_task]
                if not ep_names:
                    raise FileNotFoundError(
                        f"[RMBench] no episode .hdf5 under {ddir}")
                for ep_name in ep_names:
                    ep_idx = _episode_sort_key(ep_name)
                    try:
                        with h5py.File(os.path.join(ddir, ep_name), "r") as f:
                            n_kf = int(f["endpose/left_endpose"].shape[0])
                    except Exception as e:
                        raise RuntimeError(
                            f"[RMBench] unreadable HDF5 {ddir}/{ep_name}: {e}. "
                            f"Silently dropping episodes would shrink the "
                            f"dataset — re-download or re-render the episode."
                        ) from e
                    if n_kf < 2:
                        raise RuntimeError(
                            f"[RMBench] {ddir}/{ep_name} holds {n_kf} keyframe(s); "
                            f"a valid episode has >= 2 (start + end) — corrupt "
                            f"render, re-generate it."
                        )
                    for step_idx in range(n_kf - 1):
                        index.append({
                            "task": task,
                            "ep_idx": ep_idx,
                            "step_idx": step_idx,
                            "num_steps": n_kf,
                        })

            tmp = index_path + f".tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(index, f)
            os.replace(tmp, index_path)
            return index

    # ---- runtime IO -------------------------------------------------------

    def _load_episode(self, task: str, ep_idx: int) -> dict:
        """Load + decode all keyframes of one episode (cached, LRU)."""
        key = (task, ep_idx)
        cached = self._episode_cache.get(key)
        if cached is not None:
            self._episode_cache.move_to_end(key)
            return cached

        path = self._ep_path(task, ep_idx)
        H = W = self.image_size
        with h5py.File(path, "r") as f:
            T = int(f["endpose/left_endpose"].shape[0])
            n_cam = len(self.cameras)
            rgb_buf = np.zeros((T, n_cam, H, W, 3), dtype=np.uint8)
            pcd_buf = np.zeros((T, n_cam, H, W, 3), dtype=np.float32)
            for ci, cam in enumerate(self.cameras):
                g = CAMERA_HDF5_NAMES[cam]
                rgb_ds = f[f"observation/{g}/rgb"]
                depth_ds = f[f"observation/{g}/depth"]
                K_ds = f[f"observation/{g}/intrinsic_cv"]
                E_ds = f[f"observation/{g}/extrinsic_cv"]
                for t in range(T):
                    rgb = _decode_rgb(rgb_ds[t])
                    rgb_buf[t, ci] = _resize_rgb(rgb, self.image_size)
                    pc = depth_to_world_pcd(depth_ds[t], K_ds[t], E_ds[t])
                    # World -> render frame (+90 deg about Z) so the shared
                    # tri-view renderer's front/right cameras align with
                    # RMBench's depth / left-right axes. Origin (invalid
                    # depth) maps to origin and is still culled downstream.
                    pc = world_to_render_pos(pc)
                    pcd_buf[t, ci] = _resize_pcd(pc, self.image_size)

            # Per-arm GT actions: TCP-center xyz + quat_xyzw + grip 0/1.
            left_ep = f["endpose/left_endpose"][:]
            right_ep = f["endpose/right_endpose"][:]
            left_grip = f["endpose/left_gripper"][:]
            right_grip = f["endpose/right_gripper"][:]

        # Binarize the gripper to {0=closed, 1=open}. Keyframe values are
        # three-valued: 1.0=open, 0.0=closed, 0.2=closed on a thin object
        # (swap_T). Threshold 0.9 so only a near-fully-open gripper is open;
        # anything holding an object stays closed.
        GRIP_OPEN_THRESH = 0.9
        left_act = np.zeros((T, 8), dtype=np.float32)
        right_act = np.zeros((T, 8), dtype=np.float32)
        for t in range(T):
            l_tcp, l_quat = endpose_to_tcp_center(left_ep[t])
            r_tcp, r_quat = endpose_to_tcp_center(right_ep[t])
            # GT target POSITION goes to the render frame to match the
            # rotated point cloud; ORIENTATION (quat_xyzw) stays in the world
            # frame (the static Rz90 commutes with the yaw-only SE3 aug, so
            # the rot head learns the constant offset and the predicted quat
            # is execution-ready). See peract_utils_rmbench.world_to_render_pos.
            l_tcp = world_to_render_pos(l_tcp)
            r_tcp = world_to_render_pos(r_tcp)
            l_g = np.float32(left_grip[t] > GRIP_OPEN_THRESH)
            r_g = np.float32(right_grip[t] > GRIP_OPEN_THRESH)
            left_act[t] = np.concatenate([l_tcp, l_quat, [l_g]])
            right_act[t] = np.concatenate([r_tcp, r_quat, [r_g]])

        out = {
            "rgb": rgb_buf,           # (T, n_cam, H, W, 3) uint8
            "pcd": pcd_buf,           # (T, n_cam, H, W, 3) float32
            "left_action": left_act,  # (T, 8)
            "right_action": right_act,
            "num_steps": T,
        }
        if self.memory_enabled:
            out["mem_label"] = self._compute_mem_label(task, ep_idx, T)
        self._episode_cache[key] = out
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return out

    # ---- memory keyframe labels ------------------------------------------

    def _validate_segment_keyframes(self, task: str) -> None:
        """Fail fast if a segment-grouped task lacks ``subtask_idx`` in keyframes json."""
        kf_task = self._keyframes.get(task)
        if not kf_task:
            kf_path = os.path.join(self.keyframes_dir, f"{task}.json")
            raise FileNotFoundError(
                f"[RMBench] Task '{task}' is in SUBTASK_BY_SEGMENT_TASKS but no "
                f"keyframes json was loaded from {kf_path}."
            )
        bad_eps = []
        for ep_name, info in kf_task.items():
            kfs = info.get("keyframes", [])
            n_missing = sum(1 for k in kfs if k.get("subtask_idx") is None)
            if n_missing:
                bad_eps.append((ep_name, n_missing, len(kfs)))
        if not bad_eps:
            return
        kf_path = os.path.join(self.keyframes_dir, f"{task}.json")
        detail = ", ".join(
            f"{ep} ({miss}/{total} keyframes missing subtask_idx)"
            for ep, miss, total in bad_eps[:5]
        )
        if len(bad_eps) > 5:
            detail += f", ... and {len(bad_eps) - 5} more episodes"
        raise ValueError(
            f"[RMBench] Task '{task}' is in SUBTASK_BY_SEGMENT_TASKS and requires "
            f"'subtask_idx' on every keyframe in {kf_path}, but it is missing in "
            f"{len(bad_eps)} episode(s): {detail}. "
            f"Text-based subtask grouping would merge repeated identical language "
            f"segments and break counting memory. Regenerate keyframes with "
            f"finetune/RMBench/script/extract_key_frames/extract_keyframes.py "
            f"(language_annotation.json must be present)."
        )

    def _compute_mem_label(self, task: str, ep_idx: int, T: int) -> np.ndarray:
        """Per-keyframe binary label: 1 == "last keyframe of a subtask".

        Derived from ``keyframes/<task>.json``: each keyframe carries a
        ``language_annotation`` (the subtask language segment). Keyframes are
        grouped into subtasks by contiguous runs of identical annotation; the
        LAST keyframe of each subtask gets label 1 (the others 0).

        Exception: for tasks in ``SUBTASK_BY_SEGMENT_TASKS`` (e.g. press_button)
        keyframes are grouped by the ORIGINAL language segment index
        (``subtask_idx`` in the json) instead of by text identity, so repeated
        identical-text actions (each button press) each count as a distinct
        subtask. Missing ``subtask_idx`` raises ``ValueError`` at dataset init
        (see ``_validate_segment_keyframes``); there is no text-based fallback.

        Short-subtask filtering: a subtask whose keyframe count is below
        ``SUBTASK_MIN_KEYFRAMES[task]`` (0 if the task is absent) is SKIPPED
        entirely — none of its keyframes get label 1. This drops tiny "nudge"
        segments (e.g. blocks_ranking_try's 2-3 keyframe subtasks) so they are
        neither discriminator positives nor admitted into temporal memory.
        Note: because of this filter the episode's final keyframe is NOT
        force-labelled 1 anymore; it counts only if its own subtask is long
        enough.

        Alignment: the training HDF5 stores one slot per keyframe in
        ``sorted(frame_idx)`` order (see collect_keyframe_depth.py), so we
        sort the json keyframes by ``frame_idx`` and index slot k == the
        k-th unique frame_idx. A missing episode entry or a keyframe count
        that does not match T is a HARD ERROR: it means the keyframes json
        and the HDF5 data are out of sync, and degrading to all-zero labels
        would silently switch off memory supervision for that episode.
        """
        lbl = np.zeros((T,), dtype=np.int64)
        kf_task = self._keyframes.get(task)
        if not kf_task:
            raise RuntimeError(
                f"[RMBench] no keyframes metadata loaded for task '{task}' "
                f"(memory_enabled) — __init__ should have raised earlier."
            )
        info = kf_task.get(f"episode{ep_idx}")
        if not info:
            kf_path = os.path.join(self.keyframes_dir, f"{task}.json")
            raise KeyError(
                f"[RMBench] episode{ep_idx} of task '{task}' has HDF5 data but "
                f"no entry in {kf_path} — keyframes json out of sync with the "
                f"data. Regenerate with finetune/RMBench/script/"
                f"extract_key_frames/extract_keyframes.py."
            )
        kfs = info.get("keyframes", [])
        # Subtask grouping key per keyframe. Default = language text. For
        # SUBTASK_BY_SEGMENT_TASKS use the original segment index so repeated
        # identical-text actions stay distinct (validated at __init__).
        use_seg = task in SUBTASK_BY_SEGMENT_TASKS
        if use_seg and not all(k.get("subtask_idx") is not None for k in kfs):
            kf_path = os.path.join(self.keyframes_dir, f"{task}.json")
            missing = sum(1 for k in kfs if k.get("subtask_idx") is None)
            raise ValueError(
                f"[RMBench] episode{ep_idx} in {kf_path}: {missing}/{len(kfs)} "
                f"keyframes missing 'subtask_idx' for segment-grouped task "
                f"'{task}'. Regenerate keyframes with "
                f"finetune/RMBench/script/extract_key_frames/extract_keyframes.py."
            )
        # Map frame_idx -> grouping key; align to HDF5 slots via sorted idx.
        fi2key = {}
        for k in kfs:
            fi = k.get("frame_idx")
            if fi is not None:
                fi2key[fi] = k.get("subtask_idx") if use_seg \
                    else k.get("language_annotation")
        uniq = sorted(fi2key.keys())
        if len(uniq) != T:
            kf_path = os.path.join(self.keyframes_dir, f"{task}.json")
            raise ValueError(
                f"[RMBench] {task} episode{ep_idx}: keyframe count in json "
                f"({len(uniq)}) != HDF5 slots ({T}) — {kf_path} is out of sync "
                f"with the keyframe_data HDF5. mem_label would silently become "
                f"all-zero. Regenerate the json (extract_keyframes.py) or "
                f"re-render the episode (collect/single/run_keyframe_depth.sh)."
            )
        keys = [fi2key[fi] for fi in uniq]
        min_kf = int(SUBTASK_MIN_KEYFRAMES.get(task, 0))
        # Walk contiguous same-key runs (subtasks). Mark each run's last
        # keyframe 1, unless the run is shorter than the task's min length.
        start = 0
        for k in range(T):
            if k == T - 1 or keys[k] != keys[k + 1]:
                run_len = k - start + 1
                if run_len >= min_kf:
                    lbl[k] = 1
                start = k + 1
        return lbl

    # ---- multi-task sampling balance -------------------------------------

    def task_sampling_weights(self, alpha: float) -> np.ndarray:
        """Per-index sampling weights for temperature task balancing.

        The flat ``self.index`` has one entry per (task, ep, step) transition,
        so a uniform sampler draws each TASK with probability proportional to
        its transition count ``n_t`` (long-horizon tasks dominate). Temperature
        sampling targets task probability ``p_t ∝ n_t**alpha``; since a task
        owns ``n_t`` index entries, the per-entry weight is
        ``w_i ∝ n_{t(i)}**(alpha-1)``:

          * alpha=1 -> w_i ∝ 1            (transition-uniform, original)
          * alpha=0 -> w_i ∝ 1/n_t        (task-uniform)
          * alpha=0.5 -> w_i ∝ 1/sqrt(n_t)(tempered middle ground)

        Weights are NOT normalized (``torch.multinomial`` normalizes), and the
        per-task counts come straight from the in-memory index (no extra IO).
        """
        from collections import Counter
        counts = Counter(e["task"] for e in self.index)
        exp = float(alpha) - 1.0
        return np.array(
            [counts[e["task"]] ** exp for e in self.index], dtype=np.float64
        )

    def task_transition_counts(self) -> "OrderedDict[str, int]":
        """Ordered {task: transition_count} over the current index (for logs)."""
        out: "OrderedDict[str, int]" = OrderedDict()
        for t in self.tasks:
            out[t] = 0
        for e in self.index:
            out[e["task"]] = out.get(e["task"], 0) + 1
        return out

    # ---- dataset API ------------------------------------------------------

    def __len__(self):
        return len(self.index)

    def _cam_dict(self, ep, slot, cam_idx):
        rgb = ep["rgb"][slot, cam_idx]                 # (H, W, 3) uint8
        pcd = ep["pcd"][slot, cam_idx]                 # (H, W, 3) float32
        return {
            "rgb": np.transpose(rgb, (2, 0, 1)),       # (3, H, W)
            "pcd": np.transpose(pcd, (2, 0, 1)),
        }

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        task = entry["task"]
        step_idx = entry["step_idx"]
        num_steps = entry["num_steps"]
        ep = self._load_episode(task, entry["ep_idx"])

        sample = {}
        for ci, cam in enumerate(self.cameras):
            sample[cam] = self._cam_dict(ep, step_idx, ci)

        # GT actions come from the NEXT keyframe (k -> k+1 transition).
        sample["left_action"] = ep["left_action"][step_idx + 1].astype(np.float32)
        sample["right_action"] = ep["right_action"][step_idx + 1].astype(np.float32)

        # __init__ guarantees every task has instructions; a KeyError here
        # means construction was bypassed — let it propagate loudly.
        sample["lang_goal"] = random.choice(self.instructions_dict[task])
        sample["tasks"] = task

        if self.memory_enabled:
            K = self.memory_k_temporal       # M = max memory slots
            for ci, cam in enumerate(self.cameras):
                sample[f"anchor_{cam}"] = self._cam_dict(ep, 0, ci)
            sample["anchor_mask"] = np.array(step_idx > 0, dtype=bool)
            sample["step_idx"] = np.int64(step_idx)

            mem_label = ep.get("mem_label")
            # Discriminator target: is the CURRENT keyframe a subtask boundary?
            sample["mem_label"] = np.float32(
                mem_label[step_idx] if mem_label is not None else 0.0
            )

            if self.memory_select == "keyframe_gt":
                # NEW layout (mirrors the eval MemoryBank with select=keyframe_gt):
                #   slot 0 = previous EXECUTED keyframe (t-1)      -- always
                #   slot 1 = step-before-previous executed kf (t-2) -- always
                #   slots 2..K-1 = GT subtask-boundary "memory keyframes"
                #       strictly older than t-2 (j <= step_idx-3), most-recent
                #       first, capped at (K-2).
                # The 2-step gap means a keyframe flagged as a subtask boundary
                # only enters the memory portion once it has scrolled out of
                # the two recency slots, so it never appears twice.
                hist_mask = np.zeros((K,), dtype=bool)
                slot_src = [0] * K            # frame index per slot (0 = placeholder)

                # Recency slots: the two most-recent executed keyframes.
                if step_idx - 1 >= 0:
                    slot_src[0] = step_idx - 1
                    hist_mask[0] = True
                if K >= 2 and step_idx - 2 >= 0:
                    slot_src[1] = step_idx - 2
                    hist_mask[1] = True

                # Memory-keyframe slots: subtask boundaries older than t-2.
                if mem_label is not None:
                    sel = [j for j in range(max(0, step_idx - 2)) if mem_label[j] == 1]
                else:
                    sel = []
                n_mem_slots = max(0, K - 2)
                sel = sel[-n_mem_slots:] if n_mem_slots > 0 else []
                for i in range(len(sel)):
                    slot = 2 + i
                    slot_src[slot] = sel[-(i + 1)]   # slot 2 = most recent
                    hist_mask[slot] = True

                for slot in range(K):
                    j = slot_src[slot]
                    for ci, cam in enumerate(self.cameras):
                        sample[f"hist{slot}_{cam}"] = self._cam_dict(ep, j, ci)
                sample["hist_mask"] = hist_mask
            else:
                # LEGACY sliding window: K most-recent keyframes.
                hist_mask = np.zeros((K,), dtype=bool)
                for k in range(K):
                    hist_step = step_idx - k - 1
                    valid = hist_step >= 0
                    idx_step = hist_step if valid else 0
                    for ci, cam in enumerate(self.cameras):
                        sample[f"hist{k}_{cam}"] = self._cam_dict(ep, idx_step, ci)
                    if valid:
                        hist_mask[k] = True
                sample["hist_mask"] = hist_mask

        return sample


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=os.environ.get(
            "RMBENCH_VLA_DATA_ROOT",
            "data/bridgevla_data/RMBench/data/keyframe_data",
        ),
    )
    parser.add_argument("--ep_per_task", type=int, default=2)
    parser.add_argument("--memory_enabled", action="store_true")
    parser.add_argument("--memory_k_temporal", type=int, default=4)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--instructions_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "RMBench", "description", "task_instruction",
        ),
    )
    args = parser.parse_args()

    ds = RMBench_Dataset(
        data_root=args.data_root,
        tasks=args.tasks,
        ep_per_task=args.ep_per_task,
        memory_enabled=args.memory_enabled,
        memory_k_temporal=args.memory_k_temporal,
        instructions_dir=args.instructions_dir,
    )
    print("dataset length:", len(ds))
    print("num tasks:", ds.num_tasks, "num task paths:", ds.num_task_paths)
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, dict):
            print(k, {kk: tuple(vv.shape) for kk, vv in v.items()})
        elif isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        else:
            print(k, v)
