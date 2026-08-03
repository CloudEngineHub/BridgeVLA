"""
GemBench-style keyframe sampling on RLBench-format MemoryBench data.

Why this dataset is structured this way
---------------------------------------
GemBench's training dataset (Gembench_Dataset) reads pre-computed keyframes
from an LMDB. Each sample is a SINGLE transition (obs at keyframe k -> action
at keyframe k+1). RLBench / sam2act, in contrast, sweep through every i-th
non-keyframe in a demo and pair it with the next keyframe -- which inflates
the dataset and emits multiple obs that share the same action target.

MemoryBench follows GemBench's sampling: one sample per
keyframe transition, no recurring keypoints. We get that by:

  1. Running peract's keypoint_discovery() on each demo on first build,
  2. Caching the keyframe-only observations (RGB / point-cloud / gripper /
     language goal) to disk as one .npz per episode,
  3. __getitem__ fetches one (k, k+1) pair from the cache.

The cache makes startup amortized, but the cost of building it is one-time:
we only have 100 episodes per task and the heuristic discovers ~10 keyframes
per episode, so the on-disk cache lands around 25 MB / task at IMAGE_SIZE=128.

The emitted sample shape mirrors Gembench_Dataset.__getitem__ exactly so that
bridgevla_agent.update_gembench / _build_memory_inputs_from_replay can be
reused unchanged.
"""

import json
import os
import pickle
import random
import time
from collections import OrderedDict
from typing import List, Optional, Sequence

import fcntl
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# Importing peract_colab pulls in PyRep (low_dim_obs.pkl uses rlbench.backend.observation.Observation), so
# callers need COPPELIASIM_ROOT set and libcoppeliaSim.so loadable; train.sh / run_*.sh handle this.
from peract_colab.rlbench.utils import get_stored_demo
from helpers.demo_loading_utils import keypoint_discovery


CACHE_VERSION = 3


def _resize_rgb(img: np.ndarray, target_hw: int) -> np.ndarray:
    """uint8 (H, W, 3) -> uint8 (target_hw, target_hw, 3) via PIL bilinear."""
    if img.shape[0] == target_hw and img.shape[1] == target_hw:
        return img
    pil = Image.fromarray(img)
    pil = pil.resize((target_hw, target_hw), resample=Image.BILINEAR)
    return np.asarray(pil)


def _resize_pcd(pcd: np.ndarray, target_hw: int) -> np.ndarray:
    """(H, W, 3) float32 pointcloud -> (target_hw, target_hw, 3).

    No-op at the default IMAGE_SIZE=128 since RLBench saves at 128 natively.
    Any upsampling here fabricates 3D points that the simulator never produced:
    bilinear blends across depth discontinuities and emits phantom intermediate-
    depth points that the MVT renderer splatters as visible streaks; nearest
    just duplicates source points. Recovering true sub-pixel geometry would
    require re-running CoppeliaSim at the target resolution. So when callers
    do request a larger size we use nearest as the strictly-safer of the two,
    but the canonical path keeps target_hw == source.
    """
    if pcd.shape[0] == target_hw and pcd.shape[1] == target_hw:
        return pcd.astype(np.float32, copy=False)
    t = torch.from_numpy(np.ascontiguousarray(pcd)).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(target_hw, target_hw), mode="nearest")
    return t.squeeze(0).permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def _safe_quat(q):
    """Ensure quaternion sign convention is stable (we keep the saved xyzw as-is;
    the agent's update_gembench expects xyzw). Just normalize against zero-norm
    edge cases.
    """
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-8:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        q = q / n
    return q


class MemoryBench_Dataset(Dataset):
    """
    Args:
        data_root: e.g. /.../memorybench/data/train. Each subdir is a task,
            with all_variations/episodes/episode<i>/ underneath.
        tasks: list of task names. None -> all subdirs of data_root that match
            the MemoryBench task set.
        ep_per_task: cap demos / task. The released dataset ships 100 / task.
        cache_dir: keyframe cache lives here. Built lazily.
        image_size: target H = W to resize RGB / PCD to. Default 128, matching
            peract_utils_memorybench.IMAGE_SIZE and the raw RLBench-format save
            resolution of MemoryBench episodes -- so resize is a no-op and the
            emitted PCD is exactly what the simulator produced (no fabricated
            intermediate-depth points). This matches sam2act / RLBench's
            convention. Setting this >128 invokes nearest-neighbor PCD upsample
            (see _resize_pcd) and is supported but discouraged.
        memory_enabled / memory_k_temporal: same semantics as Gembench_Dataset.
            When enabled, anchor (frame 0) + K most-recent history slots are
            attached to each sample.
        instructions_path: optional task -> [str] map; if not provided, falls
            back to the language goal stored in variation_descriptions.pkl.
    """

    def __init__(
        self,
        data_root: str,
        # Default matches train.py's --cameras default (and the released
        # keyframe cache): the cache's camera axis is position-indexed, so
        # builder and consumer must agree on this order.
        cameras: Sequence[str] = ("left_shoulder", "right_shoulder", "wrist", "front"),
        tasks: Optional[Sequence[str]] = None,
        ep_per_task: int = 100,
        cache_dir: Optional[str] = None,
        image_size: int = 128,
        memory_enabled: bool = False,
        memory_k_temporal: int = 2,
        memory_select: str = "keyframe_gt",
        instructions_path: Optional[str] = None,
        episode_cache_size: int = 4,
        verbose: bool = True,
    ):
        self.data_root = data_root
        self.cameras = list(cameras)
        self.ep_per_task = int(ep_per_task)
        self.image_size = int(image_size)
        self.memory_enabled = bool(memory_enabled)
        self.memory_k_temporal = int(memory_k_temporal)
        # Memory frame-selection policy (mirrors RMBench / the eval MemoryBank):
        #   "keyframe_gt" -> slots 0/1 pinned to the two most-recent EXECUTED keyframes; slots 2..K-1 hold
        #       discriminator-admitted GT subtask boundaries. MemoryBench has no boundary label, so those stay
        #       empty and the memory is exactly anchor + the two recency frames.
        #   "sliding"     -> LEGACY: K most-recent keyframes.
        self.memory_select = str(memory_select)
        self.episode_cache_size = int(episode_cache_size)
        self.verbose = verbose

        if tasks is None:
            # Auto-detect tasks: all subdirs that contain all_variations/episodes.
            tasks = []
            for name in sorted(os.listdir(data_root)):
                ep_dir = os.path.join(data_root, name, "all_variations", "episodes")
                if os.path.isdir(ep_dir):
                    tasks.append(name)
        self.tasks = list(tasks)

        # Per-task language goals override (optional).
        self.instructions_dict = None
        if instructions_path is not None and os.path.exists(instructions_path):
            with open(instructions_path, "r") as f:
                self.instructions_dict = json.load(f)

        # Cache layout: <cache_dir>/<task>/episode<idx>.npz with keys rgb, pcd, gripper_pose, gripper_open,
        # keyframes, lang_goal, variation, ignore_collisions (archived, not loaded). The cache dir is resolved
        # lazily so DDP workers agree on the path, and IMAGE_SIZE is part of it so changing it invalidates old caches.
        if cache_dir is None:
            # _v3: prepend frame 0 to keyframes (the GemBench convention). Older _v1 caches put the first
            # detected keyframe in slot 0, which mismatched eval (client.py step 0 starts from task.reset())
            # and silently dropped the (frame_0 -> keyframes[0]) transition. _v3 also archives per-action
            # ignore_collisions labels — still written, but no longer loaded or trained on.
            cache_dir = os.path.join(
                self.data_root, "_keyframe_cache", f"size{self.image_size}_v{CACHE_VERSION}",
            )
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # Per-worker LRU over the loaded npz tables: we read a whole episode at once and reuse it for adjacent step_idx.
        self._episode_cache: "OrderedDict[tuple, dict]" = OrderedDict()

        # Build (or load) the global flat index, fcntl-locked so concurrent DDP ranks don't duplicate the build.
        t0 = time.time()
        self.index = self._build_or_load_index()
        if self.verbose:
            print(f"[MemoryBench] Indexed {len(self.index)} keyframe transitions across "
                  f"{len(self.tasks)} tasks in {time.time()-t0:.1f}s")

        self.num_tasks = len(self.tasks)
        self.num_task_paths = sum(1 for e in self.index if e["step_idx"] == 0)

    # ---- index build ------------------------------------------------------

    def _index_cache_path(self) -> str:
        tag = ",".join(sorted(self.tasks)) + f"|ep{self.ep_per_task}|sz{self.image_size}|cv{CACHE_VERSION}"
        # MUST be stable across processes: Python's builtin hash() randomises its seed per process unless
        # PYTHONHASHSEED is pinned, so under DDP the ranks would compute DIFFERENT cache paths and the NCCL
        # allgather in DDP() would time out 10 minutes later.
        import hashlib
        digest = hashlib.md5(tag.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"_index_{digest}.json")

    def _build_or_load_index(self) -> List[dict]:
        index_path = self._index_cache_path()

        # Under DDP only rank 0 should build. fcntl.flock is unreliable on network filesystems (a silent no-op
        # on some NFS exports), so relying on it to serialise across ranks races: two ranks enter the build,
        # one truncates a .meta mid-write and the other reads it empty -> JSONDecodeError.
        try:
            import torch.distributed as dist
            ddp_active = dist.is_available() and dist.is_initialized()
        except Exception:
            ddp_active = False

        if ddp_active:
            if dist.get_rank() == 0:
                self._build_index_locked(index_path)
            else:
                # Rank 0 may take 20-40 min to build a fresh index, far beyond NCCL's 10-min watchdog, so
                # dist.barrier() here would kill the other ranks. Poll the on-disk index instead: rank 0
                # publishes it atomically (os.replace) at the very end, so its existence means the build is done.
                deadline = time.time() + 60 * 60 * 6  # 6h hard cap
                while not os.path.exists(index_path):
                    if time.time() > deadline:
                        raise RuntimeError(
                            f"[MemoryBench] rank {dist.get_rank()} timed out "
                            f"waiting for rank 0 to publish {index_path}; "
                            f"rank 0 likely crashed mid-build."
                        )
                    time.sleep(5.0)
            with open(index_path, "r") as f:
                return json.load(f)

        # Single-process / non-DDP: keep the fcntl-locked path.
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
                ep_root = os.path.join(self.data_root, task, "all_variations", "episodes")
                if not os.path.isdir(ep_root):
                    raise FileNotFoundError(
                        f"[MemoryBench] missing episode root {ep_root}. Silently "
                        f"skipping task '{task}' would shrink the dataset — fix "
                        f"the data layout or drop the task from `tasks`."
                    )
                ep_names = sorted(os.listdir(ep_root), key=_episode_sort_key)
                ep_names = ep_names[: self.ep_per_task]
                iterator = tqdm(ep_names, desc=f"[MemoryBench] index {task}") if self.verbose else ep_names
                for ep_name in iterator:
                    ep_idx = int(ep_name.replace("episode", ""))
                    ep_path = os.path.join(ep_root, ep_name)
                    cache_file = os.path.join(self.cache_dir, task, f"episode{ep_idx}.npz")
                    meta_file = cache_file + ".meta"
                    needs_rebuild = (
                        (not os.path.exists(cache_file))
                        or (not os.path.exists(meta_file))
                        or os.path.getsize(meta_file) == 0
                    )
                    meta = None
                    if not needs_rebuild:
                        try:
                            with open(meta_file, "r") as f:
                                meta = json.load(f)
                            needs_rebuild = int(meta.get("cache_version", 0)) != CACHE_VERSION
                            # Camera axis is position-indexed at __getitem__:
                            # a cache built with a different camera order would
                            # silently permute (mislabel) the views — rebuild.
                            if not needs_rebuild and meta.get("cameras") != list(self.cameras):
                                needs_rebuild = True
                        except Exception:
                            needs_rebuild = True
                    if needs_rebuild:
                        try:
                            self._build_episode_cache(task, ep_idx, ep_path, cache_file)
                        except Exception as e:
                            # Never skip-and-continue: a dropped episode would
                            # silently shrink the dataset.
                            raise RuntimeError(
                                f"[MemoryBench] FAILED to build keyframe cache for "
                                f"{task}/{ep_name} from {ep_path}: {e}. Fix or "
                                f"re-download the episode (or fetch the released "
                                f"cache: scripts/download_checkpoints_hf.sh "
                                f"memorybench_cache)."
                            ) from e
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                    assert meta is not None
                    num_kp = int(meta["num_keyframes"])
                    if num_kp < 2:
                        # need at least one (k -> k+1) transition.
                        continue
                    for step_idx in range(num_kp - 1):
                        index.append({
                            "task": task,
                            "ep_idx": ep_idx,
                            "step_idx": step_idx,
                            "num_steps": num_kp,
                        })

            tmp = index_path + f".tmp.{os.getpid()}"
            with open(tmp, "w") as f:
                json.dump(index, f)
            os.replace(tmp, index_path)
            return index

    def _build_episode_cache(self, task: str, ep_idx: int, ep_path: str, cache_file: str):
        """One episode -> one .npz with keyframe-only arrays + meta json."""
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        # get_stored_demo's ep_root expects the parent (episodes/), not the episode directory itself.
        ep_root = os.path.dirname(ep_path)

        demo = get_stored_demo(data_path=ep_root, index=ep_idx)
        detected_keyframes = keypoint_discovery(demo)
        if len(detected_keyframes) == 0:
            raise RuntimeError(f"keypoint_discovery returned empty for {task}/episode{ep_idx}")

        # GemBench convention (`key_frames.insert(0, 0)`): the saved frame list starts with demo frame 0 (the
        # env-reset state) followed by every detected keyframe. `keypoint_discovery` explicitly skips frame 0,
        # so without the prepend slot 0 would be the first stop frame — which would mis-pair step_idx=0, lose
        # the (frame_0 -> keyframes[0]) transition eval needs, and make client.py's step_id=0 query a
        # distribution the network never saw.
        keyframes = [0] + [int(k) for k in detected_keyframes]

        cam_keys = list(self.cameras)

        # One slot per (frame_0 + every detected keyframe); slot 0 is the env-reset frame and doubles as the memory anchor.
        H = W = self.image_size
        nk = len(keyframes)

        rgb_buf = np.zeros((nk, len(cam_keys), H, W, 3), dtype=np.uint8)
        pcd_buf = np.zeros((nk, len(cam_keys), H, W, 3), dtype=np.float32)
        gp_buf = np.zeros((nk, 8), dtype=np.float32)  # x,y,z, xyzw, grip_open
        ic_buf = np.zeros((nk,), dtype=np.float32)

        # Anchor = frame 0. After the prepend it equals rgb_buf[0] / pcd_buf[0], but the dedicated anchor_*
        # arrays are still written so __getitem__'s memory path stays unchanged.
        anchor_rgb = np.zeros((len(cam_keys), H, W, 3), dtype=np.uint8)
        anchor_pcd = np.zeros((len(cam_keys), H, W, 3), dtype=np.float32)

        def _pack_obs(obs):
            cams_rgb, cams_pcd = [], []
            for c in cam_keys:
                rgb = getattr(obs, f"{c}_rgb")
                pc = getattr(obs, f"{c}_point_cloud")
                rgb = _resize_rgb(rgb, self.image_size)
                pc = _resize_pcd(pc, self.image_size)
                cams_rgb.append(rgb)
                cams_pcd.append(pc)
            return np.stack(cams_rgb, 0), np.stack(cams_pcd, 0)

        # Frame 0 anchor.
        a_rgb, a_pcd = _pack_obs(demo[0])
        anchor_rgb[:] = a_rgb
        anchor_pcd[:] = a_pcd

        for i, kf in enumerate(keyframes):
            obs = demo[kf]
            rgbs, pcs = _pack_obs(obs)
            rgb_buf[i] = rgbs
            pcd_buf[i] = pcs
            gp = np.concatenate([
                np.asarray(obs.gripper_pose[:3], dtype=np.float32),
                _safe_quat(obs.gripper_pose[3:7]),
                np.asarray([float(obs.gripper_open)], dtype=np.float32),
            ])
            gp_buf[i] = gp
            # Archived only — never loaded, emitted or trained on. Kept so the raw per-keyframe GT stays
            # available without re-parsing low_dim_obs.pkl, and so existing caches need no rebuild.
            obs_tm1 = demo[max(0, int(kf) - 1)]
            ic_buf[i] = float(getattr(obs_tm1, "ignore_collisions", 1.0))

        with open(os.path.join(ep_path, "variation_descriptions.pkl"), "rb") as f:
            descs = pickle.load(f)
        if isinstance(descs, (list, tuple)):
            lang_goal = list(descs)
        elif isinstance(descs, str):
            lang_goal = [descs]
        else:
            lang_goal = [str(descs)]

        try:
            with open(os.path.join(ep_path, "variation_number.pkl"), "rb") as f:
                variation = int(pickle.load(f))
        except Exception:
            variation = 0

        # Atomic writes (tmp file + os.replace): an interrupted run would otherwise leave a half-written .npz
        # plus a 0-byte .meta, and the next run would skip the rebuild and hit a JSONDecodeError. The .meta is
        # written LAST, so "valid .npz + valid .meta" is the only consistent state.
        pid = os.getpid()
        tmp_npz = f"{cache_file}.tmp{pid}.npz"  # keep .npz suffix so np.savez_compressed doesn't append one
        np.savez_compressed(
            tmp_npz,
            rgb=rgb_buf,
            pcd=pcd_buf,
            gripper_pose=gp_buf,
            ignore_collisions=ic_buf,
            anchor_rgb=anchor_rgb,
            anchor_pcd=anchor_pcd,
        )
        os.replace(tmp_npz, cache_file)

        meta = {
            "task": task,
            "ep_idx": ep_idx,
            "num_keyframes": int(nk),
            "keyframes": [int(k) for k in keyframes],
            "lang_goal": lang_goal,
            "variation": variation,
            "cameras": cam_keys,
            "image_size": self.image_size,
            "cache_version": CACHE_VERSION,
        }
        meta_path = cache_file + ".meta"
        tmp_meta = f"{meta_path}.tmp{pid}"
        with open(tmp_meta, "w") as f:
            json.dump(meta, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_meta, meta_path)

    # ---- runtime IO -------------------------------------------------------

    def _load_episode(self, task: str, ep_idx: int) -> dict:
        key = (task, ep_idx)
        cached = self._episode_cache.get(key)
        if cached is not None:
            self._episode_cache.move_to_end(key)
            return cached
        cache_file = os.path.join(self.cache_dir, task, f"episode{ep_idx}.npz")
        meta_file = cache_file + ".meta"
        with open(meta_file, "r") as f:
            meta = json.load(f)
        # __getitem__ indexes the camera axis by POSITION against
        # self.cameras — a cache built with a different camera order would
        # silently deliver permuted (mislabeled) views. Hard error instead
        # (the index build already rebuilds mismatched caches, so this only
        # fires for caches swapped in behind a live dataset).
        if meta.get("cameras") != list(self.cameras):
            raise RuntimeError(
                f"[MemoryBench] camera-order mismatch for {task}/episode{ep_idx}: "
                f"cache built with {meta.get('cameras')} but dataset expects "
                f"{list(self.cameras)} — the cache would be silently permuted. "
                f"Delete the cache dir to rebuild, or fix `cameras`."
            )
        npz = np.load(cache_file)
        out = {
            "rgb": npz["rgb"],
            "pcd": npz["pcd"],
            "gripper_pose": npz["gripper_pose"],
            # npz["ignore_collisions"] deliberately NOT loaded — see the cache writer and predict_collision in the yaml.
            "anchor_rgb": npz["anchor_rgb"],
            "anchor_pcd": npz["anchor_pcd"],
            "meta": meta,
        }
        self._episode_cache[key] = out
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return out

    # ---- dataset API ------------------------------------------------------

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        task = entry["task"]
        step_idx = entry["step_idx"]
        num_steps = entry["num_steps"]
        ep = self._load_episode(task, entry["ep_idx"])

        cam_list = self.cameras
        # Episode arrays are stored channel-last (H, W, 3) but the model wants channel-first, so the transpose
        # happens at __getitem__ time — the on-disk cache stays small and worker CPU does the cheap permutation.
        sample = {}
        for cam_idx, cam in enumerate(cam_list):
            rgb = ep["rgb"][step_idx, cam_idx]              # (H, W, 3) uint8
            pcd = ep["pcd"][step_idx, cam_idx]              # (H, W, 3) float32
            sample[cam] = {
                "rgb": np.transpose(rgb, (2, 0, 1)),
                "pcd": np.transpose(pcd, (2, 0, 1)),
            }

        sample["gripper_pose"] = ep["gripper_pose"][step_idx + 1].astype(np.float32)
        # GemBench's time_feat formula:
        time_feat = (1.0 - (step_idx / float(max(1, num_steps - 1)))) * 2.0 - 1.0
        sample["low_dim_state"] = np.concatenate(
            [sample["gripper_pose"], np.asarray([time_feat], dtype=np.float32)]
        ).astype(np.float32)
        # NOTE: no "ignore_collisions" key. The demo-derived label is a fixed per-task template (a
        # (task, step_idx) lookup reproduces 99.97% of it), and MemoryBench eval ignores the collision slot
        # outright (collision_checking=False, 8-D action). Matching switch: predict_collision=false in the yaml.

        # Language goal: instructions_dict[task] if provided, else the variation_descriptions text from the demo.
        if self.instructions_dict is not None and task in self.instructions_dict:
            sample["lang_goal"] = random.choice(self.instructions_dict[task])
        else:
            sample["lang_goal"] = random.choice(ep["meta"]["lang_goal"])
        sample["tasks"] = task

        # ---- episodic memory bundle: anchor + K temporal slots ------------
        if self.memory_enabled:
            K = self.memory_k_temporal

            # Anchor = frame 0 of the demo. At step_idx == 0 the anchor arrays are still emitted but masked
            # off, matching Gembench_Dataset so the spatial block falls through to identity.
            for cam_idx, cam in enumerate(cam_list):
                a_rgb = ep["anchor_rgb"][cam_idx]
                a_pcd = ep["anchor_pcd"][cam_idx]
                sample[f"anchor_{cam}"] = {
                    "rgb": np.transpose(a_rgb, (2, 0, 1)),
                    "pcd": np.transpose(a_pcd, (2, 0, 1)),
                }
            sample["anchor_mask"] = np.array(step_idx > 0, dtype=bool)
            sample["step_idx"] = np.int64(step_idx)

            # Keyframe discriminator target. MemoryBench has no "is-subtask-boundary" label, so it is always 0
            # (the discriminator is disabled). Emitted for forward compatibility only.
            sample["mem_label"] = np.float32(0.0)

            def _emit_hist_slot(slot, src_step):
                for cam_idx, cam in enumerate(cam_list):
                    h_rgb = ep["rgb"][src_step, cam_idx]
                    h_pcd = ep["pcd"][src_step, cam_idx]
                    sample[f"hist{slot}_{cam}"] = {
                        "rgb": np.transpose(h_rgb, (2, 0, 1)),
                        "pcd": np.transpose(h_pcd, (2, 0, 1)),
                    }

            # Per-slot visual history only — ``action_proj`` was removed from MemoryBlock, so temporal memory
            # is purely visual + per-slot index PE.
            hist_mask = np.zeros((K,), dtype=bool)
            if self.memory_select == "keyframe_gt":
                # RMBench layout: slot 0 = t-1, slot 1 = t-2; slots 2..K-1 = GT subtask-boundary keyframes
                # older than t-2. MemoryBench has no boundary label, so those stay masked -> anchor + two recency frames.
                slot_src = [0] * K
                if step_idx - 1 >= 0:
                    slot_src[0] = step_idx - 1
                    hist_mask[0] = True
                if K >= 2 and step_idx - 2 >= 0:
                    slot_src[1] = step_idx - 2
                    hist_mask[1] = True
                # slots 2..K-1 never admitted (no subtask-boundary labels).
                for slot in range(K):
                    _emit_hist_slot(slot, slot_src[slot])
            else:
                # LEGACY sliding window: K most-recent keyframes.
                for k in range(K):
                    hist_step = step_idx - k - 1
                    valid = hist_step >= 0
                    _emit_hist_slot(k, hist_step if valid else 0)
                    if valid:
                        hist_mask[k] = True
            sample["hist_mask"] = hist_mask
        return sample


def _episode_sort_key(name: str) -> int:
    try:
        return int(name.replace("episode", ""))
    except ValueError:
        return 1 << 30


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get(
        "MEMORYBENCH_DATA_FOLDER",
        "data/bridgevla_data/memorybench/data/train"))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--ep_per_task", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--memory_enabled", action="store_true")
    parser.add_argument("--memory_k_temporal", type=int, default=2)
    parser.add_argument("--memory_select", type=str, default="keyframe_gt",
                        choices=["keyframe_gt", "sliding"])
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    ds = MemoryBench_Dataset(
        data_root=args.data_root,
        tasks=args.tasks,
        ep_per_task=args.ep_per_task,
        cache_dir=args.cache_dir,
        image_size=args.image_size,
        memory_enabled=args.memory_enabled,
        memory_k_temporal=args.memory_k_temporal,
        memory_select=args.memory_select,
    )
    print("dataset length:", len(ds))
    print("num tasks:", ds.num_tasks)
    print("num task paths:", ds.num_task_paths)
    s = ds[0]
    for k, v in s.items():
        if isinstance(v, dict):
            print(k, {kk: tuple(vv.shape) for kk, vv in v.items()})
        elif isinstance(v, np.ndarray):
            print(k, v.shape, v.dtype)
        else:
            print(k, v)
