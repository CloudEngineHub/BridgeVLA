"""
Keyframe-only dataset for RLBench (GemBench-style sampling) with episodic memory.

Why this replaces the YARR replay-buffer pipeline
--------------------------------------------------
The original RLBench data path (formerly ``utils/dataset.py`` +
``utils/get_dataset.py``, since removed) built a PerAct/RVT-style YARR replay
buffer: it swept every ``demo_augmentation_every_n`` non-keyframe in a demo
and, for each, appended the ENTIRE remaining keyframe chain as separate
transitions. That inflated the dataset with many redundant / low-value
samples (the same target keyframe appears paired with many different
"current" frames, and the obs is usually a mid-motion non-keyframe) and made
it awkward to define the episodic-memory near-neighbour frames.

This dataset instead emits ONE sample per keyframe→keyframe transition, exactly
like GemBench / MemoryBench / RMBench: ``obs at keyframe k`` → ``action at
keyframe k+1``. No arbitrary intermediate frames, no appended keyframe chain.

EXCEPTION — ``AUG_TASKS`` (currently place_cups): these tasks deliberately go
back to the ORIGINAL BridgeVLA fill_replay pairing on top of the canonical
keyframes: every ``DEMO_AUGMENTATION_EVERY_N``-th raw frame starts a sample
paired with its next keyframe, followed by the full remaining keyframe→keyframe
chain (re-emitted per start frame, so late transitions are weighted higher,
matching the old replay distribution). All other tasks are unaffected.

Episodic memory (matches RMBench / MemoryBench ``keyframe_gt`` layout)
---------------------------------------------------------------------
When ``memory_enabled``:
  * anchor   = frame 0 of the episode (env-reset state); masked off at step 0.
  * hist slot 0/1 = the two most-recent EXECUTED keyframes (t-1, t-2) — the
    "near neighbour two frames".
  * hist slots 2..K-1 = discriminator-admitted GT subtask-boundary keyframes.
    RLBench has NO subtask-boundary label, so ``mem_label`` is always 0 and the
    keyframe discriminator is trained to a constant negative ("never admit"):
    nothing is ever admitted into slots 2..K-1, so the temporal memory stays
    exactly anchor + the two recency frames.

The emitted sample dict mirrors GemBench's ``Gembench_Dataset.__getitem__`` so
``bridgevla_agent.update_gembench`` / ``_build_memory_inputs_from_replay`` are
reused unchanged.

Cache layout: one ``.npz`` (+ ``.meta``) per episode under
``<cache_dir>/<task>/episode<idx>.npz``. Keyframe-only arrays are deduplicated
(anchor stored once, history referenced by step index), so the on-disk footprint
is a fraction of the old replay buffer.

Adapted from memoryBench/memorybench_dataset.py.
"""

import bisect
import hashlib
import json
import os
import pickle
import random
import time
from collections import Counter, OrderedDict
from typing import List, Optional, Sequence

import fcntl
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# RLBench data is stored in RLBench/PerAct format. get_stored_demo reconstructs
# Observation objects (rgb / point_cloud / gripper_pose / ...); keypoint_discovery
# is the same heuristic used by the old replay builder + visualize.py.
from peract_colab.rlbench.utils import get_stored_demo
from bridgevla.libs.peract.helpers.demo_loading_utils import keypoint_discovery

try:
    from utils.keyframe_canonicalize import canonicalize_task, episode_arrays
except ImportError:  # run as a plain script from utils/
    from keyframe_canonicalize import canonicalize_task, episode_arrays


# v2: keyframes are canonicalized per (task, variation) via majority-vote
# templates (see keyframe_canonicalize.py) — removes the spurious frame-1
# keyframe (put_item_in_drawer/close_jar), terminal pose-duplicate keyframes,
# failed-grasp retry events, and fills stop keyframes missed by the fixed
# velocity threshold, so every episode of a variation shares one structure.
CACHE_VERSION = 2

# Original BridgeVLA/PerAct-style demo augmentation, applied ONLY to these
# tasks (all other tasks keep the pure keyframe->keyframe sampling): every
# n-th raw frame becomes a sample start paired with its NEXT canonical
# keyframe, followed by the full remaining keyframe->keyframe chain — the
# chain is re-emitted for every start frame, faithful to the original
# fill_replay/_add_keypoints_to_replay builder (which weights late-episode
# transitions higher). Keyframes stay canonical (v2); only the PAIRING for
# these tasks changes. Cache impact: their .npz additionally stores the
# non-keyframe start-frame observations; staleness is gated by the
# ``aug_every_n`` meta key so other tasks' caches are untouched.
AUG_TASKS = frozenset({"place_cups"})
DEMO_AUGMENTATION_EVERY_N = 10  # matches original peract_utils DEMO_AUGMENTATION_EVERY_N


def _augmented_entries(task: str, ep_idx: int, meta: dict) -> List[dict]:
    """Index entries for an AUG_TASKS episode, mirroring fill_replay:
    for each start frame i (every n-th, before the last keyframe) emit
    (obs@i -> next keyframe) then the whole remaining kf->kf chain.
    Start frames that coincide with a keyframe reuse the keyframe obs
    (aug_slot=-1); others reference the stored extra obs by aug_slot,
    numbered in aug_frames order (matches _build_episode_cache storage)."""
    K = [int(k) for k in meta["keyframes"]]
    num_kp = len(K)
    kf_set = set(K)
    entries: List[dict] = []
    slot = 0
    for i in meta["aug_frames"]:
        i = int(i)
        j = bisect.bisect_right(K, i) - 1  # largest j with K[j] <= i
        if i in kf_set:
            first_slot = -1
        else:
            first_slot = slot
            slot += 1
        if j > num_kp - 2:
            continue  # no next keyframe (defensive; aug_frames excludes this)
        for step in range(j, num_kp - 1):
            entries.append({
                "task": task,
                "ep_idx": ep_idx,
                "step_idx": step,
                "num_steps": num_kp,
                "aug_slot": first_slot if step == j else -1,
            })
    return entries


def _resize_rgb(img: np.ndarray, target_hw: int) -> np.ndarray:
    """uint8 (H, W, 3) -> uint8 (target_hw, target_hw, 3) via PIL bilinear."""
    if img.shape[0] == target_hw and img.shape[1] == target_hw:
        return img
    pil = Image.fromarray(img)
    pil = pil.resize((target_hw, target_hw), resample=Image.BILINEAR)
    return np.asarray(pil)


def _resize_pcd(pcd: np.ndarray, target_hw: int) -> np.ndarray:
    """(H, W, 3) float32 pointcloud -> (target_hw, target_hw, 3).

    No-op at the native RLBench resolution (IMAGE_SIZE=128). Upsampling a point
    cloud fabricates 3D points the simulator never produced, so when target != src
    we fall back to nearest (the strictly-safer choice) — but the canonical path
    keeps target_hw == source.
    """
    if pcd.shape[0] == target_hw and pcd.shape[1] == target_hw:
        return pcd.astype(np.float32, copy=False)
    import torch
    t = torch.from_numpy(np.ascontiguousarray(pcd)).permute(2, 0, 1).unsqueeze(0).float()
    t = torch.nn.functional.interpolate(t, size=(target_hw, target_hw), mode="nearest")
    return t.squeeze(0).permute(1, 2, 0).contiguous().numpy().astype(np.float32)


def _safe_quat(q):
    """Normalize the xyzw quaternion (the agent's update_gembench expects xyzw
    and re-normalizes / sign-fixes downstream); guard the zero-norm edge case.
    """
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1e-8:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        q = q / n
    return q


def _episode_sort_key(name: str) -> int:
    try:
        return int(name.replace("episode", ""))
    except ValueError:
        return 1 << 30


class RLBench_Keyframe_Dataset(Dataset):
    """One sample per keyframe→keyframe transition, with optional episodic memory.

    Args:
        data_root: directory that holds ``<task>/all_variations/episodes`` (e.g.
            ``${RLBENCH_DATA_FOLDER}/train``). The flat ``<task>/...`` layout is
            also supported.
        cameras: RLBench camera names.
        tasks: list of task names. None -> autodetect every subdir that contains
            ``all_variations/episodes``.
        ep_per_task: cap demos / task (RLBench ships 100 / task; matches NUM_TRAIN).
        cache_dir: keyframe cache dir. Built lazily; defaults under data_root.
        image_size: target H = W (RLBench native = 128, so resize is a no-op).
        memory_enabled / memory_k_temporal / memory_select: episodic-memory knobs;
            same semantics as MemoryBench_Dataset / Gembench_Dataset.
        allow_build: when False (training), every episode's canonical keyframe
            cache (cache_version == CACHE_VERSION) must already exist — any
            missing/stale cache raises RuntimeError instead of being rebuilt.
            Build caches beforehand by running this module as a script.
    """

    def __init__(
        self,
        data_root: str,
        cameras: Sequence[str] = ("front", "left_shoulder", "right_shoulder", "wrist"),
        tasks: Optional[Sequence[str]] = None,
        ep_per_task: int = 100,
        cache_dir: Optional[str] = None,
        image_size: int = 128,
        memory_enabled: bool = False,
        memory_k_temporal: int = 12,
        memory_select: str = "keyframe_gt",
        episode_cache_size: int = 4,
        verbose: bool = True,
        allow_build: bool = True,
    ):
        self.allow_build = bool(allow_build)
        self.data_root = data_root
        self.cameras = list(cameras)
        self.ep_per_task = int(ep_per_task)
        self.image_size = int(image_size)
        self.memory_enabled = bool(memory_enabled)
        self.memory_k_temporal = int(memory_k_temporal)
        # "keyframe_gt": slot 0/1 = the two most-recent executed keyframes;
        #   slots 2..K-1 = discriminator-admitted boundaries (none for RLBench).
        # "sliding": legacy K most-recent keyframes.
        self.memory_select = str(memory_select)
        self.episode_cache_size = int(episode_cache_size)
        self.verbose = verbose

        if tasks is None:
            tasks = []
            for name in sorted(os.listdir(data_root)):
                ep_dir = os.path.join(data_root, name, "all_variations", "episodes")
                if os.path.isdir(ep_dir):
                    tasks.append(name)
        self.tasks = list(tasks)

        if cache_dir is None:
            cache_dir = os.path.join(
                self.data_root, "_keyframe_cache", f"size{self.image_size}_v{CACHE_VERSION}",
            )
        self.cache_dir = cache_dir
        if self.allow_build:
            os.makedirs(self.cache_dir, exist_ok=True)

        self._episode_cache: "OrderedDict[tuple, dict]" = OrderedDict()

        t0 = time.time()
        self.index = self._build_or_load_index()
        if self.verbose:
            print(f"[RLBench-kf] Indexed {len(self.index)} keyframe transitions across "
                  f"{len(self.tasks)} tasks in {time.time()-t0:.1f}s")

        self.num_tasks = len(self.tasks)
        self.num_task_paths = sum(1 for e in self.index if e["step_idx"] == 0)

    # ---- episode root resolution -----------------------------------------

    def _episodes_root(self, task: str) -> Optional[str]:
        # train/ FIRST: with the download layout (train zips -> train/<task>,
        # test zips -> <task> at the root) the root-level TEST episodes would
        # otherwise shadow the train split and training would silently run on
        # 25 test episodes per task. The flat layout (<task> at the root IS the
        # train data, no train/ subdir) still resolves via the second candidate.
        candidates = [
            os.path.join(self.data_root, "train", task, "all_variations", "episodes"),
            os.path.join(self.data_root, task, "all_variations", "episodes"),
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        return None

    # ---- index build ------------------------------------------------------

    def _index_cache_path(self) -> str:
        tag = (",".join(sorted(self.tasks))
               + f"|ep{self.ep_per_task}|sz{self.image_size}|cv{CACHE_VERSION}"
               + f"|aug{DEMO_AUGMENTATION_EVERY_N}:{','.join(sorted(AUG_TASKS))}")
        digest = hashlib.md5(tag.encode("utf-8")).hexdigest()[:8]
        return os.path.join(self.cache_dir, f"_index_{digest}.json")

    def _build_or_load_index(self) -> List[dict]:
        if not self.allow_build:
            return self._load_index_strict()
        index_path = self._index_cache_path()
        try:
            import torch.distributed as dist
            ddp_active = dist.is_available() and dist.is_initialized()
        except Exception:
            ddp_active = False

        if ddp_active:
            if dist.get_rank() == 0:
                self._build_index_locked(index_path)
            else:
                deadline = time.time() + 60 * 60 * 6  # 6h hard cap
                while not os.path.exists(index_path):
                    if time.time() > deadline:
                        raise RuntimeError(
                            f"[RLBench-kf] rank {dist.get_rank()} timed out waiting "
                            f"for rank 0 to publish {index_path}."
                        )
                    time.sleep(5.0)
            with open(index_path, "r") as f:
                return json.load(f)

        return self._build_index_locked(index_path)

    def _load_index_strict(self) -> List[dict]:
        """Assemble the index from EXISTING canonical episode caches only.

        Never unpickles demos, never runs keypoint discovery, never touches
        image data — a missing or stale (cache_version != CACHE_VERSION)
        episode cache is a hard error. Every rank scans the metas
        independently (deterministic, ~seconds), so no DDP coordination or
        lock files are needed.
        """
        index: List[dict] = []
        problems: List[str] = []
        for task in self.tasks:
            ep_root = self._episodes_root(task)
            if ep_root is None:
                problems.append(f"{task}: no episode root under {self.data_root}")
                continue
            ep_names = sorted(os.listdir(ep_root), key=_episode_sort_key)
            ep_names = [n for n in ep_names if n.startswith("episode")][: self.ep_per_task]
            for ep_name in ep_names:
                try:
                    ep_idx = int(ep_name.replace("episode", ""))
                except ValueError:
                    continue
                cache_file = os.path.join(self.cache_dir, task, f"episode{ep_idx}.npz")
                meta_file = cache_file + ".meta"
                if not os.path.exists(cache_file) or not os.path.exists(meta_file):
                    problems.append(f"{task}/episode{ep_idx}: cache file(s) missing")
                    continue
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                except Exception as e:
                    problems.append(f"{task}/episode{ep_idx}: unreadable meta ({e})")
                    continue
                got_ver = int(meta.get("cache_version", 0))
                if got_ver != CACHE_VERSION:
                    problems.append(
                        f"{task}/episode{ep_idx}: cache_version {got_ver} != {CACHE_VERSION}")
                    continue
                if task in AUG_TASKS and int(
                        meta.get("aug_every_n", 0)) != DEMO_AUGMENTATION_EVERY_N:
                    problems.append(
                        f"{task}/episode{ep_idx}: cache lacks demo-augmentation frames "
                        f"(aug_every_n={meta.get('aug_every_n')}, "
                        f"need {DEMO_AUGMENTATION_EVERY_N})")
                    continue
                if meta.get("cameras") != list(self.cameras):
                    problems.append(
                        f"{task}/episode{ep_idx}: cache camera order "
                        f"{meta.get('cameras')} != expected {list(self.cameras)} "
                        f"(views would be silently permuted)")
                    continue
                num_kp = int(meta["num_keyframes"])
                if num_kp < 2:
                    problems.append(
                        f"{task}/episode{ep_idx}: only {num_kp} keyframe(s) in "
                        f"cache (>= 2 required; corrupt cache)")
                    continue
                if task in AUG_TASKS:
                    index.extend(_augmented_entries(task, ep_idx, meta))
                    continue
                for step_idx in range(num_kp - 1):
                    index.append({
                        "task": task,
                        "ep_idx": ep_idx,
                        "step_idx": step_idx,
                        "num_steps": num_kp,
                    })
        if problems:
            shown = "\n  ".join(problems[:10])
            raise RuntimeError(
                f"[RLBench-kf] STRICT cache mode (allow_build=False): "
                f"{len(problems)} episode cache(s) missing or stale under "
                f"{self.cache_dir} (need cache_version={CACHE_VERSION}). "
                f"Training does NOT fall back to building caches. Either "
                f"download the released cache:\n"
                f"  bash scripts/download_checkpoints_hf.sh rlbench_cache\n"
                f"or build it locally (the wrapper sets up the CoppeliaSim/PyRep env):\n"
                f"  bash scripts/build_rlbench_cache.sh "
                f"--data_root {self.data_root} --ep_per_task {self.ep_per_task}\n"
                f"First problems:\n  {shown}"
            )
        return index

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
                ep_root = self._episodes_root(task)
                if ep_root is None:
                    raise FileNotFoundError(
                        f"[RLBench-kf] missing episode root for task '{task}' "
                        f"under {self.data_root}. Silently skipping a task "
                        f"would shrink the dataset — fix the data layout or "
                        f"drop the task from `tasks`."
                    )
                ep_names = sorted(os.listdir(ep_root), key=_episode_sort_key)
                ep_names = [n for n in ep_names if n.startswith("episode")][: self.ep_per_task]

                # First scan: read existing metas; decide who needs (re)build.
                # A .meta whose cache_version (and, for AUG_TASKS, aug_every_n)
                # matches is AUTHORITATIVE even when its .npz is missing — its
                # canonical keyframes are exactly what the release ships, so
                # the npz is re-materialized FROM the meta and the
                # canonicalization vote is NOT re-run (a re-vote over a
                # different episode subset could yield different keyframes).
                metas, need_vote, need_npz = {}, [], []
                for ep_name in ep_names:
                    try:
                        ep_idx = int(ep_name.replace("episode", ""))
                    except ValueError:
                        continue
                    cache_file = os.path.join(self.cache_dir, task, f"episode{ep_idx}.npz")
                    meta_file = cache_file + ".meta"
                    meta = None
                    if os.path.exists(meta_file) and os.path.getsize(meta_file) > 0:
                        try:
                            with open(meta_file, "r") as f:
                                meta = json.load(f)
                            if int(meta.get("cache_version", 0)) != CACHE_VERSION:
                                meta = None
                            elif task in AUG_TASKS and int(
                                    meta.get("aug_every_n", 0)) != DEMO_AUGMENTATION_EVERY_N:
                                meta = None  # cache predates demo augmentation
                        except Exception:
                            meta = None
                    # Keyframes in a version-matching meta stay authoritative
                    # even when the npz must be re-materialized — because it
                    # is missing, or because it was built with a different
                    # camera order (position-indexed at __getitem__, so a
                    # mismatched order would silently permute views).
                    if meta is not None and (
                            not os.path.exists(cache_file)
                            or meta.get("cameras") != list(self.cameras)):
                        need_npz.append(ep_idx)
                    elif meta is None:
                        need_vote.append(ep_idx)
                    metas[ep_idx] = meta

                # Re-materialize npz for authoritative metas that lack one
                # (e.g. a downloaded keyframe index without image caches):
                # decode the demo with the meta's exact keyframes.
                for ep_idx in need_npz:
                    meta = metas[ep_idx]
                    cache_file = os.path.join(self.cache_dir, task, f"episode{ep_idx}.npz")
                    kfs = [int(k) for k in meta.get("keyframes", [])]
                    if len(kfs) < 2 or kfs[0] != 0:
                        raise RuntimeError(
                            f"[RLBench-kf] corrupt meta for {task}/episode{ep_idx}: "
                            f"keyframes={kfs!r} (expected [0, k1, ...]).")
                    extra = {k: meta[k] for k in ("kf_conform", "kf_note") if k in meta}
                    try:
                        # _build_episode_cache prepends frame 0 itself.
                        self._build_episode_cache(
                            task, ep_idx, ep_root, cache_file,
                            keyframes=kfs[1:], extra_meta=extra)
                    except Exception as e:
                        raise RuntimeError(
                            f"[RLBench-kf] FAILED to rebuild npz for "
                            f"{task}/episode{ep_idx} from its authoritative "
                            f"meta: {e}"
                        ) from e
                    with open(cache_file + ".meta", "r") as f:
                        metas[ep_idx] = json.load(f)

                # Canonical keyframes are voted over ALL episodes of the task,
                # so run the (low-dim only) pre-pass whenever any episode has
                # no authoritative meta.
                canon = {}
                if need_vote:
                    canon = self._canonical_keyframes_for_task(task, ep_root, list(metas))

                for ep_idx in sorted(metas):
                    meta = metas[ep_idx]
                    if meta is None:
                        res = canon.get(ep_idx)
                        if res is None:
                            raise RuntimeError(
                                f"[RLBench-kf] canonicalize pre-pass produced no "
                                f"keyframes for {task}/episode{ep_idx} (see WARN "
                                f"above for the cause). Silently dropping the "
                                f"episode would shrink the dataset — fix or "
                                f"re-download it."
                            )
                        cache_file = os.path.join(self.cache_dir, task, f"episode{ep_idx}.npz")
                        meta_file = cache_file + ".meta"
                        want = [0] + [int(k) for k in res["keyframes"]]
                        old = None
                        if os.path.exists(cache_file) and os.path.exists(meta_file):
                            try:
                                with open(meta_file, "r") as f:
                                    old = json.load(f)
                            except Exception:
                                old = None
                        if old is not None and old.get("keyframes") == want \
                                and (task not in AUG_TASKS or int(
                                    old.get("aug_every_n", 0)) == DEMO_AUGMENTATION_EVERY_N):
                            # Keyframes unchanged: upgrade meta in place, keep npz.
                            meta = {**old, "cache_version": CACHE_VERSION,
                                    "kf_conform": bool(res["conform"]),
                                    "kf_note": res["note"]}
                            tmp = f"{meta_file}.tmp{os.getpid()}"
                            with open(tmp, "w") as f:
                                json.dump(meta, f)
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(tmp, meta_file)
                        else:
                            try:
                                self._build_episode_cache(
                                    task, ep_idx, ep_root, cache_file,
                                    keyframes=res["keyframes"],
                                    extra_meta={"kf_conform": bool(res["conform"]),
                                                "kf_note": res["note"]})
                            except Exception as e:
                                raise RuntimeError(
                                    f"[RLBench-kf] FAILED to build cache for "
                                    f"{task}/episode{ep_idx}: {e}. Silently "
                                    f"dropping the episode would shrink the "
                                    f"dataset — fix or re-download it."
                                ) from e
                            with open(meta_file, "r") as f:
                                meta = json.load(f)
                    num_kp = int(meta["num_keyframes"])
                    if num_kp < 2:
                        raise RuntimeError(
                            f"[RLBench-kf] cache for {task}/episode{ep_idx} holds "
                            f"{num_kp} keyframe(s); a valid episode has >= 2 "
                            f"(frame 0 + first keyframe) — corrupt cache, delete "
                            f"it and rebuild."
                        )
                    if task in AUG_TASKS:
                        index.extend(_augmented_entries(task, ep_idx, meta))
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

    def _canonical_keyframes_for_task(self, task: str, ep_root: str,
                                      ep_indices: List[int]) -> dict:
        """Low-dim pre-pass over ALL of the task's episodes: run the original
        keypoint_discovery heuristic, then canonicalize per (variation) via
        majority-vote templates. Returns {ep_idx: {keyframes, conform, note}}.
        """
        episodes = {}
        it = ep_indices
        if self.verbose:
            it = tqdm(ep_indices, desc=f"[RLBench-kf] canonicalize {task}")
        for ep_idx in it:
            ep_dir = os.path.join(ep_root, f"episode{ep_idx}")
            try:
                with open(os.path.join(ep_dir, "low_dim_obs.pkl"), "rb") as f:
                    demo = pickle.load(f)
                obs = demo._observations if hasattr(demo, "_observations") else list(demo)
                kfs = keypoint_discovery(obs)
                if len(kfs) == 0:
                    raise RuntimeError("keypoint_discovery returned empty")
                arr = episode_arrays(obs)
                try:
                    with open(os.path.join(ep_dir, "variation_number.pkl"), "rb") as f:
                        variation = int(pickle.load(f))
                except Exception:
                    variation = 0
                episodes[ep_idx] = {"arr": arr, "kfs": [int(k) for k in kfs],
                                    "variation": variation}
            except Exception as e:
                if self.verbose:
                    print(f"[RLBench-kf] WARN: canonicalize pre-pass failed for "
                          f"{task}/episode{ep_idx}: {e}")
        if not episodes:
            return {}
        result = canonicalize_task(episodes)
        if self.verbose:
            for ep_idx in sorted(result):
                if not result[ep_idx]["conform"]:
                    print(f"[RLBench-kf] NOTE: {task}/episode{ep_idx} keyframes kept "
                          f"non-canonical: {result[ep_idx]['note']}")
        return result

    def _build_episode_cache(self, task: str, ep_idx: int, ep_root: str, cache_file: str,
                             keyframes: Optional[List[int]] = None,
                             extra_meta: Optional[dict] = None):
        """One episode -> one .npz of keyframe-only arrays + meta json.

        ``keyframes`` (no leading frame 0) normally comes from the canonical
        pre-pass; the raw keypoint_discovery fallback is kept for standalone use.
        """
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)

        demo = get_stored_demo(data_path=ep_root, index=ep_idx)
        detected_keyframes = keyframes if keyframes is not None else keypoint_discovery(demo)
        if len(detected_keyframes) == 0:
            raise RuntimeError(f"keypoint_discovery returned empty for {task}/episode{ep_idx}")

        # GemBench convention: prepend demo frame 0 (env-reset state) so step_idx=0
        # pairs (frame_0 -> first detected keyframe) and slot 0 doubles as the
        # memory-bank anchor. keypoint_discovery itself skips frame 0.
        keyframes = [0] + [int(k) for k in detected_keyframes]

        cam_keys = list(self.cameras)
        H = W = self.image_size
        nk = len(keyframes)

        rgb_buf = np.zeros((nk, len(cam_keys), H, W, 3), dtype=np.uint8)
        pcd_buf = np.zeros((nk, len(cam_keys), H, W, 3), dtype=np.float32)
        gp_buf = np.zeros((nk, 8), dtype=np.float32)  # xyz, quat_xyzw, grip_open
        ic_buf = np.zeros((nk,), dtype=np.float32)

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

        a_rgb, a_pcd = _pack_obs(demo[0])
        anchor_rgb[:] = a_rgb
        anchor_pcd[:] = a_pcd

        for i, kf in enumerate(keyframes):
            obs = demo[kf]
            rgbs, pcs = _pack_obs(obs)
            rgb_buf[i] = rgbs
            pcd_buf[i] = pcs
            gp_buf[i] = np.concatenate([
                np.asarray(obs.gripper_pose[:3], dtype=np.float32),
                _safe_quat(obs.gripper_pose[3:7]),
                np.asarray([float(obs.gripper_open)], dtype=np.float32),
            ])
            # ignore_collisions: take the frame BEFORE the keyframe (matches the
            # old replay builder's obs_tm1 convention).
            obs_tm1 = demo[max(0, int(kf) - 1)]
            ic_buf[i] = float(getattr(obs_tm1, "ignore_collisions", 1.0))

        # Extra start-frame observations for the fill_replay-style pairing.
        # aug_frames lists ALL every-n start frames (incl. keyframe-coincident
        # ones); only non-keyframe frames are stored, in aug_frames order —
        # _augmented_entries assigns aug_slot with the same counter.
        aug_frames: List[int] = []
        aug_rgb = aug_pcd = None
        if task in AUG_TASKS:
            last_kf = int(keyframes[-1])
            kf_set = set(int(k) for k in keyframes)
            aug_frames = [i for i in range(0, len(demo) - 1, DEMO_AUGMENTATION_EVERY_N)
                          if i < last_kf]
            stored = [i for i in aug_frames if i not in kf_set]
            aug_rgb = np.zeros((len(stored), len(cam_keys), H, W, 3), dtype=np.uint8)
            aug_pcd = np.zeros((len(stored), len(cam_keys), H, W, 3), dtype=np.float32)
            for s, frame in enumerate(stored):
                rgbs, pcs = _pack_obs(demo[frame])
                aug_rgb[s] = rgbs
                aug_pcd[s] = pcs

        ep_dir = os.path.join(ep_root, f"episode{ep_idx}")
        with open(os.path.join(ep_dir, "variation_descriptions.pkl"), "rb") as f:
            descs = pickle.load(f)
        if isinstance(descs, (list, tuple)):
            lang_goal = list(descs)
        elif isinstance(descs, str):
            lang_goal = [descs]
        else:
            lang_goal = [str(descs)]

        try:
            with open(os.path.join(ep_dir, "variation_number.pkl"), "rb") as f:
                variation = int(pickle.load(f))
        except Exception:
            variation = 0

        pid = os.getpid()
        tmp_npz = f"{cache_file}.tmp{pid}.npz"
        arrays = dict(
            rgb=rgb_buf,
            pcd=pcd_buf,
            gripper_pose=gp_buf,
            ignore_collisions=ic_buf,
            anchor_rgb=anchor_rgb,
            anchor_pcd=anchor_pcd,
        )
        if aug_rgb is not None:
            arrays["aug_rgb"] = aug_rgb
            arrays["aug_pcd"] = aug_pcd
        np.savez_compressed(tmp_npz, **arrays)
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
        if task in AUG_TASKS:
            meta["aug_every_n"] = DEMO_AUGMENTATION_EVERY_N
            meta["aug_frames"] = [int(i) for i in aug_frames]
            meta["aug_stored"] = int(aug_rgb.shape[0])
        if extra_meta:
            meta.update(extra_meta)
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
        # silently deliver permuted (mislabeled) views. Hard error instead.
        if meta.get("cameras") != list(self.cameras):
            raise RuntimeError(
                f"[RLBench-kf] camera-order mismatch for {task}/episode{ep_idx}: "
                f"cache built with {meta.get('cameras')} but dataset expects "
                f"{list(self.cameras)} — the cache would be silently permuted. "
                f"Rebuild it (run this module as a script) or fix `cameras`."
            )
        npz = np.load(cache_file)
        out = {
            "rgb": npz["rgb"],
            "pcd": npz["pcd"],
            "gripper_pose": npz["gripper_pose"],
            "ignore_collisions": npz["ignore_collisions"],
            "anchor_rgb": npz["anchor_rgb"],
            "anchor_pcd": npz["anchor_pcd"],
            "meta": meta,
        }
        if "aug_rgb" in npz.files:
            out["aug_rgb"] = npz["aug_rgb"]
            out["aug_pcd"] = npz["aug_pcd"]
        self._episode_cache[key] = out
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return out

    # ---- multi-task sampling balance -------------------------------------

    def task_sampling_weights(self, alpha: float) -> np.ndarray:
        """Per-index sampling weights for temperature task balancing.

        The flat ``self.index`` has one entry per (task, ep, step) transition,
        so a uniform sampler draws each TASK with probability proportional to
        its transition count ``n_t`` (tasks whose demos have more keyframes
        dominate, even when ``ep_per_task`` equalizes episode COUNT across
        tasks). Temperature sampling targets task probability
        ``p_t ∝ n_t**alpha``; since a task owns ``n_t`` index entries, the
        per-entry weight is ``w_i ∝ n_{t(i)}**(alpha-1)``:

          * alpha=1 -> w_i ∝ 1            (transition-uniform, original)
          * alpha=0 -> w_i ∝ 1/n_t        (task-uniform)
          * alpha=0.5 -> w_i ∝ 1/sqrt(n_t)(tempered middle ground)

        Weights are NOT normalized (``torch.multinomial`` normalizes), and the
        per-task counts come straight from the in-memory index (no extra IO).
        """
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

    def __getitem__(self, idx: int):
        entry = self.index[idx]
        task = entry["task"]
        step_idx = entry["step_idx"]
        num_steps = entry["num_steps"]
        # fill_replay-style augmented start frame (AUG_TASKS only): obs comes
        # from a stored non-keyframe frame between keyframe step_idx and
        # step_idx+1; target/labels are identical to the plain step sample.
        aug_slot = int(entry.get("aug_slot", -1))
        ep = self._load_episode(task, entry["ep_idx"])

        cam_list = self.cameras
        # On-disk arrays are channel-last (H, W, 3); the model expects
        # channel-first (3, H, W). Transpose here (cheap, on worker CPU).
        sample = {}
        for cam_idx, cam in enumerate(cam_list):
            if aug_slot >= 0:
                rgb = ep["aug_rgb"][aug_slot, cam_idx]
                pcd = ep["aug_pcd"][aug_slot, cam_idx]
            else:
                rgb = ep["rgb"][step_idx, cam_idx]
                pcd = ep["pcd"][step_idx, cam_idx]
            sample[cam] = {
                "rgb": np.transpose(rgb, (2, 0, 1)),
                "pcd": np.transpose(pcd, (2, 0, 1)),
            }

        # Target action = the NEXT keyframe's gripper pose.
        sample["gripper_pose"] = ep["gripper_pose"][step_idx + 1].astype(np.float32)
        time_feat = (1.0 - (step_idx / float(max(1, num_steps - 1)))) * 2.0 - 1.0
        sample["low_dim_state"] = np.concatenate(
            [sample["gripper_pose"], np.asarray([time_feat], dtype=np.float32)]
        ).astype(np.float32)
        sample["ignore_collisions"] = float(ep["ignore_collisions"][step_idx + 1])

        sample["lang_goal"] = random.choice(ep["meta"]["lang_goal"])
        sample["tasks"] = task

        # ---- episodic memory bundle: anchor + K temporal slots ------------
        if self.memory_enabled:
            K = self.memory_k_temporal

            for cam_idx, cam in enumerate(cam_list):
                a_rgb = ep["anchor_rgb"][cam_idx]
                a_pcd = ep["anchor_pcd"][cam_idx]
                sample[f"anchor_{cam}"] = {
                    "rgb": np.transpose(a_rgb, (2, 0, 1)),
                    "pcd": np.transpose(a_pcd, (2, 0, 1)),
                }
            # For an augmented start (obs strictly AFTER keyframe step_idx),
            # keyframe step_idx itself is already in the past: it becomes the
            # most-recent executed keyframe, and the frame-0 anchor is always
            # a past state (aug frame 0 coincides with keyframe 0 and is
            # emitted as a plain step sample, never as aug_slot >= 0).
            last_exec = step_idx if aug_slot >= 0 else step_idx - 1
            sample["anchor_mask"] = np.array(aug_slot >= 0 or step_idx > 0, dtype=bool)
            sample["step_idx"] = np.int64(step_idx)

            # RLBench has no subtask-boundary label -> always 0. The keyframe
            # discriminator is trained toward a constant negative, so its eval
            # gate is ~0 < threshold and nothing is ever admitted into the
            # extra memory slots (memory stays anchor + the two recency frames).
            sample["mem_label"] = np.float32(0.0)

            def _emit_hist_slot(slot, src_step):
                for cam_idx, cam in enumerate(cam_list):
                    h_rgb = ep["rgb"][src_step, cam_idx]
                    h_pcd = ep["pcd"][src_step, cam_idx]
                    sample[f"hist{slot}_{cam}"] = {
                        "rgb": np.transpose(h_rgb, (2, 0, 1)),
                        "pcd": np.transpose(h_pcd, (2, 0, 1)),
                    }

            hist_mask = np.zeros((K,), dtype=bool)
            if self.memory_select == "keyframe_gt":
                # slot 0 = previous executed keyframe (t-1), slot 1 = (t-2);
                # slots 2..K-1 stay masked (no boundary labels -> never admitted).
                # For aug samples last_exec = step_idx (obs sits between
                # keyframes step_idx and step_idx+1).
                slot_src = [0] * K
                if last_exec >= 0:
                    slot_src[0] = last_exec
                    hist_mask[0] = True
                if K >= 2 and last_exec - 1 >= 0:
                    slot_src[1] = last_exec - 1
                    hist_mask[1] = True
                for slot in range(K):
                    _emit_hist_slot(slot, slot_src[slot])
            else:
                # LEGACY sliding window: K most-recent keyframes.
                for k in range(K):
                    hist_step = last_exec - k
                    valid = hist_step >= 0
                    _emit_hist_slot(k, hist_step if valid else 0)
                    if valid:
                        hist_mask[k] = True
            sample["hist_mask"] = hist_mask
        return sample


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get(
        "RLBENCH_DATA_FOLDER",
        "data/bridgevla_data/RLBench"))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--ep_per_task", type=int, default=100)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--memory_enabled", action="store_true")
    parser.add_argument("--memory_k_temporal", type=int, default=12)
    parser.add_argument("--memory_select", type=str, default="keyframe_gt",
                        choices=["keyframe_gt", "sliding"])
    parser.add_argument("--tasks", nargs="*", default=None)
    args = parser.parse_args()

    # data_root is the RLBench root; _episodes_root also resolves an optional
    # train/ sublayout, so no path juggling here. Run as a script this BUILDS
    # (or upgrades) the canonical keyframe cache — training itself loads
    # strictly (allow_build=False) and errors out on a missing cache.
    ds = RLBench_Keyframe_Dataset(
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
