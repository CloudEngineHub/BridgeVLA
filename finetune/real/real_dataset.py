"""Real_Dataset for the task-grouped Dobot + ZED layout.

Directory layout assumed:

    {data_root}/{task_group}/{episode_name}/
        pose.pkl             # full-episode trajectory as a string
        instruction.pkl      # language goal string
        extrinsic_matrix.pkl # 4x4 camera->base
        zed_rgb/{i}.jpg      # uint8 (H,W,3) camera RGB, i = 0..num_frames-1
        zed_depth/{i}.png    # uint16 (H,W) camera depth, intrinsic.pkl scale
        intrinsic.pkl        # per-frame fx/fy/cx/cy + depth_scale
        cam_img/             # optional regular camera images (unused by loader)

    The camera-frame XYZ is reconstructed from depth by pinhole deprojection
    (exact to ~1e-6 m — the X/Y channels of the ZED point cloud are precisely
    (u-cx)/fx*Z and (v-cy)/fy*Z, so storing them is pure redundancy; see
    data_collection/convert_pcd_to_depth.py). The legacy layout

        zed_rgb/{i}.pkl      # pickled uint8 (H,W,3)
        zed_pcd/{i}.pkl      # pickled float32 (H,W,3) camera-frame XYZ

    is still read when the compact files are absent, so datasets that have not
    been converted load unchanged.

Examples of task_group directories: plate_tasks, shelf_tasks, drawer_tasks.

For each frame i we produce one training sample (current=i, target=i+1).
The last frame in each episode has no next target so it is skipped.

The point cloud is transformed from camera frame to the Dobot base frame, and
the sample is built compatible with `RVTAgent.update_gembench()`:

    sample = {
        "3rd":   {"rgb": (3,H,W) uint8, "pcd": (3,H,W) float32 in base frame},
        "gripper_pose": (8,) float32 = [x, y, z (m), qx, qy, qz, qw, claw],
        "ignore_collisions": float32 scalar = 1.0,
        "low_dim_state":    (2,) float32 = [claw, time],
        "lang_goal": str,
        "tasks":     str,   # equals the task_group name
    }

The VLM language encoding wrapping ([[[text]]]) is applied in the training
loop, not here.
"""

import os
import pickle
import time
from collections import Counter, OrderedDict
from typing import List, Union

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm


# --- Image IO for the compact depth layout ---
# cv2 is the primary decoder (unambiguous uint16 PNG, fastest here) with PIL as a fallback, both imported
# lazily so unconverted datasets never touch this path.
def _read_depth_png(path: str) -> np.ndarray:
    """(H,W) uint16 depth image."""
    try:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise RuntimeError(f"cv2 failed to read depth png: {path}")
    except ImportError:
        from PIL import Image
        a = np.asarray(Image.open(path))
    if a.ndim != 2:
        raise ValueError(f"expected single-channel depth png, got shape {a.shape}: {path}")
    return a.astype(np.uint16, copy=False)


def _read_rgb_jpeg(path: str) -> np.ndarray:
    """(H,W,3) uint8 image, channel order preserved as written."""
    try:
        import cv2
        a = cv2.imread(path, cv2.IMREAD_COLOR)
        if a is None:
            raise RuntimeError(f"cv2 failed to read rgb jpeg: {path}")
    except ImportError:
        from PIL import Image
        a = np.asarray(Image.open(path))
    return a[:, :, :3]


def _parse_pose_string(data_str: str) -> List[dict]:
    """Parse a tab-/space-separated pose.pkl string.

    Each data line is: timestamp x_mm y_mm z_mm rx_deg ry_deg rz_deg claw [arm_flag]
    Returns a list of dicts (header skipped).
    """
    lines = data_str.strip().split("\n")
    entries = []
    for i, line in enumerate(lines):
        if i == 0:
            continue
        parts = line.strip().split()
        timestamp = parts[0]
        position = [float(x) for x in parts[1:4]]
        orientation = [float(x) for x in parts[4:7]]
        if len(parts) >= 8:
            claw_status = int(parts[7])
        else:
            claw_status = 1 if (i in (1, 2, 5)) else 0
        arm_flag = int(parts[8]) if len(parts) >= 9 else 0
        entries.append({
            "timestamp": timestamp,
            "position": position,
            "orientation": orientation,
            "claw_status": claw_status,
            "arm_flag": arm_flag,
        })
    return entries


def _load_pose_file(path: str) -> List[dict]:
    with open(path, "rb") as f:
        data_str = pickle.load(f)
    return _parse_pose_string(data_str)


def _convert_pcd_to_base(pcd_camera_hw3: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    """Apply 4x4 camera->base transform to an (H,W,3) camera-frame point cloud."""
    h, w, _ = pcd_camera_hw3.shape
    pc = pcd_camera_hw3.reshape(-1, 3)
    pc_h = np.concatenate([pc, np.ones((pc.shape[0], 1), dtype=pc.dtype)], axis=1)
    pc_base = (transform_4x4 @ pc_h.T).T[:, :3]
    return pc_base.reshape(h, w, 3)


# --- Memory-keyframe (subtask-boundary) labeling ---
# The real Dobot data has NO on-disk subtask-boundary annotation (unlike RMBench, which stores a per-keyframe
# subtask_idx). For the "press the <color> button <N> times, then press the <color> button" family the
# trajectory is fully regular, so boundaries are derived from the task name instead:
#   frame 0            : home
#   frame 1            : approach the (blue) button
#   frames (2k, 2k+1)  : press-down / lift-up for press k = 1..N
#   frame 2N+2         : move to the (yellow) button
#   frame 2N+3         : press the (yellow) button
# The lift-up frame ``2k+1`` completes press-k, i.e. it is the subtask-boundary "memory keyframe":
#   three_times (N=3) -> steps {3, 5, 7};   twice (N=2) -> steps {3, 5}.
# This matches RMBench's rule (mem_label == 1 at the last keyframe of each subtask) and its press_button
# special case. Number words are matched as space phrases against the normalized (underscores -> spaces)
# input, so both folder names and language instructions hit.
_PRESS_COUNT_WORDS = {
    "once": 1, "one time": 1,
    "twice": 2, "two times": 2,
    "thrice": 3, "three times": 3,
    "four times": 4, "five times": 5, "six times": 6,
}


def _press_repeat_count(name: str) -> Union[int, None]:
    """N for a ``press ... button <N> times ...`` task name/instruction, else None."""
    s = (name or "").lower().replace("_", " ")
    if "press" not in s or "button" not in s:
        return None
    for word, n in _PRESS_COUNT_WORDS.items():
        if word in s:
            return n
    return None


# Explicit per-task subtask-boundary steps for tasks whose memory keyframes are not derivable from a repeat
# count. Matched by substring against the normalized task_group / instruction.
#   value = (boundary_steps, expected_num_frames); the frame count guards against re-recorded episodes
#   (a mismatch degrades to no boundary), like the press tasks' 2N+4 guard.
_EXPLICIT_BOUNDARY_TASKS = {
    # Swap two eggplants A<->B via a temporary spot. Tracing the gripper: pick-A -> PLACE-A-at-temp (step 4)
    # -> pick-B -> PLACE-B-at-A's-spot (step 8) -> pick-A-from-temp -> place-A-final (step 12). The two
    # intermediate place-downs (4, 8) are the boundaries; step 12 is the task end, not a memory keyframe.
    "swap the two eggplants": ([4, 8], 13),
    # Put lids on the blocks, then uncover the <color> block (15-frame layout). Tracing the gripper
    # (claw 1 = grasp, 0 = release, verified on every episode): pick lid-1 -> PLACE lid-1 (step 4) ->
    # pick lid-2 -> PLACE lid-2 (step 8) -> third-lid segment -> uncover the target block (steps 12..14).
    # The two lid place-downs (4, 8) are the memory keyframes: remembering WHERE each block got covered is
    # what the final uncover depends on. Step 14 is the task end, not a boundary.
    "put lids on the blocks": ([4, 8], 15),
}


def _normalize_task_text(s: str) -> str:
    return (s or "").lower().replace("_", " ")


# --- Task categories for two-level sampling ---
# The task_group directory names encode BOTH skill and object variant
# ("put_the_banana_in_the_upper_drawer"). Sampling balances across the 5 SKILLS rather than the 19 language
# variants, so a skill with more recorded nouns (drawer: 8, shelf: 5) does not get a proportionally larger
# share. Within a skill the variants are drawn UNIFORMLY.
# Ordered list, FIRST match wins, so more specific patterns come first: (category, predicate).
_TASK_CATEGORIES = [
    # "put the <obj> in the {upper,lower} drawer"
    ("put_drawer",     lambda s: "drawer" in s),
    # "put the <obj> in the {top,bottom} shelf"
    ("put_shelf",      lambda s: "shelf" in s),
    # "press the blue button N times, then press the yellow button"
    ("press_button",   lambda s: "press" in s and "button" in s),
    # "swap the two eggplants on the plate"
    ("swap_eggplant",  lambda s: "swap" in s and "eggplant" in s),
    # "put lids on the blocks, then uncover the <color> block"
    ("put_lids",       lambda s: "lids" in s),
]


def task_category(task_group: str) -> str:
    """Coarse skill category for a task_group dir name (see _TASK_CATEGORIES).

    Unmatched task_groups fall back to their OWN name as a singleton category,
    so a newly-recorded skill is never silently folded into an existing one —
    it just gets its own bucket (and Real_Dataset prints a warning).
    """
    s = _normalize_task_text(task_group)
    for name, pred in _TASK_CATEGORIES:
        if pred(s):
            return name
    return task_group


def _frame_sort_key(stem: str):
    """Numeric sort for frame stems ('10' after '9'), tolerant of odd names."""
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def _memory_boundary_steps(task_group: str, instruction: str, num_frames: int) -> List[int]:
    """Subtask-boundary ("memory keyframe") step indices for ONE episode.

    Two sources, in order:
      1. The press-button-N-times family (formula-based): the completed-press
         "lift-up" frames ``{2k+1 : k=1..N}`` (three_times -> [3, 5, 7];
         twice -> [3, 5]), valid only for the fixed ``2N+4``-frame layout.
      2. Tasks with fixed, hand-specified boundaries in
         :data:`_EXPLICIT_BOUNDARY_TASKS` (e.g. swap-two-eggplants -> [4, 8]).

    Returns ``[]`` only for tasks not covered by either source (no boundary
    rule exists — legitimately no memory positives). A covered task whose
    episode frame count does NOT match the expected layout raises: silently
    returning ``[]`` would switch off memory supervision for that episode
    (a re-recorded / non-standard episode must be fixed or excluded, not
    silently untrained).
    """
    # 1) press-button-N-times family.
    n = _press_repeat_count(task_group)
    if n is None:
        n = _press_repeat_count(instruction)
    if n is not None:
        if num_frames != 2 * n + 4:
            raise ValueError(
                f"[Real] press-{n}-times episode has {num_frames} frames, "
                f"expected {2 * n + 4} ('{instruction}' / group "
                f"'{task_group}'). Memory boundaries would silently vanish — "
                f"fix or exclude the episode."
            )
        return [2 * k + 1 for k in range(1, n + 1) if 0 <= 2 * k + 1 < num_frames]

    # 2) explicit per-task boundaries.
    haystack = _normalize_task_text(task_group) + " || " + _normalize_task_text(instruction)
    for key, (steps, exp_frames) in _EXPLICIT_BOUNDARY_TASKS.items():
        if key in haystack:
            if num_frames != exp_frames:
                raise ValueError(
                    f"[Real] episode of '{key}' has {num_frames} frames, "
                    f"expected {exp_frames}. Memory boundaries would silently "
                    f"vanish — fix or exclude the episode."
                )
            return [s for s in steps if 0 <= s < num_frames]

    return []


class Real_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Union[str, List[str]],
        cameras: List[str] = ("3rd",),
        tasks: Union[str, List[str]] = "all",
        ep_per_task: int = 10_000,
        verbose: bool = True,
        img_stride: int = 4,
        memory_enabled: bool = False,
        memory_k_temporal: int = 4,
        memory_select: str = "keyframe_gt",
    ):
        """
        :param data_path: Root directory (or list of roots) containing
            task_group subdirectories (e.g. plate_tasks, shelf_tasks).
        :param tasks: Which task_group directories to include. Pass ``"all"``
            (default) to include every task_group under data_path, or a
            comma-separated string / list of names to select specific groups.
        :param img_stride: Spatial stride applied to ZED RGB + PCD before they
            are returned. The raw 1080x1920 ZED frames contain ~2 M points per
            sample, which is >30x the point count the sim trainer was tuned for;
            a default stride of 4 downsamples to ~130 K points per sample while
            preserving the image aspect ratio. Set to 1 to disable.
        :param memory_enabled: when True, __getitem__ also emits the episode's
            frame-0 anchor (``anchor_{cam}``) and the K most-recent history
            frames (``hist{k}_{cam}``) plus ``anchor_mask`` / ``hist_mask`` /
            ``step_idx``, matching Gembench_Dataset / MemoryBench_Dataset so
            RVTAgent.update_gembench can build the episodic-memory KV.
        :param memory_k_temporal: K = number of temporal memory slots. Must
            match the network's mvt_cfg.memory.k_temporal.
        :param memory_select: temporal-memory frame-selection policy, mirroring
            RMBench / the eval MemoryBank:
              * ``"keyframe_gt"`` (default): slot 0/1 = the two most-recent
                EXECUTED keyframes (t-1, t-2); slots 2..K-1 = GT subtask-
                boundary "memory keyframes" older than t-2. For the
                press-button-N-times tasks these boundaries are derived from
                the task name (three_times -> steps 3/5/7; twice -> 3/5; see
                :func:`_memory_boundary_steps`) and populate slots 2..K-1,
                exactly as RMBench admits its GT subtask boundaries. Tasks
                with no boundary keep ``mem_label`` all-zero, so slots 2..K-1
                stay masked and the memory is anchor + the two recency frames
                ("the two neighbouring frames").
              * ``"sliding"``: legacy K most-recent keyframes.
        """
        if isinstance(data_path, str):
            self.data_paths = [data_path]
        else:
            self.data_paths = list(data_path)
        self.cameras = list(cameras)
        self.verbose = verbose
        self.img_stride = max(1, int(img_stride))
        self.memory_enabled = bool(memory_enabled)
        self.memory_k_temporal = int(memory_k_temporal)
        self.memory_select = str(memory_select)
        # episode_path -> intrinsic.pkl contents (None for the legacy full-XYZ layout), filled lazily.
        self._intrinsic_cache: dict = {}

        # Resolve the set of task_group names to include.
        if isinstance(tasks, str):
            if tasks.strip().lower() == "all":
                self._task_filter: Union[None, set] = None  # None = accept all
            else:
                self._task_filter = {t.strip() for t in tasks.split(",") if t.strip()}
        else:
            self._task_filter = set(tasks) if tasks else None

        # Per-sample index of episode metadata; RGB/PCD are lazy-loaded in __getitem__ to keep memory flat.
        self.index: List[dict] = []
        self._build_index(ep_per_task)

    def _build_index(self, ep_per_task: int):
        t0 = time.time()
        n_episodes = 0
        skipped_episodes: List[tuple] = []
        for data_path in self.data_paths:
            if not os.path.isdir(data_path):
                raise FileNotFoundError(f"data_path not found: {data_path}")
            for task_group in sorted(os.listdir(data_path)):
                task_group_path = os.path.join(data_path, task_group)
                if not os.path.isdir(task_group_path):
                    continue
                if self._task_filter is not None and task_group not in self._task_filter:
                    continue
                for ep_name in sorted(os.listdir(task_group_path)):
                    ep_path = os.path.join(task_group_path, ep_name)
                    if not os.path.isdir(ep_path):
                        continue
                    zed_rgb_dir = os.path.join(ep_path, "zed_rgb")
                    if not os.path.isdir(zed_rgb_dir):
                        continue
                    # Count each frame once regardless of layout: a converted episode holds {i}.jpg (and {i}.pkl
                    # until the originals are deleted), an unconverted one only {i}.pkl. Keying on the stem
                    # de-duplicates; counting ".pkl" alone would drop every episode after cleanup.
                    frame_ids = {
                        os.path.splitext(f)[0]
                        for f in os.listdir(zed_rgb_dir)
                        if f.endswith((".jpg", ".pkl"))
                    }
                    num_frames = len(frame_ids)
                    if num_frames < 2:
                        # need at least one current->next transition
                        continue
                    # Depth-completeness guard. In the compact layout _load_pcd reads zed_depth/{i}.png and
                    # falls back to zed_pcd/{i}.pkl, which a converted episode does not have — so one missing
                    # depth frame would be a FileNotFoundError mid-epoch rather than at startup. Memory can pull
                    # ANY earlier frame, so a partial episode cannot be salvaged per-step: drop it whole and say so.
                    zed_depth_dir = os.path.join(ep_path, "zed_depth")
                    if os.path.isdir(zed_depth_dir):
                        depth_ids = {
                            os.path.splitext(f)[0]
                            for f in os.listdir(zed_depth_dir)
                            if f.endswith(".png")
                        }
                        missing = frame_ids - depth_ids
                        if missing:
                            skipped_episodes.append(
                                (ep_path, sorted(missing, key=_frame_sort_key))
                            )
                            continue
                    n_episodes += 1
                    # Frame k -> current observation, frame k+1 -> target action; the last frame has no target.
                    category = task_category(task_group)
                    for k in range(num_frames - 1):
                        self.index.append({
                            "episode_path": ep_path,
                            "episode_name": ep_name,
                            "task_group": task_group,
                            "category": category,
                            "step_idx": k,
                            "num_steps": num_frames,
                            "task_name": task_group,
                        })
        if self.verbose:
            task_groups_found = {e["task_group"] for e in self.index}
            print(f"[Real_Dataset] indexed {len(self.index)} samples from "
                  f"{n_episodes} episodes, task_groups={sorted(task_groups_found)}, "
                  f"across {len(self.data_paths)} root(s) in {time.time() - t0:.1f}s")
            known = {c for c, _ in _TASK_CATEGORIES}
            unmatched = sorted({e["task_group"] for e in self.index
                                if e["category"] not in known})
            if unmatched:
                print(f"[Real_Dataset] WARNING: {len(unmatched)} task_group(s) "
                      f"match no entry in _TASK_CATEGORIES and each became its "
                      f"OWN sampling category (they will each get a full "
                      f"category's share of the sampling budget — add them to "
                      f"_TASK_CATEGORIES if that is not what you want):")
                for t in unmatched:
                    print(f"    {t}")
            if skipped_episodes:
                print(f"[Real_Dataset] WARNING: skipped {len(skipped_episodes)} "
                      f"episode(s) with zed_rgb frames that have no matching "
                      f"zed_depth/*.png (incomplete conversion — would raise "
                      f"FileNotFoundError mid-training):")
                for ep_path, missing in skipped_episodes:
                    print(f"    {ep_path}  missing depth frames: {missing}")

    @property
    def num_tasks(self) -> int:
        return len({e["task_group"] for e in self.index})

    @property
    def num_task_paths(self) -> int:
        return len({entry["episode_path"] for entry in self.index})

    # ---- Multi-task sampling balance (ported from RLBench/utils/rlbench_keyframe_dataset.py; only the
    # index key differs: "task_group" here vs "task" there) ----

    def task_sampling_weights(self, alpha: float,
                              group_mode: str = "task") -> np.ndarray:
        """Per-index sampling weights for temperature balancing.

        ``self.index`` holds one entry per (episode, step) transition, so an
        unweighted sampler draws each task with probability proportional to its
        transition count ``n_t``. On the 7_19 real mix that is severely skewed
        (the two RedBull shelf tasks alone own ~48% of all transitions while
        each put_lids variant owns ~1%), so the shelf tasks would dominate the
        gradient and the long-horizon memory tasks would be starved.

        ``group_mode="task"`` (flat, the RLBench recipe) targets per-task
        probability ``p_t ∝ n_t**alpha``. Since a task owns ``n_t`` index
        entries the per-entry weight is ``w_i ∝ n_t**(alpha-1)``:

          * alpha=1   -> w_i ∝ 1             (transition-uniform, original)
          * alpha=0   -> w_i ∝ 1/n_t         (task-uniform)
          * alpha=0.5 -> w_i ∝ 1/sqrt(n_t)   (tempered middle ground)

        ``group_mode="category"`` is TWO-LEVEL and is what the real mix uses.
        The 19 task_group dirs are really 5 skills x N language variants
        (see :data:`_TASK_CATEGORIES`), and balancing flat over 19 names hands
        a skill a share proportional to how many object nouns we happened to
        record (drawer has 8 variants, swap_eggplant has 1). So instead:

          1. category c is drawn with ``p_c ∝ n_c**alpha`` (n_c = the
             category's total transitions) — the temperature acts HERE;
          2. within c every variant is equally likely, ``p(t|c) = 1/K_c``;
          3. within a variant every transition is equally likely, ``1/n_t``.

        giving ``w_i ∝ n_c**alpha / (K_c * n_t)``. alpha=0 makes the five
        skills exactly equiprobable; alpha=1 gives each skill its raw
        transition share while still averaging over variants inside it.

        Weights are NOT normalized (``torch.multinomial`` normalizes), and all
        counts come straight from the in-memory index (no extra IO).
        """
        t_counts = Counter(e["task_group"] for e in self.index)
        if group_mode == "task":
            exp = float(alpha) - 1.0
            return np.array(
                [t_counts[e["task_group"]] ** exp for e in self.index],
                dtype=np.float64,
            )
        if group_mode == "category":
            c_counts = Counter(e["category"] for e in self.index)
            variants = {}
            for e in self.index:
                variants.setdefault(e["category"], set()).add(e["task_group"])
            k_per_cat = {c: len(v) for c, v in variants.items()}
            a = float(alpha)
            return np.array(
                [(c_counts[e["category"]] ** a)
                 / (k_per_cat[e["category"]] * t_counts[e["task_group"]])
                 for e in self.index],
                dtype=np.float64,
            )
        raise ValueError(
            f"group_mode must be 'task' or 'category', got {group_mode!r}"
        )

    def task_transition_counts(self) -> "OrderedDict[str, int]":
        """Ordered {task_group: transition_count} over the index (for logs)."""
        out: "OrderedDict[str, int]" = OrderedDict()
        for e in self.index:
            out[e["task_group"]] = out.get(e["task_group"], 0) + 1
        return out

    def category_transition_counts(self) -> "OrderedDict[str, int]":
        """Ordered {category: transition_count} over the index (for logs)."""
        out: "OrderedDict[str, int]" = OrderedDict()
        for e in self.index:
            out[e["category"]] = out.get(e["category"], 0) + 1
        return out

    def category_to_tasks(self) -> "OrderedDict[str, list]":
        """Ordered {category: [task_group, ...]} over the index (for logs)."""
        out: "OrderedDict[str, list]" = OrderedDict()
        for e in self.index:
            lst = out.setdefault(e["category"], [])
            if e["task_group"] not in lst:
                lst.append(e["task_group"])
        return out

    def __len__(self) -> int:
        return len(self.index)

    def _episode_intrinsic(self, ep_path: str):
        """Per-episode camera intrinsics for the depth layout, or None.

        Cached per dataset instance (i.e. per DataLoader worker) so the pickle
        is read once per episode instead of once per frame.
        """
        if ep_path in self._intrinsic_cache:
            return self._intrinsic_cache[ep_path]
        path = os.path.join(ep_path, "intrinsic.pkl")
        meta = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                meta = pickle.load(f)
        self._intrinsic_cache[ep_path] = meta
        return meta

    def _load_pcd(self, ep_path: str, frame_step: int) -> np.ndarray:
        """Camera-frame (H,W,3) XYZ at ``img_stride``, from either layout.

        Preferred (compact) layout is ``zed_depth/{i}.png`` (uint16 depth) plus
        ``intrinsic.pkl``; the XY channels are reconstructed by pinhole
        deprojection, which is exact to ~1e-6 m (see
        data_collection/convert_pcd_to_depth.py). Falls back to the original
        ``zed_pcd/{i}.pkl`` full-XYZ dumps when the depth layout is absent, so
        unconverted datasets keep working unchanged.
        """
        depth_path = os.path.join(ep_path, "zed_depth", f"{frame_step}.png")
        meta = self._episode_intrinsic(ep_path)
        if meta is not None and os.path.exists(depth_path):
            d16 = _read_depth_png(depth_path)
            s = self.img_stride
            # Deproject AFTER striding, so only the kept pixels are computed. The u/v fed to the pinhole model
            # must be the ORIGINAL image columns/rows of those pixels (0, s, 2s, ...) — using 0..W/s would
            # silently rescale the whole cloud by a factor of s.
            h_full, w_full = d16.shape
            if s > 1:
                d16 = d16[::s, ::s]
            depth = d16.astype(np.float32) / np.float32(meta["depth_scale"])
            i = min(int(frame_step), len(meta["fx"]) - 1)
            fx, fy = float(meta["fx"][i]), float(meta["fy"][i])
            cx, cy = float(meta["cx"][i]), float(meta["cy"][i])
            u = np.arange(0, w_full, s, dtype=np.float32)[None, :]
            v = np.arange(0, h_full, s, dtype=np.float32)[:, None]
            x = ((u - np.float32(cx)) / np.float32(fx)) * depth
            y = ((v - np.float32(cy)) / np.float32(fy)) * depth
            return np.stack([x, y, depth], axis=-1)

        pcd_path = os.path.join(ep_path, "zed_pcd", f"{frame_step}.pkl")
        with open(pcd_path, "rb") as f:
            pcd_hw3 = pickle.load(f)[:, :, :3].astype(np.float32)
        if self.img_stride > 1:
            pcd_hw3 = pcd_hw3[::self.img_stride, ::self.img_stride]
        return pcd_hw3

    def _load_rgb(self, ep_path: str, frame_step: int) -> np.ndarray:
        """(H,W,3) uint8 RGB at ``img_stride``, preferring the JPEG layout."""
        jpg_path = os.path.join(ep_path, "zed_rgb", f"{frame_step}.jpg")
        if os.path.exists(jpg_path):
            rgb_hw3 = _read_rgb_jpeg(jpg_path)
        else:
            with open(os.path.join(ep_path, "zed_rgb", f"{frame_step}.pkl"), "rb") as f:
                rgb_hw3 = pickle.load(f)[:, :, :3]   # drop alpha if present
        if self.img_stride > 1:
            rgb_hw3 = rgb_hw3[::self.img_stride, ::self.img_stride]
        return rgb_hw3

    def _load_frame(self, ep_path: str, frame_step: int, extrinsic: np.ndarray):
        """Load one frame's (rgb_chw uint8, pcd_chw float32 in base frame) for
        the ZED ("3rd") camera. Shared by the current observation and the
        episodic-memory anchor / history frames so they all go through the
        identical stride + base-frame transform.
        """
        rgb_hw3 = np.ascontiguousarray(self._load_rgb(ep_path, frame_step))
        rgb_chw = np.transpose(rgb_hw3, (2, 0, 1)).astype(np.uint8)  # (3,H,W)

        pcd_hw3 = self._load_pcd(ep_path, frame_step)
        pcd_hw3 = _convert_pcd_to_base(pcd_hw3, extrinsic)
        pcd_chw = np.transpose(pcd_hw3, (2, 0, 1)).astype(np.float32)  # (3,H,W)
        return rgb_chw, pcd_chw

    def _mem_label_array(self, task_group: str, instruction: str,
                         num_steps: int) -> np.ndarray:
        """Per-step binary "is-subtask-boundary memory keyframe" labels.

        Length ``num_steps``; 1.0 at each task's subtask boundaries
        (press-button-N-times: three_times 3/5/7, twice 3/5;
        swap-two-eggplants: 4/8; put-lids-then-uncover: 4/8) and 0.0
        everywhere else and for tasks with no defined boundary. Serves two roles, exactly as in RMBench:
          * the keyframe-discriminator BCE target (``sample['mem_label']``);
          * admission of boundary frames into temporal-memory slots 2..K-1
            under ``memory_select='keyframe_gt'``.
        See :func:`_memory_boundary_steps` for the frame-layout derivation.
        """
        lbl = np.zeros((num_steps,), dtype=np.float32)
        for s in _memory_boundary_steps(task_group, instruction, num_steps):
            lbl[s] = 1.0
        return lbl

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        ep_path = entry["episode_path"]
        step = entry["step_idx"]
        num_steps = entry["num_steps"]

        # --- pose trajectory (shared file at episode level) ---
        gripper_pose = _load_pose_file(os.path.join(ep_path, "pose.pkl"))
        # GT target for this step is the next-frame pose; current is this frame.
        target_idx = min(step + 1, len(gripper_pose) - 1)
        tgt = gripper_pose[target_idx]
        cur = gripper_pose[min(step, len(gripper_pose) - 1)]

        tgt_xyz_m = np.asarray(tgt["position"], dtype=np.float32) / 1000.0  # mm->m
        tgt_quat = R.from_euler(
            "xyz", tgt["orientation"], degrees=True
        ).as_quat().astype(np.float32)  # (qx, qy, qz, qw)
        gt_gripper_pose = np.concatenate(
            [tgt_xyz_m, tgt_quat, np.array([tgt["claw_status"]], dtype=np.float32)]
        )  # shape (8,)

        # --- low-dim state: [current claw, time embedding] ---
        time_embed = (1.0 - (step / float(max(num_steps - 1, 1)))) * 2.0 - 1.0
        low_dim_state = np.asarray(
            [cur["claw_status"], time_embed], dtype=np.float32
        )

        # --- camera extrinsic (4x4 camera->base, shared at episode level) ---
        with open(os.path.join(ep_path, "extrinsic_matrix.pkl"), "rb") as f:
            extrinsic = np.asarray(pickle.load(f), dtype=np.float64)

        # --- language instruction (loaded early: it also drives the memory-keyframe labeling below) ---
        with open(os.path.join(ep_path, "instruction.pkl"), "rb") as f:
            instruction = str(pickle.load(f)).strip()

        sample = {
            "gripper_pose": gt_gripper_pose,          # (8,) float32
            "low_dim_state": low_dim_state,           # (2,) float32
            "ignore_collisions": np.float32(1.0),
        }

        # Per-__getitem__ frame cache: with keyframe_gt the K-2 unused slots all point at frame 0, so without
        # caching the same pkl would be read K times. Keyed by frame index.
        _frame_cache: dict = {}

        def _get_frame(frame_step: int):
            if frame_step not in _frame_cache:
                _frame_cache[frame_step] = self._load_frame(
                    ep_path, frame_step, extrinsic)
            return _frame_cache[frame_step]

        for cam in self.cameras:
            if cam != "3rd":
                # Only the ZED (logical "3rd") camera is present in this layout.
                raise ValueError(f"Camera {cam!r} is not available in this dataset")
            rgb_chw, pcd_chw = _get_frame(step)
            sample[cam] = {
                "rgb": rgb_chw,
                "pcd": pcd_chw,
            }

        # ---- Episodic memory bundle: anchor (frame 0) + K temporal slots ----
        # Same key layout as RMBench_Dataset / MemoryBench_Dataset. At step 0 the anchor arrays are still
        # emitted but masked off, so the spatial block falls through to identity. Visual tokens only.
        if self.memory_enabled:
            K = self.memory_k_temporal
            for cam in self.cameras:
                a_rgb, a_pcd = _get_frame(0)
                sample[f"anchor_{cam}"] = {"rgb": a_rgb, "pcd": a_pcd}
            sample["anchor_mask"] = np.array(step > 0, dtype=bool)
            sample["step_idx"] = np.int64(step)

            # Per-step subtask-boundary ("memory keyframe") labels for this episode: non-zero for the
            # press-button-N-times family and the explicit boundary tasks, all-zero otherwise. Drives both the
            # discriminator BCE target and the keyframe_gt slot admission. See _mem_label_array.
            mem_label_arr = self._mem_label_array(
                entry["task_group"], instruction, num_steps)

            # Keyframe discriminator target: is the CURRENT keyframe a subtask boundary? Supervises
            # out["mem_logit"] via BCE in the shared agent's _update_shared.
            sample["mem_label"] = np.float32(mem_label_arr[step])

            def _emit_hist_slot(slot, src_step):
                h_rgb, h_pcd = _get_frame(src_step)
                for cam in self.cameras:
                    sample[f"hist{slot}_{cam}"] = {"rgb": h_rgb, "pcd": h_pcd}

            hist_mask = np.zeros((K,), dtype=bool)
            if self.memory_select == "keyframe_gt":
                # RMBench layout: slot 0 = the previous executed keyframe (t-1), slot 1 = the one before;
                # slots 2..K-1 = GT subtask-boundary keyframes strictly older than t-2 (index <= step-3),
                # most-recent first. With no boundary label nothing is admitted -> anchor + two recency frames.
                slot_src = [0] * K
                if step - 1 >= 0:
                    slot_src[0] = step - 1
                    hist_mask[0] = True
                if K >= 2 and step - 2 >= 0:
                    slot_src[1] = step - 2
                    hist_mask[1] = True
                sel = [j for j in range(max(0, step - 2))
                       if mem_label_arr[j] == 1.0]
                n_mem_slots = max(0, K - 2)
                sel = sel[-n_mem_slots:] if n_mem_slots > 0 else []
                for i in range(len(sel)):
                    slot = 2 + i
                    slot_src[slot] = sel[-(i + 1)]  # slot 2 = most recent
                    hist_mask[slot] = True
                for slot in range(K):
                    _emit_hist_slot(slot, slot_src[slot])
            else:
                # LEGACY sliding window: K most-recent keyframes.
                for k in range(K):
                    hist_step = step - k - 1
                    valid = hist_step >= 0
                    _emit_hist_slot(k, hist_step if valid else 0)
                    if valid:
                        hist_mask[k] = True
            sample["hist_mask"] = hist_mask

        sample["lang_goal"] = instruction
        sample["tasks"] = entry["task_name"]

        return sample


if __name__ == "__main__":
    import sys
    # `python real/real_dataset.py` puts real/ on sys.path, so the sibling paths.py imports without PYTHONPATH.
    from paths import REAL_DATA_ROOT
    path = sys.argv[1] if len(sys.argv) > 1 else REAL_DATA_ROOT
    tasks_arg = sys.argv[2] if len(sys.argv) > 2 else "all"
    # 3rd positional arg: "memory" to exercise the anchor/history bundle.
    mem = len(sys.argv) > 3 and sys.argv[3].lower() in ("memory", "mem", "1", "true")
    ds = Real_Dataset(path, cameras=["3rd"], tasks=tasks_arg,
                      memory_enabled=mem, memory_k_temporal=4)
    print(f"total samples: {len(ds)}  (memory_enabled={mem})")
    # Index 0 is step 0 (anchor_mask should be False); pick a later index too
    # so a non-trivial hist_mask is visible when memory is on.
    probe = min(5, len(ds) - 1)
    s = ds[probe]
    print(f"--- sample[{probe}] ---")
    for k, v in s.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                print(f"  {k}.{k2}: {getattr(v2, 'shape', None)} {getattr(v2, 'dtype', None)}")
        elif isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} -> {v!r}")
