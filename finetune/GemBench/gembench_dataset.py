'''
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

'''
import os
import time
import random
import json
import hashlib
import fcntl
from collections import OrderedDict

import numpy as np
import torch
from tqdm import tqdm
import lmdb
import msgpack
import msgpack_numpy
msgpack_numpy.patch()


class Gembench_Dataset(torch.utils.data.Dataset):
    """Lazy-loading GemBench dataset.

    Startup only scans LMDB metadata (unpack per episode to read num_steps);
    the result is cached to disk as JSON and reused on subsequent runs.
    Heavy decoding (msgpack + transpose) happens in __getitem__ inside
    DataLoader workers, overlapping with GPU compute.
    """

    def __init__(
        self,
        data_path,
        device,
        cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
        ep_per_task=1000,
        tasks=None,
        cache_dir=None,
        episode_cache_size=4,
        memory_enabled=False,
        memory_k_temporal=2,
        memory_select="keyframe_gt",
    ):
        self.device = device
        self.data_path = data_path  # folder containing LMDB + instructions
        self.cameras = cameras
        self.tasks = tasks
        self.ep_per_task = int(ep_per_task)
        self.episode_cache_size = int(episode_cache_size)
        # Episodic memory: when enabled, __getitem__ also returns the
        # frame-0 anchor and K temporal history frames (rgb + pcd per
        # camera) plus a per-slot validity mask. All extra reads come from
        # the same cached LMDB record so it's IO-free at steady state.
        self.memory_enabled = bool(memory_enabled)
        self.memory_k_temporal = int(memory_k_temporal)
        # Memory frame-selection policy (mirrors RMBench / the eval MemoryBank):
        #   "keyframe_gt" -> slot 0/1 pinned to the two most-recent EXECUTED
        #       keyframes (t-1, t-2); slots 2..K-1 hold discriminator-admitted
        #       GT subtask-boundary keyframes. GemBench has NO boundary label,
        #       so ``mem_label`` is all-zero -> slots 2..K-1 stay empty and the
        #       memory is exactly anchor + the two recency frames.
        #   "sliding"     -> LEGACY: K most-recent keyframes.
        self.memory_select = str(memory_select)

        # Lazy-opened per-worker state. Must start empty so that values
        # populated in DataLoader workers (post-fork) are NOT shared with
        # the parent process — sharing lmdb envs across fork is unsafe.
        self._lmdb_envs = {}
        self._episode_cache = OrderedDict()  # keyed by (task, ep_key)

        self._episode_root = os.path.join(self.data_path, "keysteps_bbox/seed0")
        self._load_instructions()

        time.sleep(5)
        self.index = self._build_or_load_index(cache_dir)

        # Preserve the reporting contract used by train.py:
        #   num_tasks       = total tasks present on disk (matches old behavior)
        #   num_task_paths  = number of episodes actually included after filter
        self.num_tasks = len(os.listdir(self._episode_root))

    # ---------------------------------------------------------------- helpers

    def _load_instructions(self):
        path = os.path.join(self.data_path, "taskvars_instructions_new.json")
        with open(path, "r") as f:
            self.instruction_dict = json.load(f)

    def _cache_key(self):
        # Only inputs that affect index contents: task filter + ep_per_task.
        # `cameras` is applied at __getitem__ time, not during index build.
        tasks_part = ",".join(sorted(self.tasks)) if self.tasks else "__all__"
        payload = f"{tasks_part}|ep_per_task={self.ep_per_task}"
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def _build_or_load_index(self, cache_dir):
        cache_dir = cache_dir or os.path.join(self.data_path, "cache", "gembench_index")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"index_{self._cache_key()}.json")
        lock_file = cache_file + ".lock"

        # Fast path: cache already present — no contention, just load.
        if os.path.exists(cache_file):
            return self._load_index_from_file(cache_file)

        # Slow path: at least one rank must build. fcntl.LOCK_EX serializes
        # DDP ranks (on a shared filesystem) so only ONE scans; the rest
        # block until the lock is released, then find the cache ready.
        # Works across processes on the same node / shared mount. Safe
        # no-op on repeated acquires from the same process.
        with open(lock_file, "w") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            # Re-check inside the lock: another rank may have produced the
            # cache while we were waiting.
            if os.path.exists(cache_file):
                return self._load_index_from_file(cache_file)

            index, num_task_paths = self._scan_lmdb()
            self.num_task_paths = num_task_paths

            tmp_file = f"{cache_file}.tmp.{os.getpid()}"
            with open(tmp_file, "w") as f:
                json.dump({"num_task_paths": num_task_paths, "index": index}, f)
            os.replace(tmp_file, cache_file)
            # fcntl.flock is released automatically when lockf closes.
            return index

    def _load_index_from_file(self, cache_file):
        with open(cache_file, "r") as f:
            payload = json.load(f)
        self.num_task_paths = payload["num_task_paths"]
        return payload["index"]

    def _scan_lmdb(self):
        all_tasks = os.listdir(self._episode_root)
        if self.tasks is not None:
            all_tasks = [t for t in all_tasks if t in self.tasks]
        all_tasks = sorted(all_tasks)

        index = []
        num_task_paths = 0
        for task in tqdm(all_tasks, desc="Scanning LMDB (first-run index build)"):
            task_path = os.path.join(self._episode_root, task)
            env = lmdb.open(task_path, readonly=True, meminit=False, lock=False)
            try:
                with env.begin() as txn:
                    for i, (key, value) in enumerate(txn.cursor()):
                        if i >= self.ep_per_task:
                            break
                        num_task_paths += 1
                        episode = msgpack.unpackb(value)
                        num_steps = len(episode["key_frameids"])
                        ep_key_str = key.decode() if isinstance(key, (bytes, bytearray)) else str(key)
                        for step_idx in range(num_steps - 1):
                            index.append({
                                "task": task,
                                "ep_key": ep_key_str,
                                "step_idx": step_idx,
                                "num_steps": num_steps,
                            })
            finally:
                env.close()
        return index, num_task_paths

    # ------------------------------------------------------- runtime IO

    def _get_env(self, task):
        env = self._lmdb_envs.get(task)
        if env is None:
            env = lmdb.open(
                os.path.join(self._episode_root, task),
                readonly=True,
                meminit=False,
                lock=False,
            )
            self._lmdb_envs[task] = env
        return env

    def _get_episode(self, task, ep_key):
        cache_key = (task, ep_key)
        cached = self._episode_cache.get(cache_key)
        if cached is not None:
            self._episode_cache.move_to_end(cache_key)
            return cached

        env = self._get_env(task)
        key_bytes = ep_key.encode() if isinstance(ep_key, str) else ep_key
        with env.begin() as txn:
            raw = txn.get(key_bytes)
        if raw is None:
            raise KeyError(f"LMDB miss: task={task} ep_key={ep_key}")
        episode = msgpack.unpackb(raw)

        self._episode_cache[cache_key] = episode
        while len(self._episode_cache) > self.episode_cache_size:
            self._episode_cache.popitem(last=False)
        return episode

    # ----------------------------------------------------------- dataset API

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        entry = self.index[idx]
        task = entry["task"]
        step_idx = entry["step_idx"]
        num_steps = entry["num_steps"]

        episode = self._get_episode(task, entry["ep_key"])

        sample = {}
        for cam_idx, cam in enumerate(self.cameras):
            sample[cam] = {
                "pcd": np.transpose(episode["pc"][step_idx][cam_idx], (2, 0, 1)),
                "rgb": np.transpose(episode["rgb"][step_idx][cam_idx], (2, 0, 1)),
            }
        sample["gripper_pose"] = episode["action"][step_idx + 1]
        time_feat = (1. - (step_idx / float(num_steps - 1))) * 2. - 1.
        sample["low_dim_state"] = np.concatenate(
            [sample["gripper_pose"], [time_feat]]
        ).astype(np.float32)
        # NOTE: no "ignore_collisions" key. The 3DLoTus LMDB has no such field
        # (its action is (T, 8) = xyz + quat + gripper_open), so this used to be
        # a hardcoded constant 1.0 — a degenerate target that trained the
        # collision head toward a single class for nothing. GemBench eval never
        # reads the collision slot either (actioner requests the 8-D action and
        # the env is built with collision_checking=False), so the label is now
        # dropped end-to-end. Matching switch: predict_collision=false in
        # configs/gembench_config.yaml.
        sample["lang_goal"] = random.choice(self.instruction_dict[task])
        sample["tasks"] = task

        # ---- Episodic memory: anchor (frame 0) + K temporal slots. ----
        if self.memory_enabled:
            K = self.memory_k_temporal

            # Anchor frame is keyframe 0. For step_idx == 0 we let
            # anchor == current; the agent will set anchor_mask=False so
            # the spatial block falls through to identity at step 0.
            anchor_step = 0
            for cam_idx, cam in enumerate(self.cameras):
                sample[f"anchor_{cam}"] = {
                    "pcd": np.transpose(
                        episode["pc"][anchor_step][cam_idx], (2, 0, 1)
                    ),
                    "rgb": np.transpose(
                        episode["rgb"][anchor_step][cam_idx], (2, 0, 1)
                    ),
                }
            sample["anchor_mask"] = np.array(step_idx > 0, dtype=bool)
            sample["step_idx"] = np.int64(step_idx)

            # Keyframe discriminator target. GemBench data has no "is-subtask-
            # boundary" label, so it is always 0 (the discriminator is disabled
            # and the admission gate defaults to false). Emitted for forward-
            # compatibility; the single-arm train path does not read it.
            sample["mem_label"] = np.float32(0.0)

            def _emit_hist_slot(slot, src_step):
                # Only the visual tokens (rgb / pcd) are kept; historical
                # actions are NOT projected into the memory KV (the
                # ``action_proj`` MLP was removed from MemoryBlock — temporal
                # memory is purely visual + per-slot index PE).
                for cam_idx, cam in enumerate(self.cameras):
                    sample[f"hist{slot}_{cam}"] = {
                        "pcd": np.transpose(
                            episode["pc"][src_step][cam_idx], (2, 0, 1)
                        ),
                        "rgb": np.transpose(
                            episode["rgb"][src_step][cam_idx], (2, 0, 1)
                        ),
                    }

            hist_mask = np.zeros((K,), dtype=bool)
            if self.memory_select == "keyframe_gt":
                # RMBench layout: slot 0 = previous executed keyframe (t-1),
                # slot 1 = the one before (t-2); slots 2..K-1 = GT subtask-
                # boundary "memory keyframes" older than t-2. GemBench has no
                # boundary label (mem_label all-zero), so slots 2..K-1 stay
                # masked -> memory = anchor + the two recency frames.
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
                # LEGACY sliding window: slot k -> historical step step_idx-k-1.
                for k in range(K):
                    hist_step = step_idx - k - 1
                    valid = hist_step >= 0
                    # Safe step for dummy slots; mask=False makes the block
                    # ignore them anyway.
                    _emit_hist_slot(k, hist_step if valid else 0)
                    if valid:
                        hist_mask[k] = True
            sample["hist_mask"] = hist_mask
        return sample


if __name__ == "__main__":
    dataset = Gembench_Dataset(
        data_path="/PATH_TO_GEMBENCH_TRAIN_DATA/train_dataset",
        device="cuda:0",
        ep_per_task=3,
    )
    data = dataset[0]
    print(data["lang_goal"])
