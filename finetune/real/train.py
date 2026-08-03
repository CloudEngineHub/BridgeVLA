"""BridgeVLA++ real-robot offline trainer (Dobot + ZED, 20251011 dataset).

Loosely modeled on finetune/RLBench/train.py, but:
  * no replay buffer — we stream samples from `Real_Dataset` via DataLoader
  * uses `RVTAgent.update_gembench()` (dict-style replay_sample format)
  * passes `cameras=["3rd"]` to the agent and `SCENE_BOUNDS_REAL` at construction
  * the real dataset is small, so epochs/bs/lr are tuned accordingly in the yaml

Launch (single GPU debug):
    DEBUG=true python finetune/real/train.py \
        --exp_cfg_path finetune/real/configs/real_config.yaml \
        --mvt_cfg_path finetune/real/configs/mvt_cfg.yaml

Launch (multi-GPU):
    torchrun --nproc_per_node=4 finetune/real/train.py ...
"""

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import tqdm
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ["BITSANDBYTES_NOWELCOME"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _TeeStream:
    # Mirror Python-level stdout/stderr into a log file so a run folder always contains its own log.
    # C-library writes through fd 1/2 still need the shell tee.
    def __init__(self, primary, log_file):
        self._primary = primary
        self._log = log_file

    def write(self, data):
        self._primary.write(data)
        try:
            self._log.write(data)
        except Exception:
            pass
        return len(data) if isinstance(data, (str, bytes)) else 0

    def flush(self):
        self._primary.flush()
        try:
            self._log.flush()
        except Exception:
            pass

    def isatty(self):
        return getattr(self._primary, "isatty", lambda: False)()

    def fileno(self):
        return self._primary.fileno()

    def __getattr__(self, name):
        return getattr(self._primary, name)


def _install_tee_logging(run_dir: str, rank: int) -> None:
    # Per-rank log avoids concurrent writes on a shared FS; line-buffered so `tail -f` keeps up.
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"train_rank{rank}.log")
    log_fp = open(log_path, "a", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)
    print(f"[real/train] rank {rank} logging stdout/stderr to {log_path}", flush=True)


import bridgevla.config as exp_cfg_mod                      # noqa: E402
import bridgevla.models.bridgevla_agent as bridgevla_agent  # noqa: E402
import bridgevla.mvt.config as mvt_cfg_mod                  # noqa: E402
from bridgevla.mvt.mvt import MVT                           # noqa: E402
from bridgevla.utils.rvt_utils import get_num_feat          # noqa: E402
from real.paths import REAL_DATA_ROOT, expand as expand_path  # noqa: E402
from real.real_dataset import Real_Dataset                  # noqa: E402
from real.utils.peract_utils import (                       # noqa: E402
    CAMERAS_REAL,
    IMAGE_SIZE,
    SCENE_BOUNDS_REAL,
)
from real.visualize import visualize_epoch                  # noqa: E402

try:
    import swanlab  # noqa: F401
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

USE_SWANLAB = False


# distributed helpers

def setup_distributed(backend: str = "nccl", port=None):
    num_gpus = torch.cuda.device_count()

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(29567 + num_gpus)
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % max(num_gpus, 1))
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        pass  # set by torchrun
    elif os.getenv("DEBUG", "false").lower() == "true":
        print("[real/train] No distributed env vars — entering single-GPU debug mode.")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        # random free port to avoid clashing with concurrent debug runs
        import random as _rnd
        os.environ.setdefault("MASTER_PORT", str(_rnd.randint(29600, 29999)))
        os.environ.setdefault("LOCAL_RANK", "0")
    else:
        raise RuntimeError(
            "Distributed env vars not found. Launch with torchrun / srun, "
            "or set DEBUG=true for single-GPU mode."
        )
    # Generous timeout: the rank-0 start-of-epoch viz runs while other ranks wait at the epoch barrier, and
    # NCCL's 600 s default watchdog can fire if it exceeds 10 min.
    dist.init_process_group(
        backend=backend,
        world_size=int(os.environ["WORLD_SIZE"]),
        rank=int(os.environ["RANK"]),
        timeout=timedelta(hours=1),
    )


def reduce_mean(value) -> float:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return float(value)
    t = torch.tensor(float(value), device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


# misc helpers

def get_time_folder() -> str:
    # REAL_RUN_STAMP lets train.sh pre-compute the stamp so the shell tee and Python share one run folder.
    env_stamp = os.environ.get("REAL_RUN_STAMP")
    if env_stamp:
        return env_stamp
    import datetime
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def memory_ablation_suffix(exp_cfg):
    """Run-name suffix marking which memory group is ablated, or "" when memory
    is full / disabled. Mirrors RMBench_vla/train.py; also mirrored in train.sh
    so the shell-side tee log lands in the same run dir Python creates.
      * temporal_memory off (stage-1 coarse memory)  -> "_no_temporal_mem"
      * spatial_memory  off (stage-2 fine anchor)    -> "_no_spatial_mem"
      * both off                                     -> "_no_mem"
    """
    mem = getattr(exp_cfg, "memory", None)
    if mem is None or not bool(getattr(mem, "enabled", False)):
        return ""
    t = bool(getattr(mem, "temporal_memory", True))
    s = bool(getattr(mem, "spatial_memory", True))
    if not t and not s:
        return "_no_mem"
    if not t:
        return "_no_temporal_mem"
    if not s:
        return "_no_spatial_mem"
    return ""


def get_logdir(cmd_args, exp_cfg) -> str:
    # expand_path lets the YAML log_dir be written as ${BRIDGEVLA_DATA_ROOT}/logs_real (machine-independent);
    # the --log_dir train.sh passes is already absolute, and expand is idempotent.
    root = expand_path(cmd_args.log_dir or exp_cfg.log_dir)
    # Idempotent: only append the ablation suffix when absent, so a swanlab_run that already carries it
    # (e.g. via --exp_cfg_opts) does not get it twice.
    base = exp_cfg.swanlab_run
    suffix = memory_ablation_suffix(exp_cfg)
    if suffix and not base.endswith(suffix):
        base = f"{base}{suffix}"
    run_name = f"{base}_{get_time_folder()}"
    log_dir = os.path.join(root, "train", run_name)
    if cmd_args.debug:
        log_dir = os.path.join(log_dir, "debug")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    # stash the run name as-is so swanlab uses it even when debug appends a sub-dir
    setattr(get_logdir, "last_run_name", run_name)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir: str):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as f:
        with redirect_stdout(f):
            print(exp_cfg.dump())
    with open(f"{log_dir}/mvt_cfg.yaml", "w") as f:
        with redirect_stdout(f):
            print(mvt_cfg.dump())
    with open(f"{log_dir}/args.yaml", "w") as f:
        yaml.dump(cmd_args.__dict__, f)


def save_agent(agent, path: str, epoch: int, freeze_epochs=None,
               save_optimizer_state=False):
    """Checkpoint the agent.

    ``epoch`` / ``model_state`` are kept verbatim so eval-time loaders
    (rvt_utils.load_agent / eval_flask_app) stay byte-compatible with old
    checkpoints.

    ``global_step`` / ``freeze_epochs`` are scalars (no file-size cost) and
    are ALWAYS written — freeze_epochs feeds the resume-time consistency
    warning, and global_step documents where the LR warmup stood.

    ``save_optimizer_state`` (exp_cfg.save_optimizer_state, default False —
    mirrors RMBench_vla/train.py) gates only the heavy part:

      * optimizer_state — AdamW moment buffers + per-group lr (~2x the model
                          weights in size). Needed for resume to continue
                          training EXACTLY where it left off; without it,
                          restore_checkpoint falls back to a freshly-built
                          optimizer + global_step=0 (warmup restarts).
    """
    model = agent._network
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    ckpt = {
        "epoch": epoch,
        "model_state": model_state,
        "global_step": int(getattr(agent, "_global_step", 0)),
        "freeze_epochs": freeze_epochs,
    }
    if save_optimizer_state:
        ckpt["optimizer_state"] = agent._optimizer.state_dict()
    torch.save(ckpt, path)


def restore_checkpoint(agent, ckpt, local_rank, rank, freeze_epochs):
    """Restore model + optimizer + LR-warmup step from a training checkpoint.

    Must be called AFTER agent.build() (so agent._optimizer exists) and AFTER
    the caller has reconstructed the correct stage (Stage-1 vs Stage-2) so the
    optimizer's param-group layout matches the saved state. Returns the epoch
    to resume FROM (saved_epoch + 1). Mirrors GemBench/train.py — see there for
    the full stage-reconstruction rationale.
    """
    # 1) Model weights (handles DDP-wrapped or bare module).
    model = agent._network.module if isinstance(agent._network, DDP) else agent._network
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if rank == 0 and (missing or unexpected):
        print(f"[resume] model_state loaded with missing={len(missing)} "
              f"unexpected={len(unexpected)} keys (strict=False).")

    saved_epoch = int(ckpt["epoch"])
    start_epoch = saved_epoch + 1

    # 2) freeze_epochs consistency check — config wins, but warn loudly.
    saved_fe = ckpt.get("freeze_epochs", None)
    if rank == 0 and saved_fe is not None and int(saved_fe) != int(freeze_epochs):
        print(f"[resume] WARNING: freeze_epochs changed since checkpoint "
              f"(saved={saved_fe}, current={freeze_epochs}). Using current "
              f"config; if the checkpoint was already in Stage 2, do NOT raise "
              f"freeze_epochs above the resume epoch.")

    # 3) global_step drives the LR-warmup scale. Only restore optimizer state when it belongs to the SAME
    #    optimizer we just (re)built: at start_epoch == freeze_epochs the loop builds a fresh Stage-2
    #    optimizer, so the saved Stage-1 state would be overwritten — skip it.
    opt_state = ckpt.get("optimizer_state", None)
    can_load_opt = opt_state is not None and (
        start_epoch < freeze_epochs or start_epoch > freeze_epochs
    )
    if start_epoch == freeze_epochs:
        if rank == 0:
            print(f"[resume] start_epoch ({start_epoch}) == freeze_epochs; the "
                  f"Stage-2 optimizer will be rebuilt fresh by the loop. "
                  f"Skipping optimizer-state load; global_step stays 0.")
    elif can_load_opt:
        try:
            agent._optimizer.load_state_dict(opt_state)
            for state in agent._optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(local_rank)
            agent._global_step = int(ckpt.get("global_step", 0))
            if rank == 0:
                print(f"[resume] optimizer state restored; global_step="
                      f"{agent._global_step} (LR warmup continues).")
        except (ValueError, KeyError) as e:
            if rank == 0:
                print(f"[resume] WARNING: optimizer state load failed ({e}); "
                      f"continuing with a freshly-built optimizer, global_step=0.")
    else:
        if rank == 0:
            print("[resume] no optimizer_state in checkpoint (old format); "
                  "continuing with a freshly-built optimizer, global_step=0.")

    if rank == 0:
        print(f"[resume] resuming from epoch {saved_epoch} -> start_epoch "
              f"{start_epoch}.")
    return start_epoch


# data loading

class DistributedWeightedSampler(torch.utils.data.Sampler):
    """DDP-correct weighted (with-replacement) sampler for task balancing.

    Ported from RLBench/train.py. PyTorch ships ``WeightedRandomSampler``
    (single-process) and ``DistributedSampler`` (uniform, multi-process) but
    not their combination. This fills the gap: every rank seeds an IDENTICAL
    generator from ``seed + epoch`` and draws the SAME ``total_size``
    multinomial indices, then each rank keeps a strided
    (``rank::world_size``) disjoint slice. So across ranks the union is exactly
    one epoch's worth of draws with no overlap, and ``set_epoch`` reshuffles
    deterministically (mirrors DistributedSampler's contract).
    ``num_samples`` defaults to ``len(weights)`` so the per-epoch iteration
    count matches the old uniform sampler; sampling is WITH replacement (the
    point of reweighting), so some transitions repeat and others are skipped
    within an epoch.
    """

    def __init__(self, weights, num_replicas, rank, num_samples=None, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        n = int(num_samples) if num_samples is not None else self.weights.numel()
        # Total drawn across ALL ranks, made divisible by world_size so the strided slices are equal-length.
        self.total_size = (n // self.num_replicas) * self.num_replicas
        self.num_samples = self.total_size // self.num_replicas

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        idx = torch.multinomial(
            self.weights, self.total_size, replacement=True, generator=g
        )
        idx = idx[self.rank:self.total_size:self.num_replicas]
        return iter(idx.tolist())

    def __len__(self):
        return self.num_samples


def build_task_sampler(exp_cfg, dataset, rank: int, world_size: int,
                       local_rank: int = 0):
    """Build the task-balancing sampler from ``exp_cfg.task_sampling``.

    Returns None only for plain "transition_uniform" at epoch_size_mult=1,
    which leaves build_dataloader on the stock uniform DistributedSampler
    (old behavior). Rank 0 prints the per-task raw vs. effective share table
    and the resulting epoch length before training.
    """
    _ts_cfg = getattr(exp_cfg, "task_sampling", None)
    _ts_mode = str(getattr(_ts_cfg, "mode", "transition_uniform")) \
        if _ts_cfg is not None else "transition_uniform"
    _mult = float(getattr(_ts_cfg, "epoch_size_mult", 1.0)) if _ts_cfg is not None else 1.0
    if _mult <= 0:
        raise ValueError(
            f"task_sampling.epoch_size_mult must be > 0, got {_mult}"
        )
    # An epoch draws mult * len(dataset) samples; mult>1 lengthens epochs so per-epoch bookkeeping fires at a
    # sane cadence on this small dataset. The yaml's epoch-denominated knobs are scaled to match.
    n_draws = int(round(_mult * len(dataset)))

    counts = dataset.task_transition_counts()
    tot = sum(counts.values()) or 1

    _group = str(getattr(_ts_cfg, "group_mode", "task")) if _ts_cfg is not None else "task"

    if _ts_mode == "temperature":
        alpha = float(getattr(_ts_cfg, "alpha", 1.0))
        weights = dataset.task_sampling_weights(alpha, group_mode=_group)
        if local_rank == 0:
            if _group == "category":
                # Two-level: p_c ∝ n_c**alpha, then uniform over the variants inside c (per-variant share = p_c / K_c).
                c_counts = dataset.category_transition_counts()
                c2t = dataset.category_to_tasks()
                cw = {c: (n ** alpha) for c, n in c_counts.items()}
                csum = sum(cw.values()) or 1.0
                print(f"[task_sampling] mode=temperature group_mode=category "
                      f"alpha={alpha}: category drawn ∝ n_c**alpha, then "
                      f"UNIFORM across the variants inside it "
                      f"(alpha=1 raw-share, alpha=0 all 5 skills equal).")
                print(f"  {'category / task':62s} {'n_trans':>8s} {'raw%':>7s} {'eff%':>7s}")
                for c, n_c in sorted(c_counts.items(), key=lambda kv: -kv[1]):
                    eff_c = 100 * cw[c] / csum
                    ts = c2t[c]
                    print(f"  {c:62s} {n_c:8d} {100*n_c/tot:6.1f}% {eff_c:6.1f}%"
                          f"   ({len(ts)} variant{'s' if len(ts) != 1 else ''})")
                    for t in sorted(ts, key=lambda x: -counts[x]):
                        print(f"    {t:60s} {counts[t]:8d} "
                              f"{100*counts[t]/tot:6.1f}% {eff_c/len(ts):6.1f}%")
                print(f"  {'TOTAL':62s} {tot:8d} {100.0:6.1f}% {100.0:6.1f}%")
            else:
                # Flat: effective per-task share p_t ∝ n_t**alpha.
                pw = {t: (n ** alpha) for t, n in counts.items()}
                psum = sum(pw.values()) or 1.0
                print(f"[task_sampling] mode=temperature group_mode=task "
                      f"alpha={alpha} (alpha=1 transition-uniform, 0 task-uniform):")
                print(f"  {'task':62s} {'n_trans':>8s} {'raw%':>7s} {'eff%':>7s}")
                for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
                    print(f"  {t:62s} {n:8d} {100*n/tot:6.1f}% {100*pw[t]/psum:6.1f}%")
                print(f"  {'TOTAL':62s} {tot:8d} {100.0:6.1f}% {100.0:6.1f}%")
        return DistributedWeightedSampler(
            weights, num_replicas=world_size, rank=rank,
            num_samples=n_draws, seed=0,
        )

    if local_rank == 0:
        if _ts_mode != "transition_uniform":
            print(f"[task_sampling] WARN: unknown mode {_ts_mode!r}; falling "
                  f"back to transition_uniform.")
        print(f"[task_sampling] mode=transition_uniform "
              f"(no reweighting; each task drawn ∝ its transition count):")
        print(f"  {'task':62s} {'n_trans':>8s} {'raw%':>7s}")
        for t, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {t:62s} {n:8d} {100*n/tot:6.1f}%")
        print(f"  {'TOTAL':62s} {tot:8d} {100.0:6.1f}%")
    if _mult == 1.0:
        return None   # stock DistributedSampler: one shuffled pass, no replacement
    # epoch_size_mult only exists on DistributedWeightedSampler, so honor it here too with all-equal weights
    # rather than silently ignoring it — the yaml's epoch-denominated knobs are scaled for the LONGER epoch,
    # so a silent fallback to a 1x epoch would cut total training by that factor.
    return DistributedWeightedSampler(
        np.ones(len(dataset), dtype=np.float64), num_replicas=world_size,
        rank=rank, num_samples=n_draws, seed=0,
    )


def build_dataloader(dataset, rank: int, world_size: int,
                     batch_size: int, num_workers: int, collate_fn=None,
                     sampler=None):
    if sampler is None:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        drop_last=True,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    loader = DataLoader(dataset, **kwargs)
    return loader, sampler


# training

def move_batch_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    return batch


def train_one_epoch(agent, loader, cameras, rank, max_iter=None,
                    epoch_idx=0, global_step_base=0):
    """Run one epoch of real (up_action) grad steps.

    tqdm's total is the number of backward passes this epoch — every bar tick
    == one gradient update.
    """
    agent.train()
    epoch_losses = defaultdict(list)

    n_real_iters = len(loader) if max_iter is None else min(max_iter, len(loader))
    pbar = tqdm.tqdm(total=n_real_iters, disable=(rank != 0),
                     position=0, leave=True, desc=f"epoch {epoch_idx}")
    step_global = 0
    last_r_loss = None

    def _update_postfix(lr: float):
        if rank != 0:
            return
        postfix = {"lr": f"{lr:.2e}"}
        if last_r_loss is not None:
            postfix["loss"] = f"{last_r_loss:.3f}"
        pbar.set_postfix(**postfix)

    for step_idx, raw_batch in enumerate(loader):
        if max_iter is not None and step_idx >= max_iter:
            break
        # ---- real (up_action) ----
        batch = move_batch_to_device(raw_batch, agent._device)
        batch["lang_goal"] = [[[item]] for item in raw_batch["lang_goal"]]
        batch["tasks"] = list(raw_batch["tasks"])

        out = agent.update_gembench(
            replay_sample=batch,
            backprop=True,
            reset_log=(step_idx == 0),
            cameras=cameras,
        )
        for k, v in out.items():
            epoch_losses[f"real/{k}"].append(reduce_mean(v))

        if rank == 0 and USE_SWANLAB:
            swanlab.log(
                {f"real/{k}": v for k, v in out.items()},
                step=global_step_base + step_global,
            )
        last_r_loss = out.get("total_loss")
        _update_postfix(out.get("lr", 0.0))
        pbar.update(1)
        step_global += 1

    pbar.close()
    return {k: sum(v) / max(len(v), 1) for k, v in epoch_losses.items()}


# experiment

def experiment(cmd_args):
    setup_distributed()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)

    # ---- Resume vs pretrain mutual exclusion (resume wins) ----
    # A resume restores the full training state, so warm-starting from a pretrain checkpoint at the same time
    # would be contradictory.
    if cmd_args.resume_path:
        if cmd_args.load_pretrain and rank == 0:
            print("[resume] --resume_path is set; ignoring --load_pretrain "
                  "(resume takes priority over pretrain warm-start).")
        cmd_args.load_pretrain = False
        cmd_args.pretrain_path = None

    # ---- configs ----
    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path:
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts:
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    epochs = cmd_args.epochs if cmd_args.epochs is not None else exp_cfg.epochs

    if cmd_args.exp_cfg_opts:
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts:
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"
    # Tag the SwanLab run name when freeze_epochs is overridden from the CLI, to distinguish those runs.
    if cmd_args.freeze_epochs is not None:
        exp_cfg.swanlab_run = f"{exp_cfg.swanlab_run}_freeze{cmd_args.freeze_epochs}"
    exp_cfg.freeze()

    BATCH_SIZE = exp_cfg.bs
    if rank == 0:
        print(f"[real/train] world_size={world_size}, bs_per_gpu={BATCH_SIZE}, epochs={epochs}")

    # ---- log dir ----
    if rank == 0:
        log_dir = get_logdir(cmd_args, exp_cfg)
        log_dir_list = [log_dir]
    else:
        log_dir_list = [None]
    dist.broadcast_object_list(log_dir_list, src=0)
    log_dir = log_dir_list[0]

    # Resume: pin log_dir to the EXACT directory holding the checkpoint, so new checkpoints / logs / viz land
    # back in the original run folder regardless of the current --log_dir root.
    if cmd_args.resume_path:
        log_dir = os.path.dirname(os.path.abspath(cmd_args.resume_path))
        if rank == 0:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[resume] writing into original run folder: {log_dir}")

    # Mirror each rank's stdout/stderr into <log_dir>/train_rank{rank}.log, alongside the shell-level tee.
    _install_tee_logging(log_dir, rank)

    # ---- Episodic memory cfg (single source of truth = exp_cfg.memory) ----
    # Read here so the dataset can emit anchor / history bundles, and propagated onto mvt_cfg.memory below.
    _mem_node = getattr(exp_cfg, "memory", None)
    _mem_enabled = bool(getattr(_mem_node, "enabled", False)) if _mem_node is not None else False
    _mem_k = int(getattr(_mem_node, "k_temporal", 4)) if _mem_node is not None else 4
    # Frame-selection policy mirrors RMBench / MemoryBench. Default keyframe_gt: slots 0/1 = the two
    # most-recent executed keyframes, slots 2..K-1 = GT subtask-boundary keyframes. For the real
    # press-button-N-times tasks those are the completed-press frames derived from the task name (see
    # Real_Dataset._memory_boundary_steps); other tasks carry no boundary.
    _mem_select = str(getattr(_mem_node, "select", "keyframe_gt")) if _mem_node is not None else "keyframe_gt"

    # ---- dataset ----
    t0 = time.time()
    dataset = Real_Dataset(
        data_path=cmd_args.data_folder,
        cameras=CAMERAS_REAL,
        tasks=exp_cfg.tasks,
        verbose=(rank == 0),
        memory_enabled=_mem_enabled,
        memory_k_temporal=_mem_k,
        memory_select=_mem_select,
    )
    if rank == 0:
        print(f"[real/train] dataset: {len(dataset)} samples, "
              f"{dataset.num_task_paths} episodes")
    # Multi-task sampling balance (exp_cfg.task_sampling), the same knob as RLBench/RMBench_vla. Default
    # "transition_uniform" keeps the uniform DistributedSampler; "temperature" draws task t ∝ n_t**alpha so
    # the high-episode-count shelf tasks stop swamping the long-horizon memory tasks.
    task_sampler = build_task_sampler(
        exp_cfg, dataset, rank, world_size, local_rank=local_rank
    )
    loader, sampler = build_dataloader(
        dataset, rank, world_size, BATCH_SIZE, exp_cfg.num_workers,
        sampler=task_sampler,
    )
    if rank == 0:
        _mult = float(getattr(getattr(exp_cfg, "task_sampling", None),
                              "epoch_size_mult", 1.0))
        print(f"[task_sampling] epoch_size_mult={_mult} -> one epoch draws "
              f"{len(sampler) * world_size} samples "
              f"({_mult:g}x the {len(dataset)}-transition dataset) = "
              f"{len(loader)} steps/rank at bs={BATCH_SIZE} x {world_size} ranks "
              f"(global batch {BATCH_SIZE * world_size}).")
        print(f"[task_sampling] NOTE: epochs / save_every_n_epochs / "
              f"freeze_epochs / viz_every_n_epochs count these LONGER epochs; "
              f"warmup_steps={getattr(exp_cfg, 'warmup_steps', None)} counts "
              f"STEPS and is unaffected by epoch_size_mult.")
        print(f"[real/train] dataloader built in {time.time() - t0:.1f}s")

    # ---- mvt + agent ----
    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path:
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts:
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))
    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)

    # --- Rotation head selection (exp_cfg.rotation_representation), mirroring RMBench / memoryBench ---
    # "6d" -> continuous 6D regression head (rot_ver==2): the feat vector's rotation slice is a 6D vector
    # (Zhou et al., CVPR 2019), so feat_dim = 6 + grip(2) + collision(2) = 10, replacing mvt_cfg.yaml's
    # rot_ver=1 head. "euler_disc" keeps mvt_cfg.yaml's rot_ver and the get_num_feat feat_dim.
    _rot_repr = str(getattr(exp_cfg, "rotation_representation", "euler_disc"))
    if _rot_repr == "6d":
        mvt_cfg.rot_ver = 2
        mvt_cfg.feat_dim = 6 + 2 + 2  # 6D rot + grip(2) + collision(2)
    elif _rot_repr != "euler_disc":
        raise ValueError(
            f"Unknown rotation_representation={_rot_repr!r} "
            "(expected 'euler_disc' or '6d')"
        )

    # Propagate the exp_cfg-level toggles the network consumes onto mvt_cfg before MVT() is built (yacs
    # scopes are independent). real_config.yaml stays the single source of truth.
    if hasattr(exp_cfg, "gradient_checkpointing") and hasattr(mvt_cfg, "gradient_checkpointing"):
        mvt_cfg.gradient_checkpointing = bool(exp_cfg.gradient_checkpointing)
    if hasattr(exp_cfg, "feat_from_stage1") and hasattr(mvt_cfg, "feat_from_stage1"):
        # Rotation/grip/collision feature source: stage 1 (coarse) when True.
        mvt_cfg.feat_from_stage1 = bool(exp_cfg.feat_from_stage1)
    # Mirror the unified-yaml memory propagation from RMBench / memoryBench. ``select`` + ``discriminator``
    # go too, so the dataset layout and the eval MemoryBank stay aligned and the dumped mvt_cfg.yaml
    # self-documents the memory policy for eval-time load_agent.
    if hasattr(exp_cfg, "memory") and hasattr(mvt_cfg, "memory"):
        for _k in (
            "enabled", "k_temporal", "select",
            "spatial_at_mvt1", "spatial_at_mvt2", "temporal_at_mvt1",
            "temporal_memory", "spatial_memory",
            "share_spatial_across_stages",
            "heads", "dim_head", "num_layers", "ffn_mult", "use_fast_attn",
            "grad_through_tokens", "discriminator",
        ):
            if hasattr(exp_cfg.memory, _k):
                setattr(mvt_cfg.memory, _k, getattr(exp_cfg.memory, _k))

    mvt_cfg.freeze()

    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        **mvt_cfg,
    ).to(device_id)
    backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS_REAL,
        cameras=CAMERAS_REAL,
        log_dir=f"{log_dir}/test_run/",
        align_real_frame=True,
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 300)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    # ---- Stage 1 freeze: PaliGemma backbone frozen; heads + feat_fc trainable ----
    # vision_tower (SigLIP) is permanently frozen, and ``up_grounding`` (only built when
    # use_modified_focal_loss=True) is never trained by the real finetune.
    always_freeze = [
        "lm_head", "embed_tokens",
        "vision_tower",
    ]
    use_focal = bool(getattr(mvt_cfg, "use_modified_focal_loss", False))
    if use_focal:
        always_freeze.append("up_grounding")
    FREEZE_EPOCHS = int(
        cmd_args.freeze_epochs
        if cmd_args.freeze_epochs is not None
        else getattr(exp_cfg, "freeze_epochs", 4)
    )
    for name, param in agent._network.named_parameters():
        if any(af in name for af in always_freeze):
            param.requires_grad = False
        elif "mvt1.model" in name:
            param.requires_grad = False
    if rank == 0:
        if use_focal:
            grounding_msg = "up_grounding permanently frozen"
        else:
            grounding_msg = "no up_grounding head (use_modified_focal_loss=False)"
        print(f"[real/train] Stage 1: PaliGemma frozen for {FREEZE_EPOCHS} epochs "
              f"— heatmap head + feat_fc heads trainable "
              f"(vision_tower frozen; {grounding_msg}).")
    trainable = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    if rank == 0:
        print(f"[real/train] trainable params: {trainable / 1e9:.3f} B")

    agent.build(training=True, device=device_id)

    start_epoch = 0

    # ---- Resume from a training checkpoint (full state) ----
    # Scheduling config (freeze_epochs / save_every_n_epochs / epochs / warmup_steps / lr) always comes from
    # the current YAML+CLI; the checkpoint carries only training STATE. freeze_epochs / warmup_steps should
    # be left unchanged on resume, since they anchor the restored optimizer state.
    if cmd_args.resume_path:
        if not os.path.isfile(cmd_args.resume_path):
            raise FileNotFoundError(
                f"--resume_path is not a file: {cmd_args.resume_path}"
            )
        if rank == 0:
            print(f"[resume] loading checkpoint: {cmd_args.resume_path}")
        resume_ckpt = torch.load(cmd_args.resume_path, map_location="cpu")
        resume_start_epoch = int(resume_ckpt["epoch"]) + 1

        # Reconstruct the Stage-1 -> Stage-2 transition BEFORE loading optimizer state when the checkpoint
        # was already in Stage 2, so the param-group layout matches the saved Stage-2 optimizer.
        if resume_start_epoch > FREEZE_EPOCHS:
            for name, param in agent._network.named_parameters():
                if any(af in name for af in always_freeze):
                    param.requires_grad = False
                elif "mvt1.model" in name:
                    param.requires_grad = True
            inner = agent._network.module if isinstance(agent._network, DDP) else agent._network
            agent._network = DDP(
                inner, device_ids=[local_rank], find_unused_parameters=True,
            )
            agent.rebuild_optimizer()
            if rank == 0:
                print(f"[resume] reconstructed Stage 2 (start_epoch="
                      f"{resume_start_epoch} > freeze_epochs={FREEZE_EPOCHS}): "
                      f"unfroze PaliGemma + rebuilt DDP + optimizer before "
                      f"loading optimizer state.")

        start_epoch = restore_checkpoint(
            agent, resume_ckpt,
            local_rank=local_rank, rank=rank, freeze_epochs=FREEZE_EPOCHS,
        )
        del resume_ckpt

    # ---- dump cfgs (after construction so mvt_cfg is finalized) ----
    if rank == 0:
        exp_cfg.defrost()
        t_lr = exp_cfg.peract.lr
        t_id = exp_cfg.exp_id
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = t_lr
        exp_cfg.exp_id = t_id
        exp_cfg.freeze()

    # ---- swanlab (mirrors RLBench/train.py) ----
    # Mode comes from $SWANLAB_MODE (set by train.sh via SWANLAB_UPLOAD): "cloud" uploads to swanlab.cn
    # (needs $SWANLAB_API_KEY), "offline" writes <log_dir>/swanlog/ only, "disabled" is a no-op (also forced
    # by --debug). Cloud init can fail behind a flaky proxy, so we fall back to offline rather than silently
    # dropping every metric. View offline logs with: `swanlab watch -l <log_dir>/swanlog`
    global USE_SWANLAB
    swanlab_run_name = getattr(get_logdir, "last_run_name", os.path.basename(log_dir))
    if rank == 0 and HAS_SWANLAB:
        swanlab_project = exp_cfg.swanlab_project
        swanlab_mode = "disabled" if cmd_args.debug else os.environ.get("SWANLAB_MODE", "offline")
        # Per-run swanlog dir under the run folder (matches RLBench/train.py); an exported SWANLAB_LOG_DIR still wins.
        swanlab_logdir = os.environ.get("SWANLAB_LOG_DIR") or os.path.join(log_dir, "swanlog")
        os.makedirs(swanlab_logdir, exist_ok=True)

        def _swanlab_init(mode):
            if mode == "cloud" and os.environ.get("SWANLAB_API_KEY"):
                swanlab.login(api_key=os.environ["SWANLAB_API_KEY"])
            swanlab.init(
                project=swanlab_project,
                experiment_name=swanlab_run_name,
                mode=mode,
                logdir=swanlab_logdir,
            )

        try:
            _swanlab_init(swanlab_mode)
            USE_SWANLAB = True
            print(f"[real/train] SwanLab enabled ({swanlab_mode} mode — logs at {swanlab_logdir}), "
                  f"project={swanlab_project}, run={swanlab_run_name}")
        except Exception as e:
            # Cloud auth/init can fail without outbound HTTPS to api.swanlab.cn; drop to offline so metrics land on disk.
            if swanlab_mode == "cloud":
                print(f"[real/train] SwanLab cloud init failed ({e}); falling back to offline mode")
                try:
                    _swanlab_init("offline")
                    USE_SWANLAB = True
                    print(f"[real/train] SwanLab enabled (offline fallback — logs at {swanlab_logdir}), "
                          f"project={swanlab_project}, run={swanlab_run_name}")
                except Exception as e2:
                    print(f"[real/train] SwanLab offline fallback also failed ({e2}); continuing without.")
            else:
                print(f"[real/train] SwanLab init failed ({e}); continuing without.")

    # ---- training loop ----
    if rank == 0:
        print(f"[real/train] begin training ({epochs} epochs)", flush=True)

    save_every = int(getattr(exp_cfg, "save_every_n_epochs", 5))
    # SwanLab step counter: one grad step per real batch, so the x-axis stays monotone across epochs.
    iters_per_epoch = max(1, len(loader))

    # Per-task per-episode visualization cadence, mirroring GemBench/train.py (single source of truth =
    # exp_cfg). Rank 0 picks ONE random episode per task every `viz_every_n_epochs` epochs (plus epoch 0
    # unless initial_viz=False), dumps pred+GT heatmaps plus the memory-debug grids for every step of that
    # episode, then everyone re-syncs at a barrier.
    # `viz_tasks`: None -> all tasks (default); [] -> disabled; list[str] -> only those.
    # `--visualize`/`--no-visualize` and `--viz_per_epoch` are an additional master switch on top.
    VIZ_EVERY_N_EPOCHS = int(getattr(exp_cfg, "viz_every_n_epochs", save_every))
    _viz_tasks_cfg = getattr(exp_cfg, "viz_tasks", None)
    VIZ_TASKS = list(_viz_tasks_cfg) if _viz_tasks_cfg is not None else None
    VIZ_DISABLED = (VIZ_TASKS is not None) and (len(VIZ_TASKS) == 0)
    INITIAL_VIZ = bool(getattr(exp_cfg, "initial_viz", True))

    def run_viz(epoch_idx: int) -> None:
        """End-of-epoch / pre-epoch visualization on rank 0 only.

        Bypasses DDP (uses agent._net_mod under the hood) so non-zero ranks
        don't need to participate. We follow it with a dist.barrier() in the
        caller to re-sync before the next training step.
        """
        if (rank != 0 or not cmd_args.visualize or cmd_args.viz_per_epoch <= 0
                or VIZ_EVERY_N_EPOCHS <= 0 or VIZ_DISABLED):
            return
        # initial_viz=False skips the epoch-0 baseline pass. This guard must precede the modulo check, since
        # 0 % VIZ_EVERY_N_EPOCHS == 0 would otherwise fire at epoch 0 anyway.
        if epoch_idx == 0 and not INITIAL_VIZ:
            return
        is_viz_epoch = (
            epoch_idx == 0
            or (epoch_idx % VIZ_EVERY_N_EPOCHS == 0)
            or (epoch_idx == epochs - 1)
        )
        if not is_viz_epoch:
            return
        try:
            # visualize_epoch samples ONE episode per task and dumps every step in order.
            visualize_epoch(
                agent, dataset, epoch=epoch_idx, log_dir=log_dir,
                cameras=CAMERAS_REAL,
                seed=epoch_idx,
                stages=("mvt1", "mvt2"),
                tasks=VIZ_TASKS,
            )
        except Exception as e:
            import traceback
            print(f"[real/train] visualize_epoch failed at epoch {epoch_idx}: {e}",
                  flush=True)
            traceback.print_exc()

    for epoch in range(start_epoch, epochs):
        # Pre-epoch visualization: runs at the *start* of every epoch, so the first viz (epoch 0) shows the
        # model before any training step — a baseline for how the heatmap tightens over time.
        run_viz(epoch)
        dist.barrier()

        # Stage 2 transition: unfreeze PaliGemma (except always_freeze). DDP's reducer is pinned to the
        # requires_grad set at construction, so params flipped False->True are never all-reduced — a silent
        # multi-GPU correctness bug. Rebuilding the DDP wrapper around the same inner module fixes it.
        if epoch == FREEZE_EPOCHS:
            for name, param in agent._network.named_parameters():
                if any(af in name for af in always_freeze):
                    param.requires_grad = False
                elif "mvt1.model" in name:
                    param.requires_grad = True
            inner = agent._network.module if isinstance(agent._network, DDP) else agent._network
            agent._network = DDP(
                inner, device_ids=[local_rank], find_unused_parameters=True,
            )
            agent.rebuild_optimizer()
            trainable = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
            if rank == 0:
                if use_focal:
                    grounding_msg = "up_grounding frozen"
                else:
                    grounding_msg = "no up_grounding head (use_modified_focal_loss=False)"
                print(f"[real/train] Stage 2 @ epoch {epoch}: PaliGemma unfrozen "
                      f"(vision_tower + lm_head/embed_tokens still frozen; "
                      f"{grounding_msg}). Rebuilt DDP + optimizer. "
                      f"Trainable params: {trainable/1e9:.3f} B")

        sampler.set_epoch(epoch)
        if rank == 0:
            print(f"[real/train] epoch {epoch}/{epochs}", flush=True)
        losses = train_one_epoch(
            agent, loader, cameras=CAMERAS_REAL, rank=rank,
            max_iter=cmd_args.max_iter, epoch_idx=epoch,
            global_step_base=epoch * iters_per_epoch,
        )

        if rank == 0:
            loss_str = ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
            print(f"[real/train] epoch {epoch} done — {loss_str}", flush=True)
            if USE_SWANLAB:
                swanlab.log(
                    {f"epoch/{k}": v for k, v in losses.items()},
                    step=(epoch + 1) * iters_per_epoch - 1,
                )

        is_periodic = save_every > 0 and epoch > 0 and (epoch % save_every == 0)
        is_final = epoch == epochs - 1
        if rank == 0 and (is_periodic or is_final):
            save_opt = bool(getattr(exp_cfg, "save_optimizer_state", False))
            save_agent(agent, f"{log_dir}/model_{epoch}.pth", epoch,
                       freeze_epochs=FREEZE_EPOCHS, save_optimizer_state=save_opt)
            save_agent(agent, f"{log_dir}/model_last.pth", epoch,
                       freeze_epochs=FREEZE_EPOCHS, save_optimizer_state=save_opt)
            print(f"[real/train] saved checkpoint at epoch {epoch}", flush=True)

        dist.barrier()

    if rank == 0:
        print("[real/train] done.", flush=True)
    dist.destroy_process_group()


# entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_cfg_path", type=str,
                        default="real/configs/real_config.yaml")
    parser.add_argument("--mvt_cfg_path", type=str,
                        default="real/configs/mvt_cfg.yaml")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    # Fallback only — real/train.sh always passes --data_folder. Derived from REAL_DATA_ROOT.
    parser.add_argument("--data_folder", type=str, nargs="+",
                        default=[os.path.join(REAL_DATA_ROOT,
                                              "7_20_real_updated")])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=None,
                        help="Override freeze_epochs from the YAML config "
                             "(Stage 1 length: PaliGemma frozen for the first "
                             "N epochs, then unfrozen).")
    parser.add_argument("--max_iter", type=int, default=None,
                        help="Cap iterations per epoch (for smoke-testing).")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--load_pretrain", action="store_true",
                        help="Load BridgeVLA pretrain weights into MVT.")
    parser.add_argument("--pretrain_path", type=str, default=None,
                        help="Path to pretrain checkpoint (.pth or .safetensors dir).")
    parser.add_argument(
        "--resume_path", type=str, default=None,
        help="Path to a training checkpoint (.pth) to resume from. Restores "
             "model + optimizer + LR-warmup step + epoch, and writes new "
             "checkpoints/logs back into the checkpoint's own run folder "
             "(original timestamp preserved). Mutually exclusive with "
             "--load_pretrain (resume wins). Scheduling config (freeze_epochs, "
             "save_every_n_epochs, epochs, ...) is still read from the YAML/CLI.")
    # Visualization master switch, ON by default so a fresh `bash train.sh` dumps a pre-training baseline.
    viz_group = parser.add_mutually_exclusive_group()
    viz_group.add_argument("--visualize", dest="visualize",
                           action="store_true", default=True,
                           help="Enable start-of-epoch visualization (default).")
    viz_group.add_argument("--no-visualize", dest="visualize",
                           action="store_false",
                           help="Disable start-of-epoch visualization.")
    parser.add_argument("--viz_per_epoch", type=int, default=2,
                        help="On/off switch for the start-of-epoch visualizer "
                             "(rank 0 only). visualize_epoch samples ONE "
                             "episode per task and dumps every step in order, "
                             "so the integer value is only used as >0/<=0: "
                             "positive keeps the hook on, non-positive disables "
                             "it. Ignored when --no-visualize is set. The "
                             "actual cadence / task whitelist / epoch-0 "
                             "baseline are controlled by real_config.yaml's "
                             "viz_every_n_epochs / viz_tasks / initial_viz "
                             "(mirrors GemBench/configs/gembench_config.yaml).")
    cmd_args = parser.parse_args()
    experiment(cmd_args)
