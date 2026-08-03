"""
RMBench (dual-arm) training entrypoint.

Mirrors finetune/memoryBench/train.py 1:1 — same agent, DDP scaffolding,
SwanLab integration, Stage-1/Stage-2 freeze schedule. The deltas are:
  * dataset: RMBench_Dataset (reads pre-extracted keyframe HDF5 directly);
  * model: MVT(num_arms=2, predict_collision=False);
  * train step: agent.update_rmbench (dual-arm loss = Σ_arm trans+rot+grip).

"""
import argparse
import faulthandler
import os
import signal
import subprocess
import sys
import time
import yaml
from contextlib import redirect_stdout
from datetime import timedelta

import swanlab
import torch
import torch.distributed as dist
import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod
from bridgevla.models.bridgevla_agent import print_loss_log
from bridgevla.mvt.memory import memory_param_norms
from bridgevla.mvt.mvt import MVT
from bridgevla.utils.rvt_utils import get_num_feat

from utils.peract_utils_rmbench import CAMERAS, IMAGE_SIZE, SCENE_BOUNDS, RMBENCH_TASKS
from rmbench_dataset import RMBench_Dataset
try:
    from visualize import visualize_epoch as visualize_epoch_rmbench
except Exception:  # viz is optional; never block training on it.
    visualize_epoch_rmbench = None

USE_SWANLAB = False


# ---- per-rank stdout/stderr tee (verbatim from memoryBench/train.py) -------
class _TeeStream:
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
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, f"train_rank{rank}.log")
    log_fp = open(log_path, "a", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_fp)
    sys.stderr = _TeeStream(sys.stderr, log_fp)
    print(f"[RMBench] rank {rank} logging stdout/stderr to {log_path}", flush=True)


def _install_fault_logging(rank: int) -> None:
    """Dump Python stacks for native fatal signals into the per-rank log."""
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
        print(
            f"[RMBench] rank {rank} faulthandler enabled "
            "(fatal signals + SIGUSR1 stack dump)",
            flush=True,
        )
    except Exception as e:
        print(f"[RMBench] rank {rank} faulthandler setup failed: {e}", flush=True)


class DistributedWeightedSampler(torch.utils.data.Sampler):
    """DDP-correct weighted (with-replacement) sampler for task balancing.

    PyTorch ships ``WeightedRandomSampler`` (single-process) and
    ``DistributedSampler`` (uniform, multi-process) but not their combination.
    This fills the gap: every rank seeds an IDENTICAL generator from
    ``seed + epoch`` and draws the SAME ``total_size`` multinomial indices, then
    each rank keeps a strided (``rank::world_size``) disjoint slice. So across
    ranks the union is exactly one epoch's worth of draws with no overlap, and
    ``set_epoch`` reshuffles deterministically (mirrors DistributedSampler's
    contract). ``num_samples`` defaults to ``len(weights)`` so the per-epoch
    iteration count matches the old uniform sampler; sampling is WITH
    replacement (the point of reweighting), so some transitions repeat and
    others are skipped within an epoch.
    """

    def __init__(self, weights, num_replicas, rank, num_samples=None, seed=0):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        n = int(num_samples) if num_samples is not None else self.weights.numel()
        # total drawn across ALL ranks, made divisible by world_size so the
        # strided slices are equal-length (pairs with DataLoader drop_last).
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


def create_dataloader(dataset, rank, world_size, batch_size, num_workers,
                      collate_fn=None, sampler=None):
    if sampler is None:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  sampler=sampler, drop_last=True, pin_memory=True)
    # Space-for-time (HOST RAM, not VRAM): the dataset's per-episode LRU cache
    # is the hot path — a cold miss decodes a whole episode (RGB decode +
    # depth->world PCD for every keyframe/cam), which shows up as uneven GPU
    # util (DDP stragglers). Keep workers + their caches alive across epochs
    # and prefetch deeper so those cold-decode spikes are hidden behind
    # compute instead of stalling the allreduce.
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **kwargs), sampler


def train(agent, data_loader, epoch, cameras, rank=0):
    """One epoch over the RMBench dataloader (dual-arm, action-only)."""
    agent.train()

    def to_device(d, device):
        if isinstance(d, dict):
            return {
                k: to_device(v, device) if isinstance(v, dict)
                else v.to(device) if isinstance(v, torch.Tensor)
                else v
                for k, v in d.items()
            }
        return d

    n_iters = len(data_loader)
    pbar = tqdm.tqdm(total=n_iters, disable=(rank != 0), position=0, leave=True,
                     desc=f"epoch {epoch}")

    epoch_losses = {}
    for it, raw_batch in enumerate(data_loader):
        batch = to_device(raw_batch, agent._device)
        batch["tasks"] = raw_batch["tasks"]
        # mvt_single's text indexing expects [[[str]]].
        batch["lang_goal"] = [[[item]] for item in raw_batch["lang_goal"]]
        out = agent.update_rmbench(
            cameras=cameras,
            replay_sample=batch,
            backprop=True,
            reset_log=(it == 0),
        )
        if epoch_losses == {}:
            epoch_losses = {k: [] for k in out.keys()}
        for k in epoch_losses:
            if k in out:
                epoch_losses[k].append(out[k])

        step = epoch * n_iters + it
        if rank == 0 and USE_SWANLAB and step % 10 == 0:
            log_dict = {f"train/{k}": v for k, v in out.items()}
            log_dict.update(memory_param_norms(agent._network))
            swanlab.log(log_dict, step=step)
        if rank == 0:
            pbar.set_postfix(
                step=it, lr=f"{out.get('lr', 0.0):.2e}",
                total=f"{out.get('total_loss', 0.0):.3f}",
            )
        pbar.update(1)
    pbar.close()
    if rank == 0:
        print_loss_log(agent)
    return {f"train/{k}": sum(v) / max(1, len(v)) for k, v in epoch_losses.items() if v}


def save_agent(agent, path, epoch, freeze_epochs=None, save_optimizer_state=False):
    """Checkpoint the agent.

    ``epoch`` / ``model_state`` are kept verbatim so eval-time loaders
    (bridgevla.utils.rvt_utils.load_agent, which reads only those two keys)
    stay byte-compatible with old checkpoints.

    ``global_step`` / ``freeze_epochs`` are scalars (no file-size cost) and
    are ALWAYS written — freeze_epochs feeds the resume-time consistency
    warning (the optimizer param-group layout depends on it), and global_step
    documents where the LR warmup stood
    (lr = base_lr * min(1, global_step / warmup_steps)).

    ``save_optimizer_state`` (default False) gates only the heavy part:

      * optimizer_state — AdamW moment buffers + per-group lr (~2x the model
                          weights in size). Needed for resume to continue
                          training EXACTLY where it left off; without it,
                          restore_checkpoint falls back to a freshly-built
                          optimizer + global_step=0 (warmup restarts), same
                          path as loading an old-format checkpoint. Eval
                          loaders read only ``epoch`` + ``model_state`` and
                          are unaffected either way.
    """
    model = agent._network
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    ckpt = {
        "epoch": epoch,
        "model_state": model_state,
        "global_step": int(getattr(agent, "_global_step", 0)),
        "freeze_epochs": freeze_epochs,
    }
    if save_optimizer_state:
        ckpt["optimizer_state"] = agent._optimizer.state_dict()
    torch.save(ckpt, path)


def restore_checkpoint(agent, ckpt, local_rank, rank, freeze_epochs, always_freeze):
    """Restore model + optimizer + LR-warmup step from a training checkpoint.

    Must be called AFTER ``agent.build()`` (so ``agent._optimizer`` exists) and
    AFTER the caller has reconstructed the correct *stage* (Stage-1 vs Stage-2)
    so the optimizer's param-group layout matches the saved state. Returns the
    epoch to resume FROM (i.e. ``saved_epoch + 1``). See GemBench/train.py for
    the full rationale (this is a 1:1 mirror).
    """
    model = agent._network.module if isinstance(agent._network, DDP) else agent._network
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if rank == 0 and (missing or unexpected):
        print(f"[resume] model_state loaded with missing={len(missing)} "
              f"unexpected={len(unexpected)} keys (strict=False).")

    saved_epoch = int(ckpt["epoch"])
    start_epoch = saved_epoch + 1

    saved_fe = ckpt.get("freeze_epochs", None)
    if rank == 0 and saved_fe is not None and int(saved_fe) != int(freeze_epochs):
        print(f"[resume] WARNING: freeze_epochs changed since checkpoint "
              f"(saved={saved_fe}, current={freeze_epochs}). Using current "
              f"config; stage reconstruction follows the CURRENT value. If the "
              f"checkpoint was already in Stage 2, do NOT raise freeze_epochs "
              f"above the resume epoch.")

    opt_state = ckpt.get("optimizer_state", None)
    if start_epoch == freeze_epochs:
        if rank == 0:
            print(f"[resume] start_epoch ({start_epoch}) == freeze_epochs; "
                  f"the Stage-2 optimizer will be rebuilt fresh by the loop "
                  f"(matches uninterrupted training). Skipping optimizer-state "
                  f"load; global_step stays 0.")
    elif opt_state is not None:
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
                      f"continuing with a freshly-built optimizer and "
                      f"global_step=0. (Param-group layout likely changed.)")
    else:
        if rank == 0:
            print("[resume] no optimizer_state in checkpoint (old format); "
                  "continuing with a freshly-built optimizer, global_step=0.")

    if rank == 0:
        print(f"[resume] resuming from epoch {saved_epoch} -> start_epoch "
              f"{start_epoch}.")
    return start_epoch


def get_time():
    env_stamp = os.environ.get("RMBENCH_RUN_STAMP")
    if env_stamp:
        return env_stamp
    import datetime
    now = datetime.datetime.now()
    return f"{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}"


def memory_ablation_suffix(exp_cfg):
    """Run-name suffix marking which memory group is ablated, or "" when memory
    is full / disabled. Mirrored in train.sh so the SwanLab run name and the
    run directory stay identical (see train.sh's swanlab_run python snippet).
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


def build_run_name(exp_cfg, world_size):
    # Idempotent: train.sh may have already appended the memory-ablation suffix
    # onto swanlab_run (passed via --exp_cfg_opts); only add it when absent so a
    # direct ``python train.py`` invocation still gets the marker.
    base = exp_cfg.swanlab_run
    suffix = memory_ablation_suffix(exp_cfg)
    if suffix and not base.endswith(suffix):
        base = f"{base}{suffix}"
    return f"{base}_{get_time()}"


def get_logdir(cmd_args, exp_cfg, dist_, run_name):
    root = cmd_args.log_dir if cmd_args.log_dir else exp_cfg.log_dir
    log_dir = os.path.join(root, "train_rmbench", run_name)
    if cmd_args.debug:
        log_dir = os.path.join(log_dir, "debug")
    if dist_.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as f:
        with redirect_stdout(f):
            print(exp_cfg.dump())
    with open(f"{log_dir}/mvt_cfg.yaml", "w") as f:
        with redirect_stdout(f):
            print(mvt_cfg.dump())
    with open(f"{log_dir}/args.yaml", "w") as f:
        yaml.dump(cmd_args.__dict__, f)


def setup_distributed(backend="nccl", port=None):
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
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        pass
    elif os.getenv("DEBUG", "false").lower() == "true":
        print("Cannot find RANK and WORLD_SIZE — entering single-GPU debug mode")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "9001")
        os.environ.setdefault("LOCAL_RANK", "0")
    else:
        raise RuntimeError(
            "Distributed env vars not found. Launch with torchrun / srun, "
            "or set DEBUG=true for single-GPU mode."
        )
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # rank 0 does long single-process work (initial viz rendering, checkpointing)
    # while other ranks idle on dist.barrier(). The default NCCL watchdog timeout
    # is 10 min, which the initial viz pass can exceed and crash the whole group.
    # Bump it generously; override via NCCL_PG_TIMEOUT_MIN if needed.
    pg_timeout_min = int(os.getenv("NCCL_PG_TIMEOUT_MIN", "120"))
    dist.init_process_group(
        backend=backend, world_size=world_size, rank=rank,
        timeout=timedelta(minutes=pg_timeout_min),
    )


def experiment(cmd_args):
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)

    # cuDNN autotuner: the renderer output / ConvexUpSample / PaliGemma ViT
    # patch-embed all see FIXED input shapes here (bs, V, C, 224, 224), so the
    # one-time algorithm search amortizes immediately. Lossless: it only
    # selects among mathematically-equivalent conv kernels. (Transformer GEMMs
    # go through cuBLAS and are unaffected.)
    torch.backends.cudnn.benchmark = True

    # ---- Resume vs pretrain mutual exclusion (resume wins) ----
    # A resume restores the full training state (model + optimizer + LR-warmup
    # step) from a checkpoint, so warm-starting from a pretrain checkpoint at
    # the same time would be contradictory. If both are given, disable pretrain.
    if cmd_args.resume_path:
        if cmd_args.load_pretrain and dist.get_rank() == 0:
            print("[resume] --resume_path is set; ignoring --load_pretrain "
                  "(resume takes priority over pretrain warm-start).")
        cmd_args.load_pretrain = False
        cmd_args.pretrain_path = None

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    if dist.get_rank() == 0:
        print(f"Total devices: {dist.get_world_size()}")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id
    if cmd_args.exp_cfg_opts:
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts:
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"

    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    BATCH_SIZE_TRAIN = exp_cfg.bs
    EPOCHS = exp_cfg.epochs
    FREEZE_EPOCHS = int(getattr(exp_cfg, "freeze_epochs", 2))

    # On resume, the run name is the ORIGINAL run folder's basename so SwanLab
    # logs and log_dir keep the original timestamp.
    if dist.get_rank() == 0:
        if cmd_args.resume_path:
            run_name = os.path.basename(os.path.dirname(os.path.abspath(cmd_args.resume_path)))
        else:
            run_name = build_run_name(exp_cfg, dist.get_world_size())
        run_name_list = [run_name]
    else:
        run_name_list = [None]
    dist.broadcast_object_list(run_name_list, src=0)
    run_name = run_name_list[0]

    log_dir = get_logdir(cmd_args, exp_cfg, dist, run_name)
    # Resume: pin log_dir to the EXACT directory holding the checkpoint, so new
    # checkpoints / logs / viz land back in the original timestamped run folder.
    if cmd_args.resume_path:
        log_dir = os.path.dirname(os.path.abspath(cmd_args.resume_path))
        if dist.get_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[resume] writing into original run folder: {log_dir}")
    _install_tee_logging(log_dir, dist.get_rank())
    _install_fault_logging(dist.get_rank())

    _mem_node = getattr(exp_cfg, "memory", None)
    _mem_enabled = bool(getattr(_mem_node, "enabled", False)) if _mem_node is not None else False
    _mem_k = int(getattr(_mem_node, "k_temporal", 4)) if _mem_node is not None else 4
    _mem_select = str(getattr(_mem_node, "select", "keyframe_gt")) if _mem_node is not None else "keyframe_gt"

    # Task selection: CLI --tasks > exp_cfg.tasks (comma str; "all" -> all) .
    if cmd_args.tasks:
        tasks = list(cmd_args.tasks)
    else:
        _cfg_tasks = str(getattr(exp_cfg, "tasks", "all") or "all")
        # Accept BOTH comma- and whitespace-separated task lists (a space-only
        # list would otherwise parse as one bogus multi-word task name and
        # silently yield an empty dataset). Normalize commas to spaces, split.
        _parsed = [t for t in _cfg_tasks.replace(",", " ").split() if t]
        if not _parsed or _parsed[0] == "all":
            tasks = list(RMBENCH_TASKS)
        else:
            tasks = _parsed

    instructions_dir = cmd_args.instructions_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "RMBench", "description", "task_instruction",
    )

    t_start = time.time()
    # Per-worker episode LRU cache size. Default 4 is far too small vs the
    # shuffled access pattern (tasks*ep_per_task episodes), so most __getitem__
    # calls cold-decode a whole episode. Bump it (host RAM ~ size * one-episode
    # decoded bytes per worker) to keep hot episodes resident across the
    # shuffled stream; tune via `episode_cache_size` in the YAML if RAM-bound.
    _ep_cache = int(getattr(exp_cfg, "episode_cache_size", 64))
    train_dataset = RMBench_Dataset(
        data_root=cmd_args.data_root,
        cameras=cmd_args.cameras,
        tasks=tasks,
        ep_per_task=cmd_args.ep_per_task,
        image_size=IMAGE_SIZE,
        memory_enabled=_mem_enabled,
        memory_k_temporal=_mem_k,
        memory_select=_mem_select,
        instructions_dir=instructions_dir,
        index_cache_dir=cmd_args.index_cache_dir,
        episode_cache_size=_ep_cache,
        verbose=(dist.get_rank() == 0),
    )
    if local_rank == 0:
        print("Total tasks:", train_dataset.num_tasks)
        print("Total trajectories:", train_dataset.num_task_paths)
        print("Dataset Length:", len(train_dataset))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Fail fast on an empty / too-small dataset: with drop_last=True a dataset
    # smaller than world_size*bs yields ZERO batches, so the train loop never
    # runs and downstream code (print_loss_log) crashes with a confusing
    # AttributeError. Surface the real cause here instead.
    _min_needed = world_size * BATCH_SIZE_TRAIN
    if len(train_dataset) < _min_needed:
        raise RuntimeError(
            f"Train dataset has only {len(train_dataset)} samples for tasks "
            f"{tasks}, but world_size*bs={_min_needed} are required for at "
            f"least one full batch (drop_last=True). Check the `tasks` list in "
            f"the config (must be comma/space separated task names) and "
            f"ep_per_task / data_root."
        )

    # Multi-task sampling balance (exp_cfg.task_sampling). Default
    # "transition_uniform" reproduces the old uniform DistributedSampler. With
    # "temperature" each task is drawn ∝ n_t**alpha (alpha=0.5 -> 1/sqrt(n_t)
    # per transition), so long-horizon tasks stop dominating the gradient.
    _ts_cfg = getattr(exp_cfg, "task_sampling", None)
    _ts_mode = str(getattr(_ts_cfg, "mode", "transition_uniform")) if _ts_cfg is not None else "transition_uniform"
    _custom_sampler = None
    if _ts_mode == "temperature":
        _alpha = float(getattr(_ts_cfg, "alpha", 1.0))
        _weights = train_dataset.task_sampling_weights(_alpha)
        _custom_sampler = DistributedWeightedSampler(
            _weights, num_replicas=world_size, rank=rank, seed=0
        )
        if local_rank == 0:
            counts = train_dataset.task_transition_counts()
            tot = sum(counts.values()) or 1
            # Effective per-task share AFTER reweighting: p_t ∝ n_t**alpha.
            pw = {t: (n ** _alpha) for t, n in counts.items()}
            psum = sum(pw.values()) or 1.0
            print(f"[task_sampling] mode=temperature alpha={_alpha} "
                  f"(alpha=1 transition-uniform, 0 task-uniform):")
            print(f"  {'task':22s} {'n_trans':>8s} {'raw%':>7s} {'eff%':>7s}")
            for t, n in counts.items():
                print(f"  {t:22s} {n:8d} {100*n/tot:6.1f}% {100*pw[t]/psum:6.1f}%")
    elif _ts_mode != "transition_uniform":
        raise ValueError(
            f"[task_sampling] unknown mode {_ts_mode!r} (valid: 'temperature', "
            f"'transition_uniform'). A typo here would silently change the "
            f"task sampling distribution — refusing to fall back."
        )

    train_dataloader, train_sampler = create_dataloader(
        train_dataset, rank, world_size, BATCH_SIZE_TRAIN, exp_cfg.num_workers,
        sampler=_custom_sampler,
    )
    if local_rank == 0:
        print(f"Created Dataset. Time Cost: {(time.time()-t_start)/60.0:.1f} minutes")

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path:
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts:
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))
    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)

    # --- Rotation head selection (exp_cfg.rotation_representation). ---
    # "6d" -> continuous 6D regression head (rot_ver==2): the rotation slice of
    # the feat vector is a 6D vector (Zhou et al., CVPR 2019), so feat_dim
    # shrinks to 6 + grip(2) + collision(2) = 10. This fully replaces the
    # discrete-Euler classification head for RMBench. "euler_disc" (default)
    # keeps the original rot_ver / num_rot*3 head.
    _rot_repr = str(getattr(exp_cfg, "rotation_representation", "euler_disc"))
    if _rot_repr == "6d":
        mvt_cfg.rot_ver = 2
        mvt_cfg.feat_dim = 6 + 2 + 2  # 6D rot + grip(2) + collision(2)
    elif _rot_repr != "euler_disc":
        raise ValueError(
            f"Unknown rotation_representation={_rot_repr!r} "
            "(expected 'euler_disc' or '6d')"
        )

    if hasattr(exp_cfg, "gradient_checkpointing") and hasattr(mvt_cfg, "gradient_checkpointing"):
        mvt_cfg.gradient_checkpointing = bool(exp_cfg.gradient_checkpointing)
    if hasattr(exp_cfg, "rotate_top_ccw90") and hasattr(mvt_cfg, "rotate_top_ccw90"):
        mvt_cfg.rotate_top_ccw90 = bool(exp_cfg.rotate_top_ccw90)
    if hasattr(exp_cfg, "feat_from_stage1") and hasattr(mvt_cfg, "feat_from_stage1"):
        # Rotation/grip/collision feature source: stage 1 (coarse) when True.
        mvt_cfg.feat_from_stage1 = bool(exp_cfg.feat_from_stage1)
    if hasattr(exp_cfg, "renderer_img_sizes_w") and hasattr(mvt_cfg, "renderer_img_sizes_w"):
        mvt_cfg.renderer_img_sizes_w = exp_cfg.renderer_img_sizes_w
    if hasattr(exp_cfg, "splat_radius") and hasattr(mvt_cfg, "splat_radius"):
        mvt_cfg.splat_radius = exp_cfg.splat_radius
    if hasattr(exp_cfg, "splat_radius_mvt2") and hasattr(mvt_cfg, "splat_radius_mvt2"):
        mvt_cfg.splat_radius_mvt2 = exp_cfg.splat_radius_mvt2
    if hasattr(exp_cfg, "st_wpt_loc_aug") and hasattr(mvt_cfg, "st_wpt_loc_aug"):
        mvt_cfg.st_wpt_loc_aug = exp_cfg.st_wpt_loc_aug

    # Dual-arm propagation: exp_cfg -> mvt_cfg.
    if hasattr(exp_cfg, "num_arms"):
        mvt_cfg.num_arms = int(exp_cfg.num_arms)
    if hasattr(exp_cfg, "predict_collision"):
        mvt_cfg.predict_collision = bool(exp_cfg.predict_collision)

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

    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, (
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes,
    )

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        **mvt_cfg,
    ).to(local_rank)
    backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)

    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 1000)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    # Same always-frozen list as memoryBench; up_grounding never exists
    # with num_arms>1.
    always_freeze = ["lm_head", "embed_tokens", "vision_tower"]
    for name, param in agent._network.named_parameters():
        if any(af in name for af in always_freeze):
            param.requires_grad = False
        elif "mvt1.model" in name:
            param.requires_grad = False
    if dist.get_rank() == 0:
        print(f"[Stage 1] PaliGemma frozen for {FREEZE_EPOCHS} epochs.")

    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    if dist.get_rank() == 0:
        print(f"Total trainable parameters: {total_params / 1e9:.2f} billion")

    agent.build(training=True, device=device_id)

    start_epoch = 0
    end_epoch = EPOCHS

    # ---- Resume from a training checkpoint (full state) ----
    # Scheduling config (freeze_epochs / save_every_n_epochs / epochs /
    # warmup_steps / lr) is ALWAYS taken from the current YAML+CLI above; the
    # checkpoint only carries training STATE. Changing save_every_n_epochs on
    # resume takes effect immediately; freeze_epochs / warmup_steps should be
    # left unchanged (they anchor the restored optimizer state).
    if cmd_args.resume_path:
        if not os.path.isfile(cmd_args.resume_path):
            raise FileNotFoundError(
                f"--resume_path is not a file: {cmd_args.resume_path}"
            )
        if dist.get_rank() == 0:
            print(f"[resume] loading checkpoint: {cmd_args.resume_path}")
        resume_ckpt = torch.load(cmd_args.resume_path, map_location="cpu")
        resume_start_epoch = int(resume_ckpt["epoch"]) + 1

        # Reconstruct the Stage-1 -> Stage-2 transition BEFORE loading optimizer
        # state when the checkpoint was already in Stage 2. Mirrors the loop's
        # i==FREEZE_EPOCHS branch so the optimizer param-group layout matches.
        if resume_start_epoch > FREEZE_EPOCHS:
            for name, param in agent._network.named_parameters():
                if any(af in name for af in always_freeze):
                    param.requires_grad = False
                elif "mvt1.model" in name:
                    param.requires_grad = True
            inner = agent._network.module if isinstance(agent._network, DDP) else agent._network
            agent._network = DDP(inner, device_ids=[local_rank], find_unused_parameters=True)
            agent.rebuild_optimizer()
            if dist.get_rank() == 0:
                print(f"[resume] reconstructed Stage 2 (start_epoch="
                      f"{resume_start_epoch} > freeze_epochs={FREEZE_EPOCHS}): "
                      f"unfroze PaliGemma + rebuilt DDP + optimizer before "
                      f"loading optimizer state.")

        start_epoch = restore_checkpoint(
            agent, resume_ckpt,
            local_rank=local_rank, rank=dist.get_rank(),
            freeze_epochs=FREEZE_EPOCHS, always_freeze=always_freeze,
        )
        del resume_ckpt

    if dist.get_rank() == 0:
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()

    global USE_SWANLAB
    if dist.get_rank() == 0:
        swanlab_project = exp_cfg.swanlab_project
        swanlab_mode = "disabled" if cmd_args.debug else os.environ.get("SWANLAB_MODE", "offline")
        swanlab_logdir = os.path.join(log_dir, "swanlog")
        os.makedirs(swanlab_logdir, exist_ok=True)
        try:
            if swanlab_mode == "cloud":
                swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY", ""))
            swanlab.init(
                project=swanlab_project, experiment_name=run_name,
                mode=swanlab_mode, logdir=swanlab_logdir,
            )
            USE_SWANLAB = True
            print(f"[Info] SwanLab enabled ({swanlab_mode}) at {swanlab_logdir}")
        except Exception as e:
            # When cloud mode cannot reach the network (proxy down / no internet) it degrades to offline;
            # the logs still land in <run_dir>/swanlog and no training curves are lost.
            if swanlab_mode == "cloud":
                print(f"[Info] SwanLab cloud init failed ({e}); falling back to offline")
                try:
                    swanlab.init(
                        project=swanlab_project, experiment_name=run_name,
                        mode="offline", logdir=swanlab_logdir,
                    )
                    USE_SWANLAB = True
                    print(f"[Info] SwanLab enabled (offline) at {swanlab_logdir}")
                except Exception as e2:
                    print(f"[Info] SwanLab offline init also failed ({e2}); training continues without SwanLab")
            else:
                print(f"[Info] SwanLab init failed ({e}); training continues without SwanLab")

    if dist.get_rank() == 0:
        init_norms = memory_param_norms(agent._network)
        if init_norms:
            print("[mem_norm @ init] baseline total_to_out per block:")
            for k in sorted(init_norms.keys()):
                if k.endswith("/total_to_out"):
                    print(f"  {k}: {init_norms[k]:.4e}")
            if USE_SWANLAB:
                swanlab.log(init_norms, step=0)
        print("Start training ...", flush=True)

    # start_epoch / end_epoch are set right after agent.build() (start_epoch is
    # advanced by the resume block when --resume_path is given). Do NOT reset
    # start_epoch=0 here or a resume would silently restart from epoch 0.
    end_epoch = EPOCHS

    # Per-task per-episode dual-arm heatmap viz cadence (mirrors memoryBench).
    VIZ_EVERY_N_EPOCHS = int(getattr(exp_cfg, "viz_every_n_epochs", 20))
    _viz_tasks_cfg = getattr(exp_cfg, "viz_tasks", None)
    VIZ_TASKS = list(_viz_tasks_cfg) if _viz_tasks_cfg is not None else None
    VIZ_DISABLED = (VIZ_TASKS is not None and len(VIZ_TASKS) == 0) or visualize_epoch_rmbench is None
    INITIAL_VIZ = bool(getattr(exp_cfg, "initial_viz", True))
    # Render viz with the training-time SE3 aug (random yaw + translation,
    # co-applied to scene cloud + both arms' GT + memory anchor/history clouds)
    # so the viz reflects what the network is trained on. Default True; set
    # viz_apply_se3_aug=false in the config for the old canonical un-augmented
    # (cross-epoch comparable) view.
    VIZ_APPLY_SE3_AUG = bool(getattr(exp_cfg, "viz_apply_se3_aug", True))

    def run_viz(epoch_idx):
        if dist.get_rank() != 0 or VIZ_EVERY_N_EPOCHS <= 0 or VIZ_DISABLED:
            return
        # initial_viz=False skips the epoch-0 pre-training baseline pass.
        # The guard must come before the modulo check below, since
        # 0 % VIZ_EVERY_N_EPOCHS == 0 would otherwise still fire at epoch 0.
        if epoch_idx == 0 and not INITIAL_VIZ:
            return
        is_viz_epoch = (
            epoch_idx == 0 or (epoch_idx % VIZ_EVERY_N_EPOCHS == 0)
            or (epoch_idx == end_epoch - 1)
        )
        if not is_viz_epoch:
            return
        try:
            visualize_epoch_rmbench(
                agent, train_dataset, epoch=epoch_idx, log_dir=log_dir,
                cameras=cmd_args.cameras, seed=epoch_idx,
                stages=("mvt1", "mvt2") if mvt_cfg.stage_two else ("mvt1",),
                tasks=VIZ_TASKS, apply_se3_aug=VIZ_APPLY_SE3_AUG,
            )
        except Exception as e:
            import traceback
            print(f"[RMBench] visualize_epoch failed at epoch {epoch_idx}: {e}", flush=True)
            traceback.print_exc()

    i = start_epoch
    while True:
        if i == end_epoch:
            break
        if i == FREEZE_EPOCHS:
            for name, param in agent._network.named_parameters():
                if any(af in name for af in always_freeze):
                    param.requires_grad = False
                elif "mvt1.model" in name:
                    param.requires_grad = True
            inner = agent._network.module if isinstance(agent._network, DDP) else agent._network
            agent._network = DDP(inner, device_ids=[local_rank], find_unused_parameters=True)
            agent.rebuild_optimizer()
            total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
            if dist.get_rank() == 0:
                print(f"[Stage 2] Unfroze PaliGemma. Trainable params: {total_params/1e9:.2f}B")

        run_viz(i)
        dist.barrier()
        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training")
        train_dataloader.sampler.set_epoch(i)
        train(agent, train_dataloader, epoch=i, cameras=cmd_args.cameras,
              rank=dist.get_rank())

        save_every = int(getattr(exp_cfg, "save_every_n_epochs", 20))
        is_periodic = save_every > 0 and i > 0 and (i % save_every == 0)
        is_final = save_every > 0 and i == end_epoch - 1
        if dist.get_rank() == 0 and (is_periodic or is_final):
            save_opt = bool(getattr(exp_cfg, "save_optimizer_state", False))
            save_agent(agent, f"{log_dir}/model_{i}.pth", i,
                       freeze_epochs=FREEZE_EPOCHS, save_optimizer_state=save_opt)
            save_agent(agent, f"{log_dir}/model_last.pth", i,
                       freeze_epochs=FREEZE_EPOCHS, save_optimizer_state=save_opt)
        i += 1
        dist.barrier()

    dist.barrier()
    if dist.get_rank() == 0:
        print("[Finish]")
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvt_cfg_path", type=str,
                        default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str,
                        default="configs/rmbench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument(
        "--data_root", type=str,
        default=os.environ.get(
            "RMBENCH_VLA_DATA_ROOT",
            "data/bridgevla_data/RMBench/data/keyframe_data",
        ),
    )
    parser.add_argument("--index_cache_dir", type=str, default=None)
    parser.add_argument("--instructions_dir", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ep_per_task", type=int, default=100)
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument(
        "--resume_path", type=str, default=None,
        help="Path to a training checkpoint (.pth) to resume from. Restores "
             "model + optimizer + LR-warmup step + epoch, and writes new "
             "checkpoints/logs back into the checkpoint's own run folder "
             "(original timestamp preserved). Mutually exclusive with "
             "--load_pretrain (resume wins). Scheduling config (freeze_epochs, "
             "save_every_n_epochs, epochs, ...) is still read from the YAML/CLI."
    )
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=None,
        help="Task names to train on. Default: all RMBench tasks.",
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=list(CAMERAS),
    )
    cmd_args = parser.parse_args()
    experiment(cmd_args)
