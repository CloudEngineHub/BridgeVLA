'''
RLBench offline trainer (keyframe-only sampling + episodic memory).

This trainer was rewritten to match the CURRENT BridgeVLA++ model logic
(RMBench / MemoryBench / real):

  * Sampling: ONE sample per keyframe->keyframe transition (GemBench style),
    via ``RLBench_Keyframe_Dataset`` + a plain DataLoader. The old YARR
    replay-buffer path (formerly utils/dataset.py + utils/get_dataset.py,
    since removed) swept arbitrary intermediate frames and appended the
    whole remaining keyframe chain, inflating the dataset and making the
    episodic-memory near-neighbour frames ill-defined.
  * Multi-task sampling balance: exp_cfg.task_sampling selects between a plain
    transition-uniform DistributedSampler (default) and a temperature-weighted
    DistributedWeightedSampler (mode="temperature") that rebalances tasks
    whose demos have very different keyframe counts.
  * Episodic memory (spatial anchor + temporal episodic): the dataset emits
    the frame-0 anchor + the two most-recent executed keyframes per sample;
    ``agent.update_gembench`` builds memory_inputs automatically. RLBench has
    no subtask-boundary label, so mem_label is always 0 and the keyframe
    discriminator is trained to a constant negative ("never admit").
  * Rotation head: 6D continuous regression (rot_ver=2) + feat_from_stage1,
    selectable from the yaml (rotation_representation / feat_from_stage1).

Eval (eval.py) is unchanged: it loads mvt_cfg.yaml (which now records the
memory + rotation settings), and the shared ``RVTAgent.act`` + YARR rollout
generator already drive the per-episode MemoryBank (agent.reset() at episode
start). The post-inference action handoff to the RLBench simulator is unchanged.

Adapted from memoryBench/train.py.
'''
import argparse
import os
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
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod
from bridgevla.models.bridgevla_agent import print_loss_log
from bridgevla.mvt.memory import memory_param_norms
from bridgevla.mvt.mvt import MVT
from bridgevla.utils.rvt_utils import get_num_feat, RLBENCH_TASKS

from utils.peract_utils_rlbench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
    DATA_FOLDER,
)
from utils.rlbench_keyframe_dataset import RLBench_Keyframe_Dataset
from visualize import visualize_epoch as visualize_epoch_rlbench

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
    print(f"[RLBench] rank {rank} logging stdout/stderr to {log_path}", flush=True)


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


def create_dataloader(dataset, rank, world_size, batch_size, num_workers, sampler=None):
    if sampler is None:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True,
        )
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, sampler=sampler,
        drop_last=True, pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    return loader, sampler


def train(agent, data_loader, epoch, cameras, rank=0):
    """One epoch over the RLBench keyframe dataloader (action-only)."""
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
        out = agent.update_gembench(
            cameras=cameras,
            replay_sample=batch,
            backprop=True,
            reset_log=(it == 0),
        )
        if epoch_losses == {}:
            epoch_losses = {k: [] for k in out.keys()}
        for k in epoch_losses:
            epoch_losses[k].append(out[k])

        step = epoch * n_iters + it
        if rank == 0 and USE_SWANLAB and step % 10 == 0:
            log_dict = {f"train/{k}": v for k, v in out.items()}
            # Track memory-branch weight norms on the same step axis so the
            # mem_norm/{block}/total_to_out curves are readable alongside loss.
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
    return {f"train/{k}": sum(v) / len(v) for k, v in epoch_losses.items()}


def save_agent(agent, path, epoch, freeze_epochs=None, save_optimizer_state=False):
    """Checkpoint the agent. ``epoch`` / ``model_state`` stay byte-compatible
    with eval-time loaders; optimizer_state / global_step / freeze_epochs are
    only written when save_optimizer_state=True (exact-resume support).
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


def restore_checkpoint(agent, ckpt, local_rank, rank, freeze_epochs):
    """Restore model + optimizer + LR-warmup step from a training checkpoint.
    Mirrors memoryBench/train.py (see there for the full rationale).
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
              f"(saved={saved_fe}, current={freeze_epochs}). Using current config.")

    opt_state = ckpt.get("optimizer_state", None)
    if start_epoch == freeze_epochs:
        if rank == 0:
            print(f"[resume] start_epoch ({start_epoch}) == freeze_epochs; the "
                  f"Stage-2 optimizer is rebuilt fresh by the loop. Skipping "
                  f"optimizer-state load; global_step stays 0.")
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
                      f"continuing with a freshly-built optimizer, global_step=0.")
    else:
        if rank == 0:
            print("[resume] no optimizer_state in checkpoint (old format); "
                  "continuing with a freshly-built optimizer, global_step=0.")

    if rank == 0:
        print(f"[resume] resuming from epoch {saved_epoch} -> start_epoch {start_epoch}.")
    return start_epoch


def get_tasks(exp_cfg):
    parsed_tasks = [t.strip() for t in str(exp_cfg.tasks).split(",") if t.strip()]
    if not parsed_tasks or parsed_tasks[0] == "all":
        return list(RLBENCH_TASKS)
    return parsed_tasks


def get_time():
    # RLBENCH_RUN_STAMP lets train_*.sh pre-compute the stamp so shell-side tee
    # and Python land in the same run folder (multi-node agreement).
    env_stamp = os.environ.get("RLBENCH_RUN_STAMP")
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
    log_dir = os.path.join(root, "train", run_name)
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
    """Auto-detect launch mode: SLURM -> torchrun -> single-GPU debug."""
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
    # Generous timeout so the rank-0-only viz pass doesn't trip the NCCL
    # watchdog while other ranks wait at the post-viz barrier.
    dist.init_process_group(
        backend=backend, world_size=world_size, rank=rank, timeout=timedelta(hours=1),
    )


def experiment(cmd_args):
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)

    # ---- Resume vs pretrain mutual exclusion (resume wins) ----
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
    EPOCHS = cmd_args.epochs if cmd_args.epochs is not None else exp_cfg.epochs
    FREEZE_EPOCHS = int(getattr(exp_cfg, "freeze_epochs", 5))

    # Unified run name (shared by log folder + SwanLab). On resume, reuse the
    # checkpoint's original run folder so logs/checkpoints/viz land back there.
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
    if cmd_args.resume_path:
        log_dir = os.path.dirname(os.path.abspath(cmd_args.resume_path))
        if dist.get_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[resume] writing into original run folder: {log_dir}")
    _install_tee_logging(log_dir, dist.get_rank())

    # ---- episodic memory cfg (single source of truth = exp_cfg.memory) ----
    _mem_node = getattr(exp_cfg, "memory", None)
    _mem_enabled = bool(getattr(_mem_node, "enabled", False)) if _mem_node is not None else False
    _mem_k = int(getattr(_mem_node, "k_temporal", 12)) if _mem_node is not None else 12
    # keyframe_gt: slot 0/1 = the two most-recent executed keyframes (near
    # neighbour two frames); slots 2..K-1 = discriminator-admitted boundaries
    # (none for RLBench -> always-negative discriminator).
    _mem_select = str(getattr(_mem_node, "select", "keyframe_gt")) if _mem_node is not None else "keyframe_gt"

    tasks = get_tasks(exp_cfg)
    if dist.get_rank() == 0:
        print(f"Training on {len(tasks)} tasks: {tasks}")

    t_start = time.time()
    train_dataset = RLBench_Keyframe_Dataset(
        data_root=cmd_args.data_folder,
        cameras=cmd_args.cameras,
        tasks=tasks,
        ep_per_task=cmd_args.num_train,
        cache_dir=cmd_args.cache_dir,
        image_size=IMAGE_SIZE,
        memory_enabled=_mem_enabled,
        memory_k_temporal=_mem_k,
        memory_select=_mem_select,
        verbose=(dist.get_rank() == 0),
        # Training NEVER builds keyframe caches: a missing/stale canonical
        # cache (cache_version mismatch) is a hard error, no fallback.
        # Download (scripts/download_checkpoints_hf.sh rlbench_cache) or prebuild
        # with: bash scripts/build_rlbench_cache.sh
        allow_build=False,
    )
    if local_rank == 0:
        print("Total tasks:", train_dataset.num_tasks)
        print("Total trajectories:", train_dataset.num_task_paths)
        print("Dataset Length:", len(train_dataset))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Multi-task sampling balance (exp_cfg.task_sampling). Default
    # "transition_uniform" reproduces the old uniform DistributedSampler. With
    # "temperature" each task is drawn ∝ n_t**alpha (alpha=0.5 -> 1/sqrt(n_t)
    # per transition), so tasks with more keyframes per demo stop dominating
    # the gradient (ep_per_task only equalizes episode COUNT across tasks, not
    # keyframe count).
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
    elif _ts_mode != "transition_uniform" and local_rank == 0:
        print(f"[task_sampling] WARN: unknown mode {_ts_mode!r}; "
              f"falling back to transition_uniform (uniform DistributedSampler).")

    train_dataloader, train_sampler = create_dataloader(
        train_dataset, rank, world_size, BATCH_SIZE_TRAIN, exp_cfg.num_workers,
        sampler=_custom_sampler,
    )
    if local_rank == 0:
        print(f"Created Dataset. Time Cost: {(time.time()-t_start)/60.0:.1f} minutes")

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))
    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)

    # --- Rotation head selection (exp_cfg.rotation_representation). Mirrors
    # RMBench / real. "6d" -> continuous 6D regression head (rot_ver==2): the
    # rotation slice of the feat vector is a 6D vector (Zhou et al., CVPR 2019),
    # so feat_dim is 6 + grip(2) + collision(2) = 10. Replaces the discrete-Euler
    # head. "euler_disc" keeps mvt_cfg's rot_ver + the get_num_feat feat_dim. ---
    _rot_repr = str(getattr(exp_cfg, "rotation_representation", "euler_disc"))
    if _rot_repr == "6d":
        mvt_cfg.rot_ver = 2
        mvt_cfg.feat_dim = 6 + 2 + 2  # 6D rot + grip(2) + collision(2)
    elif _rot_repr != "euler_disc":
        raise ValueError(
            f"Unknown rotation_representation={_rot_repr!r} (expected 'euler_disc' or '6d')"
        )

    # Propagate exp_cfg-level toggles the network consumes onto mvt_cfg before
    # MVT() is built (yacs scopes are independent). Mirrors memoryBench/real.
    if hasattr(exp_cfg, "gradient_checkpointing") and hasattr(mvt_cfg, "gradient_checkpointing"):
        mvt_cfg.gradient_checkpointing = bool(exp_cfg.gradient_checkpointing)
    if hasattr(exp_cfg, "feat_from_stage1") and hasattr(mvt_cfg, "feat_from_stage1"):
        # Rotation/grip/collision feature source: stage 1 (coarse) when True.
        mvt_cfg.feat_from_stage1 = bool(exp_cfg.feat_from_stage1)
    if hasattr(exp_cfg, "img_aug_2") and hasattr(mvt_cfg, "img_aug_2"):
        mvt_cfg.img_aug_2 = float(exp_cfg.img_aug_2)
    # Unified-yaml memory propagation (select + discriminator included so the
    # dataset layout and the eval MemoryBank stay aligned, and the dumped
    # mvt_cfg.yaml self-documents the memory policy for eval-time load_agent).
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
        image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 2000)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    # ---- Stage 1 freeze: PaliGemma backbone frozen; heads + feat_fc + memory
    # blocks trainable. vision_tower (SigLIP) permanently frozen. up_grounding
    # only exists when use_modified_focal_loss=True and is unused by RLBench, so
    # freeze it permanently when present to avoid phantom AdamW moments.
    use_focal = bool(getattr(mvt_cfg, "use_modified_focal_loss", False))
    always_freeze = ["lm_head", "embed_tokens", "vision_tower"]
    if use_focal:
        always_freeze.append("up_grounding")
    for name, param in agent._network.named_parameters():
        if any(af in name for af in always_freeze):
            param.requires_grad = False
        elif "mvt1.model" in name:
            param.requires_grad = False
    if dist.get_rank() == 0:
        print(f"[Stage 1] PaliGemma frozen for {FREEZE_EPOCHS} epochs "
              "— heatmap head + feat_fc heads + memory blocks trainable "
              "(vision_tower permanently frozen).")

    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    if dist.get_rank() == 0:
        print(f"Total trainable parameters: {total_params / 1e9:.2f} billion")

    agent.build(training=True, device=device_id)

    start_epoch = 0
    end_epoch = EPOCHS

    # ---- Resume from a training checkpoint (full state) ----
    if cmd_args.resume_path:
        if not os.path.isfile(cmd_args.resume_path):
            raise FileNotFoundError(f"--resume_path is not a file: {cmd_args.resume_path}")
        if dist.get_rank() == 0:
            print(f"[resume] loading checkpoint: {cmd_args.resume_path}")
        resume_ckpt = torch.load(cmd_args.resume_path, map_location="cpu")
        resume_start_epoch = int(resume_ckpt["epoch"]) + 1

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
                      f"{resume_start_epoch} > freeze_epochs={FREEZE_EPOCHS}).")

        start_epoch = restore_checkpoint(
            agent, resume_ckpt, local_rank=local_rank, rank=dist.get_rank(),
            freeze_epochs=FREEZE_EPOCHS,
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

    # Initialize Logging =>> SwanLab (mirrors GemBench/train.py)
    # Controlled by $SWANLAB_MODE (set in train.sh via SWANLAB_UPLOAD).
    #   "cloud"    → upload to swanlab.cn (requires $SWANLAB_API_KEY)
    #   "offline"  → local-only logs under <log_dir>/swanlog/
    #   "disabled" → no-op (also forced when --debug)
    # View offline logs with: `swanlab watch -l <log_dir>/swanlog`
    global USE_SWANLAB
    if dist.get_rank() == 0:
        swanlab_project = exp_cfg.swanlab_project
        swanlab_mode = "disabled" if cmd_args.debug else os.environ.get("SWANLAB_MODE", "offline")
        swanlab_logdir = os.path.join(log_dir, "swanlog")
        os.makedirs(swanlab_logdir, exist_ok=True)

        def _swanlab_init(mode):
            if mode == "cloud":
                swanlab.login(api_key=os.environ.get("SWANLAB_API_KEY", ""))
            swanlab.init(
                project=swanlab_project, experiment_name=run_name,
                mode=mode, logdir=swanlab_logdir,
            )

        try:
            _swanlab_init(swanlab_mode)
            USE_SWANLAB = True
            print(f"[Info] SwanLab enabled ({swanlab_mode} mode — logs at {swanlab_logdir}), project={swanlab_project}, run={run_name}")
        except Exception as e:
            # Cloud auth/init can fail when the cluster has no outbound HTTPS to
            # api.swanlab.cn. Drop to offline mode so metrics still land on disk.
            if swanlab_mode == "cloud":
                print(f"[Info] SwanLab cloud init failed ({e}); falling back to offline mode")
                try:
                    _swanlab_init("offline")
                    USE_SWANLAB = True
                    print(f"[Info] SwanLab enabled (offline fallback — logs at {swanlab_logdir}), project={swanlab_project}, run={run_name}")
                except Exception as e2:
                    print(f"[Info] SwanLab offline fallback also failed ({e2}), training continues without SwanLab")
            else:
                print(f"[Info] SwanLab init failed ({e}), training continues without SwanLab")

    if dist.get_rank() == 0:
        init_norms = memory_param_norms(agent._network)
        if init_norms:
            print("[mem_norm @ init] baseline total_to_out per block:")
            for k in sorted(init_norms.keys()):
                if k.endswith("/total_to_out"):
                    print(f"  {k}: {init_norms[k]:.4e}")
            if USE_SWANLAB:
                swanlab.log(init_norms, step=0)
        else:
            print("[mem_norm @ init] memory disabled (no MemoryBlock instantiated).")
        print("Start training ...", flush=True)

    # Per-task per-episode visualization cadence (mirrors GemBench/train.py).
    # Viz samples come straight from the SAME keyframe dataset the trainer
    # consumes, so the memory bundle (anchor + hist) is visualized exactly as
    # trained (memory_grid.png per step + mvt2/anchor_memory_grid.png).
    VIZ_EVERY_N_EPOCHS = int(getattr(exp_cfg, "viz_every_n_epochs", 20))
    _viz_tasks_cfg = getattr(exp_cfg, "viz_tasks", None)
    VIZ_TASKS = list(_viz_tasks_cfg) if _viz_tasks_cfg is not None else None
    VIZ_DISABLED = (VIZ_TASKS is not None) and (len(VIZ_TASKS) == 0)
    INITIAL_VIZ = bool(getattr(exp_cfg, "initial_viz", True))

    def run_viz(epoch_idx: int) -> None:
        if dist.get_rank() != 0 or VIZ_EVERY_N_EPOCHS <= 0 or VIZ_DISABLED:
            return
        # initial_viz=False skips the epoch-0 pre-training baseline pass.
        # The guard must come before the modulo check below, since
        # 0 % VIZ_EVERY_N_EPOCHS == 0 would otherwise still fire at epoch 0.
        if epoch_idx == 0 and not INITIAL_VIZ:
            return
        is_viz_epoch = (
            epoch_idx == 0
            or (epoch_idx % VIZ_EVERY_N_EPOCHS == 0)
            or (epoch_idx == end_epoch - 1)
        )
        if not is_viz_epoch:
            return
        try:
            visualize_epoch_rlbench(
                agent, train_dataset,
                epoch=epoch_idx,
                log_dir=log_dir,
                cameras=cmd_args.cameras,
                seed=epoch_idx,
                stages=("mvt1", "mvt2") if mvt_cfg.stage_two else ("mvt1",),
                tasks=VIZ_TASKS,
            )
        except Exception as e:
            import traceback
            print(f"[RLBench] visualize_epoch failed at epoch {epoch_idx}: {e}", flush=True)
            traceback.print_exc()

    i = start_epoch
    while True:
        if i == end_epoch:
            break
        # ---- Stage-1 -> Stage-2 transition: unfreeze PaliGemma (DDP rebuild). ----
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
                print(f"[Stage 2] Unfroze PaliGemma backbone (vision_tower + "
                      f"lm_head/embed_tokens still frozen). Rebuilt DDP + optimizer. "
                      f"Trainable params: {total_params/1e9:.2f}B")

        run_viz(i)
        dist.barrier()
        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training")
        train_dataloader.sampler.set_epoch(i)
        train(agent, train_dataloader, epoch=i, cameras=cmd_args.cameras,
              rank=dist.get_rank())

        save_every = int(getattr(exp_cfg, "save_every_n_epochs", 10))
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
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())
    # Default mvt config = the original RLBench rvt2.yaml (3 axis-aligned views,
    # point renderer, img_size 224, rot_ver=1). When the yaml selects
    # rotation_representation="6d", experiment() overrides rot_ver->2 + feat_dim
    # AFTER this file is merged, so the 6D head wins. Absolute path so it
    # resolves regardless of cwd.
    _DEFAULT_MVT_CFG = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "bridgevla", "mvt", "configs", "rvt2.yaml",
    )
    parser.add_argument("--mvt_cfg_path", type=str, default=_DEFAULT_MVT_CFG)
    parser.add_argument("--exp_cfg_path", type=str, default="configs/rlbench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--exp_note", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--data_folder", type=str,
        default=os.environ.get("RLBENCH_DATA_FOLDER", DATA_FOLDER),
        help="RLBench data root. The keyframe dataset reads "
             "<data_folder>/train/<task>/all_variations/episodes.",
    )
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Keyframe cache dir (default: under <data_folder>).")
    parser.add_argument("--num_train", type=int, default=100,
                        help="Demos per task to use (matches the released 100/task).")
    parser.add_argument(
        "--cameras", type=str, nargs="+", default=list(CAMERAS),
    )
    parser.add_argument("--load_pretrain", action="store_true")
    parser.add_argument("--pretrain_path", type=str, default=None)
    parser.add_argument(
        "--resume_path", type=str, default=None,
        help="Path to a training checkpoint (.pth) to resume from. Restores "
             "model + optimizer + LR-warmup step + epoch and writes new "
             "checkpoints/logs into the checkpoint's own run folder. Mutually "
             "exclusive with --load_pretrain (resume wins).",
    )
    # --refresh_replay kept as a no-op flag for launch-script backward compat
    # (the YARR replay buffer is no longer used; the keyframe cache self-builds).
    parser.add_argument("--refresh_replay", action="store_true", default=False,
                        help="(deprecated, no-op) the keyframe cache rebuilds "
                             "automatically when its CACHE_VERSION changes.")
    parser.add_argument("--freeze_vision_tower", action="store_true")
    cmd_args = parser.parse_args()
    experiment(cmd_args)
