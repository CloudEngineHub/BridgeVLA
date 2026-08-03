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
Adapted from https://github.com/robot-colosseum/rvt_colosseum/blob/main/rvt/train.py
Therefore, the code is also under the NVIDIA Source Code License

'''
import os
import sys
import subprocess
import time
import tqdm
import yaml
import argparse
from contextlib import redirect_stdout
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import swanlab
os.environ["BITSANDBYTES_NOWELCOME"] = "1"


class _TeeStream:
    # Mirror Python-level stdout/stderr into a log file so a run folder always contains its own log, even when
    # train.sh's tee is not in the launch path. C-library writes through fd 1/2 still need the shell tee.
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
    print(f"[GemBench] rank {rank} logging stdout/stderr to {log_path}", flush=True)

import bridgevla.config as exp_cfg_mod
import bridgevla.models.bridgevla_agent as bridgevla_agent
import bridgevla.mvt.config as mvt_cfg_mod

from bridgevla.mvt.mvt import MVT
from bridgevla.models.bridgevla_agent import print_loss_log
from bridgevla.mvt.memory import memory_param_norms
from bridgevla.utils.rvt_utils import (
    get_num_feat,
)
from utils.peract_utils_gembench import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from gembench_dataset import Gembench_Dataset
from visualize import visualize_epoch as visualize_epoch_gembench

USE_SWANLAB = False


def create_dataloader(dataset, rank, world_size, batch_size, num_workers,
                      collate_fn=None):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
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
    dataloader = DataLoader(dataset, **kwargs)
    return dataloader, sampler


def train(agent, data_loader, epoch,
          cameras=["front", "left_shoulder", "right_shoulder", "wrist"],
          rank=0):
    """Run one epoch of GemBench action training. Losses are tagged `gb/`."""
    agent.train()

    def move_tensors_to_device(d, device):
        if isinstance(d, dict):
            return {k: move_tensors_to_device(v, device) if isinstance(v, dict) else v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in d.items()}
        return d

    total_grad_steps = len(data_loader)
    pbar = tqdm.tqdm(total=total_grad_steps, disable=(rank != 0),
                     position=0, leave=True, desc=f"epoch {epoch}")

    iteration = 0
    epoch_losses = {}
    step_global = 0
    last_gb_loss = None

    def _update_postfix(kind: str, lr: float):
        if rank != 0:
            return
        postfix = {"step": kind, "lr": f"{lr:.2e}"}
        if last_gb_loss is not None:
            postfix["gb_loss"] = f"{last_gb_loss:.3f}"
        pbar.set_postfix(**postfix)

    for raw_batch in data_loader:
        iteration += 1
        # ---- GemBench (up_action) ----
        batch = move_tensors_to_device(raw_batch, agent._device)
        batch["tasks"] = raw_batch["tasks"]
        batch["lang_goal"] = [[[item]] for item in raw_batch["lang_goal"]]
        update_args = {
            "cameras": cameras,
            "replay_sample": batch,
            "backprop": True,
            "reset_log": (iteration == 1),
        }
        out = agent.update_gembench(**update_args)

        if epoch_losses == {}:
            epoch_losses = {key: [] for key in out.keys()}
        for key in epoch_losses:
            epoch_losses[key].append(out[key])

        last_gb_loss = out.get("total_loss")
        step = epoch * total_grad_steps + step_global
        if rank == 0 and USE_SWANLAB and step % 10 == 0:
            log_dict = {f"gb/{k}": v for k, v in out.items()}
            # Track memory-branch weight norms on the same step axis, so ``mem_norm/{block}/total_to_out`` reads
            # alongside gb/total_loss (tags: spatial_s1, spatial_s2, temporal_s1; only blocks actually built are
            # emitted). A flat curve means no gradient is flowing through the memory branch.
            log_dict.update(memory_param_norms(agent._network))
            swanlab.log(log_dict, step=step)
        _update_postfix("gb", out.get("lr", 0.0))
        pbar.update(1)
        step_global += 1

    pbar.close()
    if rank == 0:
        print_loss_log(agent)

    return {f"gb/{k}": sum(v) / len(v) for k, v in epoch_losses.items()}


def save_agent(agent, path, epoch, freeze_epochs=None, save_optimizer_state=False):
    """Checkpoint the agent.

    ``epoch`` / ``model_state`` are kept verbatim so eval-time loaders
    (bridgevla.utils.rvt_utils.load_agent, which reads only those two keys)
    stay byte-compatible with old checkpoints.

    ``global_step`` / ``freeze_epochs`` are scalars (no file-size cost) and are
    ALWAYS written — global_step drives the manual LR-warmup scale
    (lr = base_lr * min(1, global_step / warmup_steps)) so restoring it restores
    the live LR, and freeze_epochs records the Stage-1→Stage-2 boundary in
    effect when this checkpoint was written (on resume we compare it against the
    *current* config and warn on mismatch, since the optimizer param-group
    layout depends on it).

    ``save_optimizer_state`` (default False) gates only the heavy part:

      * optimizer_state — AdamW moment buffers + per-group lr (~2x the model
                          weights in size). Needed for resume to continue
                          training EXACTLY where it left off; without it,
                          restore_checkpoint falls back to a freshly-built
                          optimizer + global_step=0 (warmup restarts).

    Older checkpoints (without these keys) still load for eval; resume just
    cannot reconstruct the optimizer from them (handled in restore_checkpoint).
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


def restore_checkpoint(agent, ckpt, local_rank, rank, freeze_epochs, always_freeze):
    """Restore model + optimizer + LR-warmup step from a training checkpoint.

    Must be called AFTER ``agent.build()`` (so ``agent._optimizer`` exists) and
    AFTER the caller has reconstructed the correct *stage* (Stage-1 vs Stage-2)
    so the optimizer's param-group layout matches the saved state. Returns the
    epoch to resume FROM (i.e. ``saved_epoch + 1``).

    Stage reconstruction (done by the caller before this fn) is keyed off
    ``start_epoch`` vs the CURRENT-config ``freeze_epochs``:
      * start_epoch <= freeze_epochs  -> Stage 1 layout; the main loop's
        ``i == freeze_epochs`` branch will rebuild the optimizer fresh, exactly
        as in an uninterrupted run, so we DON'T force the saved Stage-1
        optimizer state onto a Stage-2 optimizer.
      * start_epoch >  freeze_epochs  -> Stage 2 already active; the caller has
        unfrozen + rebuilt DDP + rebuilt the optimizer, and we load the saved
        Stage-2 optimizer state here.
    """
    # 1) Model weights (handles DDP-wrapped or bare module).
    model = agent._network.module if isinstance(agent._network, DDP) else agent._network
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)
    if rank == 0 and (missing or unexpected):
        print(f"[resume] model_state loaded with missing={len(missing)} "
              f"unexpected={len(unexpected)} keys (strict=False).")

    saved_epoch = int(ckpt["epoch"])
    start_epoch = saved_epoch + 1

    # 2) freeze_epochs consistency check — config wins, but warn loudly: a changed boundary silently breaks
    #    stage reconstruction.
    saved_fe = ckpt.get("freeze_epochs", None)
    if rank == 0 and saved_fe is not None and int(saved_fe) != int(freeze_epochs):
        print(f"[resume] WARNING: freeze_epochs changed since checkpoint "
              f"(saved={saved_fe}, current={freeze_epochs}). Using current "
              f"config; stage reconstruction follows the CURRENT value. If the "
              f"checkpoint was already in Stage 2, do NOT raise freeze_epochs "
              f"above the resume epoch.")

    # 3) global_step drives the LR-warmup scale, so restore it — but only when the saved step belongs to the
    #    SAME optimizer we just (re)built. start_epoch > freeze_epochs means the Stage-2 optimizer was rebuilt
    #    during the original run and the saved step is the post-boundary count; start_epoch < freeze_epochs
    #    means we are still on the Stage-1 optimizer the step was saved against. At exactly the boundary the
    #    loop rebuilds fresh, so the saved step no longer applies.
    opt_state = ckpt.get("optimizer_state", None)
    can_load_opt = opt_state is not None and (
        start_epoch < freeze_epochs or start_epoch > freeze_epochs
    )
    if start_epoch == freeze_epochs:
        # The main loop calls rebuild_optimizer() at i==freeze_epochs, zeroing global_step and building a fresh
        # Stage-2 optimizer, so loading the stale Stage-1 optimizer here would be overwritten anyway.
        if rank == 0:
            print(f"[resume] start_epoch ({start_epoch}) == freeze_epochs; "
                  f"the Stage-2 optimizer will be rebuilt fresh by the loop "
                  f"(matches uninterrupted training). Skipping optimizer-state "
                  f"load; global_step stays 0.")
    elif can_load_opt:
        try:
            agent._optimizer.load_state_dict(opt_state)
            # Move optimizer state tensors onto this rank's device: AdamW's exp_avg / exp_avg_sq were saved on
            # rank 0's GPU, so re-pin them locally on every rank.
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
    # GEMBENCH_RUN_STAMP lets train.sh pre-compute the stamp so the shell tee and Python share one run folder.
    env_stamp = os.environ.get("GEMBENCH_RUN_STAMP")
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
    # Idempotent: train.sh may already have appended the memory-ablation suffix via --exp_cfg_opts; only add
    # it when absent, so a direct ``python train.py`` still gets the marker.
    base = exp_cfg.swanlab_run
    suffix = memory_ablation_suffix(exp_cfg)
    if suffix and not base.endswith(suffix):
        base = f"{base}{suffix}"
    return f"{base}_{get_time()}"


def get_logdir(cmd_args, exp_cfg, dist, run_name):
    root = cmd_args.log_dir if cmd_args.log_dir else exp_cfg.log_dir
    log_dir = os.path.join(root, "train_gembench", run_name)
    if cmd_args.debug:
        log_dir = os.path.join(log_dir, "debug")
    if dist.get_rank() == 0:
        os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.

    Supports three launch modes (auto-detected in order):
      1. SLURM  — srun sets SLURM_JOB_ID / SLURM_PROCID / SLURM_NTASKS
      2. torchrun / MLP / PAI — sets RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR
      3. DEBUG  — single-GPU fallback when DEBUG=true
    """
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
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    elif os.getenv("DEBUG", "false").lower() == "true":
        print("Cannot find RANK and WORLD_SIZE — entering single-GPU debug mode")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "9001")
        os.environ.setdefault("LOCAL_RANK", "0")
        rank = 0
        world_size = 1
    else:
        raise RuntimeError(
            "Distributed env vars not found. "
            "Launch with torchrun / srun, or set DEBUG=true for single-GPU mode."
        )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Generous timeout so a rank-0-only pass (per-epoch viz / long dataset build) doesn't trip NCCL's default
    # collective timeout while the other ranks wait at a barrier.
    from datetime import timedelta
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
        timeout=timedelta(hours=1),
    )


def experiment(cmd_args):
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    cuda_device_count = 0
    for _ in range(5):
        cuda_device_count = torch.cuda.device_count()
        if cuda_device_count > 0:
            break
        time.sleep(2)
    if cuda_device_count <= 0:
        raise RuntimeError(
            "No CUDA device is visible to PyTorch. Check GPU allocation and "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')} "
            "before launching training."
        )
    if local_rank >= cuda_device_count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is out of range for the "
            f"{cuda_device_count} CUDA device(s) visible to PyTorch. "
            "Make torchrun --nproc_per_node match the visible GPU count, or "
            "fix CUDA_VISIBLE_DEVICES."
        )
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)

    # ---- Resume vs pretrain mutual exclusion (resume wins) ----
    # A resume restores the full training state, so warm-starting from a pretrain checkpoint at the same time
    # would be contradictory.
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

    ddp = int(os.environ['WORLD_SIZE']) > 1
    if dist.get_rank() == 0:
        print(f"World size: {dist.get_world_size()}")
        print(f"CUDA devices visible per process: {cuda_device_count}")
        if ddp:
            print(f"Running DDP (world_size={dist.get_world_size()}).")

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.exp_cfg_opts}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{cmd_args.mvt_cfg_opts}"

    if local_rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    BATCH_SIZE_TRAIN = exp_cfg.bs
    if local_rank == 0:
        print(f"BATCH_SIZE_TRAIN={BATCH_SIZE_TRAIN}")

    EPOCHS = exp_cfg.epochs
    FREEZE_EPOCHS = int(getattr(exp_cfg, "freeze_epochs", 2))

    # Unified run name: built on rank 0 and broadcast so every rank uses the same directory. On resume it is
    # the ORIGINAL run folder's basename, so SwanLab logs and log_dir keep the original timestamp.
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
    # Resume: pin log_dir to the EXACT directory holding the checkpoint, so new checkpoints / logs / viz land
    # back in the original timestamped run folder regardless of the current --log_dir root.
    if cmd_args.resume_path:
        log_dir = os.path.dirname(os.path.abspath(cmd_args.resume_path))
        if dist.get_rank() == 0:
            os.makedirs(log_dir, exist_ok=True)
            print(f"[resume] writing into original run folder: {log_dir}")
    # Mirror each rank's stdout/stderr into <log_dir>/train_rank{rank}.log, alongside train.sh's shell tee.
    _install_tee_logging(log_dir, dist.get_rank())

    # The unified yaml carries the memory block under exp_cfg.memory; it is propagated onto mvt_cfg.memory
    # below, and read here so the dataset can produce anchor / history bundles per sample.
    _mem_node = getattr(exp_cfg, "memory", None)
    _mem_enabled = bool(getattr(_mem_node, "enabled", False)) if _mem_node is not None else False
    _mem_k = int(getattr(_mem_node, "k_temporal", 2)) if _mem_node is not None else 2
    # Frame-selection policy mirrors RMBench / memoryBench. Default keyframe_gt: slots 0/1 = the two
    # most-recent executed keyframes, slots 2..K-1 = discriminator-admitted subtask boundaries (none for
    # GemBench, which has no boundary labels).
    _mem_select = str(getattr(_mem_node, "select", "keyframe_gt")) if _mem_node is not None else "keyframe_gt"

    t_start = time.time()
    train_dataset = Gembench_Dataset(
        cmd_args.data_folder,
        device=device_id,
        cameras=cmd_args.cameras,
        ep_per_task=cmd_args.ep_per_task,
        tasks=cmd_args.tasks,
        memory_enabled=_mem_enabled,
        memory_k_temporal=_mem_k,
        memory_select=_mem_select,
    )
    if local_rank == 0:
        print("Total tasks:", train_dataset.num_tasks)
        print("Total trajectories:", train_dataset.num_task_paths)
        print("Dataset Length:", len(train_dataset))

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    train_dataloader, train_sampler = create_dataloader(
        train_dataset, rank, world_size, BATCH_SIZE_TRAIN, exp_cfg.num_workers
    )

    t_end = time.time()
    if local_rank == 0:
        print("Created Dataset. Time Cost: {:.1f} minutes".format((t_end - t_start) / 60.0))

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path != "":
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts != "":
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)

    # --- Rotation head selection (exp_cfg.rotation_representation), mirroring RMBench / memoryBench ---
    # "6d" -> continuous 6D regression head (rot_ver==2): the feat vector's rotation slice is a 6D vector
    # (Zhou et al., CVPR 2019), so feat_dim = 6 + grip(2) + collision(2) = 10, replacing the discrete Euler
    # head. "euler_disc" keeps rvt2.yaml's rot_ver and the get_num_feat feat_dim.
    _rot_repr = str(getattr(exp_cfg, "rotation_representation", "euler_disc"))
    if _rot_repr == "6d":
        mvt_cfg.rot_ver = 2
        # The collision(2) slot is kept even though GemBench sets predict_collision=false — the width must stay
        # 10 so checkpoints remain loadable across benches. Those logits are never supervised or read.
        mvt_cfg.feat_dim = 6 + 2 + 2  # 6D rot + grip(2) + collision(2, unused)
    elif _rot_repr != "euler_disc":
        raise ValueError(
            f"Unknown rotation_representation={_rot_repr!r} "
            "(expected 'euler_disc' or '6d')"
        )

    # Top-level scalar mirrors: yacs scopes are independent, so copy the exp_cfg toggles the network consumes
    # onto mvt_cfg before MVT() is built.
    if hasattr(exp_cfg, "gradient_checkpointing") and hasattr(mvt_cfg, "gradient_checkpointing"):
        mvt_cfg.gradient_checkpointing = bool(exp_cfg.gradient_checkpointing)
    if hasattr(exp_cfg, "feat_from_stage1") and hasattr(mvt_cfg, "feat_from_stage1"):
        # Rotation/grip/collision feature source: stage 1 (coarse) when True.
        mvt_cfg.feat_from_stage1 = bool(exp_cfg.feat_from_stage1)
    # Documentation mirror only (MVT never reads it); the live gate is the ``predict_collision=`` kwarg passed
    # to RVTAgent below. Mirrored so the dumped mvt_cfg.yaml records the checkpoint's training mode.
    if hasattr(exp_cfg, "predict_collision") and hasattr(mvt_cfg, "predict_collision"):
        mvt_cfg.predict_collision = bool(exp_cfg.predict_collision)

    # ``exp_cfg.memory`` (from configs/gembench_config.yaml) is the SINGLE source of truth for memory
    # settings; the whole block is propagated onto ``mvt_cfg.memory``. Anyone using a separate
    # ``--mvt_cfg_path`` should not set ``memory:`` there, as this would overwrite it. ``select`` +
    # ``discriminator`` go too, keeping the dataset layout and the eval MemoryBank aligned.
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

    assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
        mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
    )

    backbone = MVT(
        renderer_device=device_id,
        load_pretrain=cmd_args.load_pretrain,
        pretrain_path=cmd_args.pretrain_path,
        **mvt_cfg,
    )
    backbone = backbone.to(local_rank)
    backbone = DDP(backbone, device_ids=[local_rank], find_unused_parameters=True)

    # NOTE: the historic ``image_resolution`` kwarg was dropped — RVTAgent stored but never read it. The live
    # consumer of IMAGE_SIZE is ``RLBenchEnv(image_size=…)`` in client.py, which controls CoppeliaSim's render
    # resolution at eval. Training input H/W comes from the LMDB shape (256x256).
    agent = bridgevla_agent.RVTAgent(
        network=backbone,
        stage_two=mvt_cfg.stage_two,
        rot_ver=mvt_cfg.rot_ver,
        scene_bounds=SCENE_BOUNDS,
        cameras=CAMERAS,
        log_dir=f"{log_dir}/test_run/",
        warmup_steps=int(getattr(exp_cfg, "warmup_steps", 1000)),
        predict_collision=bool(getattr(exp_cfg, "predict_collision", True)),
        **exp_cfg.peract,
        **exp_cfg.rvt,
    )

    # ---- Always-frozen modules (both stages) ----
    # PaliGemma lm_head / embed_tokens (tied, so freezing either freezes the shared Parameter; gradient still
    # flows THROUGH into the Gemma body), vision_tower (SigLIP), and up_grounding — which only exists when
    # use_modified_focal_loss=True and is warm-started from pretrain but never consumed by the gembench forward.
    always_freeze = [
        "lm_head", "embed_tokens",
        "vision_tower",
    ]
    use_focal = bool(getattr(mvt_cfg, "use_modified_focal_loss", False))
    if use_focal:
        always_freeze.append("up_grounding")
    if cmd_args.freeze_vision_tower and dist.get_rank() == 0:
        print("[Info] --freeze_vision_tower is now the default; flag is a no-op.")

    # ---- Stage 1: PaliGemma frozen (except always_freeze) ----
    for name, param in agent._network.named_parameters():
        if any(af in name for af in always_freeze):
            param.requires_grad = False
        elif "mvt1.model" in name:
            param.requires_grad = False
    if dist.get_rank() == 0:
        if use_focal:
            grounding_msg = "up_grounding loaded from pretrain but frozen"
        else:
            grounding_msg = "up_grounding head not built (use_modified_focal_loss=False)"
        print(f"[Stage 1] PaliGemma frozen for {FREEZE_EPOCHS} epochs "
              f"— heatmap head + feat_fc heads trainable "
              f"({grounding_msg}).")

    total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
    if dist.get_rank() == 0:
        print(f'Total trainable parameters: {total_params / 1e9:.2f} billion')

    agent.build(training=True, device=device_id)

    start_epoch = 0
    end_epoch = EPOCHS

    # ---- Resume from a training checkpoint (full state) ----
    # Scheduling config (freeze_epochs / save_every_n_epochs / epochs / warmup_steps / lr) always comes from
    # the current YAML+CLI; the checkpoint carries only training STATE. freeze_epochs / warmup_steps should be
    # left unchanged on resume, since they anchor the restored optimizer state.
    if cmd_args.resume_path:
        if not os.path.isfile(cmd_args.resume_path):
            raise FileNotFoundError(
                f"--resume_path is not a file: {cmd_args.resume_path}"
            )
        if dist.get_rank() == 0:
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

    # SwanLab, controlled by $SWANLAB_MODE (set by train.sh via SWANLAB_UPLOAD): "cloud" uploads (needs
    # $SWANLAB_API_KEY), "offline" writes <log_dir>/swanlog/ only, "disabled" is a no-op (also forced by
    # --debug). View offline logs with: `swanlab watch -l <log_dir>/swanlog`
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
                project=swanlab_project,
                experiment_name=run_name,
                mode=mode,
                logdir=swanlab_logdir,
            )

        try:
            _swanlab_init(swanlab_mode)
            USE_SWANLAB = True
            print(f"[Info] SwanLab enabled ({swanlab_mode} mode — logs at {swanlab_logdir}), project={swanlab_project}, run={run_name}")
        except Exception as e:
            # Cloud auth/init can fail without outbound HTTPS to api.swanlab.cn; drop to offline so metrics land on disk.
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
        print("Start training ...", flush=True)

    # Per-task per-episode visualization cadence. Rank 0 picks ONE random episode per task every
    # `viz_every_n_epochs` epochs (plus epoch 0 for the pre-training baseline), dumps pred+GT heatmaps for
    # every step of that episode, then everyone re-syncs at a barrier.
    # `viz_tasks`: None -> all tasks (default); [] -> disabled; list[str] -> only those.
    VIZ_EVERY_N_EPOCHS = int(getattr(exp_cfg, "viz_every_n_epochs", 20))
    _viz_tasks_cfg = getattr(exp_cfg, "viz_tasks", None)
    VIZ_TASKS = list(_viz_tasks_cfg) if _viz_tasks_cfg is not None else None
    VIZ_DISABLED = (VIZ_TASKS is not None) and (len(VIZ_TASKS) == 0)
    INITIAL_VIZ = bool(getattr(exp_cfg, "initial_viz", True))

    def run_viz(epoch_idx: int) -> None:
        if dist.get_rank() != 0 or VIZ_EVERY_N_EPOCHS <= 0 or VIZ_DISABLED:
            return
        # initial_viz=False skips the epoch-0 baseline pass. This guard must precede the modulo check, since
        # 0 % VIZ_EVERY_N_EPOCHS == 0 would otherwise fire at epoch 0 anyway.
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
            visualize_epoch_gembench(
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
            print(f"[GemBench] visualize_epoch failed at epoch {epoch_idx}: {e}",
                  flush=True)
            traceback.print_exc()

    i = start_epoch
    while True:
        if i == end_epoch:
            break

        # ---- Stage 2 transition: unfreeze the PaliGemma backbone (except always_freeze) ----
        # DDP's reducer is pinned to the requires_grad set at construction, so params flipped False->True are
        # never all-reduced — a silent multi-GPU correctness bug. Rebuilding the DDP wrapper fixes it.
        if i == FREEZE_EPOCHS:
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
            total_params = sum(p.numel() for p in agent._network.parameters() if p.requires_grad)
            if dist.get_rank() == 0:
                if use_focal:
                    grounding_msg = "up_grounding frozen"
                else:
                    grounding_msg = "no up_grounding head (use_modified_focal_loss=False)"
                print(f"[Stage 2] Unfroze PaliGemma (vision_tower + lm_head/embed_tokens "
                      f"still frozen; {grounding_msg}). "
                      f"Rebuilt DDP + optimizer. "
                      f"Trainable params: {total_params / 1e9:.2f}B")

        # Pre-epoch visualization on rank 0, with a barrier after so other ranks don't race ahead while rank 0 writes images.
        run_viz(i)
        dist.barrier()

        print(f"Rank [{dist.get_rank()}], Epoch [{i}]: Training on train dataset")

        train_dataloader.sampler.set_epoch(i)
        out = train(agent, train_dataloader, epoch=i, rank=dist.get_rank(),
                    cameras=cmd_args.cameras)

        save_every = int(getattr(exp_cfg, "save_every_n_epochs", 20))
        is_periodic_save = save_every > 0 and i > 0 and (i % save_every == 0)
        is_final_save = save_every > 0 and i == end_epoch - 1
        if dist.get_rank() == 0 and (is_periodic_save or is_final_save):
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
    parser.add_argument("--mvt_cfg_path", type=str, default="../bridgevla/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/gembench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--data_folder", type=str, default="/PATH_TO_TRAIN_DATA/train_dataset")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ep_per_task", type=int, default=10000)
    parser.add_argument("--freeze_vision_tower", action="store_true")
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
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="List of task names to train on. If None, use all tasks."
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["left_shoulder", "right_shoulder", "wrist", "front"],
        help="List of camera names"
    )
    cmd_args = parser.parse_args()
    experiment(cmd_args)
