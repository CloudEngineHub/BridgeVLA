"""
MemoryBench training entrypoint.

Mirrors finetune/GemBench/train.py 1:1 -- same agent, same DDP scaffolding,
same SwanLab integration, same Stage-1 / Stage-2 freeze schedule. The only
substantive change is the dataset: we read RLBench-format MemoryBench
episodes via MemoryBench_Dataset (GemBench-style keyframe sampling) instead
of GemBench's pre-built LMDB.

Why so much shared code?
The Stage-1/Stage-2 schedule, the optimizer rebuild trick across DDP, and
the per-rank tee logging are already validated in GemBench/train.py.
Re-deriving any of that for MemoryBench would risk subtle drift; we keep the
delta as small as possible (dataset only).

"""
import argparse
import os
import subprocess
import sys
import time
import yaml
from contextlib import redirect_stdout

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

from utils.peract_utils_memorybench import CAMERAS, IMAGE_SIZE, SCENE_BOUNDS, MEMORYBENCH_TASKS
from memorybench_dataset import MemoryBench_Dataset
from visualize import visualize_epoch as visualize_epoch_memorybench

USE_SWANLAB = False


# ---- per-rank stdout/stderr tee (verbatim from GemBench/train.py) ----------
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
    print(f"[MemoryBench] rank {rank} logging stdout/stderr to {log_path}", flush=True)


def create_dataloader(dataset, rank, world_size, batch_size, num_workers,
                      collate_fn=None):
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    kwargs = dict(batch_size=batch_size, num_workers=num_workers,
                  sampler=sampler, drop_last=True, pin_memory=True)
    if collate_fn is not None:
        kwargs["collate_fn"] = collate_fn
    return DataLoader(dataset, **kwargs), sampler


def train(agent, data_loader, epoch, cameras, rank=0):
    """One epoch over the MemoryBench dataloader. No mix loader -- MemoryBench
    is action-only. Otherwise mirrors GemBench/train.py's gb branch.
    """
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
            # ``mem_norm/{block}/total_to_out`` curve is directly readable
            # alongside train/total_loss. Block tags: spatial_s1, spatial_s2,
            # temporal_s1 (only the ones actually built are emitted).
            # Cost: ~18 .norm() calls on small tensors per logging tick.
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


def save_agent(agent, path, epoch, freeze_epochs=None):
    """Checkpoint the agent.

    ``epoch`` / ``model_state`` are kept verbatim so eval-time loaders
    (bridgevla.utils.rvt_utils.load_agent, which reads only those two keys)
    stay byte-compatible with old checkpoints. The extra keys below carry
    everything needed to *resume training exactly*:

      * optimizer_state — AdamW moment buffers + per-group lr.
      * global_step     — drives the manual LR-warmup scale
                          (lr = base_lr * min(1, global_step / warmup_steps)),
                          so restoring it restores the live LR.
      * freeze_epochs   — the Stage-1→Stage-2 boundary in effect when this
                          checkpoint was written; on resume we compare it
                          against the *current* config and warn on mismatch,
                          since the optimizer param-group layout depends on it.

    Older checkpoints (without these keys) still load for eval; resume just
    cannot reconstruct the optimizer from them (handled in restore_checkpoint).
    """
    model = agent._network
    model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": agent._optimizer.state_dict(),
            "global_step": int(getattr(agent, "_global_step", 0)),
            "freeze_epochs": freeze_epochs,
        },
        path,
    )


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
    env_stamp = os.environ.get("MEMORYBENCH_RUN_STAMP")
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
    log_dir = os.path.join(root, "train_memorybench", run_name)
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
    """Same auto-detect as GemBench/train.py: SLURM -> torchrun -> single-GPU debug."""
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
    dist.init_process_group(backend=backend, world_size=world_size, rank=rank)


def experiment(cmd_args):
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    device_id = f"cuda:{local_rank}"
    torch.cuda.set_device(device_id)

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

    _mem_node = getattr(exp_cfg, "memory", None)
    _mem_enabled = bool(getattr(_mem_node, "enabled", False)) if _mem_node is not None else False
    _mem_k = int(getattr(_mem_node, "k_temporal", 2)) if _mem_node is not None else 2
    # Frame-selection policy mirrors RMBench. Default keyframe_gt: slot 0/1 = the
    # two most-recent executed keyframes (the "near neighbour two frames"),
    # slots 2..K-1 = discriminator-admitted subtask-boundary keyframes (none for
    # MemoryBench, which has no boundary labels). The mvt config default is also
    # "keyframe_gt", so the dataset layout and the eval MemoryBank stay aligned.
    _mem_select = str(getattr(_mem_node, "select", "keyframe_gt")) if _mem_node is not None else "keyframe_gt"

    # Task selection precedence (matches RLBench/Colosseum trainers):
    #   1) CLI --tasks (space-separated list) wins if given.
    #   2) Else parse exp_cfg.tasks (comma-separated str). "all" -> MEMORYBENCH_TASKS.
    #   3) Else fall back to MEMORYBENCH_TASKS.
    if cmd_args.tasks:
        tasks = list(cmd_args.tasks)
    else:
        _cfg_tasks = str(getattr(exp_cfg, "tasks", "all") or "all")
        _parsed = [t.strip() for t in _cfg_tasks.split(",") if t.strip()]
        if not _parsed or _parsed[0] == "all":
            tasks = list(MEMORYBENCH_TASKS)
        else:
            tasks = _parsed
    instructions_path = cmd_args.instructions_path or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "assets", "taskvars_instructions.json"
    )

    t_start = time.time()
    train_dataset = MemoryBench_Dataset(
        data_root=cmd_args.data_folder,
        cameras=cmd_args.cameras,
        tasks=tasks,
        ep_per_task=cmd_args.ep_per_task,
        cache_dir=cmd_args.cache_dir,
        image_size=IMAGE_SIZE,
        memory_enabled=_mem_enabled,
        memory_k_temporal=_mem_k,
        memory_select=_mem_select,
        instructions_path=instructions_path,
        verbose=(dist.get_rank() == 0),
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
    if local_rank == 0:
        print(f"Created Dataset. Time Cost: {(time.time()-t_start)/60.0:.1f} minutes")

    mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
    if cmd_args.mvt_cfg_path:
        mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
    if cmd_args.mvt_cfg_opts:
        mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))
    mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)

    # --- Rotation head selection (exp_cfg.rotation_representation). Mirrors
    # RMBench. "6d" -> continuous 6D regression head (rot_ver==2): the rotation
    # slice of the feat vector is a 6D vector (Zhou et al., CVPR 2019), so
    # feat_dim is 6 + grip(2) + collision(2) = 10. This replaces the discrete /
    # autoregressive Euler head. "euler_disc" keeps the rot_ver from rvt2.yaml
    # and the get_num_feat-derived feat_dim. ---
    _rot_repr = str(getattr(exp_cfg, "rotation_representation", "euler_disc"))
    if _rot_repr == "6d":
        mvt_cfg.rot_ver = 2
        # The collision(2) slot is kept even though MemoryBench sets
        # predict_collision=false — the width must stay 10 so checkpoints remain
        # loadable across benches and across a flip of the flag. Those two
        # logits are simply never supervised and never read at eval.
        mvt_cfg.feat_dim = 6 + 2 + 2  # 6D rot + grip(2) + collision(2, unused)
    elif _rot_repr != "euler_disc":
        raise ValueError(
            f"Unknown rotation_representation={_rot_repr!r} "
            "(expected 'euler_disc' or '6d')"
        )

    # Top-level scalar mirrors. Yacs scopes are independent so we copy
    # any exp_cfg-level toggles the network actually consumes onto
    # mvt_cfg before MVT() is built.
    if hasattr(exp_cfg, "gradient_checkpointing") and hasattr(mvt_cfg, "gradient_checkpointing"):
        mvt_cfg.gradient_checkpointing = bool(exp_cfg.gradient_checkpointing)
    if hasattr(exp_cfg, "feat_from_stage1") and hasattr(mvt_cfg, "feat_from_stage1"):
        # Rotation/grip/collision feature source: stage 1 (coarse) when True.
        mvt_cfg.feat_from_stage1 = bool(exp_cfg.feat_from_stage1)
    # Documentation mirror only (MVT never reads it); the live gate is the
    # ``predict_collision=`` kwarg passed to RVTAgent below. Mirrored so the
    # dumped mvt_cfg.yaml records which mode the checkpoint was trained in.
    if hasattr(exp_cfg, "predict_collision") and hasattr(mvt_cfg, "predict_collision"):
        mvt_cfg.predict_collision = bool(exp_cfg.predict_collision)

    # Mirror the unified-yaml memory propagation from GemBench / RMBench.
    # ``select`` + ``discriminator`` are now propagated too so the dataset
    # layout (keyframe_gt) and the eval MemoryBank stay aligned, and the dumped
    # mvt_cfg.yaml self-documents the memory policy for eval-time load_agent.
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

    # NOTE: dropped the historic ``image_resolution=[IMAGE_SIZE, IMAGE_SIZE]``
    # kwarg here — RVTAgent stores it on ``self._image_resolution`` but never
    # reads it back (vestigial from the peract qattention agent class). The
    # actual training-input H/W comes from the dataset's emitted tensor shape
    # (set by ``MemoryBench_Dataset.image_size``), and the eval-time camera
    # render resolution comes from ``RLBenchEnv(image_size=...)`` in client.py.
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

    # Same always-frozen list as GemBench: up_grounding is warm-started from
    # pretrain but never trained during action finetuning.
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
            print(f"[Info] SwanLab init failed ({e}); training continues without SwanLab")

    if dist.get_rank() == 0:
        # Memory-branch init norms baseline. With default (non-zero)
        # initialisation of the residual exits these are non-zero scalars
        # at step 0; we log them so the training-time curves have a clean
        # reference point. The 3 logical blocks (spatial_s1 / spatial_s2 /
        # temporal_s1) are printed compactly; full per-layer norms go to
        # swanlab.
        init_norms = memory_param_norms(agent._network)
        if init_norms:
            print("[mem_norm @ init] baseline total_to_out per block "
                  "(non-zero with default init):")
            for k in sorted(init_norms.keys()):
                if k.endswith("/total_to_out"):
                    print(f"  {k}: {init_norms[k]:.4e}")
            if USE_SWANLAB:
                swanlab.log(init_norms, step=0)
        else:
            print("[mem_norm @ init] memory disabled (no MemoryBlock instantiated).")
        print("Start training ...", flush=True)

    # start_epoch / end_epoch are set right after agent.build() (start_epoch is
    # advanced by the resume block when --resume_path is given). Do NOT reset
    # start_epoch=0 here or a resume would silently restart from epoch 0.
    end_epoch = EPOCHS

    # Per-task per-episode visualization cadence (mirrors GemBench/train.py).
    # Rank 0 picks ONE random episode for each task every `viz_every_n_epochs`
    # epochs (plus epoch 0 and the final epoch), dumps pred+GT heatmaps for
    # every step of that episode, then everyone re-syncs at a barrier.
    #
    # `viz_tasks`:
    #   None / unset    -> all tasks (default).
    #   []              -> disabled (equivalent to viz_every_n_epochs=0).
    #   list[str]       -> only those tasks.
    VIZ_EVERY_N_EPOCHS = int(getattr(exp_cfg, "viz_every_n_epochs", 20))
    _viz_tasks_cfg = getattr(exp_cfg, "viz_tasks", None)
    VIZ_TASKS = list(_viz_tasks_cfg) if _viz_tasks_cfg is not None else None
    VIZ_DISABLED = (VIZ_TASKS is not None) and (len(VIZ_TASKS) == 0)

    def run_viz(epoch_idx: int) -> None:
        if dist.get_rank() != 0 or VIZ_EVERY_N_EPOCHS <= 0 or VIZ_DISABLED:
            return
        is_viz_epoch = (
            epoch_idx == 0
            or (epoch_idx % VIZ_EVERY_N_EPOCHS == 0)
            or (epoch_idx == end_epoch - 1)
        )
        if not is_viz_epoch:
            return
        try:
            visualize_epoch_memorybench(
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
            print(f"[MemoryBench] visualize_epoch failed at epoch {epoch_idx}: {e}",
                  flush=True)
            traceback.print_exc()

    i = start_epoch
    while True:
        if i == end_epoch:
            break
        # Stage-1 -> Stage-2 transition: same DDP rebuild trick as GemBench.
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

        # Pre-epoch visualization on rank 0. Barrier after so non-zero
        # ranks don't race ahead into the train loop while rank 0 is still
        # writing images.
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
            save_agent(agent, f"{log_dir}/model_{i}.pth", i, freeze_epochs=FREEZE_EPOCHS)
            save_agent(agent, f"{log_dir}/model_last.pth", i, freeze_epochs=FREEZE_EPOCHS)
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
                        default="configs/memorybench_config.yaml")
    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument(
        "--data_folder", type=str,
        # Defaults to the MEMORYBENCH_DATA_FOLDER environment variable; falls back to a repo-relative
        # path when unset (resolved against cwd when run via python directly).
        default=os.environ.get(
            "MEMORYBENCH_DATA_FOLDER",
            "data/bridgevla_data/memorybench/data/train",
        ),
    )
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--instructions_path", type=str, default=None)
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
        help="Task names to train on. Default: all 3 MemoryBench tasks.",
    )
    parser.add_argument(
        "--cameras", type=str, nargs="+",
        default=["left_shoulder", "right_shoulder", "wrist", "front"],
    )
    cmd_args = parser.parse_args()
    experiment(cmd_args)
