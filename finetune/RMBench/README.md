<h1 align="center">RMBench: Memory-Dependent Manipulation Benchmark</h1>

RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design. <i>Under Review</i>, [PDF](https://arxiv.org/pdf/2603.01229) | [arXiv](https://arxiv.org/abs/2603.01229) | [Website](https://rmbench.github.io/) | [Join our Community 🔥](https://robotwin-platform.github.io/doc/community/index.html)

> Tianxing Chen*, Yuran Wang*, Mingleyang Li*, Yan Qin*, Hao Shi, Zixuan Li, Yifan Hu, Yingsheng Zhang, Kaixuan Wang, Yue Chen, Hongcheng Wang, Renjing Xu, Ruihai Wu, Yao Mu, Yaodong Yang, Hao Dong†, Ping Luo†

# 🧑🏻‍💻 RMBench Usage

> This project is built upon [RoboTwin 2.0](https://github.com/robotwin-Platform/RoboTwin), and you can seamlessly transfer your policy code between the two projects.

## 1. Installation

One command — the installer creates BOTH conda envs RMBench needs:

* the shared **policy env** `bridgevla_plus_gembench` (override with
  `GEMBENCH_CONDA_ENV`; shared with GemBench/memoryBench), installed in
  `--policy-only` mode — pip stack + point-renderer build only, none of
  GemBench's CoppeliaSim / RLBench / PyRep simulator stack. Skip this step
  with `RMBENCH_SKIP_POLICY_ENV=1` if you manage that env yourself;
* the **SAPIEN sim env** `bridgevla_plus_rmbench` (python 3.10, override with
  `RMBENCH_CONDA_ENV`): SAPIEN / mplib / pytorch3d / CuRobo, plus the
  required site-packages patches.

Idempotent, safe to re-run; each part ends with an import self-check:

```
bash finetune/RMBench/install_rmbench.sh
```

(`script/_install.sh` is kept as a back-compat shim: if you already activated
your own conda env the old RoboTwin way, it reuses that env.)

Upstream RMBench repo: https://github.com/RoboTwin-Platform/RMBench

## 2. Download Assets
To download the assets, run the following command. If you encounter any rate-limit issues, please log in to your Hugging Face account by running `huggingface-cli login`:

```
bash script/_download_assets.sh
```

## 3. Download Data

**For BridgeVLA++ training/eval you do NOT need the raw demos.** Training reads
the keyframe-only HDF5 tree + keyframe metadata released with BridgeVLA++, and
eval only needs the assets from step 2:

```
bash ../../scripts/download_checkpoints_hf.sh rmbench_data
# -> data/bridgevla_data/RMBench/data/{keyframe_data,keyframes}
```

The raw `demo_clean` demos (37 GiB) are only required to RE-GENERATE that
keyframe data (or to train other policies from full trajectories):

```
bash script/_download_data.sh
```

<details>
<summary>If you need to collect the data (we actually recommend downloading it directly)</summary>

> In RMBench, we always use `demo_clean` setting.

Running the following command will first search for a random seed for the target collection quantity, and then replay the seed to collect data.

Please strictly follow our tutorial in [RoboTwin 2.0 Doc - Collect Data](https://robotwin-platform.github.io/doc/usage/collect-data.html).

```
bash collect/single/run_generic.sh ${task_name} ${task_config} ${gpu_id}
# Example: bash collect/single/run_generic.sh cover_blocks demo_clean 0
```
</details>

## 3.5 Keyframe pipeline (raw demos → training data)

The BridgeVLA++ training tree is derived from `demo_clean` in two steps, both
in the `bridgevla_plus_rmbench` env (only needed if you skip the released
`rmbench_data` and rebuild yourself):

```
# (a) trajectory-heuristic keyframe extraction (+ per-keyframe language segment
#     annotation / subtask_idx) -> data/bridgevla_data/RMBench/data/keyframes/<task>.json
python script/extract_key_frames/extract_keyframes.py \
    --data_root  ../../data/bridgevla_data/RMBench/data/data \
    --output_dir ../../data/bridgevla_data/RMBench/data/keyframes

# (b) SAPIEN re-render of ONLY the keyframes, with depth + per-frame
#     intrinsics/extrinsics + colour-corrected RGB
#     -> data/bridgevla_data/RMBench/data/keyframe_data/<task>/keyframe_depth/
bash collect/batch/run_all_keyframe_depth.sh          # all 10 tasks over 8 GPUs
bash collect/single/run_keyframe_depth.sh ${task_name} ${gpu_id}   # single task
```

Notes:
- `keyframes/<task>.json` also feeds training directly: the per-keyframe
  `language_annotation`/`subtask_idx` fields become the memory (`mem_label`)
  supervision. Missing/out-of-sync files are a **hard error** (no silent
  all-zero fallback).
- `keyframes/` must stay **next to** `keyframe_data/` — the dataset resolves it
  as a sibling of the data root.
- Step (b) is resumable and validates each written episode's `joint_action`
  against the original demo. `battery_try` legitimately has 48/50 episodes
  (2 unreplayable seeds, recorded in its `failed_episodes.json`).
- Inspection helpers live in `script/extract_key_frames/`
  (`visualize_keyframes*.py`, `validate_keyframe_depth.py`; run them with
  cwd = `data/bridgevla_data/RMBench`).

## 4. Run Policies

1. Mem-0 (ours): [See Mem-0 Document](./policy/Mem-0/README.md)
2. DP: [See DP Document](https://robotwin-platform.github.io/doc/usage/DP.html)
3. ACT: [See ACT Document](https://robotwin-platform.github.io/doc/usage/ACT.html)
4. Pi 0.5: [See Pi 0.5 Document](https://robotwin-platform.github.io/doc/usage/Pi05.html)
5. X-VLA: [See X-VLA Document](./policy/X-VLA/README.md)
6. Other Policies (Pi0, RDT, etc): [See Document](https://robotwin-platform.github.io/doc/usage) and [See Folder](./policy/)
6. **Configure your policy:** [See Tutorial Here](https://robotwin-platform.github.io/doc/usage/deploy-your-policy.html)

# 👍 Citations

If you find our work useful, please consider citing:

```
@article{chen2026rmbench,
  title={RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design},
  author={Chen, Tianxing and Wang, Yuran and Li, Mingleyang and Qin, Yan and Shi, Hao and Li, Zixuan and Hu, Yifan and Zhang, Yingsheng and Wang, Kaixuan and Chen, Yue and others},
  journal={arXiv preprint arXiv:2603.01229},
  year={2026}
}
```

# 🏷️ License

This repository is released under the MIT license. See [LICENSE](./LICENSE) for additional details.
