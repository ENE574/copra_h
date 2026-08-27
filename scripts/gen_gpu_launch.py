#!/usr/bin/env python3
"""Generate per-GPU run_config copies and a launch script for 6 dataset trainings.

Each dataset gets a dedicated GPU (1-6) so they run in parallel without conflict.
GPU 0 is left for the running P2P job. GPU 7 reserved as headroom.
"""
from pathlib import Path
import yaml

CONFIG_RUNS = Path("/home/csd/lrg/copra_h/config/runs")
CONFIG_MODELS = Path("/home/csd/lrg/copra_h/config/models")
CONFIG_DATA = Path("/home/csd/lrg/copra_h/config/datasets")
OUT_DIR = CONFIG_RUNS / "gpu_launch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (key, stage, run_config, data_config, model_config, gpu)
JOBS = [
    ("pra201", "dG",  "train_pra201_B82.yml",                    "PRA201.yml",                    "best_pra_B76.yml",        1),
    ("pra310", "dG",  "train_pra310_B84.yml",                    "PRA310.yml",                    "best_pra_B76.yml",        2),
    ("pd304",  "dG",  "train_pd304.yml",                         "PD304_train.yml",               "pd_dg_unified.yml",       3),
    ("skempi", "ddG", "train_skempi_alltrain.yml",               "SKEMPI.yml",                    "best_skempi.yml",         4),
    ("mpd",    "ddG", "train_mpd_alltrain_B42.yml",              "MPD_merged.yml",                "best_mpd.yml",            5),
    ("mcsm",   "ddG", "train_mcsm_overlap_90pct_B76_v3.yml",     "mCSM.yml",                      "best_mcsm_B76_optC.yml",  6),
]

CONDA_PY = "/home/csd/anaconda3/envs/copra_h/bin/python"
RUN_PY = "/home/csd/lrg/copra_h/run.py"

launch_lines = ["#!/bin/bash", "set -e", "cd /home/csd/lrg/copra_h", ""]

for key, stage, rc, dc, mc, gpu in JOBS:
    # validate referenced files exist
    assert (CONFIG_RUNS / rc).exists(), f"missing run_config {rc}"
    assert (CONFIG_DATA / dc).exists(), f"missing data_config {dc}"
    assert (CONFIG_MODELS / mc).exists(), f"missing model_config {mc}"

    cfg = yaml.load((CONFIG_RUNS / rc).read_text(), Loader=yaml.FullLoader)
    # assign dedicated GPU and unique output dir.
    # NOTE: CUDA_VISIBLE_DEVICES={gpu} makes only that physical GPU visible to the
    # process as device [0], so Lightning's gpus must be [0], not the physical id.
    cfg["gpus"] = [0]
    base_out = str(cfg.get("output_dir", f"/media/SSD0/csd/lrg/copra_h/outputs/{key}"))
    cfg["output_dir"] = base_out.rstrip("/") + f"_gpu{gpu}"
    new_rc = OUT_DIR / f"{key}_gpu{gpu}.yml"
    new_rc.write_text(yaml.dump(cfg, sort_keys=False))

    log = f"/home/csd/lrg/copra_h/outputs/train_{key}_gpu{gpu}.log"
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu} {CONDA_PY} {RUN_PY} finetune {stage} "
        f"--model_config {CONFIG_MODELS / mc} "
        f"--data_config {CONFIG_DATA / dc} "
        f"--run_config {new_rc} "
        f"> {log} 2>&1 &"
    )
    launch_lines.append(f"echo '[launch] {key} on gpu{gpu} -> {log}'")
    launch_lines.append(cmd)
    print(f"prepared {key}: stage={stage} gpu={gpu} rc={new_rc.name}")

launch_lines += ["", "echo 'All 6 trainings launched in background.'",
                 "sleep 3", "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv"]

launch_sh = OUT_DIR / "launch_all.sh"
launch_sh.write_text("\n".join(launch_lines) + "\n")
print(f"\nWrote launch script: {launch_sh}")
