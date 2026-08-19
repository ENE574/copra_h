#!/usr/bin/env python3
"""Train / evaluate RDE-Net Tier-1 on SKEMPI v2.

Tier-1 = frozen pretrained backbone + mutation-local readout + pc-primary loss.

Train:
  cd RDE-PPI && python train_rde_network_skempi_tier1.py \\
    configs/train/rde_ddg_skempi_tier1.yml --device cuda

Eval best checkpoint:
  python copra_h/scripts/run_rde_tier1_skempi.py --eval-only \\
    --ckpt RDE-PPI/logs_skempi_tier1/.../checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
RDE_ROOT = ROOT / "RDE-PPI"
DEFAULT_CONFIG = RDE_ROOT / "configs/train/rde_ddg_skempi_tier1.yml"
OUT_DIR = RDE_ROOT / "outputs/skempi_tier1"


def _patch_torch_load() -> None:
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]


def evaluate_ckpt(ckpt_path: Path, device: str, num_workers: int) -> pd.DataFrame:
    from rde.models.rde_ddg_tier1 import DDG_RDE_Network_Tier1
    from rde.utils.misc import get_logger
    from rde.utils.skempi import SkempiDatasetManager, eval_skempi_three_modes
    from rde.utils.train import CrossValidation, recursive_to

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    num_cvfolds = len(ckpt["model"]["models"])

    logger = get_logger("tier1_eval", None)
    dataset_mgr = SkempiDatasetManager(
        config, num_cvfolds=num_cvfolds, num_workers=num_workers, logger=logger
    )
    cv_mgr = CrossValidation(
        model_factory=DDG_RDE_Network_Tier1,
        config=config,
        num_cvfolds=num_cvfolds,
    ).to(device)
    cv_mgr.load_state_dict(ckpt["model"])

    results = []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()
            for batch in tqdm(
                dataset_mgr.get_val_loader(fold),
                desc=f"Tier1 fold {fold + 1}/{num_cvfolds}",
            ):
                batch = recursive_to(batch, device)
                _, output_dict = model(batch)
                for complex_name, mutstr, ddg_true, ddg_pred in zip(
                    batch["complex"],
                    batch["mutstr"],
                    output_dict["ddG_true"],
                    output_dict["ddG_pred"],
                ):
                    results.append(
                        {
                            "complex": complex_name,
                            "mutstr": mutstr,
                            "num_muts": len(mutstr.split(",")),
                            "ddG": ddg_true.item(),
                            "ddG_pred": ddg_pred.item(),
                            "method": "RDE-Tier1",
                        }
                    )

    df = pd.DataFrame(results)
    metrics = eval_skempi_three_modes(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "tier1_results.csv", index=False)
    metrics.to_csv(OUT_DIR / "tier1_metrics_official.csv", index=False)
    summary = {
        "ckpt": str(ckpt_path),
        "n_samples": int(len(df)),
        "iteration": ckpt.get("iteration"),
        "val_pc_pearson_at_save": ckpt.get("val/pc_pearson"),
        "official_all": metrics[metrics["mode"] == "all"].iloc[0].to_dict(),
    }
    (OUT_DIR / "tier1_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== RDE Tier-1 official metrics (mode=all) ===")
    print(metrics[metrics["mode"] == "all"].to_string(index=False))
    print(f"\nWrote {OUT_DIR}")
    return df


def train(config_path: Path, device: str, num_workers: int, tag: str) -> int:
    import subprocess

    cmd = [
        sys.executable,
        str(RDE_ROOT / "train_rde_network_skempi_tier1.py"),
        str(config_path),
        "--device",
        device,
        "--num_workers",
        str(num_workers),
    ]
    if tag:
        cmd.extend(["--tag", tag])
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(RDE_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tag", default="")
    parser.add_argument("--train", action="store_true", help="Launch Tier-1 training")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--ckpt", type=Path, default=None)
    args = parser.parse_args()

    os.chdir(RDE_ROOT)
    sys.path.insert(0, str(RDE_ROOT))
    _patch_torch_load()

    if args.train:
        return train(args.config, args.device, args.num_workers, args.tag)

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = RDE_ROOT / "logs_skempi_tier1" / "latest" / "checkpoints" / "best.pt"
    if not ckpt.is_file():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        print("Run with --train first, or pass --ckpt", file=sys.stderr)
        return 1

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        args.device = "cpu"
    evaluate_ckpt(ckpt, args.device, args.num_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
