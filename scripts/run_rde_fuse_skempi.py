#!/usr/bin/env python3
"""Train / evaluate RDE-Fuse-PIA on SKEMPI v2.

From RDE.pt only + CoPRA embedding fusion + PIA auxiliary loss.
Optional online ESM2 + ESM-IF1 via ``rde_ddg_skempi_fuse_esm.yml``.

Train (online ESM):
  cd RDE-PPI && python train_rde_network_skempi_fuse.py \\
    configs/train/rde_ddg_skempi_fuse_esm.yml --device cuda --num_workers 0

Train (RDE only):
  cd RDE-PPI && python train_rde_network_skempi_fuse.py \\
    configs/train/rde_ddg_skempi_fuse.yml --device cuda

Eval:
  python copra_h/scripts/run_rde_fuse_skempi.py --eval-only \\
    --ckpt RDE-PPI/logs_skempi_fuse/.../checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from easydict import EasyDict
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
RDE_ROOT = ROOT / "RDE-PPI"
DEFAULT_CONFIG = RDE_ROOT / "configs/train/rde_ddg_skempi_fuse.yml"
DEFAULT_CONFIG_ESM = RDE_ROOT / "configs/train/rde_ddg_skempi_fuse_esm.yml"
OUT_DIR = RDE_ROOT / "outputs/skempi_fuse"


def _patch_torch_load() -> None:
    _orig = torch.load

    def _load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(*args, **kwargs)

    torch.load = _load  # type: ignore[method-assign]


def evaluate_ckpt(ckpt_path: Path, device: str, num_workers: int) -> pd.DataFrame:
    from rde.models.rde_ddg_fuse import DDG_RDE_Network_Fuse
    from rde.utils.misc import get_logger
    from rde.utils.skempi import SkempiDatasetManager, eval_skempi_three_modes
    from rde.utils.train import CrossValidation, recursive_to

    ckpt = torch.load(ckpt_path, map_location=device)
    config = ckpt["config"]
    num_cvfolds = len(ckpt["model"]["models"])

    logger = get_logger("fuse_eval", None)
    dataset_mgr = SkempiDatasetManager(
        config, num_cvfolds=num_cvfolds, num_workers=num_workers, logger=logger
    )
    cv_mgr = CrossValidation(
        model_factory=DDG_RDE_Network_Fuse,
        config=config,
        num_cvfolds=num_cvfolds,
    ).to(device)
    if getattr(config.model, "online_esm", None) and config.model.online_esm.get("enable", False):
        for model in cv_mgr.models:
            model.bind_online_esm(device, pdb_dir=config.data.pdb_dir)

    state = ckpt["model"]
    for fold, model in enumerate(cv_mgr.models):
        model.load_state_dict(state["models"][fold])
        opt_sd = state.get("optimizers", [None] * num_cvfolds)[fold]
        if opt_sd is not None and fold < len(cv_mgr.optimizers):
            try:
                cv_mgr.optimizers[fold].load_state_dict(opt_sd)
            except ValueError:
                pass

    results = []
    with torch.no_grad():
        for fold in range(num_cvfolds):
            model, _, _ = cv_mgr.get(fold)
            model.eval()
            for batch in tqdm(
                dataset_mgr.get_val_loader(fold),
                desc=f"Fuse fold {fold + 1}/{num_cvfolds}",
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
                            "method": "RDE-Fuse-PIA",
                        }
                    )

    df = pd.DataFrame(results)
    metrics = eval_skempi_three_modes(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "fuse_results.csv", index=False)
    metrics.to_csv(OUT_DIR / "fuse_metrics_official.csv", index=False)
    summary = {
        "ckpt": str(ckpt_path),
        "n_samples": int(len(df)),
        "iteration": ckpt.get("iteration"),
        "best": ckpt.get("best"),
        "official_all": metrics[metrics["mode"] == "all"].iloc[0].to_dict(),
    }
    (OUT_DIR / "fuse_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== RDE-Fuse-PIA official metrics (mode=all) ===")
    print(metrics[metrics["mode"] == "all"].to_string(index=False))
    print(f"\nWrote {OUT_DIR}")
    return df


def train(config_path: Path, device: str, num_workers: int, tag: str) -> int:
    cmd = [
        sys.executable,
        str(RDE_ROOT / "train_rde_network_skempi_fuse.py"),
        str(config_path),
        "--device",
        device,
        "--num_workers",
        str(num_workers),
        "--tag",
        tag,
    ]
    return subprocess.call(cmd, cwd=str(RDE_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--online-esm", action="store_true", help="Use fuse_esm config (online ESM2+ESM-IF1).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tag", type=str, default="fuse")
    args = parser.parse_args()
    if args.online_esm:
        args.config = str(DEFAULT_CONFIG_ESM)

    _patch_torch_load()
    os.chdir(RDE_ROOT)
    sys.path.insert(0, str(RDE_ROOT))

    if args.eval_only:
        ckpt = Path(args.ckpt) if args.ckpt else None
        if ckpt is None or not ckpt.is_file():
            print(f"Checkpoint not found: {ckpt}")
            return 1
        evaluate_ckpt(ckpt, args.device, args.num_workers)
        return 0

    if args.train or not args.eval_only:
        return train(Path(args.config), args.device, args.num_workers, args.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
