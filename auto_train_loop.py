#!/usr/bin/env python3
"""
Auto Train Loop for MPD PEMPNI — automated hyperparameter optimization.

Orchestrates: train → analyze → adjust → retrain → repeat.

How it works:
  1. Launches training (python run.py finetune ddG ...) as a subprocess.
  2. Polls for completion every 60 s while printing latest metrics.
  3. Reads metrics.csv and res.json when done.
  4. Scores the run (val/all_pearson or train/all_pearson depending on split).
  5. If score > best_so_far, saves the config as best.
  6. Generates next trial config by picking one hyperparameter to change.
  7. Launches next trial.
  8. Repeats for N trials or until early stopping.

Usage:
  python auto_train_loop.py \
    --model_config config/models/mpd_pempni_final_v2.yml \
    --data_config config/datasets/MPD_pempni.yml \
    --run_config config/runs/finetune_mpd_pempni_final_gpu0.yml \
    --gpu 1 \
    --max_trials 10 \
    --early_stop_trials 3

Dependencies: pyyaml, pandas (both already in conda env).
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ============================================================
#  Hyperparameter search space
# ============================================================
HPARAM_SPACE = {
    "lr": [5e-5, 3e-5, 8e-5, 1e-4, 1e-5, 2e-5],
    "weight_decay": [1e-4, 1e-3, 5e-5, 0.0, 5e-4],
    "physics_aux_max_weight": [0.01, 0.005, 0.02, 0.05, 0.002, 0.0],
    "huber_delta": [2.0, 1.0, 3.0, 0.5, 1.5],
    "mutation_local_window": [16, 8, 24, 32, 6],
    "attention_dropout": [0.1, 0.2, 0.05, 0.15, 0.3],
    "residual_dropout": [0.1, 0.2, 0.05, 0.0],
    "coformer_blocks": [6, 4, 8, 10],
    "coformer_embed_dim": [320, 256, 384, 192],
    "batch_size": [1, 2],
}


class AutoTrainLoop:
    """Automated training loop with config tracking and hyperparameter adjustment."""

    def __init__(
        self,
        model_config_path: str,
        data_config_path: str,
        run_config_path: str,
        gpu: int = 0,
        max_trials: int = 10,
        early_stop_trials: int = 3,
        min_improvement: float = 0.01,
        trials_dir: str = "outputs/auto_train_trials",
        python: str = None,
    ):
        self.model_config_path = Path(model_config_path)
        self.data_config_path = Path(data_config_path)
        self.run_config_path = Path(run_config_path)
        self.gpu = gpu
        self.max_trials = max_trials
        self.early_stop_trials = early_stop_trials
        self.min_improvement = min_improvement
        self.trials_dir = Path(trials_dir)
        self.python = python or sys.executable

        # Trial history
        self.trials = []          # list of dicts with trial metadata + score
        self.best_score = float("-inf")
        self.best_trial = None
        self.best_config = None
        self.no_improve_count = 0

        # Load base configs
        self.base_model_config = self._load_yaml(model_config_path)
        self.base_data_config = self._load_yaml(data_config_path)
        self.base_run_config = self._load_yaml(run_config_path)

        # Current config (starts as copy of base)
        self.model_config = deepcopy(self.base_model_config)
        self.data_config = deepcopy(self.base_data_config)
        self.run_config = deepcopy(self.base_run_config)
        self.run_config["gpus"] = [self.gpu]  # ensure correct GPU

        self.current_workdir = None

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _load_yaml(path):
        with open(path) as f:
            return yaml.safe_load(f)

    def _dump_yaml(self, data, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=None, indent=2)
        return path

    def _pick_unused_gpu(self, exclude=None):
        """Return a free GPU index, preferring the configured one."""
        exclude = set(exclude or [])
        preferred = self.gpu
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            free = []
            for line in out.strip().split("\n"):
                idx, mem, util = line.split(", ")
                idx = int(idx)
                mem = int(mem)
                util = int(util)
                # consider GPU free if < 1000 MB used and util < 10%
                if idx not in exclude and mem < 1000 and util < 10:
                    free.append(idx)
            if preferred in free:
                return preferred
            if free:
                return free[0]
            return preferred  # fallback
        except Exception:
            return preferred

    # ---- scoring ------------------------------------------------------------

    def _score_trial(self, trial_dir: Path) -> dict:
        """Read metrics.csv and res.json, return a score dict."""
        result = {
            "best_val_pearson": None,
            "best_val_spearman": None,
            "best_val_rmse": None,
            "final_train_pearson": None,
            "test_pearson": None,
            "test_spearman": None,
            "test_rmse": None,
            "test_mae": None,
            "epochs_completed": None,
            "stopped_epoch": None,
        }

        # Try res.json first (has the primary test results)
        res_path = trial_dir / "res.json"
        if res_path.exists():
            try:
                with open(res_path) as f:
                    res_data = json.load(f)
                if res_data and isinstance(res_data, list) and len(res_data) > 0:
                    r = res_data[0]
                    result["test_pearson"] = r.get("val/all_pearson")
                    result["test_spearman"] = r.get("val/all_spearman")
                    result["test_rmse"] = r.get("val/all_rmse")
                    result["test_mae"] = r.get("val/all_mae")
                    result["best_val_pearson"] = r.get("val/pc_pearson")
                    result["best_val_spearman"] = r.get("val/pc_spearman")
            except Exception as e:
                print(f"  [WARN] Failed to read res.json: {e}")

        # Try val_metrics.json
        val_path = trial_dir / "val_metrics.json"
        if val_path.exists():
            try:
                with open(val_path) as f:
                    val_data = json.load(f)
                result["best_val_pearson"] = val_data.get("val/all_pearson", result["best_val_pearson"])
                result["best_val_spearman"] = val_data.get("val/all_spearman", result["best_val_spearman"])
                result["best_val_rmse"] = val_data.get("val/all_rmse", result["best_val_rmse"])
            except Exception:
                pass

        # Read metrics.csv for detailed training curves
        metrics_dir = trial_dir / "log_fold_0" / "lightning_logs"
        if metrics_dir.exists():
            versions = sorted(metrics_dir.glob("version_*"))
            if versions:
                csv_path = versions[-1] / "metrics.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        # Get last validation epoch metrics
                        val_cols = [c for c in df.columns if "val/all_pearson" in c]
                        if val_cols:
                            val_df = df.dropna(subset=val_cols).copy()
                            if not val_df.empty:
                                val_df["epoch"] = val_df["epoch"].astype(int)
                                ep = val_df.groupby("epoch", as_index=False).last()
                                best_idx = ep[val_cols[0]].idxmax()
                                row = ep.loc[best_idx]
                                result["best_val_pearson"] = float(row[val_cols[0]])
                                if "val/all_spearman" in row and pd.notna(row["val/all_spearman"]):
                                    result["best_val_spearman"] = float(row["val/all_spearman"])
                                if "val/all_rmse" in row and pd.notna(row["val/all_rmse"]):
                                    result["best_val_rmse"] = float(row["val/all_rmse"])

                        # Final training pearson
                        train_cols = [c for c in df.columns if "train/all_pearson" in c]
                        if train_cols:
                            last_train = df[train_cols[0]].dropna()
                            if not last_train.empty:
                                result["final_train_pearson"] = float(last_train.iloc[-1])

                        # Epochs
                        ep_col = df["epoch"].dropna()
                        if not ep_col.empty:
                            result["epochs_completed"] = int(ep_col.max())
                    except Exception as e:
                        print(f"  [WARN] Failed to parse metrics.csv: {e}")

        # Read stopped_epoch from logs if available
        val_only_path = trial_dir / "val_metrics.json"
        if val_only_path.exists():
            try:
                with open(val_only_path) as f:
                    vd = json.load(f)
                    result["stopped_epoch"] = vd.get("stopped_epoch")
            except Exception:
                pass

        return result

    # ---- config generation --------------------------------------------------

    def _generate_trial_config(self, trial_num: int, prev_result: dict):
        """Generate new hyperparameters for the next trial based on previous result."""
        config = deepcopy(self.base_model_config)
        run_cfg = deepcopy(self.base_run_config)
        run_cfg["gpus"] = [self.gpu]
        run_cfg["run_name"] = f"mpd_auto_trial_{trial_num:02d}_"

        prev_best_pearson = prev_result.get("best_val_pearson") if prev_result else None
        prev_train_pearson = prev_result.get("final_train_pearson") if prev_result else None

        changes = []

        # === Strategy: pick one change at a time based on what we observe ===

        # 1. If train pearson is high but test pearson is low → overfitting → increase dropout/weight decay
        if prev_train_pearson is not None and prev_best_pearson is not None:
            gap = prev_train_pearson - prev_best_pearson
            if gap > 0.5 and prev_best_pearson < 0.2:
                # Overfitting detected
                current_dropout = config["model"]["coformer"]["attention_dropout"]
                new_dropout = min(current_dropout + 0.1, 0.5)
                config["model"]["coformer"]["attention_dropout"] = new_dropout
                changes.append(f"attention_dropout {current_dropout} → {new_dropout} (overfit={gap:.2f})")
                config["model"]["coformer"]["residual_dropout"] = new_dropout

                # Also increase weight decay
                current_wd = config["train"]["optimizer"]["weight_decay"]
                new_wd = min(current_wd * 2, 1e-3)
                config["train"]["optimizer"]["weight_decay"] = new_wd
                changes.append(f"weight_decay {current_wd} → {new_wd} (anti-overfit)")

        # 2. If val pearson is very low (< 0.1), try lower LR for stability
        elif prev_best_pearson is not None and prev_best_pearson < 0.1:
            current_lr = config["train"]["optimizer"]["lr"]
            new_lr = current_lr * 0.6
            config["train"]["optimizer"]["lr"] = new_lr
            changes.append(f"lr {current_lr} → {new_lr} (low pearson={prev_best_pearson:.3f})")

        # 3. If val pearson is moderate (0.1-0.2) but training is unstable, adjust LR and PIA
        elif prev_best_pearson is not None and prev_best_pearson < 0.2:
            current_lr = config["train"]["optimizer"]["lr"]
            new_lr = current_lr * 0.8
            config["train"]["optimizer"]["lr"] = new_lr
            changes.append(f"lr {current_lr} → {new_lr} (moderate pearson={prev_best_pearson:.3f})")

            # Try increasing PIA weight
            current_pia = config["train"]["physics_aux_max_weight"]
            if current_pia < 0.03:
                new_pia = min(current_pia * 2, 0.05)
                config["train"]["physics_aux_max_weight"] = new_pia
                changes.append(f"physics_aux_max_weight {current_pia} → {new_pia}")
            else:
                # Lower huber delta for sharper loss
                current_delta = config["train"]["huber_delta"]
                new_delta = max(current_delta * 0.7, 0.5)
                config["train"]["huber_delta"] = new_delta
                changes.append(f"huber_delta {current_delta} → {new_delta}")

        # 4. If val pearson is > 0.2 (promising), fine-tune
        elif prev_best_pearson is not None and prev_best_pearson >= 0.2:
            current_lr = config["train"]["optimizer"]["lr"]
            new_lr = current_lr * 0.5
            config["train"]["optimizer"]["lr"] = new_lr
            changes.append(f"lr {current_lr} → {new_lr} (fine-tune, pearson={prev_best_pearson:.3f})")

            # Larger mutation window for better context
            current_win = config["model"]["mutation_local_window"]
            new_win = min(current_win + 8, 32)
            config["model"]["mutation_local_window"] = new_win
            changes.append(f"mutation_local_window {current_win} → {new_win}")

            # More coformer blocks for capacity
            current_blocks = config["model"]["coformer"]["num_blocks"]
            new_blocks = min(current_blocks + 2, 10)
            config["model"]["coformer"]["num_blocks"] = new_blocks
            changes.append(f"coformer_blocks {current_blocks} → {new_blocks}")

        # 5. Fallback: if no previous result (first trial), try varied LR
        else:
            # Already at baseline, just change LR slightly
            # Try a different LR from the space
            tried_lrs = [t.get("lr") for t in self.trials if t.get("lr") is not None]
            for candidate in HPARAM_SPACE["lr"]:
                if candidate not in tried_lrs:
                    config["train"]["optimizer"]["lr"] = candidate
                    changes.append(f"lr 5e-5 → {candidate} (exploration)")
                    break
            else:
                # All LRs tried, change something else
                config["train"]["optimizer"]["weight_decay"] = 1e-3
                changes.append("weight_decay 1e-4 → 1e-3 (exploration)")

        if not changes:
            # Try changing batch_size
            current_batch = run_cfg.get("batch_size", 1)
            # This is in dataset config, not model config
            batch_sizes = [1, 2]
            for b in batch_sizes:
                if b != current_batch:
                    run_cfg["batch_size"] = b
                    changes.append(f"batch_size {current_batch} → {b}")
                    break

        if not changes:
            config["train"]["optimizer"]["lr"] = 3e-5
            changes.append("lr 5e-5 → 3e-5 (default fallback)")

        # Build unique run_name
        tag = "_".join(changes[0].split()[:3]).replace("→", "to").replace(".", "p")
        if "-" not in tag:
            tag = changes[0].split()[0] + "_" + tag if changes else "adj"
        run_cfg["run_name"] = f"mpd_auto_t{trial_num:02d}_{tag}_"

        return config, run_cfg, changes

    # ---- monitoring ---------------------------------------------------------

    def _monitor_training(self, trial_dir: Path, process, poll_interval: int = 60):
        """Monitor training process in real-time until it finishes."""
        metrics_path = trial_dir / "log_fold_0" / "lightning_logs"
        last_step = -1
        stall_warnings = 0
        max_stall = 10  # warn after 10 polls with no progress (10 min)

        # Wait a bit for directory to be created
        for wait in range(10):
            if process.poll() is not None:
                break
            if trial_dir.exists():
                break
            time.sleep(3)

        while True:
            ret = process.poll()
            if ret is not None:
                print(f"\n  Process exited with code {ret}")
                break

            # Check latest metrics
            latest_csv = None
            if metrics_path.exists():
                versions = sorted(metrics_path.glob("version_*"))
                if versions:
                    candidate = versions[-1] / "metrics.csv"
                    if candidate.exists():
                        latest_csv = candidate

            if latest_csv:
                try:
                    df = pd.read_csv(latest_csv)
                    current_step = int(df["step"].max()) if "step" in df.columns else 0
                    train_pearson = None
                    val_pearson = None
                    if "train/all_pearson" in df.columns:
                        tp = df["train/all_pearson"].dropna()
                        if not tp.empty:
                            train_pearson = tp.iloc[-1]
                    if "val/all_pearson" in df.columns:
                        vp = df["val/all_pearson"].dropna()
                        if not vp.empty:
                            val_pearson = vp.iloc[-1]

                    if current_step > last_step:
                        last_step = current_step
                        stall_warnings = 0
                        epoch = int(df["epoch"].max()) if "epoch" in df.columns else "?"
                        line = f"  Step {current_step:>6d} | Epoch {str(epoch):>3s}"
                        if train_pearson is not None:
                            line += f" | Train Pearson: {train_pearson:.4f}"
                        if val_pearson is not None:
                            line += f" | Val Pearson: {val_pearson:.4f}"
                        print(line)
                    else:
                        stall_warnings += 1
                        if stall_warnings >= max_stall:
                            print(f"  [WARN] No progress for {max_stall * poll_interval // 60} min")
                            stall_warnings = 0
                except Exception as e:
                    pass
            else:
                print(f"  [WAIT] Metrics directory not ready yet ({poll_interval}s poll)...")

            time.sleep(poll_interval)

        # Final check - read the last metrics
        print("  Training finished. Collecting final metrics...")
        return process.returncode

    # ---- main loop ----------------------------------------------------------

    def _save_trial_configs(self, trial_dir: Path, trial_num: int):
        """Save the config files used for this trial."""
        config_dir = trial_dir / "auto_configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        self._dump_yaml(self.model_config, config_dir / "model_config.yml")
        self._dump_yaml(self.data_config, config_dir / "data_config.yml")
        self._dump_yaml(self.run_config, config_dir / "run_config.yml")

    def run(self):
        """Run the automated training loop."""
        print("=" * 70)
        print("  Auto Train Loop — MPD PEMPNI Optimization")
        print(f"  Max trials: {self.max_trials}")
        print(f"  Early stop after {self.early_stop_trials} trials without improvement")
        print(f"  Min improvement threshold: {self.min_improvement}")
        print(f"  GPU: {self.gpu}")
        print(f"  Python: {self.python}")
        print("=" * 70)

        self.trials_dir.mkdir(parents=True, exist_ok=True)

        for trial in range(1, self.max_trials + 1):
            print(f"\n{'─' * 70}")
            print(f"  Trial {trial}/{self.max_trials}")
            print(f"{'─' * 70}")

            # Create trial directory
            trial_dir = self.trials_dir / f"trial_{trial:03d}"
            trial_dir.mkdir(parents=True, exist_ok=True)

            # Generate config for this trial
            prev_result = self.trials[-1]["result"] if self.trials else None
            # Only adjust if we have a previous result (not first trial)
            if prev_result:
                model_cfg, run_cfg, changes = self._generate_trial_config(trial, prev_result)
                self.model_config = model_cfg
                self.run_config = run_cfg
                print(f"  Changes: {'; '.join(changes)}")
            else:
                print("  Baseline trial (no changes)")

            # Print current hyperparameters
            lr = self.model_config["train"]["optimizer"]["lr"]
            wd = self.model_config["train"]["optimizer"]["weight_decay"]
            pia = self.model_config["train"]["physics_aux_max_weight"]
            huber = self.model_config["train"]["huber_delta"]
            win = self.model_config["model"]["mutation_local_window"]
            drop = self.model_config["model"]["coformer"]["attention_dropout"]
            blocks = self.model_config["model"]["coformer"]["num_blocks"]
            embed = self.model_config["model"]["coformer"]["embed_dim"]
            print(f"  Hyperparameters: lr={lr}, wd={wd}, pia={pia}, huber={huber}")
            print(f"                   window={win}, dropout={drop}, blocks={blocks}, embed={embed}")

            # Save configs for this trial
            self._save_trial_configs(trial_dir, trial)

            # Build command
            cmd = [
                self.python, "run.py", "finetune", "ddG",
                "--model_config", str(self._dump_yaml(self.model_config, trial_dir / "auto_configs" / "model_config.yml")),
                "--data_config", str(self._dump_yaml(self.data_config, trial_dir / "auto_configs" / "data_config.yml")),
                "--run_config", str(self._dump_yaml(self.run_config, trial_dir / "auto_configs" / "run_config.yml")),
            ]
            cmd_str = " ".join(str(c) for c in cmd)
            print(f"  Command:\n    {cmd_str}")

            # Launch training
            print(f"  Launching training (GPU {self.gpu})...")
            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                preexec_fn=lambda: os.setsid() if os.name != "nt" else None,
            )

            # Monitor in real-time
            try:
                # Read stdout continuously
                for line in iter(process.stdout.readline, ""):
                    print(f"    {line}", end="")
                    # Check for common training milestones
                    if "Training fold 0 Finished!" in line:
                        print("  [TRIAL COMPLETE] Fold training done.")
                    elif "Exception:" in line or "Error:" in line:
                        print(f"  [ERROR] {line.strip()}")
                    elif "CUDA out of memory" in line:
                        print("  [CRITICAL] CUDA OOM — aborting trial")
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                        break
            except (BrokenPipeError, EOFError):
                pass
            finally:
                process.stdout.close()
                process.wait()

            print(f"  Trial process exited with code {process.returncode}")

            if process.returncode != 0:
                print(f"  [WARN] Trial failed with code {process.returncode}")
                trial_record = {
                    "trial": trial,
                    "status": "failed",
                    "returncode": process.returncode,
                    "result": {
                        "best_val_pearson": None,
                        "test_pearson": None,
                    },
                }
                self.trials.append(trial_record)
                # Try to read what we can
                result = self._score_trial(trial_dir)
                if result["best_val_pearson"] is not None:
                    trial_record["result"] = result
                    trial_record["status"] = "partial"
                    print(f"  Partial results: {result}")
                continue

            # Score this trial
            result = self._score_trial(trial_dir)
            score = result.get("best_val_pearson") or result.get("test_pearson") or 0.0
            print(f"\n  Trial {trial} Results:")
            print(f"    Best val Pearson:   {result.get('best_val_pearson', 'N/A')}")
            print(f"    Best val Spearman:  {result.get('best_val_spearman', 'N/A')}")
            print(f"    Best val RMSE:      {result.get('best_val_rmse', 'N/A')}")
            print(f"    Final train Pearson:{result.get('final_train_pearson', 'N/A')}")
            print(f"    Test Pearson:       {result.get('test_pearson', 'N/A')}")
            print(f"    Test Spearman:      {result.get('test_spearman', 'N/A')}")
            print(f"    Test RMSE:          {result.get('test_rmse', 'N/A')}")
            print(f"    Test MAE:           {result.get('test_mae', 'N/A')}")
            print(f"    Epochs completed:   {result.get('epochs_completed', 'N/A')}")

            # Save trial record
            trial_record = {
                "trial": trial,
                "status": "completed",
                "score": score,
                "hyperparameters": {
                    "lr": self.model_config["train"]["optimizer"]["lr"],
                    "weight_decay": self.model_config["train"]["optimizer"]["weight_decay"],
                    "physics_aux_max_weight": self.model_config["train"]["physics_aux_max_weight"],
                    "huber_delta": self.model_config["train"]["huber_delta"],
                    "mutation_local_window": self.model_config["model"]["mutation_local_window"],
                    "attention_dropout": self.model_config["model"]["coformer"]["attention_dropout"],
                    "coformer_blocks": self.model_config["model"]["coformer"]["num_blocks"],
                    "coformer_embed_dim": self.model_config["model"]["coformer"]["embed_dim"],
                },
                "result": result,
            }
            self.trials.append(trial_record)

            # Check for improvement
            if score > self.best_score + self.min_improvement:
                improvement = score - self.best_score if self.best_score != float("-inf") else score
                print(f"\n  ✓ NEW BEST! {self.best_score:.4f} → {score:.4f} (+{improvement:.4f})")
                self.best_score = score
                self.best_trial = trial
                self.best_config = {
                    "model_config": deepcopy(self.model_config),
                    "run_config": deepcopy(self.run_config),
                    "trial_dir": str(trial_dir),
                }
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                print(f"\n  No improvement (best={self.best_score:.4f}, count={self.no_improve_count})")

            # Save checkpoint
            self._save_checkpoint()

            # Early stopping
            if self.no_improve_count >= self.early_stop_trials and self.best_score > -0.5:
                print(f"\n  Early stopping: {self.early_stop_trials} trials without improvement.")
                break

            print(f"\n  Proceeding to trial {trial + 1}...")
            # Small delay between trials
            time.sleep(10)

        # Final summary
        self._print_summary()

    def _save_checkpoint(self):
        """Save current state to resume later."""
        ckpt = {
            "trials": self.trials,
            "best_score": self.best_score,
            "best_trial": self.best_trial,
            "best_config": self.best_config,
            "no_improve_count": self.no_improve_count,
        }
        with open(self.trials_dir / "checkpoint.json", "w") as f:
            json.dump(ckpt, f, indent=2, default=str)
        print(f"  Checkpoint saved: {self.trials_dir / 'checkpoint.json'}")

    def _print_summary(self):
        """Print final summary of all trials."""
        print("\n" + "=" * 70)
        print("  FINAL SUMMARY")
        print("=" * 70)

        completed = [t for t in self.trials if t["status"] == "completed"]
        if not completed:
            print("  No completed trials.")
            return

        df = pd.DataFrame(completed)
        summary_cols = ["trial", "score"]
        for col in ["best_val_pearson", "best_val_spearman", "best_val_rmse",
                     "test_pearson", "test_spearman", "test_rmse", "test_mae",
                     "final_train_pearson", "epochs_completed"]:
            vals = [t["result"].get(col) for t in completed]
            if any(v is not None for v in vals):
                df[col] = vals

        print("\n  Trial Results:")
        print(f"  {'Trial':>6s} | {'Score':>8s} | {'Val P':>7s} | {'Val S':>7s} | {'Val RMSE':>9s} | {'Test P':>7s} | {'Train P':>8s} | {'Epochs':>7s}")
        print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}-+-{'-'*7}-+-{'-'*8}-+-{'-'*7}")

        for t in completed:
            r = t["result"]
            print(f"  {t['trial']:>6d} | {t.get('score', 0):>8.4f} | "
                  f"{r.get('best_val_pearson', -1) or -1:>7.4f} | "
                  f"{r.get('best_val_spearman', -1) or -1:>7.4f} | "
                  f"{r.get('best_val_rmse', -1) or -1:>9.4f} | "
                  f"{r.get('test_pearson', -1) or -1:>7.4f} | "
                  f"{r.get('final_train_pearson', -1) or -1:>8.4f} | "
                  f"{r.get('epochs_completed', -1) or -1:>7d}")

        print(f"\n  Best Trial: #{self.best_trial} (val Pearson = {self.best_score:.4f})")
        if self.best_config:
            print(f"  Best config dir: {self.best_config.get('trial_dir', 'N/A')}")
            print(f"  Best hyperparameters:")
            hp = self.best_config.get("model_config", {}).get("train", {})
            print(f"    lr={hp.get('optimizer', {}).get('lr', '?')}, "
                  f"wd={hp.get('optimizer', {}).get('weight_decay', '?')}, "
                  f"pia={hp.get('physics_aux_max_weight', '?')}")
            model_hp = self.best_config.get("model_config", {}).get("model", {})
            print(f"    window={model_hp.get('mutation_local_window', '?')}, "
                  f"blocks={model_hp.get('coformer', {}).get('num_blocks', '?')}")

        # Save best configs
        if self.best_config:
            best_dir = self.trials_dir / "best_config"
            best_dir.mkdir(parents=True, exist_ok=True)
            if "model_config" in self.best_config:
                self._dump_yaml(self.best_config["model_config"], best_dir / "model_config.yml")
                self._dump_yaml(self.best_config["run_config"], best_dir / "run_config.yml")

        print("\n  All results saved to:", (self.trials_dir / "checkpoint.json"))
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Auto Train Loop for MPD PEMPNI hyperparameter optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python auto_train_loop.py --gpu 1 --max_trials 10 --early_stop_trials 3
  python auto_train_loop.py --max_trials 5 --trials_dir outputs/my_sweep
  python auto_train_loop.py --resume outputs/auto_train_trials/checkpoint.json
        """,
    )
    parser.add_argument("--model_config", default="config/models/mpd_pempni_final_v2.yml",
                        help="Base model config YAML")
    parser.add_argument("--data_config", default="config/datasets/MPD_pempni.yml",
                        help="Base dataset config YAML")
    parser.add_argument("--run_config", default="config/runs/finetune_mpd_pempni_final_gpu0.yml",
                        help="Base run config YAML")
    parser.add_argument("--gpu", type=int, default=1,
                        help="Preferred GPU index (auto-fallback if busy)")
    parser.add_argument("--max_trials", type=int, default=10,
                        help="Maximum number of training trials")
    parser.add_argument("--early_stop_trials", type=int, default=3,
                        help="Stop after N trials without improvement")
    parser.add_argument("--min_improvement", type=float, default=0.01,
                        help="Minimum score improvement to count as progress")
    parser.add_argument("--trials_dir", default="outputs/auto_train_trials",
                        help="Directory to store trial results")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint.json to resume")

    args = parser.parse_args()

    loop = AutoTrainLoop(
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        run_config_path=args.run_config,
        gpu=args.gpu,
        max_trials=args.max_trials,
        early_stop_trials=args.early_stop_trials,
        min_improvement=args.min_improvement,
        trials_dir=args.trials_dir,
    )

    # Resume support (simple: just adjust starting trial count if needed)
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            import json
            with open(ckpt_path) as f:
                ckpt = json.load(f)
            loop.trials = ckpt.get("trials", [])
            loop.best_score = ckpt.get("best_score", float("-inf"))
            loop.best_trial = ckpt.get("best_trial")
            loop.best_config = ckpt.get("best_config")
            loop.no_improve_count = ckpt.get("no_improve_count", 0)
            print(f"Resumed from checkpoint: {ckpt_path}")
            print(f"  Existing trials: {len(loop.trials)}, best: {loop.best_score:.4f}")

    loop.run()


if __name__ == "__main__":
    main()
