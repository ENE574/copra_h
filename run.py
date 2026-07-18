import json
import os
os.environ["NUMEXPR_MAX_THREADS"] = '56'
os.environ["MKL_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
import fire
import pathlib
from pathlib import Path
import pandas as pd

import numpy as np
import yaml
import wandb
import time
from easydict import EasyDict
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, ModelSummary
from pytorch_lightning.strategies.ddp import DDPStrategy
from pl_modules import ModelModule, DataModule, DDGModule
from pl_modules.multitask_data_module import MultiTaskDataModule
from utils.task_profile import apply_task_profile
from collections import defaultdict

torch.set_num_threads(16)


class ConstrainedCheckpoint(pl.Callback):
    """Save the checkpoint with the best pc metric among epochs where all > all_min."""

    def __init__(
        self,
        dirpath,
        all_monitor="val/all_pearson",
        pc_monitor="val/pc_pearson",
        all_min=0.0,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.all_monitor = all_monitor
        self.pc_monitor = pc_monitor
        self.all_min = float(all_min)
        self.best_pc = float("-inf")
        self.best_all = float("-inf")
        self.best_epoch = -1
        self.best_model_path = ""
        self._ckpt_path = self.dirpath / "best-constrained.ckpt"

    @staticmethod
    def _metric_tag(monitor):
        return monitor.replace("/", "_")

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if self.all_monitor not in metrics or self.pc_monitor not in metrics:
            return
        all_v = float(metrics[self.all_monitor])
        pc_v = float(metrics[self.pc_monitor])
        if all_v <= self.all_min or pc_v <= self.best_pc:
            return
        epoch = int(trainer.current_epoch)
        trainer.save_checkpoint(str(self._ckpt_path))
        self.best_pc = pc_v
        self.best_all = all_v
        self.best_epoch = epoch
        self.best_model_path = str(self._ckpt_path)
        if trainer.is_global_zero:
            print(
                "Constrained checkpoint updated:",
                f"epoch={epoch}",
                f"{self._metric_tag(self.all_monitor)}={all_v:.4f}",
                f"{self._metric_tag(self.pc_monitor)}={pc_v:.4f}",
            )


def parse_yaml(yaml_dir):
    with open(yaml_dir, 'r') as f:
        content = f.read()
        config_dict = EasyDict(yaml.load(content, Loader=yaml.FullLoader))
        # args = Namespace(**config_dict)
    return config_dict
def init_pytorch_settings():
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([pathlib.PosixPath, EasyDict])
    # Multiprocess Setting to speedup dataloader
    torch.multiprocessing.set_start_method('forkserver')
    torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.set_float32_matmul_precision('high')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class LightningRunner(object):
    def __init__(self, model_config='./config/models/esm2_rinalmo.yaml', data_config=None,
                 run_config='./config/runs/finetune_sequence.yaml'):
        super(LightningRunner, self).__init__()
        self.model_args = apply_task_profile(parse_yaml(model_config))
        self.run_args = parse_yaml(run_config)
        
        if getattr(self.run_args, "multitask_sources", None):
            self.dataset_args = None
            print("Multitask mode: dataset configs will be loaded from run_config")
        elif data_config:
            self.dataset_args = parse_yaml(data_config)
        else:
            raise ValueError("data_config is required unless using multitask mode")
            
        init_pytorch_settings()
        if getattr(self.model_args, "task_profile", None):
            print("Applied task profile:", self.model_args.task_profile)

    def _checkpoint_paths(self, ckpt_callbacks):
        paths = {}
        for cb in ckpt_callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                paths[cb.monitor] = cb.best_model_path
        return paths

    def _make_data_module(self, col_group: str):
        dm_kwargs = dict(self.dataset_args)
        dm_kwargs["col_group"] = col_group
        return DataModule(dataset_args=self.dataset_args, **dm_kwargs)

    @staticmethod
    def _remap_val_monitor(monitor):
        if not monitor:
            return monitor
        if monitor.startswith("val/"):
            return "train/" + monitor.split("/", 1)[1]
        if monitor == "val_loss":
            return "train_loss"
        return monitor

    def _split_limit_val_batches(self, col_group: str):
        if self.dataset_args is None:
            return 1.0
        split_df = pd.read_csv(self.dataset_args.df_path)
        return 0.0 if (split_df[col_group] == "val").sum() == 0 else 1.0

    @staticmethod
    def _resolve_trainer_strategy(gpus):
        """Use DDP only for multi-GPU; single-GPU runs avoid NCCL overhead/failures."""
        if gpus is None:
            return "auto"
        n_devices = len(gpus) if isinstance(gpus, (list, tuple)) else 1
        if n_devices <= 1:
            return "auto"
        return DDPStrategy(find_unused_parameters=True)

    def _patch_scheduler_for_no_val(self, has_val: bool = True):
        sched = getattr(self.model_args.train, "scheduler", None)
        if sched is None:
            return None
        if has_val:
            return None
        old_type = getattr(sched, "type", None)
        if old_type == "plateau":
            sched.type = "multistep"
            sched.milestones = [50, 100]
            sched.gamma = 0.5
            print(f"No val split: scheduler type {old_type} -> multistep (milestones=[50,100], gamma=0.5)")
            return old_type
        return None

    def _make_multitask_data_module(self, col_group: str):
        sources = getattr(self.run_args, "multitask_sources", None)
        if not sources:
            raise ValueError("multitask_sources missing in run config")
        return MultiTaskDataModule(
            multitask_sources=list(sources),
            col_group=col_group,
            primary_task=getattr(self.run_args, "multitask_primary_task", None),
        )

    def _resolve_finetune_test_ckpt(self, trainer, ckpt_callbacks, constrained_callback=None):
        strategy = getattr(self.run_args, "checkpoint_select", "primary")
        if strategy == "constrained" and constrained_callback is not None:
            if constrained_callback.best_model_path:
                print(
                    "Using constrained checkpoint:",
                    f"epoch={constrained_callback.best_epoch}",
                    f"all={constrained_callback.best_all:.4f}",
                    f"pc={constrained_callback.best_pc:.4f}",
                )
                return constrained_callback.best_model_path
            print(
                "No constrained checkpoint (no epoch with "
                f"{constrained_callback.all_monitor} > {constrained_callback.all_min}); "
                "fallback to primary best."
            )
        paths = self._checkpoint_paths(ckpt_callbacks)
        ckpt_monitor = getattr(self.run_args, "checkpoint_monitor", None)
        if ckpt_monitor in paths:
            return paths[ckpt_monitor]
        return "best"

    def _run_fold_tests(self, trainer, model, data_module, ckpt_callbacks):
        ckpt_monitor = getattr(self.run_args, "checkpoint_monitor", "val/pc_pearson")
        ckpt_monitor_all = getattr(self.run_args, "checkpoint_monitor_all", None)
        paths = self._checkpoint_paths(ckpt_callbacks)
        test_both = bool(getattr(self.run_args, "test_both_checkpoints", False))

        primary_path = paths.get(ckpt_monitor)
        if primary_path is None:
            primary_path = "best"
        print(f"Primary test ({ckpt_monitor}): {primary_path}")
        model.test_results_csv = "results_test.csv"
        _ = trainer.test(model=model, ckpt_path=primary_path, datamodule=data_module)
        primary_res = dict(model.res)
        primary_res["checkpoint"] = primary_path if primary_path != "best" else paths.get(ckpt_monitor, "best")
        primary_res["checkpoint_monitor"] = ckpt_monitor

        fold_result = {"fold_primary": primary_res}
        if test_both and ckpt_monitor_all and ckpt_monitor_all in paths:
            secondary_path = paths[ckpt_monitor_all]
            if secondary_path != primary_path:
                print(f"Secondary test ({ckpt_monitor_all}): {secondary_path}")
                model.test_results_csv = "results_test_all.csv"
                _ = trainer.test(model=model, ckpt_path=secondary_path, datamodule=data_module)
                secondary_res = dict(model.res)
                secondary_res["checkpoint"] = secondary_path
                secondary_res["checkpoint_monitor"] = ckpt_monitor_all
                fold_result["fold_secondary"] = secondary_res
                model.test_results_csv = "results_test.csv"
        return fold_result, primary_path

    @staticmethod
    def _read_val_metrics_from_csv(log_dir):
        """Best epoch-end val metrics from Lightning CSVLogger."""
        csv_paths = sorted(Path(log_dir).glob("lightning_logs/version_*/metrics.csv"))
        if not csv_paths:
            return None
        df = pd.read_csv(csv_paths[-1])
        if "val/all_pearson" not in df.columns:
            return None
        val_df = df.dropna(subset=["val/all_pearson"]).copy()
        if val_df.empty:
            return None
        val_df["epoch"] = val_df["epoch"].astype(int)
        ep = val_df.groupby("epoch", as_index=False).last()
        best_idx = ep["val/all_pearson"].idxmax()
        row = ep.loc[best_idx]
        out = {
            "best_val_epoch": int(row["epoch"]),
            "val/all_pearson": float(row["val/all_pearson"]),
        }
        for col in (
            "val/all_spearman",
            "val/all_rmse",
            "val/all_mae",
            "val/pc_pearson",
            "val/pc_spearman",
            "val/pc_rmse",
            "val/pc_mae",
            "train/all_pearson",
        ):
            if col in row and pd.notna(row[col]):
                out[col] = float(row[col])
        return out

    def _save_val_only_results(self, log_dir, output_dir, ckpt_callbacks, stopped_epoch=None):
        val_metrics = self._read_val_metrics_from_csv(log_dir)
        paths = self._checkpoint_paths(ckpt_callbacks)
        ckpt_monitor = getattr(self.run_args, "checkpoint_monitor", "val/all_pearson")
        ckpt_path = paths.get(ckpt_monitor, "")
        payload = {
            "selection_metric": ckpt_monitor,
            "checkpoint": ckpt_path,
            **(val_metrics or {}),
        }
        if stopped_epoch is not None:
            payload["stopped_epoch"] = int(stopped_epoch)
        out_path = Path(output_dir) / "val_metrics.json"
        out_path.write_text(json.dumps(payload, indent=2))
        print("Val-only selection metrics:", json.dumps(payload, indent=2))
        return payload

    def save_model(self, model, output_dir, trainer, ckpt_path="best"):
        if ckpt_path == "best":
            ckpt_path = trainer.checkpoint_callback.best_model_path
        print("Best Model Path:", ckpt_path)
        
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        if trainer.global_rank == 0:
            (output_dir / 'model_data.json').write_text(json.dumps(vars(self.dataset_args), indent=2))
            torch.save({
                "model_state_dict": checkpoint['state_dict'],
                "dataset_args": vars(self.dataset_args),
            }, str(output_dir / "model.pt"))
    
    def select_module(self, stage, log_dir):
        if stage=='dG':
            model = ModelModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args, run_args=self.run_args)
        elif stage=='ddG':
            model = DDGModule(output_dir=log_dir, model_args=self.model_args, data_args=self.dataset_args, run_args=self.run_args)
        else:
            raise NotImplementedError
        return model

    def finetune(self, stage='dG'):
        print("Run args:", self.run_args, "\n")
        print("Model args:", self.model_args, "\n")
        print("Dataset args:", self.dataset_args, "\n")
        output_base_dir, gpus = (self.run_args.output_dir, self.run_args.gpus)
        self.model_args.model.stage = stage
        # Setup datamodule
        run_results = []
        run_results_dual = []
        run_id = self.run_args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path(output_base_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        fold_indices = getattr(self.run_args, "fold_indices", None)
        if fold_indices is not None:
            fold_indices = [int(i) for i in fold_indices]
        default_col_group = getattr(self.dataset_args, "col_group", None)
        for k in range(self.run_args.num_folds):
            if fold_indices is not None and k not in fold_indices:
                continue
            col_group = default_col_group if default_col_group is not None else f"fold_{k}"
            print(f"Training fold {k} Started! (split column: {col_group})")
            log_dir = output_dir / f'log_fold_{k}'
            if getattr(self.run_args, "multitask_sources", None):
                data_module = self._make_multitask_data_module(col_group)
            else:
                data_module = self._make_data_module(col_group)
            limit_val_batches = self._split_limit_val_batches(col_group)
            has_val = limit_val_batches > 0
            orig_sched_monitor = self._patch_scheduler_for_no_val(has_val=has_val)

            # Setup model module
            model = self.select_module(stage, log_dir)

            # Two-stage calibration (B28): load a pretrained ddG checkpoint's
            # backbone + pred_head, keep the identity-initialized output
            # calibration layer, freeze everything else, and train ONLY the
            # calibration (scale+bias) with a fresh optimizer. This forces the
            # affine to undo B25's systematic under-scaling (pred = 0.559*true + 0.319),
            # driving val RMSE toward the recalibration floor (~0.55) with pearson intact.
            if getattr(self.model_args.train, "freeze_except_calib", False):
                import torch as _torch
                _ckpt_path = getattr(self.run_args, "ckpt", None)
                if _ckpt_path is not None:
                    print("[freeze_except_calib] loading backbone+pred_head from %s" % _ckpt_path)
                    _ckpt = _torch.load(_ckpt_path, map_location="cpu", weights_only=False)
                    _sd = _ckpt.get("state_dict", _ckpt)
                    _sd = {k.replace("model.", "", 1): v for k, v in _sd.items()}
                    # drop any old calibration params so they stay identity-initialized
                    _sd = {k: v for k, v in _sd.items() if not k.startswith("ddg_calib")}
                    _miss = model.model.load_state_dict(_sd, strict=False)
                    print("  loaded: %d missing (ddg_calib expected), %d unexpected"
                          % (len(_miss.missing_keys), len(_miss.unexpected_keys)))
                    # prevent the normal finetune resume path from re-loading / restoring optimizer
                    self.run_args.ckpt = None
                    if getattr(self.run_args, "ckpts", None) is not None:
                        self.run_args.ckpts = None
                frozen = trainable = 0
                for n, p in model.named_parameters():
                    if "ddg_calib" in n:
                        p.requires_grad_(True); trainable += 1
                    else:
                        p.requires_grad_(False); frozen += 1
                print("[freeze_except_calib] frozen=%d trainable=%d (only ddg_calib_* will update)"
                      % (frozen, trainable))

            # Trainer setting
            if self.run_args.wandb:
                wandb.init(project='copra', name=run_id)
                logger = WandbLogger()
            else:
                logger = CSVLogger(str(log_dir))
            # version_dir = Path(logger_csv.log_dir)
            pl.seed_everything(self.model_args.train.seed)
            print("Successfully initialized, start trainer...")
            strategy = self._resolve_trainer_strategy(gpus)
            es_monitor = getattr(self.run_args, "early_stop_monitor", "val_loss")
            es_mode = getattr(self.run_args, "early_stop_mode", None)
            if es_mode is None:
                es_mode = "min" if es_monitor in ("val_loss", "val/all_rmse", "val/all_mae") else "max"
            ckpt_monitor = getattr(self.run_args, "checkpoint_monitor", es_monitor)
            ckpt_mode = getattr(self.run_args, "checkpoint_mode", es_mode)
            ckpt_callbacks = [
                ModelCheckpoint(
                    dirpath=(log_dir / 'checkpoint'),
                    filename='best-{epoch}-{' + ckpt_monitor + ':.3f}',
                    monitor=ckpt_monitor,
                    mode=ckpt_mode,
                    save_last=True,
                    save_top_k=1,
                ),
            ]
            ckpt_monitor_all = getattr(self.run_args, "checkpoint_monitor_all", None)
            if ckpt_monitor_all is not None:
                ckpt_mode_all = getattr(self.run_args, "checkpoint_mode_all", "max")
                ckpt_callbacks.append(
                    ModelCheckpoint(
                        dirpath=(log_dir / 'checkpoint'),
                        filename='best-all-{epoch}-{' + ckpt_monitor_all + ':.3f}',
                        monitor=ckpt_monitor_all,
                        mode=ckpt_mode_all,
                        save_top_k=1,
                    )
                )
            constrained_callback = None
            if getattr(self.run_args, "checkpoint_select", "primary") == "constrained":
                constrained_callback = ConstrainedCheckpoint(
                    dirpath=(log_dir / "checkpoint"),
                    all_monitor=ckpt_monitor,
                    pc_monitor=ckpt_monitor_all or "val/pc_pearson",
                    all_min=float(getattr(self.run_args, "checkpoint_select_all_min", 0.0)),
                )
            trainer_callbacks = [
                EarlyStopping(
                    monitor=es_monitor,
                    mode=es_mode,
                    patience=self.run_args.patience,
                    strict=False,
                ),
                *ckpt_callbacks,
            ]
            if constrained_callback is not None:
                trainer_callbacks.append(constrained_callback)
            trainer = pl.Trainer(
                devices=gpus,
                # max_steps=self.run_args.iters,
                max_epochs=self.run_args.epochs,
                accumulate_grad_batches=int(getattr(self.run_args, "accumulate_grad_batches", 1)),
                logger=logger,
                callbacks=trainer_callbacks,
                gradient_clip_val=self.model_args.train.max_grad_norm if self.model_args.train.max_grad_norm is not None else None,
                gradient_clip_algorithm='norm' if self.model_args.train.max_grad_norm is not None else None,
                strategy=strategy,
                log_every_n_steps=3,
                limit_val_batches=limit_val_batches,
            )
            fit_ckpt = None
            if getattr(self.run_args, "ckpt", None) is not None:
                fit_ckpt = self.run_args.ckpt
            else:
                ckpts = getattr(self.run_args, "ckpts", None)
                if ckpts is not None and k < len(ckpts):
                    fit_ckpt = ckpts[k]
            # Transfer learning for ddG: route ckpt through model_args.resume so
            # DDGModule loads backbone weights with strict=False, allowing
            # DDG-specific layers (GNN, mutation_local, local_fuse, etc.) to be
            # randomly initialized on top of the pretrained backbone.
            if stage == 'ddG' and fit_ckpt is not None:
                print("Transfer learning: routing ckpt through model_args.resume (strict=False)")
                self.model_args.resume = fit_ckpt
                trainer.fit(model=model, datamodule=data_module)
            else:
                trainer.fit(model=model, datamodule=data_module, ckpt_path=fit_ckpt)
            print(f"Training fold {k} Finished!")
            trainer.strategy.barrier()
            self._last_trainer = trainer
            skip_test = bool(getattr(self.run_args, "skip_test", False))
            if skip_test:
                print("skip_test=True: recording val metrics only (no MPD48 test).")
                fold_result = self._save_val_only_results(
                    log_dir, output_dir, ckpt_callbacks, stopped_epoch=trainer.current_epoch
                )
                run_results_dual.append({"fold": k, "val_only": fold_result})
                run_results.append(fold_result)
            else:
                print("Best Validation Results:")
                fold_result, test_ckpt = self._run_fold_tests(
                    trainer, model, data_module, ckpt_callbacks
                )
                run_results_dual.append({"fold": k, **fold_result})
                run_results.append(fold_result["fold_primary"])
                if trainer.global_rank == 0:
                    self.save_model(model, output_dir, trainer, ckpt_path=test_ckpt)
            if orig_sched_monitor is not None:
                self.model_args.train.scheduler.monitor = orig_sched_monitor
        with open(output_dir / 'res.json', 'w') as f:
            json.dump(run_results, f)
        with open(output_dir / 'res_dual.json', 'w') as f:
            json.dump(run_results_dual, f, indent=2)
        results_df = pd.DataFrame(run_results)
        print(results_df.describe())
        if len(results_df) > 0:
            print("Fold-average metrics (mean across folds):")
            print(results_df.mean(numeric_only=True))

    def _resolve_test_ckpts(self):
        ckpts = getattr(self.run_args, "ckpts", None)
        ckpt = getattr(self.run_args, "ckpt", None)
        if ckpts is None and ckpt is not None:
            ckpts = [ckpt]
        if ckpts is None:
            raise ValueError(
                "test() requires a checkpoint. Set run_config ckpt (one path) or ckpts (list, one per fold)."
            )
        if isinstance(ckpts, str):
            ckpts = [ckpts]
        ckpts = list(ckpts)
        num_folds = int(self.run_args.num_folds)
        if len(ckpts) < num_folds:
            raise ValueError(
                f"Need {num_folds} checkpoint path(s) (ckpts), but got {len(ckpts)}."
            )
        return ckpts

    def test(self, stage='dG'):
        print("Args:", self.run_args, self.dataset_args, self.model_args)
        output_base_dir, gpus = (
            self.run_args.output_dir,
            self.run_args.gpus,
        )
        ckpts = self._resolve_test_ckpts()
        # create a timestamped run directory under output_dir, same as finetune
        run_id = self.run_args.run_name + time.strftime("%Y-%m-%d-%H-%M-%S")
        output_dir = Path(output_base_dir) / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        run_results = []
        default_col_group = getattr(self.dataset_args, "col_group", None)
        for k in range(self.run_args.num_folds):
            col_group = default_col_group if default_col_group is not None else f"fold_{k}"
            log_dir = output_dir / f'log_fold_{k}'
            data_module = self._make_data_module(col_group)
            # data_module.setup()
            model = self.select_module(stage, log_dir)
            logger = CSVLogger(str(log_dir))
            strategy = self._resolve_trainer_strategy(gpus)
            trainer = pl.Trainer(
                devices=gpus,
                max_epochs=0,
                logger=[
                    logger,
                ],
                callbacks=[
                    TQDMProgressBar(refresh_rate=1),
                ],
                strategy=strategy,
            )

            _ = trainer.test(model=model, ckpt_path=ckpts[k], datamodule=data_module)
            res = model.res
            run_results.append(res)
        if trainer.global_rank == 0:
            results_df = pd.DataFrame(run_results)
            print(results_df.describe())
            if len(results_df) > 0:
                print("Fold-average metrics (mean across folds):")
                print(results_df.mean(numeric_only=True))
            

if __name__ == '__main__':
    fire.Fire(LightningRunner)