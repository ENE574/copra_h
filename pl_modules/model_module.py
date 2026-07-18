import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from models import ModelRegister
from utils.metrics import ScalarMetricAccumulator, cal_pearson, cal_spearman, cal_rmse, cal_mae, get_loss
from utils.torch_compat import torch_load_compat
def get_model(model_args:dict=None):
    register = ModelRegister()
    model_args_ori = {}
    model_args_ori.update(model_args)
    model_cls = register[model_args['model_type']]
    model = model_cls(**model_args_ori)
    return model

class ModelModule(pl.LightningModule):
    def __init__(self, output_dir=None, model_args=None, data_args=None, run_args=None):
        super().__init__()
        self.save_hyperparameters()
        if model_args is None:
            model_args = {}
        if data_args is None:
            data_args = {}
        self.output_dir = output_dir
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir) / 'pred'
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.l_type = data_args.loss_type
        self.model = get_model(model_args=model_args.model)
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.optimizers_cfg = self.model_args.train.optimizer
        self.scheduler_cfg = self.model_args.train.scheduler
        self.valid_it = 0
        self.batch_size = data_args.batch_size

        self.train_loss = None

        def _get_hp(name, default=None):
            if hasattr(self.run_args, name):
                return getattr(self.run_args, name)
            if hasattr(self.model_args, "train") and hasattr(self.model_args.train, name):
                return getattr(self.model_args.train, name)
            return default

        # Label standardization (z-score)
        self.standardize_label = bool(_get_hp("standardize_label", False))
        self.label_stats_from_data = bool(_get_hp("label_stats_from_data", False))
        self.label_mean = float(_get_hp("label_mean", 0.0))
        self.label_std = float(_get_hp("label_std", 1.0))
        self._label_stats_logged = False

        # Auxiliary physics loss weight scheduling (warmup + decay)
        self.physics_aux_max_weight = float(_get_hp("physics_aux_max_weight", 0.0))
        self.physics_aux_warmup_epochs = int(_get_hp("physics_aux_warmup_epochs", 0))
        self.physics_aux_decay_epochs = int(_get_hp("physics_aux_decay_epochs", 0))
        self.physics_aux_min_weight = float(_get_hp("physics_aux_min_weight", 0.0))

        self.main_loss_type = str(_get_hp("main_loss_type", "mse")).lower()
        self.huber_delta = float(_get_hp("huber_delta", 2.0))
        self.physics_aux_normalize = bool(_get_hp("physics_aux_normalize", True))
        self.physics_aux_clip = float(_get_hp("physics_aux_clip", 0.0))
        self.physics_aux_stop_grad = bool(_get_hp("physics_aux_stop_grad", False))
        self.physics_aux_start_epoch = int(_get_hp("physics_aux_start_epoch", 0))
        self.physics_aux_prob = float(_get_hp("physics_aux_prob", 1.0))
        self.physics_aux_terms = _get_hp("physics_aux_terms", None)
        if self.physics_aux_terms is not None:
            if isinstance(self.physics_aux_terms, str):
                self.physics_aux_terms = [self.physics_aux_terms]
            self.physics_aux_terms = [str(x) for x in list(self.physics_aux_terms)]

        # Z-score physics targets vs preds using CSV population stats (same space as FoldX kcal/mol).
        self.physics_aux_standardize = bool(_get_hp("physics_aux_standardize", True))
        _pia_names = list(self.model.physics_names)
        _mu = torch.zeros(len(_pia_names), dtype=torch.float32)
        _std = torch.ones(len(_pia_names), dtype=torch.float32)
        if self.physics_aux_standardize:
            csv_path = getattr(self.data_args, "physics_targets_csv", None)
            if csv_path and Path(str(csv_path)).is_file():
                from data.foldx_physics import compute_physics_target_stats_from_csv

                mu_d, std_d = compute_physics_target_stats_from_csv(csv_path)
                for i, n in enumerate(_pia_names):
                    _mu[i] = float(mu_d.get(n, 0.0))
                    _std[i] = max(float(std_d.get(n, 1.0)), 1e-6)
        self.register_buffer("_pia_target_mu", _mu, persistent=False)
        self.register_buffer("_pia_target_std", _std, persistent=False)
        self._pia_name_to_idx = {n: i for i, n in enumerate(_pia_names)}

    def on_fit_start(self) -> None:
        if not self.standardize_label:
            return
        if not self.label_stats_from_data:
            return
        if self.trainer is None or self.trainer.datamodule is None:
            return
        dm = self.trainer.datamodule
        if not hasattr(dm, "train_dataset") or dm.train_dataset is None:
            return

        ys = []
        for i in range(len(dm.train_dataset)):
            item = dm.train_dataset[i]
            if isinstance(item, dict) and "labels" in item:
                y = item["labels"]
            elif isinstance(item, dict) and "label" in item:
                y = item["label"]
            else:
                continue
            try:
                ys.append(float(y))
            except Exception:
                try:
                    ys.append(float(y.item()))
                except Exception:
                    continue

        if len(ys) == 0:
            return

        y_t = torch.tensor(ys, dtype=torch.float32)
        mean = float(y_t.mean().item())
        std = float(y_t.std(unbiased=False).clamp_min(1e-8).item())
        self.label_mean = mean
        self.label_std = std
        # NOTE: Lightning forbids `self.log()` inside `on_fit_start`; we will log once in the first training step.

    def _physics_aux_weight(self) -> float:
        """Warm up to max, then linearly decay to min."""
        e = int(self.current_epoch)
        if self.physics_aux_max_weight <= 0:
            return 0.0
        if self.physics_aux_warmup_epochs > 0 and e < self.physics_aux_warmup_epochs:
            # Linear warmup from 0 -> max
            return float(self.physics_aux_max_weight * (e + 1) / float(self.physics_aux_warmup_epochs))
        # If decay is disabled, keep a constant weight at max (after warmup).
        # Returning min_weight here would silently disable the aux loss for common
        # configs where min_weight defaults to 0.
        if self.physics_aux_decay_epochs <= 0:
            return float(self.physics_aux_max_weight)
        t = min(1.0, (e - self.physics_aux_warmup_epochs) / float(self.physics_aux_decay_epochs))
        return float(self.physics_aux_max_weight + t * (self.physics_aux_min_weight - self.physics_aux_max_weight))

    def _main_loss(self, pred, y):
        if self.l_type != 'regression':
            return get_loss(self.l_type, pred, y, reduction='mean')
        if self.main_loss_type == 'huber' or self.main_loss_type == 'smoothl1':
            return F.huber_loss(pred, y, delta=self.huber_delta, reduction='mean')
        return F.mse_loss(pred, y, reduction='mean')

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def configure_optimizers(self):
        opt_type = str(self.optimizers_cfg.type).lower()
        wd = float(getattr(self.optimizers_cfg, "weight_decay", 0.0))
        b1 = float(getattr(self.optimizers_cfg, "beta1", 0.9))
        b2 = float(getattr(self.optimizers_cfg, "beta2", 0.999))
        betas = (b1, b2)
        if opt_type == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.optimizers_cfg.lr,
                betas=betas,
                weight_decay=wd,
            )
        elif opt_type == "adamw":
            # Decoupled weight decay (Loshchilov & Hutter); prefer smaller wd than legacy Adam+L2.
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.optimizers_cfg.lr,
                betas=betas,
                weight_decay=wd,
            )
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.optimizers_cfg.lr, weight_decay=wd)
        elif opt_type == "rmsprop":
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.optimizers_cfg.lr, weight_decay=wd)
        else:
            raise NotImplementedError('Optimizer not supported: %s' % self.optimizers_cfg.type)

        if self.scheduler_cfg.type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                                   factor=self.scheduler_cfg.factor, 
                                                                   patience=self.scheduler_cfg.patience, 
                                                                   min_lr=self.scheduler_cfg.min_lr)
        elif self.scheduler_cfg.type == 'multistep':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                             milestones=self.scheduler_cfg.milestones, 
                                                             gamma=self.scheduler_cfg.gamma)
        elif self.scheduler_cfg.type == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                               gamma=self.scheduler_cfg.gamma)
        else:
            raise NotImplementedError('Scheduler not supported: %s' % self.scheduler_cfg.type)

        if getattr(self.model_args, 'resume', None) is not None:
            print("Resuming from checkloint: %s" % self.model_args.resume)
            ckpt = torch_load_compat(self.model_args.resume, map_location=self.model_args.device)
            it_first = ckpt['iteration']
            lsd_result = self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
            print(
                'Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

            print('Resuming optimizer states...')
            optimizer.load_state_dict(ckpt['optimizer'])
            print('Resuming scheduler states...')
            scheduler.load_state_dict(ckpt['scheduler'])
            
        if self.scheduler_cfg.type == 'plateau':
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": 'val_loss'
                }
            }
        else:
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                }
            }
        return optim_dict

    def on_train_start(self):
        log_hyperparams = {'model_args':self.model_args, 'data_args': self.data_args, 'run_args': self.run_args}
        self.logger.log_hyperparams(log_hyperparams)

    def on_before_optimizer_step(self, optimizer) -> None:
        pass
        # for name, param in self.named_parameters():
        #     if param.grad is None:
        #         print(name)
        #         print("Found Unused Parameters")

    @staticmethod
    def _unwrap_multitask_batch(batch):
        """Extract the actual task batch from CombinedLoader dict format.
        
        Lightning 2.0 CombinedLoader (max_size_cycle mode) yields dicts like
        {task_name: batch_dict}.  Extract the first non-empty task batch.
        """
        if not isinstance(batch, dict):
            return batch
        # Check if this looks like a CombinedLoader dict (values are dicts with 'labels')
        for task_name, task_batch in batch.items():
            if isinstance(task_batch, dict) and 'labels' in task_batch:
                return task_batch
        return batch

    def training_step(self, batch, batch_idx):
        batch = self._unwrap_multitask_batch(batch)
        y = batch['labels']
        if self.standardize_label and self.label_stats_from_data and (not self._label_stats_logged):
            self.log("train/label_mean", float(self.label_mean), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log("train/label_std", float(self.label_std), batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self._label_stats_logged = True
        if self.standardize_label:
            y = (y - self.label_mean) / (self.label_std + 1e-8)
        output = self.model(batch, self.data_args.strategy)
        if isinstance(output, dict):
            pred = output['pred']
            pred_for_loss = pred
            if self.standardize_label:
                pred_for_loss = (pred - self.label_mean) / (self.label_std + 1e-8)
            # Basic ΔG loss
            main_loss = self._main_loss(pred_for_loss, y)
            loss = main_loss
            
            # Physics Decomposition Loss (Auxiliary)
            if 'physics_targets' in batch:
                physics_loss = 0
                physics_terms = 0
                targets = batch['physics_targets']
                physics_pred = output['physics']
                allowed_terms = None
                if self.physics_aux_terms is not None:
                    allowed_terms = set(self.physics_aux_terms)
                if self.physics_aux_stop_grad:
                    pooled = output.get('pooled', None)
                    if pooled is None:
                        raise RuntimeError("physics_aux_stop_grad requires model to return 'pooled' in output dict")
                    pooled = pooled.detach()
                    physics_pred = {name: self.model.physics_heads[name](pooled).squeeze(-1) for name in self.model.physics_names}

                for name in physics_pred:
                    if allowed_terms is not None and name not in allowed_terms:
                        continue
                    if name in targets:
                        t = targets[name].float()
                        p = physics_pred[name]
                        if self.physics_aux_standardize:
                            i = self._pia_name_to_idx[name]
                            m = self._pia_target_mu[i].to(device=p.device, dtype=p.dtype)
                            s = self._pia_target_std[i].to(device=p.device, dtype=p.dtype)
                            p_loss = F.mse_loss((p - m) / (s + 1e-8), (t - m) / (s + 1e-8))
                        else:
                            p_loss = F.mse_loss(p, t)
                        physics_loss += p_loss
                        physics_terms += 1
                        self.log(f"train/loss_{name}", p_loss, batch_size=self.batch_size, on_step=True, sync_dist=True)

                if physics_terms > 0:
                    physics_loss = physics_loss / float(physics_terms)

                if self.physics_aux_normalize:
                    denom = main_loss.detach().clamp_min(1e-8)
                    physics_loss = physics_loss / denom

                if self.physics_aux_clip and self.physics_aux_clip > 0:
                    physics_loss = physics_loss.clamp(max=self.physics_aux_clip)

                w = self._physics_aux_weight()
                if int(self.current_epoch) < int(self.physics_aux_start_epoch):
                    w = 0.0
                if self.physics_aux_prob < 1.0:
                    if float(torch.rand(1, device=pred.device).item()) > float(self.physics_aux_prob):
                        w = 0.0
                self.log("train/physics_aux_weight", w, batch_size=self.batch_size, on_step=True, sync_dist=True)
                self.log("train/physics_aux_loss", physics_loss, batch_size=self.batch_size, on_step=True, sync_dist=True)
                self.log("train/main_loss", main_loss, batch_size=self.batch_size, on_step=True, sync_dist=True)
                loss = loss + w * physics_loss
            
            # Log individual physical components for monitoring
            for name, val in output['physics'].items():
                self.log(f"train/physics_{name}_val", val.mean(), batch_size=self.batch_size, on_step=True, sync_dist=True)
            self.log("train/main_refinement", output['main_refinement'].mean(), batch_size=self.batch_size, on_step=True, sync_dist=True)
        else:
            pred = output
            loss = self._main_loss(pred, y)

        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        y = batch['labels']
        output = self.model(batch, self.data_args.strategy)
        if isinstance(output, dict):
            pred = output['pred']
        else:
            pred = output

        val_loss = get_loss(self.l_type, pred, y, reduction='mean')
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=self.batch_size, mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for y_true, y_pred in zip(batch['labels'], pred):
            result = {}
            result['y_true'] = y_true.item()
            result['y_pred'] = y_pred.item()
            self.results.append(result)
        return val_loss
    
    def on_validation_epoch_end(self):
        results = pd.DataFrame(self.results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)
        y_pred = np.array(results[f'y_pred'])
        y_true = np.array(results[f'y_true'])
        pearson_all = np.abs(cal_pearson(y_pred, y_true))
        spearman_all = np.abs(cal_spearman(y_pred, y_true))
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        
        self.log(f'val/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
    
        val_loss = rmse_all * rmse_all
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.valid_it += 1
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.results = []
        self.scalar_accum = ScalarMetricAccumulator()
        
    def test_step(self, batch, batch_idx):
        y = batch['labels']
        output = self.model(batch, self.data_args.strategy)
        if isinstance(output, dict):
            pred = output['pred']
        else:
            pred = output

        test_loss = get_loss(self.l_type, pred, y, reduction='mean')
        self.scalar_accum.add(name='loss', value = test_loss, batchsize=self.batch_size, mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for y_true, y_pred in zip(batch['labels'], pred):
            result = {}
            result['y_true'] = y_true.item()
            result['y_pred'] = y_pred.item()
            self.results.append(result)
        return test_loss

    def on_test_epoch_end(self):
        results = pd.DataFrame(self.results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_test.csv'), index=False)
        y_pred = np.array(results[f'y_pred'])
        y_true = np.array(results[f'y_true'])
        pearson_all = np.abs(cal_pearson(y_pred, y_true))
        spearman_all = np.abs(cal_spearman(y_pred, y_true))
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        
        self.log(f'test/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all}
        print("Self.Res:", self.res)
        # test_loss = self.scalar_accum.get_average('loss')
        test_loss = rmse_all * rmse_all
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        
        return test_loss