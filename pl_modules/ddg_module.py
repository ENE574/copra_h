import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from models import ModelRegister
from utils.metrics import ScalarMetricAccumulator, cal_pearson, cal_spearman, cal_rmse, cal_mae, get_loss, per_complex_corr
from utils.torch_compat import torch_load_compat
def get_model(model_args:dict=None):
    register = ModelRegister()
    model_args_ori = {}
    model_args_ori.update(model_args)
    model_cls = register[model_args['model_type']]
    model = model_cls(**model_args_ori)
    return model

class DDGModule(pl.LightningModule):
    def __init__(self, output_dir=None, model_args=None, data_args=None, run_args=None):
        super().__init__()
        self.strict_loading = False
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
        # self.ddg_head = nn.Sequential(
        #     nn.Linear(320, 320), nn.ReLU(),
        #     nn.Linear(320, 320), nn.ReLU(),
        #     nn.Linear(320, 1)
        # )
        self.train_loss = None
        print("Initializing DDG Module!")

        def _get_hp(name, default=None):
            if self.run_args is not None and hasattr(self.run_args, name):
                return getattr(self.run_args, name)
            if hasattr(self.model_args, "train") and hasattr(self.model_args.train, name):
                return getattr(self.model_args.train, name)
            return default

        self.physics_aux_max_weight = float(_get_hp("physics_aux_max_weight", 0.0))
        self.physics_aux_warmup_epochs = int(_get_hp("physics_aux_warmup_epochs", 0))
        self.physics_aux_decay_epochs = int(_get_hp("physics_aux_decay_epochs", 0))
        self.physics_aux_min_weight = float(_get_hp("physics_aux_min_weight", 0.0))
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

    def _physics_aux_weight(self) -> float:
        e = int(self.current_epoch)
        if self.physics_aux_max_weight <= 0:
            return 0.0
        if self.physics_aux_warmup_epochs > 0 and e < self.physics_aux_warmup_epochs:
            return float(self.physics_aux_max_weight * (e + 1) / float(self.physics_aux_warmup_epochs))
        if self.physics_aux_decay_epochs <= 0:
            return float(self.physics_aux_max_weight)
        t = min(1.0, (e - self.physics_aux_warmup_epochs) / float(self.physics_aux_decay_epochs))
        return float(self.physics_aux_max_weight + t * (self.physics_aux_min_weight - self.physics_aux_max_weight))

    def _physics_aux_loss(self, batch, pooled_mut: torch.Tensor, main_loss: torch.Tensor) -> torch.Tensor:
        """MSE on standardized FoldX terms vs physics heads (mutant pooled), mirroring ModelModule."""
        if self.physics_aux_max_weight <= 0:
            return pooled_mut.new_zeros(())
        if "physics_targets" not in batch:
            return pooled_mut.new_zeros(())
        targets = batch["physics_targets"]
        pooled = pooled_mut.detach() if self.physics_aux_stop_grad else pooled_mut
        physics_pred = {name: self.model.physics_heads[name](pooled).squeeze(-1) for name in self.model.physics_names}
        allowed_terms = set(self.physics_aux_terms) if self.physics_aux_terms is not None else None
        physics_loss = pooled.new_zeros(())
        physics_terms = 0
        for name in physics_pred:
            if allowed_terms is not None and name not in allowed_terms:
                continue
            if name not in targets:
                continue
            t = targets[name].float()
            p = physics_pred[name]
            if self.physics_aux_standardize:
                i = self._pia_name_to_idx[name]
                m = self._pia_target_mu[i].to(device=p.device, dtype=p.dtype)
                s = self._pia_target_std[i].to(device=p.device, dtype=p.dtype)
                p_loss = F.mse_loss((p - m) / (s + 1e-8), (t - m) / (s + 1e-8))
            else:
                p_loss = F.mse_loss(p, t)
            physics_loss = physics_loss + p_loss
            physics_terms += 1
            self.log(f"train/loss_{name}", p_loss.detach(), batch_size=self.batch_size, on_step=True, sync_dist=True)
        if physics_terms == 0:
            return pooled_mut.new_zeros(())
        physics_loss = physics_loss / float(physics_terms)
        if self.physics_aux_normalize:
            denom = main_loss.detach().clamp_min(1e-8)
            physics_loss = physics_loss / denom
        if self.physics_aux_clip and self.physics_aux_clip > 0:
            physics_loss = physics_loss.clamp(max=self.physics_aux_clip)
        return physics_loss

    def get_progress_bar_dict(self):
        tqdm_dict = super().get_progress_bar_dict()
        tqdm_dict.pop('v_num', None)
        return tqdm_dict

    def _ddg_optimizer_param_groups(self):
        """Higher LR for pred_head and optional offline stem (per-model proj + q/k/v group projs)."""
        base_lr = float(self.optimizers_cfg.lr)
        head_mult = float(getattr(self.optimizers_cfg, "pred_head_lr_mult", 1.0))
        stem_mult = float(getattr(self.optimizers_cfg, "offline_fusion_stem_lr_mult", 1.0))

        head_ids = set()
        stem_ids = set()
        groups = []

        head_params = list(self.model.pred_head.parameters())
        if head_mult != 1.0:
            head_ids = {id(p) for p in head_params}
            groups.append({"params": head_params, "lr": base_lr * head_mult})

        stem_params = []
        if stem_mult != 1.0 and getattr(self.model, "use_offline_embeddings", False):
            for name in ("offline_proj", "offline_q_proj", "offline_k_proj", "offline_v_proj"):
                mod = getattr(self.model, name, None)
                if mod is not None:
                    stem_params.extend(list(mod.parameters()))
        if stem_mult != 1.0 and stem_params:
            stem_ids = {id(p) for p in stem_params}
            groups.append({"params": stem_params, "lr": base_lr * stem_mult})

        rest = [p for p in self.parameters() if id(p) not in head_ids and id(p) not in stem_ids]
        groups.append({"params": rest, "lr": base_lr})

        if head_mult != 1.0 or stem_mult != 1.0:
            nh, ns, nr = (len(head_params) if head_mult != 1.0 else 0), (
                len(stem_params) if stem_mult != 1.0 and stem_params else 0
            ), len(rest)
            print(
                "DDG param groups: base_lr=%s pred_head_lr_mult=%s offline_fusion_stem_lr_mult=%s "
                "-> tensors: pred_head=%d offline_stem=%d other=%d"
                % (base_lr, head_mult, stem_mult, nh, ns, nr)
            )
        return groups

    def configure_optimizers(self):
        print("Configuring Optimizers...")
        opt_type = str(self.optimizers_cfg.type).lower()
        wd = float(getattr(self.optimizers_cfg, "weight_decay", 0.0))
        b1 = float(getattr(self.optimizers_cfg, "beta1", 0.9))
        b2 = float(getattr(self.optimizers_cfg, "beta2", 0.999))
        betas = (b1, b2)
        param_groups = self._ddg_optimizer_param_groups()
        if opt_type == "adam":
            optimizer = torch.optim.Adam(param_groups, betas=betas, weight_decay=wd)
        elif opt_type == "adamw":
            optimizer = torch.optim.AdamW(param_groups, betas=betas, weight_decay=wd)
        elif opt_type == "sgd":
            optimizer = torch.optim.SGD(param_groups, weight_decay=wd)
        elif opt_type == "rmsprop":
            optimizer = torch.optim.RMSprop(param_groups, weight_decay=wd)
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

        if self.model_args.resume is not None:
            print("Resuming from checkloint: %s" % self.model_args.resume)
            ckpt = torch_load_compat(self.model_args.resume, map_location=self.model_args.device)
            it_first = ckpt['iteration']
            lsd_result = self.model.load_state_dict(ckpt['state_dict'], strict=False)
            print('Missing keys (%d): %s' % (len(lsd_result.missing_keys), ', '.join(lsd_result.missing_keys)))
            print(
                'Unexpected keys (%d): %s' % (len(lsd_result.unexpected_keys), ', '.join(lsd_result.unexpected_keys)))

            print('Resuming optimizer states...')
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as ex:
                print(
                    "Warning: could not load optimizer from resume checkpoint (%s). "
                    "This often happens when param groups changed (e.g. pred_head_lr_mult). "
                    "Continuing with a fresh optimizer state."
                    % (ex,)
                )
            print('Resuming scheduler states...')
            try:
                scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as ex:
                print("Warning: could not load scheduler from resume checkpoint (%s)." % (ex,))
            
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
    def on_train_epoch_start(self):
        self.train_results = []
        
    def training_step(self, batch, batch_idx):
        y = batch['labels']
        mut_out = self.model(batch, self.data_args.strategy, 'mutation')
        ddg_pred = mut_out['ddg_pred']
        ddg_pred_inv = mut_out['ddg_pred_inv']
        pooled_mut = mut_out['pooled_mut']
        loss = get_loss(self.l_type, ddg_pred, y, reduction='mean')
        loss_inv = get_loss(self.l_type, ddg_pred_inv, -y, reduction='mean')
        loss = 0.5 * (loss + loss_inv)

        bs_std = ddg_pred.detach().float().std(unbiased=False)
        self.log(
            "train/ddg_pred_batch_std",
            bs_std,
            batch_size=self.batch_size,
            on_step=True,
            sync_dist=True,
        )

        if self.physics_aux_max_weight > 0 and pooled_mut is not None:
            physics_loss = self._physics_aux_loss(batch, pooled_mut, loss)
            w = self._physics_aux_weight()
            if int(self.current_epoch) < int(self.physics_aux_start_epoch):
                w = 0.0
            if self.physics_aux_prob < 1.0:
                if float(torch.rand(1, device=loss.device).item()) > float(self.physics_aux_prob):
                    w = 0.0
            self.log("train/physics_aux_weight", w, batch_size=self.batch_size, on_step=True, sync_dist=True)
            self.log("train/physics_aux_loss", physics_loss.detach(), batch_size=self.batch_size, on_step=True, sync_dist=True)
            self.log("train/ddg_main_loss", loss.detach(), batch_size=self.batch_size, on_step=True, sync_dist=True)
            loss = loss + w * physics_loss

        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        for complex, y_true, y_pred in zip(batch['complex'], batch['labels'], ddg_pred):
            result = {}
            result['complex'] = complex
            result['y_true'] = y_true.item()
            result['y_pred'] = y_pred.item()
            self.train_results.append(result)
        return loss

    def on_train_epoch_end(self):
        results = pd.DataFrame(self.train_results)
        # print("Validation:", results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)
        y_pred = np.array(results[f'y_pred'])
        y_true = np.array(results[f'y_true'])
        pearson_all = cal_pearson(y_pred, y_true)
        spearman_all = cal_spearman(y_pred, y_true)
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred', true_attr='y_true')
        print(f'[Train All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        print(f'[Train PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
        
        self.log(f'train/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
    
    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        y = batch['labels']
        mut_out = self.model(batch, self.data_args.strategy, 'mutation')
        ddg_pred, ddg_pred_inv = mut_out['ddg_pred'], mut_out['ddg_pred_inv']
        # batch['prot'] = batch['prot_mut']
        # batch['restype'] = batch['mut_restype']
        # feat_mut = self.model(batch, self.data_args.strategy, 'mutation')
        # ddg_pred = self.ddg_head(feat_wild - feat_mut).squeeze(1)
        # ddg_pred_inv = self.ddg_head(feat_mut - feat_wild).squeeze(1)
        # print(y.shape, pred.shape)
        loss = get_loss(self.l_type, ddg_pred, y, reduction='mean')
        loss_inv = get_loss(self.l_type, ddg_pred_inv, -y, reduction='mean')
        val_loss = 0.5 * (loss + loss_inv)
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=self.batch_size, mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for complex, y_true, y_pred in zip(batch['complex'], batch['labels'], ddg_pred):
            result = {}
            result['complex'] = complex
            result['y_true'] = y_true.item()
            result['y_pred'] = y_pred.item()
            self.results.append(result)
        return val_loss
    
    def on_validation_epoch_end(self):
        results = pd.DataFrame(self.results)
        # print("Validation:", results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_{self.valid_it}.csv'), index=False)
        y_pred = np.array(results[f'y_pred'])
        y_true = np.array(results[f'y_true'])
        pearson_all = cal_pearson(y_pred, y_true)
        spearman_all = cal_spearman(y_pred, y_true)
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred', true_attr='y_true')
        print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        print(f'[PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
        
        self.log(f'val/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
    
        val_loss = self.scalar_accum.get_average('val_loss')
        self.log('val_loss', val_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        # Trigger scheduler
        self.valid_it += 1
        return val_loss

    def on_test_epoch_start(self) -> None:
        self.results = []
        self.scalar_accum = ScalarMetricAccumulator()
        
    def test_step(self, batch, batch_idx):
        y = batch['labels']
        mut_out = self.model(batch, self.data_args.strategy, 'mutation')
        ddg_pred, ddg_pred_inv = mut_out['ddg_pred'], mut_out['ddg_pred_inv']
        # batch['prot'] = batch['prot_mut']
        # batch['restype'] = batch['mut_restype']
        # feat_mut = self.model(batch, self.data_args.strategy, 'mutation')
        # ddg_pred = self.ddg_head(feat_wild - feat_mut).squeeze(1)
        # ddg_pred_inv = self.ddg_head(feat_mut - feat_wild).squeeze(1)
        # print(y.shape, pred.shape)
        loss = get_loss(self.l_type, ddg_pred, y, reduction='mean')
        loss_inv = get_loss(self.l_type, ddg_pred_inv, -y, reduction='mean')
        test_loss = 0.5 * (loss + loss_inv)
        self.scalar_accum.add(name='loss', value = test_loss, batchsize=self.batch_size, mode='mean')
        self.log("test_loss_step", test_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        for complex, y_true, y_pred in zip(batch['complex'], batch['labels'], ddg_pred):
            result = {}
            result['y_true'] = y_true.item()
            result['y_pred'] = y_pred.item()
            result['complex'] = complex
            self.results.append(result)
        return test_loss

    def on_test_epoch_end(self):
        results = pd.DataFrame(self.results)
        if self.output_dir is not None:
            results.to_csv(os.path.join(self.output_dir, f'results_test.csv'), index=False)
        y_pred = np.array(results[f'y_pred'])
        y_true = np.array(results[f'y_true'])
        pearson_all = cal_pearson(y_pred, y_true)
        spearman_all = cal_spearman(y_pred, y_true)
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred', true_attr='y_true')
        print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        print(f'[PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
        
        self.log(f'test/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all, 
                    "pc_pearson": pearson_pc,"pc_spearman": spearman_pc, "rmse_pc": rmse_pc, "mae_pc": mae_pc}
        print("Self.Res:", self.res)
        test_loss = self.scalar_accum.get_average('loss')
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        
        return test_loss