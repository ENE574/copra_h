import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from models import ModelRegister
from utils.metrics import ScalarMetricAccumulator, cal_pearson, cal_spearman, cal_rmse, cal_mae, get_loss, per_complex_corr, global_auroc
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
        
        if hasattr(data_args, "loss_type"):
            self.l_type = data_args.loss_type
        elif run_args and getattr(run_args, "multitask_sources", None):
            self.l_type = "regression"
        else:
            self.l_type = "regression"
            
        self.model = get_model(model_args=model_args.model)
        self.model_args = model_args
        self.data_args = data_args
        self.run_args = run_args
        self.optimizers_cfg = self.model_args.train.optimizer
        self.scheduler_cfg = self.model_args.train.scheduler
        if self.scheduler_cfg is not None and getattr(self.scheduler_cfg, "type", None) == "plateau":
            sched_monitor = getattr(self.scheduler_cfg, "monitor", "val_loss")
            if sched_monitor.startswith("val/"):
                sched_monitor = "train/" + sched_monitor.split("/", 1)[1]
            elif sched_monitor == "val_loss":
                sched_monitor = "train_loss"
            self.scheduler_cfg.monitor = sched_monitor
        self.valid_it = 0
        
        if hasattr(data_args, "batch_size"):
            self.batch_size = data_args.batch_size
        elif run_args and getattr(run_args, "multitask_sources", None):
            sources = run_args.multitask_sources
            self.batch_size = sum(s.get("batch_size", 1) for s in sources)
        else:
            self.batch_size = 4
        # self.ddg_head = nn.Sequential(
        #     nn.Linear(320, 320), nn.ReLU(),
        #     nn.Linear(320, 320), nn.ReLU(),
        #     nn.Linear(320, 1)
        # )
        self.train_loss = None
        print("Initializing DDG Module!")

        def _get_hp(name, default=None):
            if self.run_args is not None and hasattr(self.run_args, name):
                val = getattr(self.run_args, name)
                if val is not None:
                    return val
            if hasattr(self.model_args, "train") and hasattr(self.model_args.train, name):
                val = getattr(self.model_args.train, name)
                if val is not None:
                    return val
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

        self.main_loss_type = str(_get_hp("main_loss_type", "huber")).lower()
        self.huber_delta = float(_get_hp("huber_delta", 2.0))
        self.complex_demeaned_loss_weight = float(_get_hp("complex_demeaned_loss_weight", 0.0))
        self.complex_demean_min_group_size = int(_get_hp("complex_demean_min_group_size", 2))
        self.ranking_loss_weight = float(_get_hp("ranking_loss_weight", 0.0))
        self.ranking_loss_margin = float(_get_hp("ranking_loss_margin", 0.0))
        self.accum_window_loss = bool(_get_hp("accum_window_loss", False))
        self.ddg_pred_std_floor = float(_get_hp("ddg_pred_std_floor", 0.0))
        self.ddg_pred_std_reg_weight = float(_get_hp("ddg_pred_std_reg_weight", 0.0))
        self.ddg_mag_loss_weight = float(_get_hp("ddg_mag_loss_weight", 0.0))
        self.ddg_mag_loss_mode = str(_get_hp("ddg_mag_loss_mode", "linear")).lower()
        self.interface_aux_weight = float(_get_hp("interface_aux_weight", 0.0))
        self._pred_std_buffer = []
        self._accum_buffer = []
        if self.complex_demeaned_loss_weight > 0:
            print(
                "DDG complex-demeaned loss: weight=%s min_group_size=%s accum_window=%s main_loss=%s huber_delta=%s"
                % (
                    self.complex_demeaned_loss_weight,
                    self.complex_demean_min_group_size,
                    self.accum_window_loss,
                    self.main_loss_type,
                    self.huber_delta,
                )
            )
        if self.ranking_loss_weight > 0:
            print(
                "DDG pairwise ranking loss weight: %s margin=%s accum_window=%s"
                % (self.ranking_loss_weight, self.ranking_loss_margin, self.accum_window_loss)
            )
        if self.ddg_pred_std_reg_weight > 0:
            print(
                "DDG pred-std regularizer: floor=%s weight=%s"
                % (self.ddg_pred_std_floor, self.ddg_pred_std_reg_weight)
            )
        if self.interface_aux_weight > 0:
            print("DDG interface self-supervision weight: %s" % self.interface_aux_weight)

        self.use_ddg_residual = bool(getattr(self.model, "mutation_ddg_residual", False))
        if self.use_ddg_residual:
            cdim = self.model.complex_dim
            fsize = self.model.feat_size
            out_dim = int(getattr(self.model, "output_dim", 1))
            self.ddg_complex_head = nn.Sequential(
                nn.Linear(cdim, fsize), nn.ReLU(),
                nn.Linear(fsize, out_dim),
            )
            self.ddg_delta_head = nn.Sequential(
                nn.Linear(cdim, fsize), nn.ReLU(),
                nn.Linear(fsize, fsize), nn.ReLU(),
                nn.Linear(fsize, out_dim),
            )
            print("DDG residual head: complex_bias(wt) + delta(local diff)")
        else:
            self.ddg_complex_head = None
            self.ddg_delta_head = None

    def _accum_steps(self):
        if self.trainer is not None:
            return max(1, int(getattr(self.trainer, "accumulate_grad_batches", 1)))
        if self.run_args is not None:
            return max(1, int(getattr(self.run_args, "accumulate_grad_batches", 1)))
        return 1

    def _pred_std_regularizer(self, pred):
        if self.ddg_pred_std_reg_weight <= 0:
            return pred.new_zeros(())
        self._pred_std_buffer.extend(pred.detach().float().reshape(-1).tolist())
        target_n = max(4, self.batch_size * self._accum_steps())
        if len(self._pred_std_buffer) < target_n:
            return pred.new_zeros(())
        vals = torch.tensor(self._pred_std_buffer, device=pred.device, dtype=torch.float32)
        self._pred_std_buffer.clear()
        std = vals.std(unbiased=False)
        if std >= self.ddg_pred_std_floor:
            return pred.new_zeros(())
        gap = self.ddg_pred_std_floor - std
        return self.ddg_pred_std_reg_weight * gap * gap

    def _unwrap_multitask_batch(self, batch):
        if isinstance(batch, dict) and "labels" not in batch:
            task_name = next(iter(batch.keys()))
            return str(task_name), batch[task_name]
        return None, batch

    def _interface_aux_loss(self, batch, pooled_mut):
        """Light interface geometry self-supervision from cross-entity contact density."""
        if self.interface_aux_weight <= 0 or pooled_mut is None:
            return pooled_mut.new_zeros(()) if pooled_mut is not None else batch["labels"].new_zeros(())
        atom_min = batch.get("atom_min_dist")
        idf = batch.get("identifier")
        if atom_min is None or idf is None:
            return pooled_mut.new_zeros(())
        losses = []
        for b in range(atom_min.shape[0]):
            idb = idf[b]
            dist = atom_min[b]
            a_idx = (idb == 0).nonzero(as_tuple=True)[0]
            b_idx = (idb == 1).nonzero(as_tuple=True)[0]
            if len(a_idx) == 0 or len(b_idx) == 0:
                continue
            sub = dist.index_select(0, a_idx).index_select(1, b_idx)
            finite = torch.isfinite(sub)
            if not finite.any():
                continue
            contact = (sub[finite] < 8.0).float().mean()
            rep_norm = pooled_mut[b].float().norm()
            target = contact.detach() * (rep_norm.detach() + 1e-6)
            losses.append((rep_norm - target).pow(2))
        if not losses:
            return pooled_mut.new_zeros(())
        return self.interface_aux_weight * torch.stack(losses).mean()

    def _predict_ddg(self, mut_out):
        if self.use_ddg_residual:
            feat = mut_out["ddg_mut_feat"]
            ctx = mut_out["ddg_wt_context"]
            bias = self.ddg_complex_head(ctx).squeeze(-1)
            delta = self.ddg_delta_head(feat).squeeze(-1)
            delta_inv = self.ddg_delta_head(-feat).squeeze(-1)
            return bias + delta, -bias + delta_inv
        return mut_out["ddg_pred"], mut_out["ddg_pred_inv"]

    def _ddg_point_loss(self, pred, y, weights=None):
        if self.main_loss_type in {"huber", "smoothl1"}:
            loss = F.huber_loss(pred, y, delta=self.huber_delta, reduction="none")
        elif self.main_loss_type == "mse":
            loss = F.mse_loss(pred, y, reduction="none")
        else:
            loss = get_loss(self.l_type, pred, y, reduction="none")
        if weights is not None:
            loss = loss * weights
        return loss.mean()

    def _complex_demeaned_ddg_loss(self, pred, y, complexes):
        """Huber/MSE on complex-demeaned preds and labels (batch groups with >= min size)."""
        if self.complex_demeaned_loss_weight <= 0:
            return pred.new_zeros(())

        groups = {}
        for i, c in enumerate(complexes):
            groups.setdefault(str(c), []).append(i)

        demean_pred = []
        demean_y = []
        for indices in groups.values():
            if len(indices) < self.complex_demean_min_group_size:
                continue
            idx = torch.as_tensor(indices, device=pred.device, dtype=torch.long)
            p = pred.index_select(0, idx)
            t = y.index_select(0, idx)
            demean_pred.append(p - p.mean())
            demean_y.append(t - t.mean())

        if not demean_pred:
            return pred.new_zeros(())

        dp = torch.cat(demean_pred, dim=0)
        dt = torch.cat(demean_y, dim=0)
        return self._ddg_point_loss(dp, dt)

    def _ddg_symmetric_loss(self, pred, pred_inv, y, complexes, weights=None):
        use_inbatch_demean = self.complex_demeaned_loss_weight > 0 and not self.accum_window_loss
        if not use_inbatch_demean:
            loss = self._ddg_point_loss(pred, y, weights)
            loss_inv = self._ddg_point_loss(pred_inv, -y, weights)
            return 0.5 * (loss + loss_inv)

        loss = self._ddg_point_loss(pred, y, weights)
        loss_inv = self._ddg_point_loss(pred_inv, -y, weights)
        loss = 0.5 * (loss + loss_inv)

        demean = self._complex_demeaned_ddg_loss(pred, y, complexes)
        demean_inv = self._complex_demeaned_ddg_loss(pred_inv, -y, complexes)
        demean = 0.5 * (demean + demean_inv)
        return loss + self.complex_demeaned_loss_weight * demean

    def _pairwise_ranking_loss(self, pred, y):
        if pred.numel() < 2:
            return pred.new_zeros(())
        y_diff = y.unsqueeze(1) - y.unsqueeze(0)
        p_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        upper = torch.triu(torch.ones(pred.numel(), pred.numel(), device=pred.device), diagonal=1).bool()
        y_diff = y_diff[upper]
        p_diff = p_diff[upper]
        if y_diff.numel() == 0:
            return pred.new_zeros(())
        product = y_diff * p_diff
        return F.softplus(self.ranking_loss_margin - product).mean()

    def _accum_complex_demeaned_loss(self, pred, y, complexes, buffer):
        """Complex-demeaned Huber over current batch + prior micro-batches in accum window."""
        if self.complex_demeaned_loss_weight <= 0:
            return pred.new_zeros(())

        all_preds = []
        all_ys = []
        all_c = []
        all_is_cur = []

        for i, c in enumerate(complexes):
            all_preds.append(pred[i])
            all_ys.append(y[i])
            all_c.append(str(c))
            all_is_cur.append(True)
        for p_buf, y_buf, c_list in buffer:
            for i, c in enumerate(c_list):
                all_preds.append(p_buf[i])
                all_ys.append(y_buf[i])
                all_c.append(c)
                all_is_cur.append(False)

        groups = {}
        for idx, c in enumerate(all_c):
            groups.setdefault(c, []).append(idx)

        losses = []
        for indices in groups.values():
            if len(indices) < self.complex_demean_min_group_size:
                continue
            gp = torch.stack([all_preds[i] for i in indices])
            gy = torch.stack([all_ys[i] for i in indices])
            mp = gp.mean()
            my = gy.mean()
            for i in indices:
                if all_is_cur[i]:
                    losses.append(self._ddg_point_loss(all_preds[i] - mp, all_ys[i] - my))

        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()

    def _accum_cross_ranking_loss(self, pred, y, complexes, buffer):
        """Ranking pairs between current batch and detached preds in accum window (same complex)."""
        if not buffer:
            return pred.new_zeros(())

        losses = []
        cur_c = [str(c) for c in complexes]
        for i, ci in enumerate(cur_c):
            for p_buf, y_buf, c_list in buffer:
                for j, cj in enumerate(c_list):
                    if ci != cj:
                        continue
                    yd = y[i] - y_buf[j]
                    pd = pred[i] - p_buf[j]
                    losses.append(F.softplus(self.ranking_loss_margin - yd * pd))
        if not losses:
            return pred.new_zeros(())
        return torch.stack(losses).mean()

    def _ddg_training_loss(self, pred, pred_inv, y, complexes, accum_buffer=None):
        weights = None
        if self.ddg_mag_loss_weight > 0:
            # Magnitude-aware weighting: up-weight large-|ddG| samples so the
            # model is forced to fit extreme mutations instead of regressing to
            # the mean (which inflates RMSE on large ddG). Symmetric for inv branch.
            mag = y.abs()
            if self.ddg_mag_loss_mode == "sqrt":
                weights = 1.0 + self.ddg_mag_loss_weight * mag.sqrt()
            elif self.ddg_mag_loss_mode == "log1p":
                weights = 1.0 + self.ddg_mag_loss_weight * torch.log1p(mag)
            else:  # linear (default)
                weights = 1.0 + self.ddg_mag_loss_weight * mag
        loss = self._ddg_symmetric_loss(pred, pred_inv, y, complexes, weights=weights)
        if accum_buffer is not None and self.accum_window_loss:
            if self.complex_demeaned_loss_weight > 0:
                dem = self._accum_complex_demeaned_loss(pred, y, complexes, accum_buffer)
                dem_inv = self._accum_complex_demeaned_loss(pred_inv, -y, complexes, accum_buffer)
                dem = 0.5 * (dem + dem_inv)
                loss = loss + self.complex_demeaned_loss_weight * dem
                self.log("train/accum_demean_loss", dem.detach(), batch_size=self.batch_size, on_step=True, sync_dist=True)
            if self.ranking_loss_weight > 0:
                rank = 0.5 * (
                    self._accum_cross_ranking_loss(pred, y, complexes, accum_buffer)
                    + self._accum_cross_ranking_loss(pred_inv, -y, complexes, accum_buffer)
                )
                loss = loss + self.ranking_loss_weight * rank
                self.log("train/accum_ranking_loss", rank.detach(), batch_size=self.batch_size, on_step=True, sync_dist=True)
        else:
            if self.ranking_loss_weight > 0:
                rank = 0.5 * (self._pairwise_ranking_loss(pred, y) + self._pairwise_ranking_loss(pred_inv, -y))
                loss = loss + self.ranking_loss_weight * rank
                self.log("train/ranking_loss", rank.detach(), batch_size=self.batch_size, on_step=True, sync_dist=True)
        return loss

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
        freeze_backbone = bool(getattr(self.optimizers_cfg, "freeze_backbone", False))

        head_ids = set()
        stem_ids = set()
        groups = []

        # Collect ALL DDG-specific parameters (including GNN, which lives inside self.model)
        head_params = list(self.model.pred_head.parameters())
        if getattr(self.model, "ddg_local_fuse", None) is not None:
            head_params.extend(list(self.model.ddg_local_fuse.parameters()))
        if getattr(self.model, "ddg_wt_delta", None) is not None:
            head_params.extend(list(self.model.ddg_wt_delta.parameters()))
        if getattr(self.model, "mut_local_gnn", None) is not None:
            head_params.extend(list(self.model.mut_local_gnn.parameters()))
        if self.ddg_complex_head is not None:
            head_params.extend(list(self.ddg_complex_head.parameters()))
        if self.ddg_delta_head is not None:
            head_params.extend(list(self.ddg_delta_head.parameters()))
        if getattr(self.model, "cross_attn_proj", None) is not None:
            head_params.extend(list(self.model.cross_attn_proj.parameters()))

        # When freezing backbone, use head_mult=1.0 internally (DDG layers always at base_lr)
        effective_head_mult = 1.0 if freeze_backbone else head_mult
        if effective_head_mult != 1.0 or freeze_backbone:
            head_ids = {id(p) for p in head_params}
            groups.append({"params": head_params, "lr": base_lr * effective_head_mult})

        stem_params = []
        if stem_mult != 1.0 and getattr(self.model, "use_offline_embeddings", False) and not freeze_backbone:
            for name in ("offline_proj", "offline_q_proj", "offline_k_proj", "offline_v_proj"):
                mod = getattr(self.model, name, None)
                if mod is not None:
                    stem_params.extend(list(mod.parameters()))
        if stem_mult != 1.0 and stem_params:
            stem_ids = {id(p) for p in stem_params}
            groups.append({"params": stem_params, "lr": base_lr * stem_mult})

        if not freeze_backbone:
            rest = [p for p in self.parameters() if id(p) not in head_ids and id(p) not in stem_ids]
            groups.append({"params": rest, "lr": base_lr})
        else:
            # Freeze everything that is NOT in the DDG head groups
            frozen_ids = head_ids | stem_ids
            n_frozen = 0
            for p in self.parameters():
                if id(p) not in frozen_ids:
                    p.requires_grad_(False)
                    n_frozen += 1
            rest = []
            print("freeze_backbone: %d backbone tensors frozen, %d DDG tensors trainable"
                  % (n_frozen, len(head_params)))

        if (head_mult != 1.0 or stem_mult != 1.0) and not freeze_backbone:
            nh = len(head_params) if head_mult != 1.0 else 0
            ns = len(stem_params) if stem_mult != 1.0 and stem_params else 0
            nr = len(rest)
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
            sched_monitor = getattr(self.scheduler_cfg, "monitor", "val_loss")
            sched_mode = getattr(self.scheduler_cfg, "mode", None)
            if sched_monitor.startswith("val/"):
                sched_monitor = "train/" + sched_monitor.split("/", 1)[1]
            elif sched_monitor == "val_loss":
                sched_monitor = "train_loss"
            if sched_mode is None:
                sched_mode = "min" if sched_monitor in ("train_loss", "val_loss", "val/all_rmse", "val/all_mae", "train/all_rmse", "train/all_mae") else "max"
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=self.scheduler_cfg.factor,
                patience=self.scheduler_cfg.patience,
                min_lr=self.scheduler_cfg.min_lr,
                mode=sched_mode,
            )
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
            print("Loading backbone weights from: %s" % self.model_args.resume)
            ckpt = torch_load_compat(self.model_args.resume, map_location='cpu')
            state_dict = ckpt['state_dict']
            # Strip 'model.' prefix from PL checkpoint keys for backbone loading.
            # In PL state_dict, keys look like 'model.prot_embedding', but
            # self.model.load_state_dict expects 'prot_embedding' (no prefix).
            is_pl_checkpoint = any(k.startswith('model.') for k in state_dict.keys())
            if is_pl_checkpoint:
                state_dict = {k.replace('model.', '', 1): v for k, v in state_dict.items()}
            print("  Stripped 'model.' prefix → %d backbone keys" % len(state_dict))
            
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('pred_head')}
            print("  Removed pred_head keys → %d backbone keys" % len(state_dict))
            
            lsd_result = self.model.load_state_dict(state_dict, strict=False)
            n_missing = len(lsd_result.missing_keys)
            n_unexpected = len(lsd_result.unexpected_keys)
            print('  Loaded backbone: %d missing, %d unexpected keys' % (n_missing, n_unexpected))
            if n_missing > 0:
                ddg_missing = [k for k in lsd_result.missing_keys if any(t in k for t in ('ddg', 'mut_local', 'gnn', 'wt_context', 'wt_delta', 'local_fuse', 'mutation'))]
                other_missing = [k for k in lsd_result.missing_keys if k not in ddg_missing]
                if ddg_missing:
                    print('  DDG-specific layers (initialized random): %d keys' % len(ddg_missing))
                if other_missing:
                    print('  Other missing (!!): %s' % ', '.join(other_missing[:10]))
            # Only restore optimizer/scheduler when resuming the same model.
            # For transfer learning, backbone weights are loaded but optimizer starts fresh.
            backbone_loaded = n_unexpected == 0 and len(state_dict) > 0
            if backbone_loaded and 'optimizer' in ckpt:
                print('  Attempting optimizer restore...')
                try:
                    optimizer.load_state_dict(ckpt['optimizer'])
                    print('  Optimizer restored.')
                except Exception as ex:
                    print("  Optimizer NOT restored (fresh start; this is normal for transfer learning): %s" % ex)
                try:
                    scheduler.load_state_dict(ckpt['scheduler'])
                    print('  Scheduler restored.')
                except Exception as ex:
                    print("  Scheduler NOT restored: %s" % ex)
            
        if self.scheduler_cfg.type == 'plateau':
            sched_monitor = getattr(self.scheduler_cfg, "monitor", "val_loss")
            if sched_monitor.startswith("val/"):
                sched_monitor = "train/" + sched_monitor.split("/", 1)[1]
            elif sched_monitor == "val_loss":
                sched_monitor = "train_loss"
            optim_dict = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": sched_monitor,
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
        self._accum_buffer = []
        self._pred_std_buffer = []

    def training_step(self, batch, batch_idx):
        task_name, batch = self._unwrap_multitask_batch(batch)
        if task_name is not None:
            self.log("train/multitask_source", float(hash(task_name) % 1000), batch_size=1, on_step=True)
        
        noise_scale = float(getattr(self.model_args.train, "input_noise_scale", 0.0))
        if noise_scale > 0 and self.training:
            for key in ['offline_embeddings', 'offline_embeddings_mut']:
                if key in batch:
                    emb_dict = batch[key]
                    for group in emb_dict:
                        for model_name in emb_dict[group]:
                            emb = emb_dict[group][model_name]
                            noise = torch.randn_like(emb) * noise_scale
                            emb_dict[group][model_name] = emb + noise
        
        reverse_aug_prob = float(getattr(self.model_args.train, "reverse_aug_prob", 0.0))
        y = batch['labels']
        if reverse_aug_prob > 0 and self.training and 'offline_embeddings_mut' in batch:
            if torch.rand(1).item() < reverse_aug_prob:
                batch['offline_embeddings'], batch['offline_embeddings_mut'] = \
                    batch['offline_embeddings_mut'], batch['offline_embeddings']
                y = -y
        strategy = batch.get('strategy', 'separate')
        mut_out = self.model(batch, strategy, 'mutation')
        ddg_pred, ddg_pred_inv = self._predict_ddg(mut_out)
        pooled_mut = mut_out['pooled_mut']
        accum_buffer = self._accum_buffer if self.accum_window_loss else None
        complex_group = batch.get('complex_group', batch['complex'])
        loss = self._ddg_training_loss(ddg_pred, ddg_pred_inv, y, complex_group, accum_buffer=accum_buffer)
        std_reg = self._pred_std_regularizer(ddg_pred)
        if float(std_reg.detach()) > 0:
            loss = loss + std_reg
            self.log(
                "train/ddg_pred_std_reg",
                std_reg.detach(),
                batch_size=self.batch_size,
                on_step=True,
                sync_dist=True,
            )

        if self.accum_window_loss:
            cg_list = [str(c) for c in complex_group]
            self._accum_buffer.append(
                (ddg_pred.detach(), y.detach(), cg_list)
            )
            if len(self._accum_buffer) >= self._accum_steps():
                self._accum_buffer.clear()

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

        iface = self._interface_aux_loss(batch, pooled_mut)
        if float(iface.detach()) > 0:
            loss = loss + iface
            self.log(
                "train/interface_aux_loss",
                iface.detach(),
                batch_size=self.batch_size,
                on_step=True,
                sync_dist=True,
            )

        self.train_loss = loss.detach()
        self.log("train_loss", float(self.train_loss), batch_size=self.batch_size, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)
        complex_group_rec = batch.get('complex_group', batch['complex'])
        for complex, cg, y_true, y_pred in zip(batch['complex'], complex_group_rec, batch['labels'], ddg_pred):
            result = {}
            result['complex'] = complex
            result['complex_group'] = str(cg)
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
        auroc_global = global_auroc(results, pred_attr='y_pred', true_attr='y_true')
        print(f'[Train All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        print(f'[Train PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
        print(f'[Train Global_AUROC] AUROC {auroc_global:.6f}')
        
        self.log(f'train/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'train/global_auroc', auroc_global, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
    
    def on_validation_epoch_start(self):
        self.scalar_accum = ScalarMetricAccumulator()
        self.results = []

    def validation_step(self, batch, batch_idx):
        _, batch = self._unwrap_multitask_batch(batch)
        y = batch['labels']
        strategy = batch.get('strategy', 'separate')
        mut_out = self.model(batch, strategy, 'mutation')
        ddg_pred, ddg_pred_inv = self._predict_ddg(mut_out)
        val_loss = self._ddg_symmetric_loss(ddg_pred, ddg_pred_inv, y, batch['complex'])
        self.scalar_accum.add(name='val_loss', value=val_loss, batchsize=self.batch_size, mode='mean')
        self.log("val_loss_step", val_loss, batch_size=self.batch_size, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)

        complex_group = batch.get('complex_group', batch['complex'])
        for complex, cg, y_true, y_pred in zip(batch['complex'], complex_group, batch['labels'], ddg_pred):
            result = {}
            result['complex'] = complex
            result['complex_group'] = str(cg)
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
        auroc_global = global_auroc(results, pred_attr='y_pred', true_attr='y_true')
        print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        print(f'[PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
        print(f'[Global_AUROC] AUROC {auroc_global:.6f}')
        
        self.log(f'val/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'val/global_auroc', auroc_global, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
    
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
        strategy = batch.get('strategy', 'separate')
        mut_out = self.model(batch, strategy, 'mutation')
        ddg_pred, ddg_pred_inv = self._predict_ddg(mut_out)
        test_loss = self._ddg_symmetric_loss(ddg_pred, ddg_pred_inv, y, batch['complex'])
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
            test_csv = getattr(self, "test_results_csv", "results_test.csv")
            results.to_csv(os.path.join(self.output_dir, test_csv), index=False)
        y_pred = np.array(results[f'y_pred'])
        y_true = np.array(results[f'y_true'])
        pearson_all = cal_pearson(y_pred, y_true)
        spearman_all = cal_spearman(y_pred, y_true)
        rmse_all = cal_rmse(y_pred, y_true)
        mae_all = cal_mae(y_pred, y_true)
        pearson_pc, spearman_pc, rmse_pc, mae_pc = per_complex_corr(results, pred_attr='y_pred', true_attr='y_true')
        auroc_global = global_auroc(results, pred_attr='y_pred', true_attr='y_true')
        print(f'[All_Task] Pearson {pearson_all:.6f} Spearman {spearman_all:.6f} RMSE {rmse_all:.6f} MAE {mae_all:.6f}')
        print(f'[PC_Task]  Pearson {pearson_pc:.6f} Spearman {spearman_pc:.6f} RMSE {rmse_pc:.6f} MAE {mae_pc:.6f}')
        print(f'[Global_AUROC] AUROC {auroc_global:.6f}')
        
        self.log(f'test/all_pearson', pearson_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_spearman', spearman_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_rmse', rmse_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/all_mae', mae_all, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_pearson', pearson_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_spearman', spearman_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_rmse', rmse_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/pc_mae', mae_pc, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.log(f'test/global_auroc', auroc_global, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        self.res = {"pearson": pearson_all,"spearman": spearman_all, "rmse": rmse_all, "mae": mae_all, 
                    "pc_pearson": pearson_pc,"pc_spearman": spearman_pc, "rmse_pc": rmse_pc, "mae_pc": mae_pc,
                    "global_auroc": auroc_global}
        print("Self.Res:", self.res)
        test_loss = self.scalar_accum.get_average('loss')
        self.log('test_loss', test_loss, batch_size=self.batch_size, on_epoch=True, sync_dist=True)
        
        return test_loss