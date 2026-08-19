"""Generate B27 train-split predictions with the best checkpoint (frozen, eval, no noise).
Outputs a CSV (complex, complex_group, y_true, y_pred) in raw ddG units, used to fit a
nonlinear output calibrator that is then evaluated on the val split (no leakage).
"""
import sys, os
sys.path.insert(0, '/home/csd/lrg/copra_h')
import torch
import pandas as pd
from run import LightningRunner
from pl_modules import DataModule, DDGModule

CKPT = "/media/SSD0/csd/lrg/copra_h/outputs/MPD_merged_pdb_overlap/mpd_pdb_overlap_B27_2026-07-06-18-47-28/log_fold_0/checkpoint/best-epoch=98-val/all_pearson=0.714.ckpt"
OUT = "/home/csd/lrg/copra_h/scripts/b27_train_preds.csv"

runner = LightningRunner(
    model_config='config/models/best_mpd_B27.yml',
    data_config='config/datasets/MPD_all_train_B12.yml',
    run_config='config/runs/train_mpd_alltrain_B27.yml',
)
model_args = runner.model_args
data_args = runner.dataset_args
run_args = runner.run_args

dm_kwargs = dict(data_args)
dm_kwargs['col_group'] = data_args.col_group
dm = DataModule(dataset_args=data_args, **dm_kwargs)
dm.setup()
train_loader = dm.train_dataloader()

model = DDGModule(output_dir='/tmp', model_args=model_args, data_args=data_args, run_args=run_args)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
miss = model.load_state_dict(ckpt['state_dict'], strict=False)
print("load_state_dict missing:", len(miss.missing_keys), "unexpected:", len(miss.unexpected_keys))
model.eval()

rows = []
with torch.no_grad():
    for batch in train_loader:
        strategy = batch.get('strategy', 'separate')
        mut_out = model.model(batch, strategy, 'mutation')
        ddg_pred, _ = model._predict_ddg(mut_out)
        y = batch['labels']
        cg = batch.get('complex_group', batch['complex'])
        for c, g, yt, yp in zip(batch['complex'], cg, y, ddg_pred):
            rows.append({'complex': str(c), 'complex_group': str(g),
                         'y_true': float(yt), 'y_pred': float(yp)})
df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("Saved", OUT, "n=", len(df))
print("train pred: pearson=%.3f RMSE=%.3f" % (
    __import__('scipy.stats').pearsonr(df.y_pred, df.y_true)[0],
    ((df.y_pred - df.y_true) ** 2).mean() ** 0.5))
