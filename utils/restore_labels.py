"""Per-PDB label restoration utilities.

After training/evaluation with per-PDB standardized labels, 
use these functions to restore predictions to original scale.
"""
import pandas as pd
import numpy as np
import os


def load_pdb_stats(stats_path=None):
    """Load per-PDB mean/std statistics."""
    if stats_path is None:
        stats_path = '/media/SSD0/csd/lrg/copra_h/datasets/SKEMPI/splits/skempi_pdb_stats.csv'
    stats = pd.read_csv(stats_path, index_col=0)
    return stats['mean'].to_dict(), stats['std'].to_dict()


def restore_labels_from_csv(pred_csv_path, output_csv_path=None):
    """Restore per-PDB standardized predictions to original scale."""
    df = pd.read_csv(pred_csv_path)
    means, stds = load_pdb_stats()
    
    # Check if 'pdb' column exists
    pdb_col = None
    for col in ['pdb', 'PDB', 'complex']:
        if col in df.columns:
            pdb_col = col
            break
    
    if pdb_col is None:
        print("Warning: No PDB column found in predictions CSV. Cannot restore labels.")
        return df
    
    for idx, row in df.iterrows():
        pdb = str(row[pdb_col])
        # Try to extract PDB from complex name if needed (e.g., '1ACB_LI38G' -> '1ACB')
        if pdb not in means and '_' in pdb:
            pdb = pdb.split('_')[0]
        
        if pdb in means:
            df.at[idx, 'y_true'] = row['y_true'] * stds[pdb] + means[pdb]
            df.at[idx, 'y_pred'] = row['y_pred'] * stds[pdb] + means[pdb]
    
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
    
    print(f"Restored labels for {len(df)} samples using {len(means)} PDB stats.")
    return df


def compute_metrics_from_csv(csv_path, pdb_col='pdb'):
    """Compute all metrics from a predictions CSV (with restored labels)."""
    df = pd.read_csv(csv_path)
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import roc_auc_score
    
    # Global metrics
    p, _ = pearsonr(y_true, y_pred)
    s, _ = spearmanr(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mae = np.mean(np.abs(y_true - y_pred))
    
    # AUROC
    y_binary = (y_true > 0).astype(int)
    try:
        auroc = roc_auc_score(y_binary, y_pred)
    except:
        auroc = float('nan')
    
    # Per-complex metrics (if pdb_col exists)
    if pdb_col in df.columns:
        pc_results = []
        for pdb, group in df.groupby(pdb_col):
            gt = group['y_true'].values
            pr = group['y_pred'].values
            if len(gt) >= 3:
                pc_p, _ = pearsonr(gt, pr)
                pc_s, _ = spearmanr(gt, pr)
                pc_rmse = np.sqrt(np.mean((gt - pr)**2))
                pc_mae = np.mean(np.abs(gt - pr))
                pc_results.append({'pc_pearson': abs(pc_p), 'pc_spearman': abs(pc_s), 
                                  'pc_rmse': pc_rmse, 'pc_mae': pc_mae})
        
        pc_df = pd.DataFrame(pc_results)
        print(f"\nPer-complex metrics ({len(pc_df)} complexes with >=3 mutations):")
        for m in ['pc_pearson', 'pc_spearman', 'pc_rmse', 'pc_mae']:
            print(f"  {m}: {pc_df[m].mean():.4f}")
    
    print(f"\nGlobal metrics:")
    print(f"  all_pearson: {abs(p):.4f}")
    print(f"  all_spearman: {abs(s):.4f}")
    print(f"  all_rmse: {rmse:.4f}")
    print(f"  all_mae: {mae:.4f}")
    print(f"  AUROC: {auroc:.4f}")
    
    return {
        'all_pearson': abs(p), 'all_spearman': abs(s),
        'all_rmse': rmse, 'all_mae': mae, 'AUROC': auroc
    }
