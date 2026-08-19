import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, '/home/csd/lrg/copra_h')

from models.copra_offline12 import CopraOffline12
from pl_modules.ddg_module import DDGModule
from data.datasets import StructureDDGDataset

print("Testing data loading and model forward...")

dataset = StructureDDGDataset(
    df_path='/media/SSD0/csd/lrg/copra_h/datasets/mCSM_RNA/splits/crossvalidation.csv',
    data_root='/media/SSD0/csd/lrg/copra_h/datasets/mCSM_RNA/PDBs',
    batch_size=2,
    use_offline_embeddings=True,
    offline_embedding_root='/media/SSD0/csd/lrg/copra_h/outputs/feature_extraction_mCSM_RNA',
    offline_mutant_subdir='mutant',
    entity_b_type='rna',
    mutation_task='ddg_prot_na'
)

dataloader = dataset.train_dataloader()
print(f"Train dataloader length: {len(dataloader)}")

for batch in dataloader:
    print("\n=== Batch structure ===")
    print("Keys:", list(batch.keys()))
    
    if 'offline_embeddings' in batch:
        emb = batch['offline_embeddings']
        print("offline_embeddings keys:", list(emb.keys()))
        for k, v in emb.items():
            if isinstance(v, dict):
                for mk, mv in v.items():
                    print(f"  {k}/{mk}: {mv.shape}")
            else:
                print(f"  {k}: {v.shape}")
    
    if 'offline_embeddings_mut' in batch:
        emb_mut = batch['offline_embeddings_mut']
        print("offline_embeddings_mut keys:", list(emb_mut.keys()))
    
    if 'labels' in batch:
        print("labels:", batch['labels'].shape, batch['labels'][:5])
    
    break

print("\n=== Model test ===")
model = CopraOffline12(
    d_model=512,
    seq_prot_models=['esm2'],
    str_prot_models=['esm_if1'],
    seq_rna_models=['rna_fm'],
    str_rna_models=['rhofold'],
    use_diff_pred=True
)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

model.eval()
with torch.no_grad():
    output = model(batch)
    print("Output keys:", list(output.keys()))
    if 'ddg_pred' in output:
        print("ddg_pred shape:", output['ddg_pred'].shape)
        print("ddg_pred:", output['ddg_pred'])

print("\nTest complete!")