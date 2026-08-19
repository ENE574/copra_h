import torch
import torch.nn as nn
import os
import sys
sys.path.insert(0, '/home/csd/lrg/copra_h')

from models.copra_offline12 import CopraOffline12

print("Testing model forward pass with dummy data...")

model = CopraOffline12(
    d_model=512,
    seq_prot_models=['esm2'],
    str_prot_models=['esm_if1'],
    seq_rna_models=['rna_fm'],
    str_rna_models=['rhofold'],
    use_diff_pred=True
)

print(f"\nModel parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

print("\n=== ModuleDict keys ===")
print("pre_lns keys:", list(model.pre_lns.keys()))
print("projs keys:", list(model.projs.keys()))
print("fusion_q keys:", list(model.fusion_q.keys()))

batch_size = 2
prot_seq_len = 100
rna_seq_len = 50

dummy_batch = {
    'offline_embeddings': {
        'prot_seq': {'esm2': torch.randn(batch_size, prot_seq_len, 1280)},
        'prot_struct': {'esm_if1': torch.randn(batch_size, prot_seq_len, 512)},
        'rna_seq': {'rna_fm': torch.randn(batch_size, rna_seq_len, 640)},
        'rna_struct': {'rhofold': torch.randn(batch_size, rna_seq_len, 384)},
    },
    'offline_embeddings_mut': {
        'prot_seq': {'esm2': torch.randn(batch_size, prot_seq_len, 1280)},
        'prot_struct': {'esm_if1': torch.randn(batch_size, prot_seq_len, 512)},
        'rna_seq': {'rna_fm': torch.randn(batch_size, rna_seq_len, 640)},
        'rna_struct': {'rhofold': torch.randn(batch_size, rna_seq_len, 384)},
    },
    'labels': torch.tensor([-0.5, 1.2]),
    'complex': ['test1', 'test2']
}

print("\n=== Testing forward pass ===")
model.train()
output = model(dummy_batch)
print("Output keys:", list(output.keys()))
print("ddg_pred shape:", output['ddg_pred'].shape)
print("ddg_pred:", output['ddg_pred'])
print("ddg_pred_inv:", output['ddg_pred_inv'])
print("pooled_mut shape:", output['pooled_mut'].shape)
print("pooled_mut norm:", output['pooled_mut'].norm(dim=-1))

print("\n=== Testing gradient flow ===")
model.zero_grad()
loss = output['ddg_pred'].abs().mean()
loss.backward()

print("\nGradient norms:")
for name, param in model.named_parameters():
    if param.requires_grad and param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm > 0:
            print(f"  {name}: {grad_norm:.6f}")

print("\nTest complete!")