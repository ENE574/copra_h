# CoPRA Project Overview

## Purpose
CoPRA predicts protein–RNA binding affinity by combining pretrained protein and RNA language models with structure-aware inputs. The repository includes instructions for installing dependencies such as PyTorch Lightning, flash-attn, and RiNALMo, and provides Hugging Face links for PRA and PRI datasets plus pretrained checkpoints for evaluation or fine-tuning.

## Training and Evaluation Entry Points
- **`run.py`** exposes a `LightningRunner` with `finetune` and `test` commands (via `fire.Fire`). It loads model, data, and runtime YAML configurations, initializes multiprocessing and CUDA defaults, and orchestrates k-fold training with `pytorch_lightning.Trainer` callbacks for early stopping and checkpointing.
- Stage selection determines which Lightning module runs:
  - `PretuneModule` for bi-scope pretraining on PRI30k.
  - `ModelModule` for dG regression tasks (e.g., PRA310/PRA201 finetuning or testing).
  - `DDGModule` for mutation effect (ddG) evaluation.

## Data Handling
- **`pl_modules/data_module.py`** provides a `DataModule` wrapper that reads fold annotations from CSV splits, constructs train/val/test subsets, and chooses dataset classes through `data.DataRegister`.
- Collation is strategy-aware: sequence batches use `CustomSeqCollate`, structure batches use `CustomStructCollate`, and PRI30k structural batches use `PRI30kStructCollate`. Graph datasets fall back to a `torch_geometric` `DataLoader`.

## Configuration Files
- Model, dataset, and run hyperparameters live under `config/models/`, `config/datasets/`, and `config/runs/` respectively. Typical invocations pass these YAML files directly to `run.py`, for example the README recipes for PRA310 finetuning or zero-shot mutation testing.

## Repository Layout Highlights
- `pl_modules/`: Lightning modules for each training stage plus the shared `DataModule` wrapper.
- `data/`: Dataset implementations, protein/RNA utilities, and transform helpers registered via `DataRegister`.
- `assets/`: Figures summarizing model architecture and benchmark results included in the README.
- `models/`, `utils/`, and `weights/` (when populated): supporting model components, preprocessing utilities, and external checkpoints referenced by the training scripts.
