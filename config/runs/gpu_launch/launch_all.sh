#!/bin/bash
set -e
cd /home/csd/lrg/copra_h

echo '[launch] pra201 on gpu1 -> /home/csd/lrg/copra_h/outputs/train_pra201_gpu1.log'
CUDA_VISIBLE_DEVICES=1 /home/csd/anaconda3/envs/copra_h/bin/python /home/csd/lrg/copra_h/run.py finetune dG --model_config /home/csd/lrg/copra_h/config/models/best_pra_B76.yml --data_config /home/csd/lrg/copra_h/config/datasets/PRA201.yml --run_config /home/csd/lrg/copra_h/config/runs/gpu_launch/pra201_gpu1.yml > /home/csd/lrg/copra_h/outputs/train_pra201_gpu1.log 2>&1 &
echo '[launch] pra310 on gpu2 -> /home/csd/lrg/copra_h/outputs/train_pra310_gpu2.log'
CUDA_VISIBLE_DEVICES=2 /home/csd/anaconda3/envs/copra_h/bin/python /home/csd/lrg/copra_h/run.py finetune dG --model_config /home/csd/lrg/copra_h/config/models/best_pra_B76.yml --data_config /home/csd/lrg/copra_h/config/datasets/PRA310.yml --run_config /home/csd/lrg/copra_h/config/runs/gpu_launch/pra310_gpu2.yml > /home/csd/lrg/copra_h/outputs/train_pra310_gpu2.log 2>&1 &
echo '[launch] pd304 on gpu3 -> /home/csd/lrg/copra_h/outputs/train_pd304_gpu3.log'
CUDA_VISIBLE_DEVICES=3 /home/csd/anaconda3/envs/copra_h/bin/python /home/csd/lrg/copra_h/run.py finetune dG --model_config /home/csd/lrg/copra_h/config/models/pd_dg_unified.yml --data_config /home/csd/lrg/copra_h/config/datasets/PD304_train.yml --run_config /home/csd/lrg/copra_h/config/runs/gpu_launch/pd304_gpu3.yml > /home/csd/lrg/copra_h/outputs/train_pd304_gpu3.log 2>&1 &
echo '[launch] skempi on gpu4 -> /home/csd/lrg/copra_h/outputs/train_skempi_gpu4.log'
CUDA_VISIBLE_DEVICES=4 /home/csd/anaconda3/envs/copra_h/bin/python /home/csd/lrg/copra_h/run.py finetune ddG --model_config /home/csd/lrg/copra_h/config/models/best_skempi.yml --data_config /home/csd/lrg/copra_h/config/datasets/SKEMPI.yml --run_config /home/csd/lrg/copra_h/config/runs/gpu_launch/skempi_gpu4.yml > /home/csd/lrg/copra_h/outputs/train_skempi_gpu4.log 2>&1 &
echo '[launch] mpd on gpu5 -> /home/csd/lrg/copra_h/outputs/train_mpd_gpu5.log'
CUDA_VISIBLE_DEVICES=5 /home/csd/anaconda3/envs/copra_h/bin/python /home/csd/lrg/copra_h/run.py finetune ddG --model_config /home/csd/lrg/copra_h/config/models/best_mpd.yml --data_config /home/csd/lrg/copra_h/config/datasets/MPD_merged.yml --run_config /home/csd/lrg/copra_h/config/runs/gpu_launch/mpd_gpu5.yml > /home/csd/lrg/copra_h/outputs/train_mpd_gpu5.log 2>&1 &
echo '[launch] mcsm on gpu6 -> /home/csd/lrg/copra_h/outputs/train_mcsm_gpu6.log'
CUDA_VISIBLE_DEVICES=6 /home/csd/anaconda3/envs/copra_h/bin/python /home/csd/lrg/copra_h/run.py finetune ddG --model_config /home/csd/lrg/copra_h/config/models/best_mcsm_B76_optC.yml --data_config /home/csd/lrg/copra_h/config/datasets/mCSM.yml --run_config /home/csd/lrg/copra_h/config/runs/gpu_launch/mcsm_gpu6.yml > /home/csd/lrg/copra_h/outputs/train_mcsm_gpu6.log 2>&1 &

echo 'All 6 trainings launched in background.'
sleep 3
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
