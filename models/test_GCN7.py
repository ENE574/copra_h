# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:37:09 2022

@author: lllli
"""
import os
import time
import json
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

from timm.optim import create_optimizer_v2, optimizer_kwargs

import networkx as nx
import torch_geometric
import matplotlib.pyplot as plt
#from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
import numpy as np
#from sk_metrics import calculate_metrics

#from create_dataset_1 import AA_Dataset
from create_dataset_1217 import AA_Dataset_con, AA_Dataset_MT

#from model_GCN_ import Net2azalea5 as Net
from model_GCN_ import Net2lily5 as Net
#from model_GCN_ import Net2lily6 as Net

from loss_def import mse_weighted, mse_w_rect
from lr_scheduler import LR_Scheduler


#############
import argparse
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial


import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP


from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

numworkersgiven = 0
batchsizegiven = 48
batchsizegiventest = 48
epochsgiven = 300

save_num = '5_1'
save_path = '00_results/' + save_num + '/'
saveornot = 'yes' 
try:
    os.makedirs(save_path)
    print(f"文件夹已创建: {save_path}")

except OSError as e:
    print(f"创建文件夹失败: {e}")

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                   metavar='N N N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
group.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='lamb', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "adamw")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "step"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                   help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=0, metavar='S',
                   help='random seed (default: 0)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--save-images', action='store_true', default=False,
                   help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                   help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                   help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text




torch.backends.cudnn.benchmark = True


torch.manual_seed(0)



def root_mean_squared_error(true, pred):  
    squared_error = np.square(true - pred)   
    sum_squared_error = np.sum(squared_error)  
    rmse_loss = np.sqrt(sum_squared_error / true.size)  
    return rmse_loss  


def split_and_get_num(input_list):
    result_list = []
    for item in input_list:
        parts = item.split('__')
        result_list.append(int(parts[1])-1) #样本编号是从1开始， 而dataset的索引是从0开始，故减一
    return result_list


def save_txt_file(results, save_path, output_filename):
    with open(output_filename, 'w') as txt:
        for ls in results:
            for line in ls:
                txt.write(f"{line}")

    




# 读取保存有数据集划分json文件
data_dir_split = '../data_MpbPPI/'
set_name = 'S4169'
# data_split_mode: 'CV10_random'/'complex'
data_split_mode = 'CV10_random'
# seed for retrieving corresponding downstream dataset splitting
splitting_seed = 256
# indicate the MT PPI complex source for retrieving corresponding data source file
mutation_source = '_foldx'

with open(data_dir_split + f'{set_name}_{data_split_mode}_data_split_{splitting_seed}.jsonl') as f:
    splits = f.readlines()
split_list = []
for split in splits:
    split_list.append(json.loads(split))



results = []
best_results = []
best_matrices = []

    
# 数据集
dataset_WT = AA_Dataset_con(root = 'data_1217/data_per_mutation_con/')
dataset_MT = AA_Dataset_MT(root = 'data_1217/data_per_mutation_MT/')
#dataset_WT = AA_Dataset(root = 'data_per_mutation_test/')
#dataset_MT = AA_Dataset(root = 'data_per_mutation_test/')


#for fold in range(1, 2):
for fold in range(len(split_list)):
    # 按fold导入数据集
    print('current fold:', fold + 1)
    dataset_splits = split_list[fold]
    train_list, val_list = dataset_splits['train'], dataset_splits['val']
    train_num_ls, val_num_ls = split_and_get_num(train_list), split_and_get_num(val_list) 
    
    train_dataset_WT, test_dataset_WT = dataset_WT[train_num_ls], dataset_WT[val_num_ls]
    train_dataset_MT, test_dataset_MT = dataset_MT[train_num_ls], dataset_MT[val_num_ls]

    best_pcc = 0
    ######################### 定义训练的设备############    
    args, args_text = _parse_args()
    
    batch_size_given = batchsizegiven
    batch_size_given_test = batchsizegiventest
    args.lr =  3.75e-04 * batch_size_given/32.
    args.weight_decay = 0.05 #0.01 (0.01 is okay)
    
    num_workers_given = numworkersgiven
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    # model def
    model = Net().to(device)
    #model.load_state_dict(torch.load('model_121.pth'))
    learning_rate = args.lr
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-04)
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
    #optimizer = torch.optim.lamb(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.05)
    #loss_f = torch.nn.CrossEntropyLoss()
    
    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    #optimizer.load_state_dict(torch.load('opt_121.pth'))
    #loss_f = mse_w_rect() #
    loss_f = torch.nn.MSELoss() #mse_w_rect()
    
    # 设置训练网络的一些参数
    # 记录训练/测试的次数
    total_train_step, total_test_step = 0, 0
    # 训练的轮数
    warm = 20
    #epochs = 300 # 240
    start_epoch = 0
    epochs = epochsgiven
    epoch_results = ''
    
    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        # random list for shuffled
        train_list = [i for i in range(len(train_dataset_WT))]
        train_list_shuffled = random.sample(train_list, len(train_list))
        trainset_wt = train_dataset_WT[train_list_shuffled]
        trainset_mt = train_dataset_MT[train_list_shuffled]
        # testset 不shuffle
        #test_list = [i for i in range(len(test_dataset_WT))]
        #test_list_shuffled = random.sample(test_list, len(test_list))
        testset_wt = test_dataset_WT#[test_list_shuffled]
        testset_mt = test_dataset_MT#[test_list_shuffled]
        
        ##################### 利用 DataLoader 来加载数据集 #############
        trainbatchsize, testbatchsize = batch_size_given, batch_size_given_test #32, 32
        train_loader_wt = DataLoader(trainset_wt, batch_size = trainbatchsize, shuffle=False, num_workers= num_workers_given)
        test_loader_wt  = DataLoader(testset_wt,  batch_size = testbatchsize,  shuffle=False, num_workers= num_workers_given)
        
        train_loader_mt = DataLoader(trainset_mt, batch_size = trainbatchsize, shuffle=False, num_workers= num_workers_given)
        test_loader_mt  = DataLoader(testset_mt,  batch_size = testbatchsize,  shuffle=False, num_workers= num_workers_given)
    
        scheduler = LR_Scheduler('cos', learning_rate, epochs, len(train_loader_wt), warmup_epochs=warm)####cos学习率###
        print("-------Fold {}---Epoch {} -------".format(fold + 1, epoch + 1))
    
        # 训练步骤开始
        model.train()
        y_pred_train, y_real_train = [], []
        counter = 0###cos学习率###    
        
        # batch train
        for batch_train_wt,batch_train_mt in zip(train_loader_wt, train_loader_mt):
            
            batch_real_train = batch_train_wt.y.numpy()
            y_real_train.append(batch_train_wt.y) # 保存每个 batch 的groundtruth
            
            batch_train_wt, batch_train_mt = batch_train_wt.to(device), batch_train_mt.to(device)
            #optimizer.zero_grad()
    
            output_TR = model(batch_train_wt, batch_train_mt)#训练数据送入网络中
            label = batch_train_wt.y.to(device)
            #print(output_TR, label)
            # tanh
            #output_TR = torch.tanh(output_TR)
            #label = torch.tanh(label)
            
            loss = loss_f(output_TR, label) 
            
            scheduler(optimizer, counter, epoch)###cos学习率###
            print('lr:', optimizer.param_groups[-1]['lr'])
            counter = counter + 1
            
            # 优化器优化模型
            optimizer.zero_grad()##先清零上一步的梯度
            loss.backward()#反向传播
            optimizer.step()#优化器更新梯度
    
            total_train_step = total_train_step + 1
            if total_train_step % 10 == 0:##逢10的时候打印
                end_time = time.time()
                print("Fold({}), Epoch({}), Time:{:.2f}, 训练次数：{}, Loss: {:.2f}".format((fold+1), (epoch+1), (end_time - start_time), total_train_step, loss.item()))
                #loss.item()打印tensor数据会显示tensor，加上item()再打印就是打印数值
            
            # 评价指标每个batch
            batch_pred_train = output_TR.cpu().detach().numpy()
            y_pred_train.append(batch_pred_train)# 保存每个 batch 的预测值
    
            MSE_batch_tr  = mean_squared_error(batch_real_train, batch_pred_train)
            RMSE_batch_tr = np.sqrt(MSE_batch_tr)
            MAE_batch_tr  = mean_absolute_error(batch_real_train, batch_pred_train)
            #RMSE_batch_tr = root_mean_squared_error(batch_real_train, batch_pred_train)
            pccs_batch_tr = pearsonr(batch_real_train.reshape(-1), batch_pred_train.reshape(-1))[0]
            print(f"batch training--MSE: {MSE_batch_tr:.4f}, RMSE: {RMSE_batch_tr:.4f}, MAE: {MAE_batch_tr:.4f}, Pearson: {pccs_batch_tr:.4f}")
            
        # 评价指标
        all_real_tr = np.concatenate(y_real_train)# 将每个 batch 的预测值合并为一个数组
        all_pred_tr = np.concatenate(y_pred_train)# 将每个 batch 的groundtruth合并为一个数组
        
        #RMSE_tr = root_mean_squared_error(all_real_tr, all_pred_tr)
        MSE_tr  = mean_squared_error(all_real_tr, all_pred_tr)
        RMSE_tr = np.sqrt(MSE_tr)
        MAE_tr  = mean_absolute_error(all_real_tr, all_pred_tr)
        pccs_tr = pearsonr(all_real_tr, all_pred_tr)[0]
        epoch_result_train = f'MSE: {MSE_tr:.4f}, RMSE: {RMSE_tr:.4f}, MAE: {MAE_tr:.4f}, Pearson: {pccs_tr:.4f}'
        print("training:", epoch_result_train)
        
        # 测试步骤开始
        model.eval()
        y_pred_test, y_real_test = [], []

        with torch.no_grad():####不再调整梯度 这一步必须
            for batch_test_wt, batch_test_mt in zip(test_loader_wt, test_loader_mt):
                batch_real_test = batch_test_wt.y.numpy()
                y_real_test.append(batch_test_wt.y)
                
                batch_test_wt = batch_test_wt.to(device)
                batch_test_mt = batch_test_mt.to(device)
                
                outputs = model(batch_test_wt, batch_test_mt)
                batch_test_y = batch_test_wt.y.to(device)
    
                # 评价指标每个batch
                batch_pred_test = outputs.cpu().detach().numpy()
                y_pred_test.append(batch_pred_test)

                MSE_batch_tt  = mean_squared_error(batch_real_test, batch_pred_test)
                RMSE_batch_tt = np.sqrt(MSE_batch_tt)
                MAE_batch_tt  = mean_absolute_error(batch_real_test, batch_pred_test)
                #RMSE_batch_tt = root_mean_squared_error(batch_real_test, batch_pred_test)
                pccs_batch_tt = pearsonr(batch_real_test, batch_pred_test)[0]
                print(f"batch test--MSE: {MSE_batch_tt:.4f}, RMSE: {RMSE_batch_tt:.4f}, MAE: {MAE_batch_tt:.4f}, Pearson: {pccs_batch_tt:.4f}")
    
        # 评价指标
        all_real_tt = np.concatenate(y_real_test)# 将每个 batch 的预测值合并为一个数组
        all_pred_tt = np.concatenate(y_pred_test)# 将每个 batch 的groundtruth合并为一个数组
        
        MSE_tt  = mean_squared_error(all_real_tt, all_pred_tt)
        RMSE_tt = np.sqrt(MSE_tt)
        MAE_tt  = mean_absolute_error(all_real_tt, all_pred_tt)
        #RMSE_tt = root_mean_squared_error(all_real_tt, all_pred_tt)
        pccs_tt = pearsonr(all_real_tt.reshape(-1), all_pred_tt.reshape(-1))[0]
        epoch_result_test = f'MSE: {MSE_tt:.4f}, RMSE: {RMSE_tt:.4f}, MAE: {MAE_tt:.4f}, Pearson: {pccs_tt:.4f}'
        print("test:", epoch_result_test)  
        
        
        # for best model save
        pcc = pccs_tt
        if pcc > best_pcc:
            print(f'Best model...Fold:{fold + 1}, Epoch:{epoch +1}')
            best_pcc, bmse, brmse, bmae= pcc, MSE_tt, RMSE_tt, MAE_tt
            best_epoch = (epoch + 1)
            best_model = model.state_dict()
            if saveornot == 'yes':
                print("Saving best model...")
                torch.save(best_model, save_path + 'fold_{}_bestmodel.pth'.format(fold + 1))                
        
        best_epoch_result = f'Best epoch:{best_epoch}--test--MSE: {bmse:.4f}, RMSE: {brmse:.4f}, MAE: {bmae:.4f}, Pearson: {best_pcc:.4f}'
        print(best_epoch_result)
        # for txt result save
        epoch_result = f'Epoch:{epoch+1}--train:' + epoch_result_train + '--test:'+epoch_result_test
        epoch_results = epoch_results + epoch_result + '\n'
    
    # epochs结束 保存

        
    #best_epoch_result = f'Best epoch:{best_epoch}--test--MSE: {bmse:.4f}, RMSE: {brmse:.4f}, MAE: {bmae:.4f}, Pearson: {best_pcc:.4f}'
    info = f'--- Fold:{fold+1} ---'
    parameter = 'lr:{}, warm:{}, epochs:{}, trainbatchsize:{}, testbatchsize:{}'.format(args.lr, warm, epochs, trainbatchsize, testbatchsize)
    fold_result = info + '\n' + parameter + '\n' + best_epoch_result + '\n' + epoch_results
    # 保存环节
    if saveornot == 'yes':
        save_txt_file(fold_result, save_path, save_path + '{}_result_{}.txt'.format(save_num, fold+1))
    results.append(fold_result)
    best_result = info + '\n' + parameter + '\n' + best_epoch_result + '\n'
    best_results.append(best_result)
    best_matrices.append([bmse, brmse, bmae, best_pcc])
    
    
# 各fold的均值
best_matrices = np.array(best_matrices)
best_matrices_mean = np.mean(best_matrices, axis = 0) # 按列求平均
best_matrices_mean_result = f'--- Mean result of {len(best_matrices)} folds:---\ntest--MSE: {best_matrices_mean[0]:.4f}, RMSE: {best_matrices_mean[1]:.4f},\
 MAE: {best_matrices_mean[2]:.4f}, Pearson: {best_matrices_mean[3]:.4f}'
print(best_results)
print(best_matrices_mean_result)
# 保存环节
if saveornot == 'yes':
    results.append(best_results)
    results.append(best_matrices_mean_result)
    save_txt_file(results, save_path, save_path + '{}_results.txt'.format(save_num))
    
    results_mean = []
    results_mean.append(best_results)
    results_mean.append(best_matrices_mean_result)
    save_txt_file(results_mean, save_path, save_path + '{}_results_mean.txt'.format(save_num))

'''    
for epoch in range(100):
#for epoch in range(1):
    tmp = train()
    if epoch % 20 == 0:
        print(tmp) 

# 测试
from  sklearn.metrics import accuracy_score
def evaluate(loader):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data).detach().cpu().numpy()
            label = data.y.detach().cpu().numpy()        
    return pred , label

loader = DataLoader(test_dataset, batch_size=20, shuffle=False)
pred , label = evaluate(loader)
print(pred.shape , label.shape)

preds = []
# list_a.index(max(list_a)) 
for i in range(pred.shape[0]):
    tmp = pred[i].tolist()
    preds.append(tmp.index(max(tmp)))
len(preds)

print(accuracy_score(label, preds))
'''