# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from datasets import build_dataset
from engine import train_one_epoch, evaluate
from losses import DistillationLoss
from samplers import RASampler
from augment import new_data_aug_generator

import models
import models_v2

import utils

from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy
    
def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    
    # Model parameters
    parser.add_argument('--model', default='deit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
           
    parser.add_argument('--src', action='store_true') #simple random crop
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
  
    # Dataset parameters
    parser.add_argument('--data-path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data-set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--inat-category', default='name',
                        choices=['kingdom', 'phylum', 'class', 'order', 'supercategory', 'family', 'genus', 'name'],
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    parser.add_argument('--mask_donor', default='')
    parser.add_argument('--mask_receiver', default = '')
    parser.add_argument('--mask_resume', action='store_true')
    parser.add_argument('--donor_classes', default=1000, type=int)
    parser.add_argument('--receiver_classes', default=100, type=int)
    
    return parser

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)
#     device = torch.device("cpu")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True

    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.receiver_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )         
    model.to(device)
    
    print(f"Creating donor model: {args.model}")
    model2 = create_model(
        args.model,
        pretrained=False,
        num_classes=args.donor_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )         
    model2.to(device)
    
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume='')

    model_without_ddp = model
    model_without_ddp2 = model2
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        model2 = torch.nn.parallel.DistributedDataParallel(model2, device_ids=[args.gpu])
        model_without_ddp2 = model2.module

    # preparation
    for name, mod in model.named_modules():
        if(hasattr(mod, 'weight') and name != 'module.head'):
            prune.identity(mod, 'weight')
    for name, mod in model2.named_modules():
        if(hasattr(mod, 'weight') and name != 'head'):
            prune.identity(mod, 'weight')
    checkpoint = torch.load(args.mask_receiver, map_location='cpu')
    model_without_ddp.load_state_dict(checkpoint['model'])
    donor_checkpoint = torch.load(args.mask_donor, map_location='cpu')
    model_without_ddp2.load_state_dict(donor_checkpoint['model'])

    # preparation stage 2
    for name, mod in model.named_modules():
        if(hasattr(mod, 'weight') and name != 'head'):
            prune.identity(mod, 'weight')
    for name, mod in model2.named_modules():
        if(hasattr(mod, 'weight') and name != 'head'):
            prune.identity(mod, 'weight')
    total_zero = 0
    total = 0
    for name, mod in model.named_modules():
        if(hasattr(mod, 'weight') and name != 'module.head' and name != 'head'):
            total_zero += float(torch.sum(mod.weight == 0))
            total += float(mod.weight.nelement())
    print("Sparsity 1: {:.2f}%".format(100.*float(total_zero)/float(total)))
    total_zero = 0
    total = 0
    for name, mod in model2.named_modules():
        if(hasattr(mod, 'weight') and name != 'module.head' and name != 'head'):
            total_zero += float(torch.sum(mod.weight == 0))
            total += float(mod.weight.nelement())
    print("Sparsity 2: {:.2f}%".format(100.*float(total_zero)/float(total)))
    similar = 0
    total = 0
    for (name1, mod1), (name2, mod2) in zip(model.named_modules(), model2.named_modules()):
        if(hasattr(mod1, 'weight') and name1 != 'head' and name1 != 'module.head'):
            similar += float(torch.sum(torch.eq(mod1.weight, mod2.weight)))
            total += float(mod.weight.nelement())
    print("Shared Sparsity: {:.2f}%".format(100. * float(similar)/float(total)))
    print(similar)
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
