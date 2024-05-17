import os
import sys
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from yacs.config import CfgNode as CN
from datetime import datetime, timezone, timedelta

import torch
import torch.distributed
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import models
import dataset


def get_cfg(cfg_path):
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    return cfg

def save_cfg_to_txt(cfg, save_dir):
    with open(f"{save_dir}/config.yaml", "w") as f:
        for key, value in cfg.items():
            f.write(f"{key}: {value}\n")
            
def set_random_seed(seed):
    # Python
    random.seed(seed)
    
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Numpy
    np.random.seed(seed)

def train(args: argparse, cfg: CN):
    print(f'{args.local_rank}: {torch.cuda.get_device_name()} | {args.device}')
    torch.set_float32_matmul_precision(cfg.gpu_precision)
    
    # Prepare dataset
    dataset_provider = getattr(dataset, cfg.dataset.name).dataset_provider.DatasetProvider(data_dir=args.data_dir, **cfg.dataset.params)
    
    # Prepare model
    stereo_depth_model = getattr(models, cfg.model.name)(cfg.model, dataset=dataset_provider)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True)
    
    # Logger
    logger = TensorBoardLogger(save_dir=args.save_dir+'/logs/')
    
    # Prepare trainer
    trainer = pl.Trainer(**cfg.trainer.params, 
                         callbacks=[lr_monitor], 
                         logger=logger)
    trainer.fit(model=stereo_depth_model)
    trainer.test(model=stereo_depth_model)
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='/root/code/configs/config.yaml', help='Path to dataset details')
    parser.add_argument('--save_dir', default='/root/code/save', type=str, help='Path to save result')
    parser.add_argument('--exp_name', default='main', type=str, help='Experiment name')
    parser.add_argument('--checkpoint', action='store_true', help='Continue learning from saved checkpoint')
    DEFAULT_LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    parser.add_argument('--local_rank', type=int, default=DEFAULT_LOCAL_RANK)
    args = parser.parse_args()
    
    # Set config
    cfg = get_cfg(Path(args.cfg_path))
    assert cfg.dataset.name in ['MVSEC', 'DSEC']
    args.data_dir = cfg.dataset.dir + cfg.dataset.name
    assert Path(args.data_dir).is_dir()
    assert cfg.dataset.params.representation in ['voxel', 'on_off', 'raw']
    
    # Set seed
    # seed = np.random.randint(0, 2**32 - 1)
    seed = 41
    set_random_seed(seed)
    
    try:
        assert int(os.environ['WORLD_SIZE']) >= 1
    except:
        print('\n####################################################')
        print('### Excute launch/~~~~.sh file, not ~~~~.py file ###')
        print('###      Enter terminal command like below       ###')
        print('###          $ cd launch/ && . ~~~~.sh           ###')
        print('####################################################\n')
        sys.exit()
        
    if int(os.environ['WORLD_SIZE']) > 1:
        # Multi GPUs
        print('Use Multi GPUs')
        args.is_distributed = True
    else:
        # Or only one GPU
        print('Use Single GPU')
        args.is_distributed = False
        
    # Device config
    args.is_master   = args.local_rank == 0
    args.device      = 'cuda:%d' % args.local_rank
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size  = torch.distributed.get_world_size()
    args.rank        = torch.distributed.get_rank()
    args.num_workers = 4 * args.world_size
    time_now = datetime.now(tz=timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S"),
    args.save_dir = args.save_dir + '/' + str(time_now[0]) + '/' + args.exp_name + '_' + str(seed)
    
    if args.is_master:
        # Save scripts
        source_folder = '/root/code/scripts'
        if os.path.exists(args.save_dir):
            shutil.rmtree(args.save_dir)
        shutil.copytree(source_folder, args.save_dir+'/scripts')
        
        # Save config
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        cfg.defrost()
        cfg.seed = seed
        cfg.freeze()
        save_cfg_to_txt(cfg, args.save_dir)
    
    train(args, cfg)
    print('\n', '# Save dir: ', args.save_dir, '\n')