import os
import yaml
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from datetime import datetime, timezone, timedelta

import torch
import torch.distributed
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

import models
import dataset


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return OmegaConf.create(config)

def save_config(cfg, save_dir):
    with open(f'{save_dir}/config.yaml', 'w') as file:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), file)
            
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

def train(args: argparse, cfg):
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
    args = parser.parse_args()
    
    # Set config
    cfg = load_config(Path(args.cfg_path))
    assert cfg.dataset.name in ['MVSEC', 'DSEC']
    args.data_dir = cfg.dataset.dir + cfg.dataset.name
    assert Path(args.data_dir).is_dir()
    assert cfg.dataset.params.representation in ['voxel', 'on_off', 'raw']
    
    # Set seed
    # seed = np.random.randint(0, 2**32 - 1)
    if cfg.seed is None: seed = 41
    set_random_seed(seed)
    
    # Update save_dir
    time_now = datetime.now(tz=timezone(timedelta(hours=9))).strftime("%Y%m%d-%H%M%S"),
    args.save_dir = args.save_dir + '/' + str(time_now[0]) + '/' + args.exp_name + '_' + str(seed)
    
    # Save scripts
    source_folder = '/root/code/scripts'
    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    shutil.copytree(source_folder, args.save_dir+'/scripts')
    
    # Save config
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    cfg.seed = seed
    save_config(cfg, args.save_dir)
    
    train(args, cfg)
    print('\n', '# Save dir: ', args.save_dir, '\n')