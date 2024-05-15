import time
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lightning as pl

# from .concentration import ConcentrationNet
from .rose import RoSE
from .stereo_matching import StereoMatchingNetwork
from utils.metrics import *


class StereoDepthLightningModule(pl.LightningModule):
    def __init__(self, config, dataset=None):
        super(StereoDepthLightningModule, self).__init__()
        self.cfg = config
        
        self.rose = RoSE(**config.event_processor.params)
        self.stereo_matching_net = StereoMatchingNetwork(**config.disparity_estimator.params)
        self.max_disp = config.disparity_estimator.params.max_disp
        
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        
        self.dataset = dataset
        
    def forward(self, stereo_event, stereo_image):
        rose_img = {}
        ei_frame = {}
        for loc in ['left', 'right']:
            # event_stack[loc] = rearrange(event_stack[loc], 'b c h w t s -> b (c s t) h w')
            rose_img[loc] = self.rose(stereo_event[loc])
            ei_frame[loc] = torch.cat([rose_img[loc], stereo_image[loc]], dim=1)
            
        pred_disparity_pyramid = self.stereo_matching_net(
            ei_frame['left'],
            ei_frame['right']
        )
        
        return pred_disparity_pyramid
    
    def train_dataloader(self):
        train_dataset = self.dataset.get_train_dataset()
        train_loader = DataLoader(dataset=train_dataset, **self.cfg.dataloader.train.params)
        return train_loader
        
    def val_dataloader(self):
        valid_dataset = self.dataset.get_valid_dataset()
        valid_loader = DataLoader(dataset=valid_dataset, **self.cfg.dataloader.validation.params)
        return valid_loader
        
    def test_dataloader(self):
        test_dataset = self.dataset.get_test_dataset()
        test_loader = DataLoader(dataset=test_dataset, **self.cfg.dataloader.test.params)
        return test_loader
        
    def training_step(self, batch, batch_idx):
        stereo_event = batch['event'] if 'event' in batch else None     # ['left', 'right'], [N, C, H, W]
        stereo_image = batch['image'] if 'image' in batch else None     # ['left', 'right'], [N, C, H, W]
        disparity_gt = batch['disparity_gt'] if 'disparity_gt' in batch else None   # [N, H, W]
        file_index = batch['file_index'] if 'file_index' in batch else None         # List, length=batch_size
        # seq_name = batch['seq_name'] if 'seq_name' in batch else None
        batch_size = len(file_index)
        
        # Forward pass
        pred_disparity_pyramid = self.forward(stereo_event, stereo_image)   # List: [[N, H, W], ...]
        
        # Calculate loss
        loss_disp = self._cal_loss(pred_disparity_pyramid, disparity_gt)
        loss = loss_disp.mean()
        
        # Calculate metrics
        pred = pred_disparity_pyramid[-1].detach()
        gt   = disparity_gt.detach()
        mask = (gt > 0) & (gt < self.max_disp)
        train_mde = train_mdise = train_1pa = 0.
        for p, g, m in zip(pred, gt, mask):
            p, g = p[m], g[m]
            train_mde += mean_depth_error(p, g).item()
            train_mdise += mean_disparity_error(p, g).item()
            train_1pa += n_pixel_accuracy(p, g, 1).item()
        train_mde, train_mdise, train_1pa = train_mde/batch_size, train_mdise/batch_size, train_1pa/batch_size
        
        # Train log dict
        self.log_dict(
            {'train/loss': loss, 'train/mde': train_mde, 'train/mdise': train_mdise, 'train/1pa': train_1pa}, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True, 
            batch_size=batch_size, 
            sync_dist=True, 
        )
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        stereo_event = batch['event'] if 'event' in batch else None     # ['left', 'right'], [N, C, H, W]
        stereo_image = batch['image'] if 'image' in batch else None     # ['left', 'right'], [N, C, H, W]
        disparity_gt = batch['disparity_gt'] if 'disparity_gt' in batch else None   # [N, H, W]
        file_index = batch['file_index'] if 'file_index' in batch else None
        # seq_name = batch['seq_name'] if 'seq_name' in batch else None
        batch_size = len(file_index)
        
        # Forward pass
        start_t = time.time()
        pred_disparity_pyramid = self.forward(stereo_event, stereo_image)
        fps = 1 / ((time.time() - start_t) / batch_size)
        
        # Calculate loss
        loss_disp = None
        loss_disp = self._cal_loss(pred_disparity_pyramid, disparity_gt)
        loss = loss_disp.mean()
        
        # Calculate metrics
        pred = pred_disparity_pyramid[-1].detach()
        gt   = disparity_gt.detach()
        mask = (gt > 0) & (gt < self.max_disp)
        val_mde = val_mdise = val_1pa = 0.
        for p, g, m in zip(pred, gt, mask):
            p, g = p[m], g[m]
            val_mde += mean_depth_error(p, g).item()
            val_mdise += mean_disparity_error(p, g).item()
            val_1pa += n_pixel_accuracy(p, g, 1).item()
        val_mde, val_mdise, val_1pa = val_mde/batch_size, val_mdise/batch_size, val_1pa/batch_size
        
        # Validation log dict
        self.log_dict(
            {'val/loss': loss, 'fps': fps, 'val/mde': val_mde, 'val/mdise': val_mdise, 'val/1pa': val_1pa}, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            logger=True, 
            batch_size=batch_size, 
            sync_dist=True, 
        )
        
        return loss
    
    def configure_optimizers(self):
        params_group = self._get_params_group(self.optimizer.params.lr)
        optimizer = getattr(torch.optim, self.optimizer.name)(params_group, **self.optimizer.params)
        if self.scheduler.name == 'CosineAnnealingWarmupRestarts':
            from utils.scheduler import CosineAnnealingWarmupRestarts
            scheduler = {'scheduler': CosineAnnealingWarmupRestarts(optimizer, **self.scheduler.params), 
                         'name': 'lr_scheduler'}
        else:
            scheduler = {'scheduler': getattr(torch.optim.lr_scheduler, self.scheduler.name)(optimizer, **self.scheduler.params), 
                         'name': 'lr_scheduler'}
            
        return [optimizer], [scheduler]
    
    def _cal_loss(self, pred_disparity_pyramid, gt_disparity):
        pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]
        criterion = nn.SmoothL1Loss(reduction='none')

        loss = 0.0
        mask = gt_disparity > 0
        for idx in range(len(pred_disparity_pyramid)):
            pred_disp = pred_disparity_pyramid[idx]
            weight = pyramid_weight[idx]

            if pred_disp.size(-1) != gt_disparity.size(-1):
                pred_disp = pred_disp.unsqueeze(1)
                pred_disp = F.interpolate(pred_disp, size=(gt_disparity.size(-2), gt_disparity.size(-1)),
                                          mode='bilinear', align_corners=False) * (
                                    gt_disparity.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)

            cur_loss = criterion(pred_disp[mask], gt_disparity[mask])
            loss += weight * cur_loss

        return loss
    
    def _get_params_group(self, learning_rate):
        def filter_specific_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return True
            return False
        
        def filter_base_params(kv):
            specific_layer_name = ['offset_conv.weight', 'offset_conv.bias']
            for name in specific_layer_name:
                if name in kv[0]:
                    return False
            return True
        
        specific_params = list(filter(filter_specific_params,
                                      self.named_parameters()))
        base_params = list(filter(filter_base_params,
                                  self.named_parameters()))
        
        specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
        base_params = [kv[1] for kv in base_params]
        
        specific_lr = learning_rate * 0.1
        params_group = [
            {'params': base_params, 'lr': learning_rate},
            {'params': specific_params, 'lr': specific_lr},
        ]
        
        return params_group