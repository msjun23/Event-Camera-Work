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
        
        # Train, validation and test step outputs buffer
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
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
        file_index = batch['file_index'] if 'file_index' in batch else None
        # seq_name = batch['seq_name'] if 'seq_name' in batch else None
        
        # Forward pass
        start_t = time.time()
        pred_disparity_pyramid = self.forward(stereo_event, stereo_image)   # List: [[N, H, W], ...]
        fps = 1 / ((time.time() - start_t) / pred_disparity_pyramid[-1].shape[0])
        
        # Calculate loss
        loss_disp = self._cal_loss(pred_disparity_pyramid, disparity_gt)
        loss = loss_disp.mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # Calculate metrics
        pred = pred_disparity_pyramid[-1].detach()
        gt   = disparity_gt.detach()
        num_imgs = len(file_index)
        mask = (gt > 0) & (gt < self.max_disp)
        step_mde = mean_depth_error(pred, gt, mask, on_epoch=False)
        step_mdise = mean_disparity_error(pred, gt, mask, on_epoch=False)
        step_1pa = n_pixel_accuracy(pred, gt, mask, 1, on_epoch=False)
        
        # Logging
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('lr', cur_lr, prog_bar=True, on_step=True)         # Learning rate
        self.log('fps', fps, prog_bar=True, on_step=True)           # FPS
        
        # Training step outputs
        output_dict = {'loss': loss, 'FPS': fps, 'MDE': step_mde, 'MDisE': step_mdise, '1PA': step_1pa, 'num_imgs': num_imgs}
        self.training_step_outputs.append(output_dict)
        return output_dict
    
    def on_train_epoch_end(self):
        # Gather all batch metrics
        num_imgs = 0
        epoch_mde = 0.
        epoch_mdise = 0.
        epoch_1pa = 0.
        for batch_output in self.training_step_outputs:
            num_imgs += batch_output['num_imgs']
            epoch_mde += batch_output['MDE']
            epoch_mdise += batch_output['MDisE']
            epoch_1pa += batch_output['1PA']
        
        # Metrics on an epoch
        epoch_mde /= num_imgs
        epoch_mdise /= num_imgs
        epoch_1pa /= num_imgs
        
        # Log epoch metrics
        self.log('MDE', epoch_mde, prog_bar=True, on_epoch=True)
        self.log('MDisE', epoch_mdise, prog_bar=True, on_epoch=True)
        self.log('1PA', epoch_1pa, prog_bar=True, on_epoch=True)
        
        # Free up the memory
        self.training_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        stereo_event = batch['event'] if 'event' in batch else None     # ['left', 'right'], [N, C, H, W]
        stereo_image = batch['image'] if 'image' in batch else None     # ['left', 'right'], [N, C, H, W]
        disparity_gt = batch['disparity_gt'] if 'disparity_gt' in batch else None   # [N, H, W]
        # file_index = batch['file_index'] if 'file_index' in batch else None
        # seq_name = batch['seq_name'] if 'seq_name' in batch else None
        
        # Forward pass
        pred_disparity_pyramid = self.forward(stereo_event, stereo_image)
        
        # Calculate loss
        loss_disp = None
        if disparity_gt is not None:
            loss_disp = self._cal_loss(pred_disparity_pyramid, disparity_gt)
            loss = loss_disp.mean()
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
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