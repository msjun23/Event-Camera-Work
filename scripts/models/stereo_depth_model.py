import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl

from einops import rearrange

# from .concentration import ConcentrationNet
from .rose import RoSE
from .stereo_matching import StereoMatchingNetwork

class StereoDepthLightningModule(pl.LightningModule):
    def __init__(self, 
                 event_processor=None,
                 disparity_estimator=None, 
                 optimizer=None, 
                 scheduler=None):
        super(StereoDepthLightningModule, self).__init__()
        self.rose = RoSE(**event_processor.params)
        self.stereo_matching_net = StereoMatchingNetwork(**disparity_estimator.params)
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.SmoothL1Loss(reduction='none')
        
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
        
    def training_step(self, batch, batch_idx):
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
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            
        # Logging learning rate
        cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("lr", cur_lr, prog_bar=True, on_step=True)
        return loss
    
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

            cur_loss = self.criterion(pred_disp[mask], gt_disparity[mask])
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