import os
import time
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

import lightning as pl
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info

# from .concentration import ConcentrationNet
# from .rose import RoSE
# from .stereo_matching import StereoMatchingNetwork
from .refinement import StereoDRNetRefinement

from .sequence_encoder import SequenceEncoder
from .s4 import S4Block

from .ev_feature_extractor import EventFeatureExtractor
from .feature_extractor import FeatureExtractor
from .cost import CostVolumePyramid
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid
from utils import metrics

import warnings
# Ignore pytorch lightning sync_dist warning
warnings.filterwarnings('ignore', '.*It is recommended to use.*', category=UserWarning)


class StereoDepthLightningModule(pl.LightningModule):
    def __init__(self, config, dataset=None):
        super(StereoDepthLightningModule, self).__init__()
        self.cfg = config
        self.model_init(**config.network)
        
        # self.rose = RoSE(**config.event_processor.params)
        # self.stereo_matching_net = StereoMatchingNetwork(**config.disparity_estimator.params, **config.event_processor.params)
        # self.max_disp = config.disparity_estimator.params.max_disp
        self.best_val_loss = float('inf')
        
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        
        self.dataset = dataset
        
        self.show = config.show
        
        self.metrics = config.metric
        
    def model_init(self, max_disp,
                   ev_in_channels=2,
                   img_in_channels=3,
                   num_downsample=2,
                   no_mdconv=False,
                   feature_similarity='correlation',
                   num_scales=3,
                   num_fusions=6,
                   deformable_groups=2,
                   mdconv_dilation=2,
                   no_intermediate_supervision=False,
                   num_stage_blocks=1,
                   num_deform_blocks=3,
                   refine_channels=None, 
                   **kwargs, ):
        refine_channels = img_in_channels if refine_channels is None else refine_channels
        self.num_downsample = num_downsample
        self.num_scales = num_scales
        self.max_disp = max_disp

        # Feature extractor
        self.ev_feature_extractor = EventFeatureExtractor(in_channels=ev_in_channels)
        self.feature_extractor = FeatureExtractor(in_channels=img_in_channels)
        max_disp = max_disp // 3

        # Event sequence processor
        self.seq_encoder = SequenceEncoder(**kwargs['seq_encoder'])
        self.s4 = S4Block(**kwargs['event_processor'])

        # Cost volume construction
        self.cost_volume_constructor = CostVolumePyramid(max_disp, feature_similarity=feature_similarity)

        # Cost aggregation
        self.aggregation = AdaptiveAggregation(max_disp=max_disp,
                                               num_scales=num_scales,
                                               num_fusions=num_fusions,
                                               num_stage_blocks=num_stage_blocks,
                                               num_deform_blocks=num_deform_blocks,
                                               no_mdconv=no_mdconv,
                                               mdconv_dilation=mdconv_dilation,
                                               deformable_groups=deformable_groups,
                                               intermediate_supervision=not no_intermediate_supervision)

        # Disparity estimation
        self.disparity_estimation = DisparityEstimationPyramid(max_disp)

        # Refinement
        refine_module_list = nn.ModuleList()
        for i in range(num_downsample):
            refine_module_list.append(StereoDRNetRefinement(img_channels=refine_channels))

        self.refinement = refine_module_list
        
    def disparity_refinement(self, left_img, right_img, disparity):
        disparity_pyramid = []
        for i in range(self.num_downsample):
            scale_factor = 1. / pow(2, self.num_downsample - i - 1)

            if scale_factor == 1.0:
                curr_left_img = left_img
                curr_right_img = right_img
            else:
                curr_left_img = F.interpolate(left_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
                curr_right_img = F.interpolate(right_img,
                                                scale_factor=scale_factor,
                                                mode='bilinear', align_corners=False)
            inputs = (disparity, curr_left_img, curr_right_img)
            disparity = self.refinement[i](*inputs)
            disparity_pyramid.append(disparity)  # [H/2, H]

        return disparity_pyramid
        
    def on_train_start(self):
        model_info = str(self)
        total_params = sum(p.numel() for p in self.parameters())
        log_dir = self.trainer.log_dir
        model_info_f = os.path.join(log_dir, 'model.txt')
        with open(model_info_f, 'w') as f:
            f.write(model_info)
            f.write(f'\n\nTotal parameters: {total_params}\n')
            
    def forward(self, stereo_event, stereo_image):
        # _stereo_event = {}
        # rose_img = {}
        # ei_frame = {}
        # for loc in ['left', 'right']:
        #     _stereo_event[loc] = stereo_event[loc].clone()
        #     _stereo_event[loc] = rearrange(_stereo_event[loc], 'n t c h w -> t n c h w')
        #     functional.reset_net(self.rose)
        #     rose_img[loc] = self.rose(_stereo_event[loc])[-1]   # [N, C, H, W]
        #     ei_frame[loc] = torch.cat([rose_img[loc], stereo_image[loc]], dim=1)
            
        # pred_disparity_pyramid = self.stereo_matching_net(
        #     ei_frame['left'],
        #     ei_frame['right']
        # )
        
        # return rose_img, pred_disparity_pyramid
    
    ##########################################################################################
    
        # pred_disparity_pyramid = self.stereo_matching_net(
        #     stereo_event['left'], 
        #     stereo_event['right'], 
        #     stereo_image['left'], 
        #     stereo_image['right']
        # )
        
        # return {'left': stereo_event['left'].sum(dim=1).sum(dim=1), 'right': stereo_event['right'].sum(dim=1).sum(dim=1)}, pred_disparity_pyramid
        
    ##########################################################################################
        
        left_ev, right_ev = stereo_event['left'], stereo_event['right']
        left_img, right_img = stereo_image['left'], stereo_image['right']
        
        left_ev = rearrange(left_ev, 'b t c h w -> t b c h w')
        right_ev = rearrange(right_ev, 'b t c h w -> t b c h w')

        # Extract event and image features
        left_ev_features = self.ev_feature_extractor(left_ev)
        right_ev_features = self.ev_feature_extractor(right_ev)
        left_features = self.feature_extractor(left_img)
        right_features = self.feature_extractor(right_img)

        # Process each feature level
        for f_idx, (left_ev_f, right_ev_f) in enumerate(zip(left_ev_features, right_ev_features)):
            left_ev_seq = rearrange(self.seq_encoder(left_ev_f), 't b c -> b t c')
            right_ev_seq = rearrange(self.seq_encoder(right_ev_f), 't b c -> b t c')

            left_ev_seq, _ = self.s4(left_ev_seq)
            right_ev_seq, _ = self.s4(right_ev_seq)

            left_ev_t_score = F.softmax(left_ev_seq, dim=1)
            right_ev_t_score = F.softmax(right_ev_seq, dim=1)

            left_ev_t_score = rearrange(left_ev_t_score, 'b t c -> b t c 1 1')
            right_ev_t_score = rearrange(right_ev_t_score, 'b t c -> b t c 1 1')

            left_ev_f = rearrange(left_ev_f, 't b c h w -> b t c h w')
            right_ev_f = rearrange(right_ev_f, 't b c h w -> b t c h w')

            left_ev_f = torch.sum(left_ev_t_score * left_ev_f, dim=1)
            right_ev_f = torch.sum(right_ev_t_score * right_ev_f, dim=1)

            left_features[f_idx] = left_features[f_idx] + left_ev_f  # Avoid in-place operation
            right_features[f_idx] = right_features[f_idx] + right_ev_f  # Avoid in-place operation

        # Construct cost volume and aggregate
        cost_volume = self.cost_volume_constructor(left_features, right_features)
        aggregation = self.aggregation(cost_volume)

        # Estimate disparity and refine
        disparity_pyramid = self.disparity_estimation(aggregation)
        disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])

        return {'left': stereo_event['left'].sum(dim=1).sum(dim=1), 'right': stereo_event['right'].sum(dim=1).sum(dim=1)}, disparity_pyramid
    
    '''
    DataLoader settngs
    '''
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
    
    '''
    Train, validaion and test process settings
    '''
    def on_train_epoch_start(self):
        self.train_start_time = time.time()
        
    def training_step(self, batch):
        stereo_event = batch['event'] if 'event' in batch else None     # ['left', 'right'], [N, C, H, W]
        stereo_image = batch['image'] if 'image' in batch else None     # ['left', 'right'], [N, C, H, W]
        disparity_gt = batch['disparity_gt'] if 'disparity_gt' in batch else None   # [N, H, W]
        file_index = batch['file_index'] if 'file_index' in batch else None         # List, length=batch_size
        # seq_name = batch['seq_name'] if 'seq_name' in batch else None
        batch_size = len(file_index)
        
        # Forward pass
        rose_img, pred_disparity_pyramid = self.forward(stereo_event, stereo_image)   # List: [[N, H, W], ...]
        
        # Calculate loss
        loss_disp = self._cal_loss(pred_disparity_pyramid, disparity_gt)
        loss = loss_disp.mean()
        
        # Calculate metrics
        pred = pred_disparity_pyramid[-1].detach()
        gt   = disparity_gt.detach()
        mask = (gt > 0) & (gt < self.max_disp)
        metrics_dict = {getattr(self.metrics, metric).name:0. for metric in self.metrics}
        for p, g, m in zip(pred, gt, mask):
            p, g = p[m], g[m]
            for _m, detail in self.metrics.items():
                if _m.startswith('n_pixel_error'):
                    _m = '_'.join(_m.split('_')[:-1])
                metrics_dict[detail.name] += getattr(metrics, _m)(p, g).item() if 'params' not in detail else getattr(metrics, _m)(p, g, **detail.params).item()
        metrics_dict = {k: v/batch_size for k, v in metrics_dict.items()}
        
        # Train log dict
        self.log('train/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {'train/'+k: v for k, v in metrics_dict.items()}, 
            prog_bar=False, 
            logger=True, 
            on_step=True, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=batch_size, 
        )
        
        # Show the results
        if self.show.train:
            frames = [rose_img['left'][0], rose_img['right'][0], stereo_image['left'][0], stereo_image['right'][0], pred[0], gt[0]]
            names  = ['left_event', 'right_event', 'left_image', 'right_image', 'pred', 'gt']
            self._show_figs('train', frames, names)
            
        return loss
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        time_per_epoch = time.time() - self.train_start_time
        
        # Formatting time for logging
        hours, rem = divmod(time_per_epoch, 3600)
        minutes, seconds = divmod(rem, 60)
        time_per_epoch_str = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'
        
        # Calculate ETA
        remaining_epochs = self.trainer.max_epochs - (self.current_epoch + 1)
        eta_seconds = time_per_epoch * remaining_epochs
        eta_hours, eta_rem = divmod(eta_seconds, 3600)
        eta_minutes, eta_seconds = divmod(eta_rem, 60)
        eta_str = f'{int(eta_hours):02}:{int(eta_minutes):02}:{int(eta_seconds):02}'
        
        train_metrics = self.trainer.logged_metrics
        epoch = self.current_epoch
        log_dir = self.trainer.log_dir
        train_log_f = os.path.join(log_dir, f'train_log.txt')
        
        # Save as .txt file
        if self.trainer.is_global_zero:     # Master proc. only
            with open(train_log_f, 'a') as f:
                f.write(f'Epoch: {epoch} | time per epoch: {time_per_epoch_str} | eta: {eta_str}\n')
                log_msg = 'Train'
                for key, value in train_metrics.items():
                    if key.startswith('train/') and key.endswith('_epoch'):
                        _key = key.split('/')[1].split('_')[0]
                        log_msg += f' | {_key}: {value.item():.3f}'
                log_msg += '\n'
                f.write(log_msg)
                
        # Save the last checkpoint
        last_ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, f'{epoch}.ckpt')
        self.trainer.save_checkpoint(last_ckpt_path)
        
    def on_validation_epoch_start(self):
        self.val_start_time = time.time()
        
    def validation_step(self, batch):
        stereo_event = batch['event'] if 'event' in batch else None     # ['left', 'right'], [N, C, H, W]
        stereo_image = batch['image'] if 'image' in batch else None     # ['left', 'right'], [N, C, H, W]
        disparity_gt = batch['disparity_gt'] if 'disparity_gt' in batch else None   # [N, H, W]
        file_index = batch['file_index'] if 'file_index' in batch else None
        # seq_name = batch['seq_name'] if 'seq_name' in batch else None
        batch_size = len(file_index)
        
        # Forward pass
        start_t = time.time()
        rose_img, pred_disparity_pyramid = self.forward(stereo_event, stereo_image)   # List: [[N, H, W], ...]
        fps = 1 / ((time.time() - start_t) / batch_size)
        
        # Calculate loss
        loss_disp = self._cal_loss(pred_disparity_pyramid, disparity_gt)
        loss = loss_disp.mean()
        
        # Calculate metrics
        pred = pred_disparity_pyramid[-1].detach()
        gt   = disparity_gt.detach()
        mask = (gt > 0) & (gt < self.max_disp)
        metrics_dict = {getattr(self.metrics, metric).name:0. for metric in self.metrics}
        for p, g, m in zip(pred, gt, mask):
            p, g = p[m], g[m]
            for _m, detail in self.metrics.items():
                if _m.startswith('n_pixel_error'):
                    _m = '_'.join(_m.split('_')[:-1])
                metrics_dict[detail.name] += getattr(metrics, _m)(p, g).item() if 'params' not in detail else getattr(metrics, _m)(p, g, **detail.params).item()
        metrics_dict = {k: v/batch_size for k, v in metrics_dict.items()}
        
        # Validation log dict
        self.log('val/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
        self.log_dict(
            {'val/'+k: v for k, v in metrics_dict.items()} | {'val/fps': fps}, 
            prog_bar=False, 
            logger=True, 
            on_step=True, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=batch_size, 
        )
        
        # Show the results
        if self.show.validation:
            frames = [rose_img['left'][0], rose_img['right'][0], stereo_image['left'][0], stereo_image['right'][0], pred[0], gt[0]]
            names  = ['left_event', 'right_event', 'left_image', 'right_image', 'pred', 'gt']
            self._show_figs('validation', frames, names)
            
        return loss
    
    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        super().on_validation_epoch_end()
        time_per_val = time.time() - self.val_start_time
        
        # Formatting time for logging
        hours, rem = divmod(time_per_val, 3600)
        minutes, seconds = divmod(rem, 60)
        time_per_val_str = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'
        
        val_metrics = self.trainer.logged_metrics
        epoch = self.current_epoch
        log_dir = self.trainer.log_dir
        val_log_f = os.path.join(log_dir, f'val_log.txt')
        
        # Save as .txt file
        if self.trainer.is_global_zero:     # Master proc. only
            with open(val_log_f, 'a') as f:
                f.write(f'Epoch: {epoch} | time for validation: {time_per_val_str}\n')
                log_msg = 'Validation'
                for key, value in val_metrics.items():
                    if key.startswith('val/') and key.endswith('_epoch'):
                        _key = key.split('/')[1].split('_')[0]
                        log_msg += f' | {_key}: {value.item():.3f}'
                log_msg += '\n'
                f.write(log_msg)
                
        # Save the best checkpoint based on validation loss
        val_loss = self.trainer.callback_metrics.get('val/loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_ckpt_path = os.path.join(self.trainer.checkpoint_callback.dirpath, 'best.ckpt')
            self.trainer.save_checkpoint(best_ckpt_path)
            
    def on_test_epoch_start(self):
        self.test_start_time = time.time()
        
    def test_step(self, batch):
        stereo_event = batch['event'] if 'event' in batch else None     # ['left', 'right'], [N, C, H, W]
        stereo_image = batch['image'] if 'image' in batch else None     # ['left', 'right'], [N, C, H, W]
        disparity_gt = batch['disparity_gt'] if 'disparity_gt' in batch else None   # [N, H, W]
        file_index = batch['file_index'] if 'file_index' in batch else None
        # seq_name = batch['seq_name'] if 'seq_name' in batch else None
        batch_size = len(file_index)
        
        # Forward pass
        start_t = time.time()
        rose_img, pred_disparity_pyramid = self.forward(stereo_event, stereo_image)   # List: [[N, H, W], ...]
        fps = 1 / ((time.time() - start_t) / batch_size)
        
        if disparity_gt is not None:
            # Calculate loss
            loss_disp = self._cal_loss(pred_disparity_pyramid, disparity_gt)
            loss = loss_disp.mean()
            
            # Calculate metrics
            pred = pred_disparity_pyramid[-1].detach()
            gt   = disparity_gt.detach()
            mask = (gt > 0) & (gt < self.max_disp)
            metrics_dict = {getattr(self.metrics, metric).name:0. for metric in self.metrics}
            for p, g, m in zip(pred, gt, mask):
                p, g = p[m], g[m]
                for _m, detail in self.metrics.items():
                    if _m.startswith('n_pixel_error'):
                        _m = '_'.join(_m.split('_')[:-1])
                    metrics_dict[detail.name] += getattr(metrics, _m)(p, g).item() if 'params' not in detail else getattr(metrics, _m)(p, g, **detail.params).item()
            metrics_dict = {k: v/batch_size for k, v in metrics_dict.items()}
            
            # Test log dict
            self.log('test/loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=True, batch_size=batch_size)
            self.log_dict(
                {'test/'+k: v for k, v in metrics_dict.items()} | {'test/fps': fps}, 
                prog_bar=False, 
                logger=True, 
                on_step=True, 
                on_epoch=True, 
                sync_dist=True, 
                batch_size=batch_size, 
            )
            
            # Show the results
            if self.show.test:
                frames = [rose_img['left'][0], rose_img['right'][0], stereo_image['left'][0], stereo_image['right'][0], pred[0], gt[0]]
                names  = ['left_event', 'right_event', 'left_image', 'right_image', 'pred', 'gt']
                self._show_figs('test', frames, names)
            return loss
        else:       # No disparity gt
            pred = pred_disparity_pyramid[-1].detach()
            gt = torch.zeros_like(pred)
            # Show the results
            if self.show.test:
                frames = [rose_img['left'][0], rose_img['right'][0], stereo_image['left'][0], stereo_image['right'][0], pred[0], gt[0]]
                names  = ['left_event', 'right_event', 'left_image', 'right_image', 'pred', 'gt']
                self._show_figs('test', frames, names)
            return None
        
    def on_test_epoch_end(self):
        super().on_test_epoch_end()
        time_per_test = time.time() - self.test_start_time
        
        # Formatting time for logging
        hours, rem = divmod(time_per_test, 3600)
        minutes, seconds = divmod(rem, 60)
        time_per_test_str = f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'
        
        test_metrics = self.trainer.logged_metrics
        log_dir = self.trainer.log_dir
        test_log_f = os.path.join(log_dir, f'test_log.txt')
        
        # Save as .txt file
        if self.trainer.is_global_zero:     # Master proc. only
            with open(test_log_f, 'a') as f:
                f.write(f'Time for test: {time_per_test_str}\n')
                log_msg = 'Test'
                for key, value in test_metrics.items():
                    if key.startswith('test/') and key.endswith('_epoch'):
                        _key = key.split('/')[1].split('_')[0]
                        log_msg += f' | {_key}: {value.item():.3f}'
                log_msg += '\n'
                f.write(log_msg)
                
    '''
    Hyper-parameteres settings
    '''
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
    
    def _show_figs(self, exp, frames: list, names: list):
        def _to_numpy(frame):
            if frame.dtype == torch.bfloat16:
                frame = frame.to(torch.float32)
            if frame is None:
                return np.zeros((256, 256, 3), dtype=np.uint8)  # Assuming 256x256 for example
            if frame.ndim == 3:  # C*H*W
                frame = frame.detach().cpu().numpy().transpose(1, 2, 0)  # Convert C*H*W to H*W*C
            elif frame.ndim == 2:  # H*W
                frame = frame.detach().cpu().numpy()
            return frame
        
        # Filter out None frames and their corresponding names
        frames_names = [(_to_numpy(frame), name) for frame, name in zip(frames, names) if frame is not None]
        if not frames_names:
            return
        
        frames, names = zip(*frames_names)
        num_rows = len(frames) // 2
        fig, axes = plt.subplots(nrows=num_rows, ncols=2, figsize=(10, 5 * num_rows))
        
        for i in range(num_rows):
            ax_left, ax_right = axes[i] if num_rows > 1 else axes
            
            left_frame = frames[2 * i]
            right_frame = frames[2 * i + 1]
            left_name = names[2 * i]
            right_name = names[2 * i + 1]
            
            ax_left.imshow(left_frame)
            ax_left.set_title(left_name)
            ax_left.axis('off')
            
            ax_right.imshow(right_frame)
            ax_right.set_title(right_name)
            ax_right.axis('off')
            
        plt.tight_layout()
        save_file = os.path.join(self.trainer.log_dir, f'{exp}_comparison.png')
        plt.savefig(save_file)
        plt.close(fig)