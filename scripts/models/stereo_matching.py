import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .refinement import StereoDRNetRefinement

from .sequence_encoder import SequenceEncoder
from .s4 import S4Block

from .ev_feature_extractor import EventFeatureExtractor
from .feature_extractor import FeatureExtractor
from .cost import CostVolumePyramid
from .aggregation import AdaptiveAggregation
from .estimation import DisparityEstimationPyramid

class StereoMatchingNetwork(nn.Module):
    def __init__(self, max_disp,
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
                 **s4_args):
        super(StereoMatchingNetwork, self).__init__()

        refine_channels = img_in_channels if refine_channels is None else refine_channels
        self.num_downsample = num_downsample
        self.num_scales = num_scales

        self.seq_encoder = SequenceEncoder()
        self.s4 = S4Block(**s4_args)

        # Feature extractor
        self.ev_feature_extractor = EventFeatureExtractor(in_channels=ev_in_channels)
        self.feature_extractor = FeatureExtractor(in_channels=img_in_channels)
        max_disp = max_disp // 3

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

    # def forward(self, left_img, right_img):
    #     ''' E.g.,
    #     left_feature torch.Size([16, 128, 86, 112]) torch.Size([16, 128, 43, 56]) torch.Size([16, 128, 22, 28]) 3
    #     cost_volume torch.Size([16, 40, 86, 112]) torch.Size([16, 20, 43, 56]) torch.Size([16, 10, 22, 28]) 3
    #     aggregation torch.Size([16, 40, 86, 112]) torch.Size([16, 20, 43, 56]) torch.Size([16, 10, 22, 28]) 3
    #     disparity_pyramid torch.Size([16, 22, 28]) torch.Size([16, 43, 56]) torch.Size([16, 86, 112]) 3
    #     disparity_pyramid torch.Size([16, 22, 28]) torch.Size([16, 43, 56]) torch.Size([16, 86, 112]) torch.Size([16, 128, 168]) torch.Size([16, 256, 336]) 5
    #     '''
    #     # feature size order in list: larger -> smaller
    #     left_feature = self.feature_extractor(left_img)                             # left_img, right_img: [B, C, H, W], tensor
    #     right_feature = self.feature_extractor(right_img)                           # ([B, C_f, H/n1, W/n1], [B, C_f, H/n2, W/n2], ...), list of tensors
    #     cost_volume = self.cost_volume_constructor(left_feature, right_feature)     # ([B, C_v, H/n1, W/n1], [B, C_v, H/n2, W/n2], ...), list of tensors
    #     aggregation = self.aggregation(cost_volume)                                 # ([B, C_v, H/n1, W/n1], [B, C_v, H/n2, W/n2], ...), list of tensors
    #     # disparity_pyramid becomes reverse order: smaller -> larger
    #     # and has no channel dimension; disparity map is 1-C image
    #     disparity_pyramid = self.disparity_estimation(aggregation)                  # ([B, H/n_{-1}, W/n_{-1}], [B, H/n_{-2}, W/n_{-2}], ...), list of tensors
    #     # Refine disparity_maps to original size, disparity_pyramid[-1] is a final output (disparity map in original size)
    #     disparity_pyramid += self.disparity_refinement(left_img, right_img, disparity_pyramid[-1])

    #     return disparity_pyramid

    def forward(self, left_ev, right_ev, left_img, right_img):
        torch.autograd.set_detect_anomaly(True)
        
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

        return disparity_pyramid