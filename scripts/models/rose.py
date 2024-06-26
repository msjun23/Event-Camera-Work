import torch
import torch.nn as nn
import torch.nn.functional as F

import snntorch as snn
from snntorch import surrogate

from spikingjelly.activation_based import neuron, functional, surrogate, layer


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, padding, bias=False):
#         super(ConvBlock, self).__init__()
#         self.block = nn.Sequential(*[
#             nn.Conv2d(in_channels=in_channels, 
#                       out_channels=out_channels, 
#                       kernel_size=kernel_size, 
#                       padding=padding, 
#                       bias=bias), 
#             nn.BatchNorm2d(out_channels)
#         ])
        
#     def forward(self, x):
#         return self.block(x)
      
      
class RoSE(nn.Module):
    def __init__(self, in_channels=5, base_channels=32, kernel_size=3, padding=1, stride=1):
        super(RoSE, self).__init__()
        # self.attention_method = attention_method
        
        # # Spiking Neuron and simulation params
        # # Surrogate gradient function
        # spike_grad = surrogate.atan(alpha=2.0)
        # # Decay rate beta
        # beta = 0.8
        
        # [timestep, batch_size, in_channels, base_h, base_w]
        # self.conv1 = ConvBlock(in_channels=in_channels, out_channels=base_channels, kernel_size=1, padding=0)
        # self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, learn_threshold=True)
        # self.conv2 = ConvBlock(in_channels=base_channels, out_channels=1, kernel_size=1, padding=0)
        # self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, learn_beta=True, learn_threshold=True)
        
        tau = 2.
        v_thr = 1.
        v_rst = 0.
        backend = 'cupy'
        # [T, N, C, H, W]
        self.rose = nn.Sequential(
            layer.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=kernel_size, padding=padding, stride=stride), 
            layer.BatchNorm2d(base_channels), 
            # neuron.IFNode(v_threshold=v_thr, v_reset=v_rst, surrogate_function=surrogate.ATan()), 
            neuron.LIFNode(tau=tau, v_threshold=v_thr, v_reset=v_rst, surrogate_function=surrogate.ATan()), 
            layer.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=kernel_size, padding=padding, stride=stride), 
            layer.BatchNorm2d(1), 
            # neuron.IFNode(v_threshold=v_thr, v_reset=v_rst, surrogate_function=surrogate.ATan()), 
            neuron.LIFNode(tau=tau, v_threshold=v_thr, v_reset=v_rst, surrogate_function=surrogate.ATan()), 
        )
        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend=backend)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        # print(torch.unique(x.cpu().to(torch.float32)))
        # Initialize membrane potentials at each layer
        # mem1 = self.lif1.init_leaky()
        # mem2 = self.lif2.init_leaky()
        
        # for t in range(x.size(0)):
        #     # x[t]: [N, C=1, H, W]
        #     spk1, mem1 = self.lif1(self.conv1(x[t]), mem1)
        #     spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
            
        # for noforloop:
        # spk1, mem1 = self.lif1(self.conv1(x), mem1)
        # spk2, mem2 = self.lif2(self.conv2(spk1), mem2)
        
        # output = spk2+mem2
        # return output
        
        return self.rose(x)