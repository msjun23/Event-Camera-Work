import torch.nn as nn

from spikingjelly.activation_based import layer # Use spikingjelly to conv with T dim (5D data)

    
class SequenceEncoder(nn.Module):
    def __init__(self, in_channels=128, base_channels=128, d_model=128):
        super(SequenceEncoder, self).__init__()
        
        # Define layers without hardcoding the input dimensions
        self.conv1 = layer.Conv2d(in_channels=in_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, step_mode='m')
        self.conv2 = layer.Conv2d(in_channels=base_channels, out_channels=base_channels, kernel_size=3, stride=1, padding=1, step_mode='m')
        
        self.fc = layer.Linear(in_features=base_channels, out_features=d_model)  # This will be dynamically adjusted in the forward pass
        
    def forward(self, x):
        # Apply the convolutional layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global average pooling to reduce H and W to 1
        x = x.mean(dim=[-2, -1])        # [T, B, C, H, W] -> [T, B, C]
        
        # Final fully connected layer
        x = self.fc(x)
        
        return x