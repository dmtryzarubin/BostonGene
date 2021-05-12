import torch
import torch.nn as nn

def double_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    """
    Double-Convolution layer with RELU activation function
    param: in_channels : number of input channels
    param: out_channels : number of output channels
    """
    conv = nn.Sequential(
        nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
            ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False
            ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Left side of Unet
        self.down_conv_1 = double_conv(in_channels=3, out_channels=16)#

        self.down_conv_2 = double_conv(in_channels=16, out_channels=32)#
        
        self.down_conv_3 = double_conv(in_channels=32, out_channels=64)#
        
        self.down_conv_4 = double_conv(in_channels=64, out_channels=96)#
        
        self.down_conv_5 = double_conv(in_channels=96, out_channels=128)
        
        # Right side of Unet
        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=96,
            kernel_size=2,
            stride=2)
        self.up_conv_1 = double_conv(96*2, 96)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=96,
            out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_conv_2 = double_conv(128, 64)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=2,
            stride=2)
        self.up_conv_3 = double_conv(64, 32)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=2,
            stride=2)
        self.up_conv_4 = double_conv(32, 16)

        self.out = nn.Conv2d(
            in_channels=16,
            out_channels=6,
            kernel_size=1)
    
    def forward(self, image):
        """
        Implements forward pass
        """
        # Encoder

        # Input image [3, 128, 128]
        l_1 = self.down_conv_1(image)
        # Out shape: [16, 128, 128]
        l_2 = self.max_pool_2(l_1)
        # [16, 64, 64]
        l_3 = self.down_conv_2(l_2)
        # [32, 64, 64]
        l_4 = self.max_pool_2(l_3)
        # [32, 32, 32]
        l_5 = self.down_conv_3(l_4) 
        # [64, 32, 32]
        l_6 = self.max_pool_2(l_5)
        # [64, 16, 16]
        l_7 = self.down_conv_4(l_6) 
        # [96, 16, 16]
        l_8 = self.max_pool_2(l_7)
        # [96, 8, 8]
        l_9 = self.down_conv_5(l_8)
        # [128, 8, 8]
        
        # Decoder
        l_10 = self.up_trans_1(l_9)
        # [96, 16, 16]
        l_11 = self.up_conv_1(torch.cat([l_10, l_7], axis=1))
        # [96, 16, 16]
        l_12 = self.up_trans_2(l_11)
        # [64, 32, 32]
        l_13 = self.up_conv_2(torch.cat([l_12, l_5], axis=1))
        # [64, 32, 32]
        l_14 = self.up_trans_3(l_13)
        # [64, 32, 32]
        l_15 = self.up_conv_3(torch.cat([l_14, l_3], axis=1))
        # [32, 64, 64]
        l_16 = self.up_trans_4(l_15)
        # [16, 128, 128]
        l_17 = self.up_conv_4(torch.cat([l_16, l_1], axis=1))
        # [16, 128, 128]
        out = self.out(l_17)
        # [6, 128, 128]
        return out