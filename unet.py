#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 08:59:13 2025

@author: vilmalehto

The UNet model. The code is done based on the original UNet architecture 
(Ronneberger et al. 2015). 
"""

import torch
import torch.nn as nn

#tupla konvoluutio, eli ne kaksi vaakatason nuolta mallin kuvassa
class ConvolutionBlock(nn.Module):
    """
    A double convolution block. Each convolution consists of:
    Conv2D -> BatchNorm -> ReLU
        
    Used in both encode and decoder parts of the Unet and on its own 
    in the bottle neck.
        
    Parameters
    ----------
    in_channels: int
        Number of input channels
    out_channels: int
        Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv_output = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace =  True))
        
    def forward(self, x):
        """
        Forward pass for the double convolution block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C_in, H, W)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (B, C_out, H, W)

        """
        return self.conv_output(x)
        
    
class EncoderBlock(nn.Module):
    """
    Encoder block for downsampling the input.
    Applies a ConvolutionBlock followed by MaxPooling.
    Prepares the output for next encoder block and skipped connection. 
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels. 
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        #utilizes the previously created double convolution block
        self.convolution = ConvolutionBlock(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size = 2, stride = 2) 
        
        #Encoder block has two outputs one for the U-path and one for skip connection
        #The skip connection output is the convolution output
        #and the U-path output goes through the MaxPooling
    def forward(self, x):
        """
        Forward pass through the encoder block.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        
        Returns
        -------
        conv_output : torch.Tensor
            Output from convolution block (used for skip connection).
        pooling : torch.Tensor
            Downsampled output (used for next encoder block).
        """
        conv_output = self.convolution(x)
        pooling = self.downsample(conv_output)
        
        return conv_output, pooling
        
class DecoderBlock(nn.Module):
    """
    Decoder block for upsampling the feature maps.
    Upsamples the input, concatenates with skip connection, and applies ConvolutionBlock.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size = 2, stride = 2)
        self.conv = ConvolutionBlock(in_channels, out_channels)
        
    #the inputs for decoder block are the output from previous 
    #encoder/decoder block and the input from skipconnection
    def forward(self, x1, x2): 
        """
        Forward pass through the decoder block.
        
        Parameters
        ----------
        x1 : torch.Tensor
            Input from the previous decoder or bottleneck (U-path).
        x2 : torch.Tensor
            Skip connection input from encoder block.
        
        Returns
        -------
        torch.Tensor
            Output feature map after upsampling and convolution.
        """
        up = self.up(x1) #upsample the input form u-path
        comb = torch.cat([up, x2], 1) #combine the inputs
        op = self.conv(comb) #do a convolution for the combined inputs
        return op
            

class UNet(nn.Module):
    """
    UNet architecture for segmentation. Calls the blocks in right order, to 
    create the architecture.
    
    Parameters
    ----------
    in_channels : int
        Number of input channels (e.g., 3 for RGB images).
    num_classes : int
        Number of output classes for segmentation.
    """
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        #The downsample path
        self.down_conv1 = EncoderBlock(in_channels, 64)
        self.down_conv2 = EncoderBlock(64, 128)
        self.down_conv3 = EncoderBlock(128, 256)
        self.down_conv4 = EncoderBlock(256, 512)
        
        #The bottle neck
        self.bottle_neck = ConvolutionBlock(512, 1024)
        
        #The upsampling path
        self.up_conv1 = DecoderBlock(1024, 512)
        self.up_conv2 = DecoderBlock(512, 256)
        self.up_conv3 = DecoderBlock(256, 128)
        self.up_conv4 = DecoderBlock(128, 64)
        
        #The output layer
        self.output = nn.Conv2d(in_channels = 64, out_channels = num_classes, kernel_size = 1)
        
    def forward(self, x):
        """
        Forward pass through the full UNet architecture.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
        
        Returns
        -------
        torch.Tensor
            Output segmentation map of shape (B, num_classes, H, W).
        """
        
        skip1, p1 = self.down_conv1(x)
        skip2, p2 = self.down_conv2(p1)
        skip3, p3 = self.down_conv3(p2)
        skip4, p4 = self.down_conv4(p3)
        
        b = self.bottle_neck(p4)
        
        up1 = self.up_conv1(b, skip4)
        up2 = self.up_conv2(up1, skip3)
        up3 = self.up_conv3(up2, skip2)
        up4 = self.up_conv4(up3, skip1)
        
        output = self.output(up4)
        return output
        
        
        
    