"""
---
title: U-Net
summary: >
    PyTorch implementation and tutorial of U-Net model.
---

# U-Net

This is an implementation of the U-Net model from the paper,
[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).

U-Net consists of a contracting path and an expansive path.
The contracting path is a series of convolutional layers and pooling layers,
where the resolution of the feature map gets progressively reduced.
Expansive path is a series of up-sampling layers and convolutional layers
where the resolution of the feature map gets progressively increased.

At every step in the expansive path the corresponding feature map from the contracting path
concatenated with the current feature map.

![U-Net diagram from paper](unet.png)

Here is the [training code](experiment.html) for an experiment that trains a U-Net
on [Carvana dataset](carvana.html).
"""
import torch
import torch.nn.functional as F
from torch import nn


class DoubleConvolution(nn.Module):
    """
    ### Two $3 \times 3$ Convolution Layers

    Each step in the contraction path and expansive path have two $3 \times 3$
    convolutional layers followed by ReLU activations.

    In the U-Net paper they used $0$ padding,
    but we use $1$ padding so that final feature map is not cropped.
    """

    def __init__(self, in_channels: int, out_channels: int):
        """
        :param in_channels: is the number of input channels
        :param out_channels: is the number of output channels
        """
        super().__init__()

        # First $3 \times 3$ convolutional layer
        self.first = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()
        # Second $3 \times 3$ convolutional layer
        self.second = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

    def forward(self, x: torch.Tensor):
        # Apply the two convolution layers and activations
        x = self.first(x)
        x = self.act1(x)
        x = self.second(x)
        return self.act2(x)


class DownSample(nn.Module):
    """
    ### Down-sample

    Each step in the contracting path down-samples the feature map with
    a $2 \times 2$ max pooling layer.
    """

    def __init__(self):
        super().__init__()
        # Max pooling layer
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        return self.pool(x)


class UpSample(nn.Module):
    """
    ### Up-sample

    Each step in the expansive path up-samples the feature map with
    a $2 \times 2$ up-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        # Up-convolution
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor):
        return self.up(x)
    

class ViTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, dropout, squeeze_factor):
        super().__init__()
        patch_depth = in_channels * patch_size * patch_size
        squeeze_channels = out_channels // squeeze_factor
        d_model = squeeze_channels * patch_size * patch_size
        self.first_linear = nn.Linear(patch_depth, d_model)
        # self.positional_encodings = nn.Parameter(torch.zeros(1, 5000, d_model), requires_grad=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=patch_size, 
                dim_feedforward=d_model * 2, 
                dropout=dropout, 
                batch_first=True, 
            ), 
            num_layers=1
        )
        self.second_linear = nn.Linear(d_model, out_channels * patch_size * patch_size)
        self.patch_size = patch_size

    def forward(self, x: torch.Tensor):
        '''
        Input:
            x shape: B, C, H, W
        Return:
            shape: B, C, H, W
        '''
        B, C, H, W = x.shape
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size) # B, patch_depth, num_patch
        x = self.first_linear(x.transpose(1, 2)) # B, num_patch, d_model
        # pe = self.positional_encodings[:, :x.shape[1], :]
        # x = pe + x
        x = self.second_linear(self.transformer(x)).transpose(1, 2) # B, out_channels * patch_size * patch_size, num_patch
        return F.fold(x, output_size=(H, W), kernel_size=self.patch_size, stride=self.patch_size)
    

class CropAndConcat(nn.Module):
    """
    ### Crop and Concatenate the feature map

    At every step in the expansive path the corresponding feature map from the contracting path
    concatenated with the current feature map.
    """
    def __init__(self, channels, patch_size, dropout, squeeze_factor) -> None:
        super().__init__()
        self.conv = ViTBlock(channels * 5, channels * 20, patch_size, dropout, squeeze_factor)

    def forward(self, x: torch.Tensor, contracting_x: torch.Tensor):
        """
        :param x: current feature map in the expansive path
        :param contracting_x: corresponding feature map from the contracting path
        """
        BT, C, H, W = contracting_x.shape
        contracting_x = contracting_x.reshape(-1, 5, C, H, W)
        contracting_x = self.conv(contracting_x.reshape(-1, 5 * C, H, W)).reshape(-1, 20, C, H, W)
        contracting_x = contracting_x.reshape(-1, C, H, W)
        # Crop the feature map from the contracting path to the size of the current feature map
        # contracting_x = torchvision.transforms.functional.center_crop(contracting_x, [x.shape[2], x.shape[3]])
        # Concatenate the feature maps
        x = torch.cat([x, contracting_x], dim=1)
        #
        return x


class Model(nn.Module):
    """
    ## U-Net
    """
    def __init__(self, configs):
        """
        :param in_channels: number of channels in the input image
        :param out_channels: number of channels in the result feature map
        """
        super().__init__()
        in_channels = configs.enc_in
        out_channels = configs.c_out
        self.pred_len = configs.pred_len
        scale = configs.scale

        self.pad = nn.ReplicationPad3d((0, 0, 8, 7, 0, 0))
        # Double convolution layers for the contracting path.
        # The number of features gets doubled at each step starting from $64$.
        self.down_conv = nn.ModuleList([DoubleConvolution(i, o) for i, o in
                                        [(in_channels, 8 * scale), (8 * scale, 16 * scale), (16 * scale, 32 * scale), (32 * scale, 64 * scale)]])
        # Down sampling layers for the contracting path
        self.down_sample = nn.ModuleList([DownSample() for _ in range(4)])

        # The two convolution layers at the lowest resolution (the bottom of the U).
        self.middle_conv = ViTBlock(64 * configs.seq_len * scale, 128 * configs.pred_len * scale, patch_size=1, dropout=configs.dropout, squeeze_factor=configs.squeeze)

        # Up sampling layers for the expansive path.
        # The number of features is halved with up-sampling.
        self.up_sample = nn.ModuleList([UpSample(i * scale, o * scale) for i, o in
                                        [(128, 64), (64, 32), (32, 16), (16, 8)]])
        # Double convolution layers for the expansive path.
        # Their input is the concatenation of the current feature map and the feature map from the
        # contracting path. Therefore, the number of input features is double the number of features
        # from up-sampling.
        self.up_conv = nn.ModuleList([DoubleConvolution(i * scale, o * scale) for i, o in
                                      [(128, 64), (64, 32), (32, 16), (16, 8)]])
        # Crop and concatenate layers for the expansive path.
        self.concat = nn.ModuleList([CropAndConcat(i * scale, j, configs.dropout, configs.squeeze) for i, j in [(64, 1), (32, 2), (16, 4), (8, 8)]])
        # Final $1 \times 1$ convolution layer to produce the output
        self.final_conv = nn.Conv2d(8 * scale, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: input image
        """
        x = self.pad(x)
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        # To collect the outputs of contracting path for later concatenation with the expansive path.
        pass_through = []
        # Contracting path
        for i in range(len(self.down_conv)):
            # Two $3 \times 3$ convolutional layers
            x = self.down_conv[i](x)
            # Collect the output
            pass_through.append(x)
            # Down-sample
            x = self.down_sample[i](x)

        x = x.reshape(B, T, x.shape[1], x.shape[2], x.shape[3])
        # Two $3 \times 3$ convolutional layers at the bottom of the U-Net
        x = self.middle_conv(x.reshape(B, -1, x.shape[3], x.shape[4]))
        x = x.reshape(B, self.pred_len, -1, x.shape[2], x.shape[3])
        x = x.reshape(B * self.pred_len, -1, x.shape[3], x.shape[4])

        # Expansive path
        for i in range(len(self.up_conv)):
            # Up-sample
            x = self.up_sample[i](x)
            # Concatenate the output of the contracting path
            x = self.concat[i](x, pass_through.pop())
            # Two $3 \times 3$ convolutional layers
            x = self.up_conv[i](x)

        # Final $1 \times 1$ convolution layer
        x = self.final_conv(x)

        #
        return x.reshape(B, self.pred_len, -1, H, W)[:, :, :, 7:-8, :]