import torch
import numpy as np
import pytest

from models import UNet, UNetViT


def random_tensor(shape):
    return torch.randn(*shape)


def test_double_conv_down_up_sample():
    x = random_tensor((1, 3, 16, 16))
    # DoubleConvolution
    conv = UNet.DoubleConvolution(3, 5)
    y = conv(x)
    assert y.shape[1] == 5
    # DownSample halves spatial dims
    down = UNet.DownSample()
    y2 = down(x)
    assert y2.shape[2] == x.shape[2] // 2
    # UpSample doubles spatial dims
    up = UNet.UpSample(3, 1)
    y3 = up(x)
    assert y3.shape[2] == x.shape[2] * 2


def test_crop_and_concat():
    # create contracting map larger than x in height
    x = torch.randn(1, 2, 8, 8)
    contracting = torch.randn(1, 2, 10, 8)
    concat = UNet.CropAndConcat(channels=2)
    out = concat(x, contracting)
    # output channels should be sum
    assert out.shape[1] == 4
    # output height equals x height
    assert out.shape[2] == 8


def test_unet_forward_shape(dummy_config):
    cfg = dummy_config
    model = UNet.Model(cfg)
    B, T, C, H, W = 1, 2, cfg.enc_in, 16, 16
    inp = torch.randn(B, T, C, H, W)
    out = model(inp)
    # expected shape [B, pred_len, c_out, H-15, W]
    expected_H = H - 15
    assert out.shape == (B, cfg.pred_len, cfg.c_out, expected_H, W)


def test_vitblock_identity_shape():
    block = UNetViT.ViTBlock(in_channels=2, out_channels=2, patch_size=1, dropout=0, squeeze_factor=1)
    x = torch.randn(1, 2, 8, 8)
    y = block(x)
    assert y.shape == x.shape


def test_unetvit_forward_shape(dummy_config):
    cfg = dummy_config
    model = UNetViT.Model(cfg)
    B, T, C, H, W = 1, 2, cfg.enc_in, 16, 16
    inp = torch.randn(B, T, C, H, W)
    out = model(inp)
    expected_H = H - 15
    assert out.shape == (B, cfg.pred_len, cfg.c_out, expected_H, W)
