# Adapted from Topdu/OpenOCR at 1ccfc6ee6161f7133b192e16af3a9265273997ba.
# Licensed under Apache-2.0; see LICENSE and NOTICE in this directory.
# Changes: local imports and extraction of the required network components.

import numpy as np
import torch
from torch import nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_
from .layers import DropPath, Identity, Mlp

class ConvBNLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class ConvMixer(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        HW=[8, 25],
        local_k=[3, 3],
    ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim,
                                     dim,
                                     local_k,
                                     1, [local_k[0] // 2, local_k[1] // 2],
                                     groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose(1, 2).reshape([x.shape[0], self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        mixer='Global',
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(H * W,
                              H + hk - 1,
                              W + wk - 1,
                              dtype=torch.float32,
                              requires_grad=False)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.0
            mask = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask[mask >= 1] = -np.inf
            self.register_buffer('mask', mask[None, None, :, :])
        self.mixer = mixer

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         attn_mask=mask,
        #         dropout_p=self.attn_drop.p
        #     )
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.mixer == 'Local':
            attn += self.mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mixer='Global',
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer='nn.LayerNorm',
        eps=1e-6,
        prenorm=True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            if norm_layer != 'nn.LayerNorm':
                raise ValueError('Unsupported CTC normalization layer')
            self.norm1 = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == 'Conv':
            self.mixer = ConvMixer(dim,
                                   num_heads=num_heads,
                                   HW=HW,
                                   local_k=local_mixer)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
