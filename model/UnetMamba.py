import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor
from einops.layers.torch import Reduce
from mamba_ssm.modules.mamba_simple import Mamba


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class SpatialBlock(nn.Module):
    def __init__(self, dim, mlp_ratio, act_layer, norm_layer, kernel_size_spa, drop_ratio=0., drop_path_ratio=0.5):
        super(SpatialBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.spa_attn = SpatialAttention(kernel_size_spa=kernel_size_spa)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.spa_attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size_spa):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size_spa, stride=(1, 1), padding='same')
        self.tanh = nn.Tanh()

    def forward(self, x):
        source = x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.tanh(x)
        x = x * source
        x = x + source
        return x


class Spatial(nn.Module):
    def __init__(self, in_c_spa, depth_spa, embed_dim, kernel_size_spa):
        super(Spatial, self).__init__()
        self.phase = nn.Conv2d(in_channels=2, out_channels=in_c_spa, kernel_size=(1, 1), stride=(1, 1))
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
        self.blocks = nn.Sequential(*[
            SpatialBlock(dim=embed_dim, mlp_ratio=4, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                         kernel_size_spa=kernel_size_spa)
            for _ in range(depth_spa)])
        self.dropout = nn.Dropout(p=0.2)
        self.norm = nn.LayerNorm(embed_dim)
        

    def forward(self, x):
        x = self.dropout(self.relu(self.norm(self.phase(x))))
        x = self.blocks(x)
        x = self.norm(x)
        return x

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, dim):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Conv1d(in_channels * 4, out_channels, kernel_size=3, padding=1, bias=False),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.double_conv(x)


class MambaNet(nn.Module):
    def __init__(self, dim, in_channels, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mamba = Mamba(d_model=dim)
        self.conv = nn.Conv1d(in_channels, in_channels, 3, 1, padding=1)
        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, 1, padding=1)

        self.act = nn.GELU()

        self.dropout = nn.Dropout(0.2) # JFPM:0.5
        self.num_heads = num_heads
        
        self.mamba_mh = Mamba(d_model=dim // num_heads)
        self.mambalist = nn.ModuleList(
            self.mamba_mh for _ in range(num_heads)
        )

        
    def forward(self, x):
        x = self.norm1(x)
        x = self.act(self.conv(x))
        x = self.dropout(x)
        x = self.mamba(x)

        b, l, d = x.shape
        x = x.view(b, l, d // self.num_heads, self.num_heads)
        x_new = torch.zeros_like(x)
        for i, layer in enumerate(self.mambalist):
            x_new[:, :, :, i] = layer(x[:, :, :, i])
        x = x_new.view(b, l, d)

        x = self.act(self.conv1(x))
        x = self.norm2(x)
        x = self.dropout(x)
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, dim, num_heads):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            MambaNet(dim, in_channels, num_heads),
            DoubleConv(in_channels, out_channels, dim)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dim, num_heads):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dim)
        self.mamba = MambaNet(dim // 2, in_channels, num_heads)
        

    def forward(self, x1, x2):
        x1 = self.mamba(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=6):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        source = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        x = avg_out + max_out
        x = self.sigmoid(x)
        x = x * source
        return x

class UnetMamba(nn.Module):
    def __init__(self, num_classes, img_size, embed_dim, num_heads, in_c_spa, depth_spa, kernel_size_spa):
        super(UnetMamba, self).__init__()
        self.inc = DoubleConv(img_size[0], img_size[0], embed_dim * 2)
        self.down1 = Down(img_size[0], img_size[0] * 2, embed_dim, num_heads)
        self.down2 = Down(img_size[0] * 2, img_size[0] * 4, embed_dim // 2, num_heads)
        self.down3 = Down(img_size[0] * 4, img_size[0] * 8, embed_dim // 4, num_heads)

        self.up1 = Up(img_size[0] * 8, img_size[0] * 4, embed_dim // 2, num_heads)
        self.up2 = Up(img_size[0] * 4, img_size[0] * 2, embed_dim, num_heads)
        self.up3 = Up(img_size[0] * 2, img_size[0], embed_dim * 2, num_heads)
        self.linear1 = nn.Linear(embed_dim * img_size[0], embed_dim * 5)
        self.linear2 = nn.Linear(embed_dim * 5, embed_dim)
        self.linear3 = nn.Linear(embed_dim, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten = nn.Flatten()
        self.act = nn.GELU()
        self.spatial = Spatial(in_c_spa=in_c_spa, depth_spa=depth_spa, embed_dim=embed_dim,
                               kernel_size_spa=kernel_size_spa)
        
        self.conv = nn.Conv1d(img_size[0], img_size[0], 2, 2)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = ChannelAttention(img_size[0])
    
        
    def forward(self, x):
        x_spa = self.spatial(x)
        x_spa = Reduce("b c t e -> b t e", "mean")(x_spa)

        x1 = self.inc(torch.cat([x[:, 0, :, :], x[:, 1, :, :]], dim=2))
        x2 = self.dropout(self.down1(x1))
        x3 = self.dropout(self.down2(x2))
        x4 = self.dropout(self.down3(x3))

        x = self.dropout(self.up1(x4, x3))
        x = self.dropout(self.up2(x, x2))
        x = self.dropout(self.up3(x, x1))
        
        x = self.dropout(self.conv(x))

        x = self.dropout(self.attn(x))
        x = self.flatten(x)
        x = self.dropout(self.act(self.linear1(x)))
        x = self.dropout(self.act(self.linear2(x)))
        x = self.linear3(x)
        return x


def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = UnetMamba(num_classes=args.num_classes,
                          img_size=(30, 256),
                          embed_dim=args.embed_dim,
                          num_heads=args.num_heads,
                          in_c_spa=args.in_c_spa,
                          depth_spa=args.depth_spa,
                          kernel_size_spa=args.kernel_size_spa
                          )
        
    elif args.dataset_name == 'JFPM':
        model = UnetMamba(num_classes=args.num_classes,
                          img_size=(8, 256),
                          embed_dim=args.embed_dim,
                          num_heads=args.num_heads,
                          in_c_spa=args.in_c_spa,
                          depth_spa=args.depth_spa,
                          kernel_size_spa=args.kernel_size_spa)
    else:
        return None
    return model
