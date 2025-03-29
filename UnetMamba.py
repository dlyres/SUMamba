import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import Tensor
# from zeta.nn import SSM
from einops.layers.torch import Reduce
from model.xlstm import xLSTM as xlstm
from mamba_ssm.modules.mamba_simple import Mamba


# class SSVEPMambaBlock(nn.Module):

#     def __init__(
#         self,
#         dim: int,
#         dt_rank: int,
#         dim_inner: int,
#         d_state: int,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.dt_rank = dt_rank
#         self.dim_inner = dim_inner
#         self.d_state = d_state

#         self.forward_conv1d = nn.Conv1d(
#             in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding='same'
#         )
#         self.backward_conv1d = nn.Conv1d(
#             in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding='same'
#         )
#         self.norm = nn.LayerNorm(dim)
#         self.silu = nn.SiLU()
#         self.ssm = SSM(dim, dt_rank, dim_inner, d_state)

#         # Linear layer for z and x
#         self.proj = nn.Linear(dim, dim)

#         # Softplus
#         self.softplus = nn.Softplus()


#     def forward(self, x: torch.Tensor):
#         b, s, d = x.shape

#         # Skip connection
#         skip = x

#         # Normalization
#         x = self.norm(x)

#         # Split x into x1 and x2 with linears
#         z1 = self.proj(x)
#         x = self.proj(x)

#         # forward con1d
#         x1 = self.process_direction(
#             x,
#             self.forward_conv1d,
#             self.ssm,
#         )

#         # backward conv1d
#         x2 = self.process_direction(
#             x,
#             self.backward_conv1d,
#             self.ssm,
#         )

#         # Activation
#         z = self.silu(z1)

#         # Matmul
#         x1 *= z
#         x2 *= z

#         # Residual connection
#         return x1 + x2 + skip

#     def process_direction(
#         self,
#         x: Tensor,
#         conv1d: nn.Conv1d,
#         ssm: SSM,
#     ):
#         x = rearrange(x, "b s d -> b d s")
#         x = self.softplus(conv1d(x))
#         # print(f"Conv1d: {x}")
#         x = rearrange(x, "b d s -> b s d")
#         x = ssm(x)
#         return x

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


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim, norm_layer=None):
        super().__init__()
        self.in_c = in_c
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(self.in_c, self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # flatten: [B, C, H, W] -> [B, C, num_patches]
        # transpose: [B, C, num_patches] -> [B, num_patches, C]
        x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.norm(x)
        return x




class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        # [batch_size, num_patches + 1, total_embed_dim]
        B, N, C = x.shape

        # qkv(): -> [batch_size, num_patches + 1, 3 * total_embed_dim]
        # reshape: -> [batch_size, num_patches + 1, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size, num_heads, embed_dim_per_head, num_patches + 1]
        # @: multiply -> [batch_size, num_heads, num_patches + 1, num_patches + 1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size, num_heads, num_patches + 1, embed_dim_per_head]
        # transpose: -> [batch_size, num_patches + 1, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size, num_patches + 1, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



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


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_ratio=0., attn_drop_ratio=0.,
                 drop_path_ratio=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _init_vit_weights(m):
    if isinstance(m, nn.Linear or nn.Conv1d or nn.Conv2d):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class Vit_attn(nn.Module):
    def __init__(self, img_size, patch_size, in_c, embed_dim, depth, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.2, attn_drop_ratio=0.2, drop_path_ratio=0.5, norm_layer=None, act_layer=None):

        super(Vit_attn, self).__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_c = in_c
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        num_patches = (self.img_size[0] // self.patch_size[0]) * (self.img_size[1] // self.patch_size[1])

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_ratio)

        self.blocks = nn.Sequential(*[
            Block(dim=self.embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio,
                  norm_layer=norm_layer, act_layer=act_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(self.embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.01)

        self.apply(_init_vit_weights)

    def forward(self, x):
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
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
        # self.double_conv = nn.Sequential(
        #     # MambaNet(dim, in_channels, out_channels),
        #     # nn.Dropout(0.2),
        #     nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.LayerNorm(dim),
        #     nn.GELU(),
        # )
        
    def forward(self, x):
        return self.double_conv(x)


class MambaNet(nn.Module):
    def __init__(self, dim, in_channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.mamba = Mamba(d_model=dim)
        self.conv = nn.Conv1d(in_channels, in_channels, 3, 1, padding=1)
        self.conv1 = nn.Conv1d(in_channels, in_channels, 3, 1, padding=1)

        self.act = nn.GELU()

        self.dropout = nn.Dropout(0.2)
        self.attn = Attention(dim, num_heads=8, attn_drop_ratio=0.2, proj_drop_ratio=0.2)
        self.mamba_mh = Mamba(d_model=dim // 4)
        self.mambalist = nn.ModuleList(
            self.mamba_mh for _ in range(4)
        )
        
    def forward(self, x):
        # x = self.norm1(x)
        # x = self.act(self.conv(x))
        # x = self.dropout(x)
        # # x = self.mamba(x)

        # b, l, d = x.shape
        # x = x.view(b, l, d // 4, 4)
        # x_new = torch.zeros_like(x)
        # for i, layer in enumerate(self.mambalist):
        #     x_new[:, :, :, i] = layer(x[:, :, :, i])
        # x = x_new.view(b, l, d)

        # x = self.act(self.conv1(x))
        # x = self.norm2(x)
        # x = self.dropout(x)
        x = self.attn(x)
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, dim):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2),
            MambaNet(dim, in_channels),
            DoubleConv(in_channels, out_channels, dim)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, dim):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, dim)
        # self.mamba = SSVEPMambaBlock(dim=dim // 2, dt_rank=dim // 16, dim_inner=dim // 2, d_state=16)
        self.mamba = nn.Sequential(
            MambaNet(dim // 2, in_channels)
        )

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
    def __init__(self, num_classes, in_c, img_size, patch_size, embed_dim, dt_rank, dim_inner, d_state):
        super(UnetMamba, self).__init__()
        self.inc = DoubleConv(30, 30, 512)
        self.down1 = Down(30, 60, 256)
        self.down2 = Down(60, 120, 128)
        self.down3 = Down(120, 240, 64)

        self.up1 = Up(240, 120, 128)
        self.up2 = Up(120, 60, 256)
        self.up3 = Up(60, 30, 512)
        self.linear1 = nn.Linear(256 * 30, 256 * 5)
        self.linear2 = nn.Linear(256 * 5, 256)
        self.linear3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.4)
        self.flatten = nn.Flatten()
        self.act = nn.GELU()
        self.spatial = Spatial(in_c_spa=4, depth_spa=6, embed_dim=embed_dim,
                               kernel_size_spa=(7, 7))
        
        self.conv = nn.Conv1d(30, 30, 2, 2)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = ChannelAttention(30)
    
        
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

        x = self.dropout(self.attn(x + x_spa))
        x = self.flatten(x)
        x = self.dropout(self.act(self.linear1(x)))
        x = self.dropout(self.act(self.linear2(x)))
        x = self.linear3(x)
        return x


def make_model(args):
    if args.dataset_name == 'BETA' or args.dataset_name == 'Benchmark':
        model = UnetMamba(num_classes=args.num_classes,
                          in_c=args.in_c,
                          img_size=(30, 256),
                          patch_size=args.patch_size,
                          embed_dim=args.embed_dim,
                          d_state=args.d_state,
                          dt_rank=args.dt_rank,
                          dim_inner=args.dim_inner)
        
    elif args.dataset_name == 'JFPM':
        model = UnetMamba(num_classes=args.num_classes,
                          in_c=args.in_c,
                          img_size=(8, 256),
                          patch_size=args.patch_size,
                          embed_dim=args.embed_dim,
                          d_state=args.d_state,
                          dt_rank=args.dt_rank,
                          dim_inner=args.dim_inner)
    else:
        return None
    return model
