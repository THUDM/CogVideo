# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
from einops import rearrange

from .movq_enc_3d import CausalConv3d, Upsample3D, DownSample3D


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False
        self.add_conv = add_conv
        if self.add_conv:
            # self.conv = nn.Conv3d(zq_channels, zq_channels, kernel_size=3, stride=1, padding=1)
            self.conv = CausalConv3d(zq_channels, zq_channels, kernel_size=3, pad_mode=pad_mode)
        # self.conv_y = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        # self.conv_b = nn.Conv3d(zq_channels, f_channels, kernel_size=1, stride=1, padding=0)
        self.conv_y = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)
        self.conv_b = CausalConv3d(zq_channels, f_channels, kernel_size=1, pad_mode=pad_mode)

    def forward(self, f, zq):
        if zq.shape[2] > 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
            zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
            zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
            zq = torch.cat([zq_first, zq_rest], dim=2)
        else:
            zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")
        if self.add_conv:
            zq = self.conv(zq)
        norm_f = self.norm_layer(f)
        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


def Normalize3D(in_channels, zq_ch, add_conv):
    return SpatialNorm3D(
        in_channels,
        zq_ch,
        norm_layer=nn.GroupNorm,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
    )


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        zq_ch=None,
        add_conv=False,
        pad_mode="constant",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        # self.conv1 = torch.nn.Conv3d(in_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize3D(out_channels, zq_ch, add_conv=add_conv)
        self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = torch.nn.Conv3d(out_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                # self.conv_shortcut = torch.nn.Conv3d(in_channels,
                #                                      out_channels,
                #                                      kernel_size=3,
                #                                      stride=1,
                #                                      padding=1)
                self.conv_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
                # self.nin_shortcut = CausalConv3d(in_channels, out_channels, kernel_size=1, pad_mode=pad_mode)

    def forward(self, x, temb, zq):
        h = x
        h = self.norm1(h, zq)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h, zq)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock2D(nn.Module):
    def __init__(self, in_channels, zq_ch=None, add_conv=False):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize3D(in_channels, zq_ch, add_conv=add_conv)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, zq):
        h_ = x
        h_ = self.norm(h_, zq)

        t = h_.shape[2]
        h_ = rearrange(h_, "b c t h w -> (b t) c h w")

        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)

        return x + h_


class MOVQDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        pad_mode="first",
        temporal_compress_times=4,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        if zq_ch is None:
            zq_ch = z_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # self.conv_in = torch.nn.Conv3d(z_channels,
        #                                block_in,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1)
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, pad_mode=pad_mode)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )
        # remove attention block
        # self.mid.attn_1 = AttnBlock2D(block_in, zq_ch, add_conv=add_conv)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        pad_mode=pad_mode,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock2D(block_in, zq_ch, add_conv=add_conv))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=False)
                else:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = Normalize3D(block_in, zq_ch, add_conv=add_conv)
        # self.conv_out = torch.nn.Conv3d(block_in,
        #                                 out_ch,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=1)
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, pad_mode=pad_mode)

    def forward(self, z, use_cp=False):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        t = z.shape[2]
        # z to block_in

        zq = z
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, zq)
        # h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.conv.weight


class NewDecoder3D(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        pad_mode="first",
        temporal_compress_times=4,
        post_quant_conv=False,
        **ignorekwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        if zq_ch is None:
            zq_ch = z_channels
        if post_quant_conv:
            self.post_quant_conv = CausalConv3d(zq_ch, z_channels, kernel_size=3, pad_mode=pad_mode)
        else:
            self.post_quant_conv = None

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print("Working with z of shape {} = {} dimensions.".format(self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        # self.conv_in = torch.nn.Conv3d(z_channels,
        #                                block_in,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1)
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3, pad_mode=pad_mode)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )
        # remove attention block
        # self.mid.attn_1 = AttnBlock2D(block_in, zq_ch, add_conv=add_conv)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            pad_mode=pad_mode,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        pad_mode=pad_mode,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock2D(block_in, zq_ch, add_conv=add_conv))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=False)
                else:
                    up.upsample = Upsample3D(block_in, resamp_with_conv, compress_time=True)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        self.norm_out = Normalize3D(block_in, zq_ch, add_conv=add_conv)
        # self.conv_out = torch.nn.Conv3d(block_in,
        #                                 out_ch,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=1)
        self.conv_out = CausalConv3d(block_in, out_ch, kernel_size=3, pad_mode=pad_mode)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        t = z.shape[2]
        # z to block_in

        zq = z
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z)
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, zq)
        # h = self.mid.attn_1(h, zq)
        h = self.mid.block_2(h, temb, zq)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.conv.weight
