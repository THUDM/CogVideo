# pytorch_diffusion + derived encoder decoder
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
from einops import rearrange


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


class CausalConv3d(nn.Module):
    @beartype
    def __init__(
        self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], pad_mode="constant", **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        if self.pad_mode == "constant":
            causal_padding_3d = (self.time_pad, 0, self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_3d, mode="constant", value=0)
        elif self.pad_mode == "first":
            pad_x = torch.cat([x[:, :, :1]] * self.time_pad, dim=2)
            x = torch.cat([pad_x, x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode="constant", value=0)
        elif self.pad_mode == "reflect":
            # reflect padding
            reflect_x = x[:, :, 1 : self.time_pad + 1, :, :].flip(dims=[2])
            if reflect_x.shape[2] < self.time_pad:
                reflect_x = torch.cat(
                    [torch.zeros_like(x[:, :, :1, :, :])] * (self.time_pad - reflect_x.shape[2]) + [reflect_x], dim=2
                )
            x = torch.cat([reflect_x, x], dim=2)
            causal_padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
            x = F.pad(x, causal_padding_2d, mode="constant", value=0)
        else:
            raise ValueError("Invalid pad mode")
        return self.conv(x)


def Normalize3D(in_channels):  # same for 3D and 2D
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv, compress_time=False):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            if x.shape[2] > 1:
                # split first frame
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0, mode="nearest")
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0, mode="nearest")
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            else:
                x = x.squeeze(2)
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
                x = x[:, :, None, :, :]
        else:
            # only interpolate 2D
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.with_conv:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class DownSample3D(nn.Module):
    def __init__(self, in_channels, with_conv, compress_time=False, out_channels=None):
        super().__init__()
        self.with_conv = with_conv
        if out_channels is None:
            out_channels = in_channels
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=0)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time:
            h, w = x.shape[-2:]
            x = rearrange(x, "b c t h w -> (b h w) c t")

            # split first frame
            x_first, x_rest = x[..., 0], x[..., 1:]

            if x_rest.shape[-1] > 0:
                x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=2, stride=2)
            x = torch.cat([x_first[..., None], x_rest], dim=-1)
            x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)

        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = self.conv(x)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        else:
            t = x.shape[2]
            x = rearrange(x, "b c t h w -> (b t) c h w")
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
            x = rearrange(x, "(b t) c h w -> b c t h w", t=t)
        return x


class ResnetBlock3D(nn.Module):
    def __init__(
        self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512, pad_mode="constant"
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize3D(in_channels)
        # self.conv1 = torch.nn.Conv3d(in_channels,
        #                              out_channels,
        #                              kernel_size=3,
        #                              stride=1,
        #                              padding=1)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, pad_mode=pad_mode)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize3D(out_channels)
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

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        h = self.norm2(h)
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
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize3D(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)

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

        # # original version, nan in fp16
        # w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # w_ = w_ * (int(c)**(-0.5))
        # # implement c**-0.5 on q
        q = q * (int(c) ** (-0.5))
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]

        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        h_ = rearrange(h_, "(b t) c h w -> b c t h w", t=t)

        return x + h_


class Encoder3D(nn.Module):
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
        double_z=True,
        pad_mode="first",
        temporal_compress_times=4,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # log2 of temporal_compress_times
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        # downsampling
        # self.conv_in = torch.nn.Conv3d(in_channels,
        #                                self.ch,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1)
        self.conv_in = CausalConv3d(in_channels, self.ch, kernel_size=3, pad_mode=pad_mode)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        pad_mode=pad_mode,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock2D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=True)
                else:
                    down.downsample = DownSample3D(block_in, resamp_with_conv, compress_time=False)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, pad_mode=pad_mode
        )
        # remove attention block
        # self.mid.attn_1 = AttnBlock2D(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in, out_channels=block_in, temb_channels=self.temb_ch, dropout=dropout, pad_mode=pad_mode
        )

        # end
        self.norm_out = Normalize3D(block_in)
        # self.conv_out = torch.nn.Conv3d(block_in,
        #                                 2*z_channels if double_z else z_channels,
        #                                 kernel_size=3,
        #                                 stride=1,
        #                                 padding=1)
        self.conv_out = CausalConv3d(
            block_in, 2 * z_channels if double_z else z_channels, kernel_size=3, pad_mode=pad_mode
        )

    def forward(self, x, use_cp=False):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)
        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        # h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h
