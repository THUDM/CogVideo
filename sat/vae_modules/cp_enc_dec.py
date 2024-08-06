import math
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from beartype import beartype
from beartype.typing import Union, Tuple, Optional, List
from einops import rearrange

from sgm.util import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,
    get_context_parallel_group_rank,
)

# try:
from vae_modules.utils import SafeConv3d as Conv3d
# except:
#     # Degrade to normal Conv3d if SafeConv3d is not available
#     from torch.nn import Conv3d


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def exists(v):
    return v is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


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


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def _split(input_, dim):
    cp_world_size = get_context_parallel_world_size()

    if cp_world_size == 1:
        return input_

    cp_rank = get_context_parallel_rank()

    # print('in _split, cp_rank:', cp_rank, 'input_size:', input_.shape)

    inpu_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()
    dim_size = input_.size()[dim] // cp_world_size

    input_list = torch.split(input_, dim_size, dim=dim)
    output = input_list[cp_rank]

    if cp_rank == 0:
        output = torch.cat([inpu_first_frame_, output], dim=dim)
    output = output.contiguous()

    # print('out _split, cp_rank:', cp_rank, 'output_size:', output.shape)

    return output


def _gather(input_, dim):
    cp_world_size = get_context_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()

    # print('in _gather, cp_rank:', cp_rank, 'input_size:', input_.shape)

    input_first_frame_ = input_.transpose(0, dim)[:1].transpose(0, dim).contiguous()
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[1:].transpose(0, dim).contiguous()

    tensor_list = [torch.empty_like(torch.cat([input_first_frame_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]

    if cp_rank == 0:
        input_ = torch.cat([input_first_frame_, input_], dim=dim)

    tensor_list[cp_rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    output = torch.cat(tensor_list, dim=dim).contiguous()

    # print('out _gather, cp_rank:', cp_rank, 'output_size:', output.shape)

    return output


def _conv_split(input_, dim, kernel_size):
    cp_world_size = get_context_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    # print('in _conv_split, cp_rank:', cp_rank, 'input_size:', input_.shape)

    cp_rank = get_context_parallel_rank()

    dim_size = (input_.size()[dim] - kernel_size) // cp_world_size

    if cp_rank == 0:
        output = input_.transpose(dim, 0)[: dim_size + kernel_size].transpose(dim, 0)
    else:
        # output = input_.transpose(dim, 0)[cp_rank * dim_size + 1:(cp_rank + 1) * dim_size + kernel_size].transpose(dim, 0)
        output = input_.transpose(dim, 0)[
            cp_rank * dim_size + kernel_size : (cp_rank + 1) * dim_size + kernel_size
        ].transpose(dim, 0)
    output = output.contiguous()

    # print('out _conv_split, cp_rank:', cp_rank, 'input_size:', output.shape)

    return output


def _conv_gather(input_, dim, kernel_size):
    cp_world_size = get_context_parallel_world_size()

    # Bypass the function if context parallel is 1
    if cp_world_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()

    # print('in _conv_gather, cp_rank:', cp_rank, 'input_size:', input_.shape)

    input_first_kernel_ = input_.transpose(0, dim)[:kernel_size].transpose(0, dim).contiguous()
    if cp_rank == 0:
        input_ = input_.transpose(0, dim)[kernel_size:].transpose(0, dim).contiguous()
    else:
        input_ = input_.transpose(0, dim)[max(kernel_size - 1, 0) :].transpose(0, dim).contiguous()

    tensor_list = [torch.empty_like(torch.cat([input_first_kernel_, input_], dim=dim))] + [
        torch.empty_like(input_) for _ in range(cp_world_size - 1)
    ]
    if cp_rank == 0:
        input_ = torch.cat([input_first_kernel_, input_], dim=dim)

    tensor_list[cp_rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=group)

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim).contiguous()

    # print('out _conv_gather, cp_rank:', cp_rank, 'input_size:', output.shape)

    return output


def _pass_from_previous_rank(input_, dim, kernel_size):
    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    # print('in _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)

    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    input_ = input_.transpose(0, dim)

    # pass from last rank
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
    if cp_rank > 0:
        recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)

    if cp_rank == 0:
        input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
    else:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    input_ = input_.transpose(0, dim).contiguous()

    # print('out _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)

    return input_


def _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding=None):
    # Bypass the function if kernel size is 1
    if kernel_size == 1:
        return input_

    group = get_context_parallel_group()
    cp_rank = get_context_parallel_rank()
    cp_group_rank = get_context_parallel_group_rank()
    cp_world_size = get_context_parallel_world_size()

    # print('in _pass_from_previous_rank, cp_rank:', cp_rank, 'input_size:', input_.shape)

    global_rank = torch.distributed.get_rank()
    global_world_size = torch.distributed.get_world_size()

    input_ = input_.transpose(0, dim)

    # pass from last rank
    send_rank = global_rank + 1
    recv_rank = global_rank - 1
    if send_rank % cp_world_size == 0:
        send_rank -= cp_world_size
    if recv_rank % cp_world_size == cp_world_size - 1:
        recv_rank += cp_world_size

    # req_send = torch.distributed.isend(input_[-kernel_size + 1:].contiguous(), send_rank, group=group)
    # recv_buffer = torch.empty_like(input_[-kernel_size + 1:]).contiguous()
    # req_recv = torch.distributed.recv(recv_buffer, recv_rank, group=group)
    # req_recv.wait()
    recv_buffer = torch.empty_like(input_[-kernel_size + 1 :]).contiguous()
    if cp_rank < cp_world_size - 1:
        req_send = torch.distributed.isend(input_[-kernel_size + 1 :].contiguous(), send_rank, group=group)
    if cp_rank > 0:
        req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)
    # req_send = torch.distributed.isend(input_[-kernel_size + 1:].contiguous(), send_rank, group=group)
    # req_recv = torch.distributed.irecv(recv_buffer, recv_rank, group=group)

    if cp_rank == 0:
        if cache_padding is not None:
            input_ = torch.cat([cache_padding.transpose(0, dim).to(input_.device), input_], dim=0)
        else:
            input_ = torch.cat([input_[:1]] * (kernel_size - 1) + [input_], dim=0)
    else:
        req_recv.wait()
        input_ = torch.cat([recv_buffer, input_], dim=0)

    input_ = input_.transpose(0, dim).contiguous()
    return input_


def _drop_from_previous_rank(input_, dim, kernel_size):
    input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim)
    return input_


class _ConvolutionScatterToContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_split(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_gather(grad_output, ctx.dim, ctx.kernel_size), None, None


class _ConvolutionGatherFromContextParallelRegion(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _conv_gather(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_split(grad_output, ctx.dim, ctx.kernel_size), None, None


class _ConvolutionPassFromPreviousRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _pass_from_previous_rank(input_, dim, kernel_size)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None


class _FakeCPConvolutionPassFromPreviousRank(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, kernel_size, cache_padding):
        ctx.dim = dim
        ctx.kernel_size = kernel_size
        return _fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding)

    @staticmethod
    def backward(ctx, grad_output):
        return _drop_from_previous_rank(grad_output, ctx.dim, ctx.kernel_size), None, None, None


def conv_scatter_to_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionScatterToContextParallelRegion.apply(input_, dim, kernel_size)


def conv_gather_from_context_parallel_region(input_, dim, kernel_size):
    return _ConvolutionGatherFromContextParallelRegion.apply(input_, dim, kernel_size)


def conv_pass_from_last_rank(input_, dim, kernel_size):
    return _ConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size)


def fake_cp_pass_from_previous_rank(input_, dim, kernel_size, cache_padding):
    return _FakeCPConvolutionPassFromPreviousRank.apply(input_, dim, kernel_size, cache_padding)


class ContextParallelCausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride=1, **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)

        time_pad = time_kernel_size - 1
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_kernel_size = time_kernel_size
        self.temporal_dim = 2

        stride = (stride, stride, stride)
        dilation = (1, 1, 1)
        self.conv = Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        self.cache_padding = None

    def forward(self, input_, clear_cache=True):
        # if input_.shape[2] == 1: # handle image
        #     # first frame padding
        #     input_parallel = torch.cat([input_] * self.time_kernel_size, dim=2)
        # else:
        #     input_parallel = conv_pass_from_last_rank(input_, self.temporal_dim, self.time_kernel_size)

        # padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        # input_parallel = F.pad(input_parallel, padding_2d, mode = 'constant', value = 0)

        # output_parallel = self.conv(input_parallel)
        # output = output_parallel
        # return output

        input_parallel = fake_cp_pass_from_previous_rank(
            input_, self.temporal_dim, self.time_kernel_size, self.cache_padding
        )

        del self.cache_padding
        self.cache_padding = None
        if not clear_cache:
            cp_rank, cp_world_size = get_context_parallel_rank(), get_context_parallel_world_size()
            global_rank = torch.distributed.get_rank()
            if cp_world_size == 1:
                self.cache_padding = (
                    input_parallel[:, :, -self.time_kernel_size + 1 :].contiguous().detach().clone().cpu()
                )
            else:
                if cp_rank == cp_world_size - 1:
                    torch.distributed.isend(
                        input_parallel[:, :, -self.time_kernel_size + 1 :].contiguous(),
                        global_rank + 1 - cp_world_size,
                        group=get_context_parallel_group(),
                    )
                if cp_rank == 0:
                    recv_buffer = torch.empty_like(input_parallel[:, :, -self.time_kernel_size + 1 :]).contiguous()
                    torch.distributed.recv(
                        recv_buffer, global_rank - 1 + cp_world_size, group=get_context_parallel_group()
                    )
                    self.cache_padding = recv_buffer.contiguous().detach().clone().cpu()

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        input_parallel = F.pad(input_parallel, padding_2d, mode="constant", value=0)

        output_parallel = self.conv(input_parallel)
        output = output_parallel
        return output


class ContextParallelGroupNorm(torch.nn.GroupNorm):
    def forward(self, input_):
        gather_flag = input_.shape[2] > 1
        if gather_flag:
            input_ = conv_gather_from_context_parallel_region(input_, dim=2, kernel_size=1)
        output = super().forward(input_)
        if gather_flag:
            output = conv_scatter_to_context_parallel_region(output, dim=2, kernel_size=1)
        return output


def Normalize(in_channels, gather=False, **kwargs):  # same for 3D and 2D
    if gather:
        return ContextParallelGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    else:
        return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class SpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        gather=False,
        **norm_layer_params,
    ):
        super().__init__()
        if gather:
            self.norm_layer = ContextParallelGroupNorm(num_channels=f_channels, **norm_layer_params)
        else:
            self.norm_layer = torch.nn.GroupNorm(num_channels=f_channels, **norm_layer_params)
        # self.norm_layer = norm_layer(num_channels=f_channels, **norm_layer_params)
        if freeze_norm_layer:
            for p in self.norm_layer.parameters:
                p.requires_grad = False

        self.add_conv = add_conv
        if add_conv:
            self.conv = ContextParallelCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
            )

        self.conv_y = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )
        self.conv_b = ContextParallelCausalConv3d(
            chan_in=zq_channels,
            chan_out=f_channels,
            kernel_size=1,
        )

    def forward(self, f, zq, clear_fake_cp_cache=True):
        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]
            zq_first = torch.nn.functional.interpolate(zq_first, size=f_first_size, mode="nearest")
            zq_rest = torch.nn.functional.interpolate(zq_rest, size=f_rest_size, mode="nearest")
            zq = torch.cat([zq_first, zq_rest], dim=2)
        else:
            zq = torch.nn.functional.interpolate(zq, size=f.shape[-3:], mode="nearest")

        if self.add_conv:
            zq = self.conv(zq, clear_cache=clear_fake_cp_cache)

        # f = conv_gather_from_context_parallel_region(f, dim=2, kernel_size=1)
        norm_f = self.norm_layer(f)
        # norm_f = conv_scatter_to_context_parallel_region(norm_f, dim=2, kernel_size=1)

        new_f = norm_f * self.conv_y(zq) + self.conv_b(zq)
        return new_f


def Normalize3D(
    in_channels,
    zq_ch,
    add_conv,
    gather=False,
):
    return SpatialNorm3D(
        in_channels,
        zq_ch,
        gather=gather,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
    )


class Upsample3D(nn.Module):
    def __init__(
        self,
        in_channels,
        with_conv,
        compress_time=False,
    ):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.compress_time = compress_time

    def forward(self, x):
        if self.compress_time and x.shape[2] > 1:
            if x.shape[2] % 2 == 1:
                # split first frame
                x_first, x_rest = x[:, :, 0], x[:, :, 1:]

                x_first = torch.nn.functional.interpolate(x_first, scale_factor=2.0, mode="nearest")
                x_rest = torch.nn.functional.interpolate(x_rest, scale_factor=2.0, mode="nearest")
                x = torch.cat([x_first[:, :, None, :, :], x_rest], dim=2)
            else:
                x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")

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
        if self.compress_time and x.shape[2] > 1:
            h, w = x.shape[-2:]
            x = rearrange(x, "b c t h w -> (b h w) c t")

            if x.shape[-1] % 2 == 1:
                # split first frame
                x_first, x_rest = x[..., 0], x[..., 1:]

                if x_rest.shape[-1] > 0:
                    x_rest = torch.nn.functional.avg_pool1d(x_rest, kernel_size=2, stride=2)
                x = torch.cat([x_first[..., None], x_rest], dim=-1)
                x = rearrange(x, "(b h w) c t -> b c t h w", h=h, w=w)
            else:
                x = torch.nn.functional.avg_pool1d(x, kernel_size=2, stride=2)
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


class ContextParallelResnetBlock3D(nn.Module):
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
        gather_norm=False,
        normalization=Normalize,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )

        self.conv1 = ContextParallelCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
            add_conv=add_conv,
            gather=gather_norm,
        )
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = ContextParallelCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ContextParallelCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                )
            else:
                self.nin_shortcut = Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb, zq=None, clear_fake_cp_cache=True):
        h = x

        # if isinstance(self.norm1, torch.nn.GroupNorm):
        #     h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)
        if zq is not None:
            h = self.norm1(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            h = self.norm1(h)
        # if isinstance(self.norm1, torch.nn.GroupNorm):
        #     h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)

        h = nonlinearity(h)
        h = self.conv1(h, clear_cache=clear_fake_cp_cache)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        # if isinstance(self.norm2, torch.nn.GroupNorm):
        #     h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)
        if zq is not None:
            h = self.norm2(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        else:
            h = self.norm2(h)
        # if isinstance(self.norm2, torch.nn.GroupNorm):
        #     h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)

        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h, clear_cache=clear_fake_cp_cache)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, clear_cache=clear_fake_cp_cache)
            else:
                x = self.nin_shortcut(x)

        return x + h


class ContextParallelEncoder3D(nn.Module):
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
        gather_norm=False,
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

        self.conv_in = ContextParallelCausalConv3d(
            chan_in=in_channels,
            chan_out=self.ch,
            kernel_size=3,
        )

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
                    ContextParallelResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        dropout=dropout,
                        temb_channels=self.temb_ch,
                        gather_norm=gather_norm,
                    )
                )
                block_in = block_out
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
        self.mid.block_1 = ContextParallelResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
        )

        self.mid.block_2 = ContextParallelResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            gather_norm=gather_norm,
        )

        # end
        self.norm_out = Normalize(block_in, gather=gather_norm)

        self.conv_out = ContextParallelCausalConv3d(
            chan_in=block_in,
            chan_out=2 * z_channels if double_z else z_channels,
            kernel_size=3,
        )

    def forward(self, x):
        # timestep embedding
        temb = None

        # downsampling
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.block_2(h, temb)

        # end
        # h = conv_gather_from_context_parallel_region(h, dim=2, kernel_size=1)
        h = self.norm_out(h)
        # h = conv_scatter_to_context_parallel_region(h, dim=2, kernel_size=1)

        h = nonlinearity(h)
        h = self.conv_out(h)

        return h


class ContextParallelDecoder3D(nn.Module):
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
        gather_norm=False,
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

        self.conv_in = ContextParallelCausalConv3d(
            chan_in=z_channels,
            chan_out=block_in,
            kernel_size=3,
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ContextParallelResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=Normalize3D,
            gather_norm=gather_norm,
        )

        self.mid.block_2 = ContextParallelResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=Normalize3D,
            gather_norm=gather_norm,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ContextParallelResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        normalization=Normalize3D,
                        gather_norm=gather_norm,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = Upsample3D(block_in, with_conv=resamp_with_conv, compress_time=False)
                else:
                    up.upsample = Upsample3D(block_in, with_conv=resamp_with_conv, compress_time=True)
            self.up.insert(0, up)

        self.norm_out = Normalize3D(block_in, zq_ch, add_conv=add_conv, gather=gather_norm)

        self.conv_out = ContextParallelCausalConv3d(
            chan_in=block_in,
            chan_out=out_ch,
            kernel_size=3,
        )

    def forward(self, z, clear_fake_cp_cache=True):
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        t = z.shape[2]
        # z to block_in

        zq = z
        h = self.conv_in(z, clear_cache=clear_fake_cp_cache)

        # middle
        h = self.mid.block_1(h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        h = self.mid.block_2(h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, zq, clear_fake_cp_cache=clear_fake_cp_cache)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq, clear_fake_cp_cache=clear_fake_cp_cache)
        h = nonlinearity(h)
        h = self.conv_out(h, clear_cache=clear_fake_cp_cache)

        return h

    def get_last_layer(self):
        return self.conv_out.conv.weight
