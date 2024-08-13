import math
import torch
import torch.distributed
import torch.nn as nn
from ..util import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_context_parallel_world_size,

)

_USE_CP = True


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
        output = input_.transpose(dim, 0)[cp_rank * dim_size + 1 : (cp_rank + 1) * dim_size + kernel_size].transpose(
            dim, 0
        )
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
        input_ = input_.transpose(0, dim)[kernel_size - 1 :].transpose(0, dim).contiguous()

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