from typing import Any, Union
from math import log2
from beartype import beartype

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import grad as torch_grad
from torch.cuda.amp import autocast

import torchvision
from torchvision.models import VGG16_Weights
from einops import rearrange, einsum, repeat
from einops.layers.torch import Rearrange
from kornia.filters import filter3d

from ..magvit2_pytorch import Residual, FeedForward, LinearSpaceAttention
from .lpips import LPIPS

from sgm.modules.autoencoding.vqvae.movq_enc_3d import CausalConv3d, DownSample3D
from sgm.util import instantiate_from_config


def exists(v):
    return v is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def leaky_relu(p=0.1):
    return nn.LeakyReLU(p)


def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()


def hinge_gen_loss(fake):
    return -fake.mean()


@autocast(enabled=False)
@beartype
def grad_layer_wrt_loss(loss: Tensor, layer: nn.Parameter):
    return torch_grad(outputs=loss, inputs=layer, grad_outputs=torch.ones_like(loss), retain_graph=True)[0].detach()


def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, "b c f ... -> b f c ...")
    batch_indices = torch.arange(batch, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")
    images = video[batch_indices, frame_indices]
    images = rearrange(images, "b 1 c ... -> b c ...")
    return images


def gradient_penalty(images, output):
    batch_size = images.shape[0]

    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size(), device=images.device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = rearrange(gradients, "b ... -> b (...)")
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


# discriminator with anti-aliased downsampling (blurpool Zhang et al.)


class Blur(nn.Module):
    def __init__(self):
        super().__init__()
        f = torch.Tensor([1, 2, 1])
        self.register_buffer("f", f)

    def forward(self, x, space_only=False, time_only=False):
        assert not (space_only and time_only)

        f = self.f

        if space_only:
            f = einsum("i, j -> i j", f, f)
            f = rearrange(f, "... -> 1 1 ...")
        elif time_only:
            f = rearrange(f, "f -> 1 f 1 1")
        else:
            f = einsum("i, j, k -> i j k", f, f, f)
            f = rearrange(f, "... -> 1 ...")

        is_images = x.ndim == 4

        if is_images:
            x = rearrange(x, "b c h w -> b c 1 h w")

        out = filter3d(x, f, normalized=True)

        if is_images:
            out = rearrange(out, "b c 1 h w -> b c h w")

        return out


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True, antialiased_downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1, stride=(2 if downsample else 1))

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = (
            nn.Sequential(
                Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2), nn.Conv2d(filters * 4, filters, 1)
            )
            if downsample
            else None
        )

    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only=True)

            x = self.downsample(x)

        x = (x + res) * (2**-0.5)
        return x


class Discriminator(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        channels=3,
        max_dim=512,
        attn_heads=8,
        attn_dim_head=32,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False,
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)

        blocks = []

        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []
        attn_blocks = []

        image_resolution = min_image_resolution

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan, out_chan, downsample=is_not_last, antialiased_downsample=antialiased_downsample
            )

            attn_block = nn.Sequential(
                Residual(LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)),
                Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
            )

            blocks.append(nn.ModuleList([block, attn_block]))

            image_resolution //= 2

        self.blocks = nn.ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2**num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            nn.Linear(latent_dim, 1),
            Rearrange("b 1 -> b"),
        )

    def forward(self, x):
        for block, attn_block in self.blocks:
            x = block(x)
            x = attn_block(x)

        return self.to_logits(x)


class DiscriminatorBlock3D(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        antialiased_downsample=True,
    ):
        super().__init__()
        self.conv_res = nn.Conv3d(input_channels, filters, 1, stride=2)

        self.net = nn.Sequential(
            nn.Conv3d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv3d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = nn.Sequential(
            Rearrange("b c (f p1) (h p2) (w p3) -> b (c p1 p2 p3) f h w", p1=2, p2=2, p3=2),
            nn.Conv3d(filters * 8, filters, 1),
        )

    def forward(self, x):
        res = self.conv_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only=True)

            x = self.downsample(x)

        x = (x + res) * (2**-0.5)
        return x


class DiscriminatorBlock3DWithfirstframe(nn.Module):
    def __init__(
        self,
        input_channels,
        filters,
        antialiased_downsample=True,
        pad_mode="first",
    ):
        super().__init__()
        self.downsample_res = DownSample3D(
            in_channels=input_channels,
            out_channels=filters,
            with_conv=True,
            compress_time=True,
        )

        self.net = nn.Sequential(
            CausalConv3d(input_channels, filters, kernel_size=3, pad_mode=pad_mode),
            leaky_relu(),
            CausalConv3d(filters, filters, kernel_size=3, pad_mode=pad_mode),
            leaky_relu(),
        )

        self.maybe_blur = Blur() if antialiased_downsample else None

        self.downsample = DownSample3D(
            in_channels=filters,
            out_channels=filters,
            with_conv=True,
            compress_time=True,
        )

    def forward(self, x):
        res = self.downsample_res(x)

        x = self.net(x)

        if exists(self.downsample):
            if exists(self.maybe_blur):
                x = self.maybe_blur(x, space_only=True)

            x = self.downsample(x)

        x = (x + res) * (2**-0.5)
        return x


class Discriminator3D(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        frame_num,
        channels=3,
        max_dim=512,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False,
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)
        temporal_num_layers = int(log2(frame_num))
        self.temporal_num_layers = temporal_num_layers

        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []

        image_resolution = min_image_resolution
        frame_resolution = frame_num

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            if ind < temporal_num_layers:
                block = DiscriminatorBlock3D(
                    in_chan,
                    out_chan,
                    antialiased_downsample=antialiased_downsample,
                )

                blocks.append(block)

                frame_resolution //= 2
            else:
                block = DiscriminatorBlock(
                    in_chan,
                    out_chan,
                    downsample=is_not_last,
                    antialiased_downsample=antialiased_downsample,
                )
                attn_block = nn.Sequential(
                    Residual(
                        LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)
                    ),
                    Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
                )

                blocks.append(nn.ModuleList([block, attn_block]))

            image_resolution //= 2

        self.blocks = nn.ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2**num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            nn.Linear(latent_dim, 1),
            Rearrange("b 1 -> b"),
        )

    def forward(self, x):
        for i, layer in enumerate(self.blocks):
            if i < self.temporal_num_layers:
                x = layer(x)
                if i == self.temporal_num_layers - 1:
                    x = rearrange(x, "b c f h w -> (b f) c h w")
            else:
                block, attn_block = layer
                x = block(x)
                x = attn_block(x)

        return self.to_logits(x)


class Discriminator3DWithfirstframe(nn.Module):
    @beartype
    def __init__(
        self,
        *,
        dim,
        image_size,
        frame_num,
        channels=3,
        max_dim=512,
        linear_attn_dim_head=8,
        linear_attn_heads=16,
        ff_mult=4,
        antialiased_downsample=False,
    ):
        super().__init__()
        image_size = pair(image_size)
        min_image_resolution = min(image_size)

        num_layers = int(log2(min_image_resolution) - 2)
        temporal_num_layers = int(log2(frame_num))
        self.temporal_num_layers = temporal_num_layers

        layer_dims = [channels] + [(dim * 4) * (2**i) for i in range(num_layers + 1)]
        layer_dims = [min(layer_dim, max_dim) for layer_dim in layer_dims]
        layer_dims_in_out = tuple(zip(layer_dims[:-1], layer_dims[1:]))

        blocks = []

        image_resolution = min_image_resolution
        frame_resolution = frame_num

        for ind, (in_chan, out_chan) in enumerate(layer_dims_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(layer_dims_in_out) - 1)

            if ind < temporal_num_layers:
                block = DiscriminatorBlock3DWithfirstframe(
                    in_chan,
                    out_chan,
                    antialiased_downsample=antialiased_downsample,
                )

                blocks.append(block)

                frame_resolution //= 2
            else:
                block = DiscriminatorBlock(
                    in_chan,
                    out_chan,
                    downsample=is_not_last,
                    antialiased_downsample=antialiased_downsample,
                )
                attn_block = nn.Sequential(
                    Residual(
                        LinearSpaceAttention(dim=out_chan, heads=linear_attn_heads, dim_head=linear_attn_dim_head)
                    ),
                    Residual(FeedForward(dim=out_chan, mult=ff_mult, images=True)),
                )

                blocks.append(nn.ModuleList([block, attn_block]))

            image_resolution //= 2

        self.blocks = nn.ModuleList(blocks)

        dim_last = layer_dims[-1]

        downsample_factor = 2**num_layers
        last_fmap_size = tuple(map(lambda n: n // downsample_factor, image_size))

        latent_dim = last_fmap_size[0] * last_fmap_size[1] * dim_last

        self.to_logits = nn.Sequential(
            nn.Conv2d(dim_last, dim_last, 3, padding=1),
            leaky_relu(),
            Rearrange("b ... -> b (...)"),
            nn.Linear(latent_dim, 1),
            Rearrange("b 1 -> b"),
        )

    def forward(self, x):
        for i, layer in enumerate(self.blocks):
            if i < self.temporal_num_layers:
                x = layer(x)
                if i == self.temporal_num_layers - 1:
                    x = x.mean(dim=2)
                    # x = rearrange(x, "b c f h w -> (b f) c h w")
            else:
                block, attn_block = layer
                x = block(x)
                x = attn_block(x)

        return self.to_logits(x)


class VideoAutoencoderLoss(nn.Module):
    def __init__(
        self,
        disc_start,
        perceptual_weight=1,
        adversarial_loss_weight=0,
        multiscale_adversarial_loss_weight=0,
        grad_penalty_loss_weight=0,
        quantizer_aux_loss_weight=0,
        vgg_weights=VGG16_Weights.DEFAULT,
        discr_kwargs=None,
        discr_3d_kwargs=None,
    ):
        super().__init__()

        self.disc_start = disc_start
        self.perceptual_weight = perceptual_weight
        self.adversarial_loss_weight = adversarial_loss_weight
        self.multiscale_adversarial_loss_weight = multiscale_adversarial_loss_weight
        self.grad_penalty_loss_weight = grad_penalty_loss_weight
        self.quantizer_aux_loss_weight = quantizer_aux_loss_weight

        if self.perceptual_weight > 0:
            self.perceptual_model = LPIPS().eval()
            # self.vgg = torchvision.models.vgg16(pretrained = True)
            # self.vgg.requires_grad_(False)
        # if self.adversarial_loss_weight > 0:
        #     self.discr = Discriminator(**discr_kwargs)
        # else:
        #     self.discr = None
        # if self.multiscale_adversarial_loss_weight > 0:
        #     self.multiscale_discrs = nn.ModuleList([*multiscale_discrs])
        # else:
        #     self.multiscale_discrs = None
        if discr_kwargs is not None:
            self.discr = Discriminator(**discr_kwargs)
        else:
            self.discr = None
        if discr_3d_kwargs is not None:
            # self.discr_3d = Discriminator3D(**discr_3d_kwargs)
            self.discr_3d = instantiate_from_config(discr_3d_kwargs)
        else:
            self.discr_3d = None
        # self.multiscale_discrs = nn.ModuleList([*multiscale_discrs])

        self.register_buffer("zero", torch.tensor(0.0), persistent=False)

    def get_trainable_params(self) -> Any:
        params = []
        if self.discr is not None:
            params += list(self.discr.parameters())
        if self.discr_3d is not None:
            params += list(self.discr_3d.parameters())
        # if self.multiscale_discrs is not None:
        #     for discr in self.multiscale_discrs:
        #         params += list(discr.parameters())
        return params

    def get_trainable_parameters(self) -> Any:
        return self.get_trainable_params()

    def forward(
        self,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        aux_losses=None,
        last_layer=None,
        split="train",
    ):
        batch, channels, frames = inputs.shape[:3]

        if optimizer_idx == 0:
            recon_loss = F.mse_loss(inputs, reconstructions)

            if self.perceptual_weight > 0:
                frame_indices = torch.randn((batch, frames)).topk(1, dim=-1).indices

                input_frames = pick_video_frame(inputs, frame_indices)
                recon_frames = pick_video_frame(reconstructions, frame_indices)

                perceptual_loss = self.perceptual_model(input_frames.contiguous(), recon_frames.contiguous()).mean()
            else:
                perceptual_loss = self.zero

            if global_step >= self.disc_start or not self.training or self.adversarial_loss_weight == 0:
                gen_loss = self.zero
                adaptive_weight = 0
            else:
                # frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices
                # recon_video_frames = pick_video_frame(reconstructions, frame_indices)

                # fake_logits = self.discr(recon_video_frames)
                fake_logits = self.discr_3d(reconstructions)
                gen_loss = hinge_gen_loss(fake_logits)

                adaptive_weight = 1
                if self.perceptual_weight > 0 and last_layer is not None:
                    norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_layer).norm(p=2)
                    norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_layer).norm(p=2)
                    adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min=1e-3)
                    adaptive_weight.clamp_(max=1e3)

                    if torch.isnan(adaptive_weight).any():
                        adaptive_weight = 1

            # multiscale discriminator losses

            # multiscale_gen_losses = []
            # multiscale_gen_adaptive_weights = []
            # if self.multiscale_adversarial_loss_weight > 0:
            #     if not exists(recon_video_frames):
            #         frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices
            #         recon_video_frames = pick_video_frame(reconstructions, frame_indices)
            #     for discr in self.multiscale_discrs:
            #         fake_logits = recon_video_frames

            #         multiscale_gen_loss = hinge_gen_loss(fake_logits)
            #         multiscale_gen_losses.append(multiscale_gen_loss)

            #         multiscale_adaptive_weight = 1.

            #         if exists(norm_grad_wrt_perceptual_loss):
            #             norm_grad_wrt_gen_loss = grad_layer_wrt_loss(multiscale_gen_loss, last_layer).norm(p = 2)
            #             multiscale_adaptive_weight = norm_grad_wrt_perceptual_loss / norm_grad_wrt_gen_loss.clamp(min = 1e-5)
            #             multiscale_adaptive_weight.clamp_(max = 1e3)

            #         multiscale_gen_adaptive_weights.append(multiscale_adaptive_weight)
            #     weighted_multiscale_gen_losses = sum(loss * weight for loss, weight in zip(multiscale_gen_losses, multiscale_gen_adaptive_weights))
            # else:
            #     weighted_multiscale_gen_losses = self.zero

            if aux_losses is None:
                aux_losses = self.zero

            total_loss = (
                recon_loss
                + aux_losses * self.quantizer_aux_loss_weight
                + perceptual_loss * self.perceptual_weight
                + gen_loss * self.adversarial_loss_weight
            )
            # gen_loss * adaptive_weight * self.adversarial_loss_weight + \
            # weighted_multiscale_gen_losses * self.multiscale_adversarial_loss_weight

            log = {
                "{}/total_loss".format(split): total_loss.detach(),
                "{}/recon_loss".format(split): recon_loss.detach(),
                "{}/perceptual_loss".format(split): perceptual_loss.detach(),
                "{}/gen_loss".format(split): gen_loss.detach(),
                "{}/aux_losses".format(split): aux_losses.detach(),
                # "{}/weighted_multiscale_gen_losses".format(split): weighted_multiscale_gen_losses.detach(),
                "{}/adaptive_weight".format(split): adaptive_weight,
                # "{}/multiscale_adaptive_weights".format(split): sum(multiscale_gen_adaptive_weights),
            }

            return total_loss, log

        if optimizer_idx == 1:
            # frame_indices = torch.randn((batch, frames)).topk(1, dim = -1).indices

            # real = pick_video_frame(inputs, frame_indices)
            # fake = pick_video_frame(reconstructions, frame_indices)

            # apply_gradient_penalty = self.grad_penalty_loss_weight > 0
            # if apply_gradient_penalty:
            #     real = real.requires_grad_()

            # real_logits = self.discr(real)
            # fake_logits = self.discr(fake.detach())

            apply_gradient_penalty = self.grad_penalty_loss_weight > 0
            if apply_gradient_penalty:
                inputs = inputs.requires_grad_()
            real_logits = self.discr_3d(inputs)
            fake_logits = self.discr_3d(reconstructions.detach())

            discr_loss = hinge_discr_loss(fake_logits, real_logits)

            # # multiscale discriminators
            # multiscale_discr_losses = []
            # if self.multiscale_adversarial_loss_weight > 0:
            #     for discr in self.multiscale_discrs:
            #         multiscale_real_logits = discr(inputs)
            #         multiscale_fake_logits = discr(reconstructions.detach())

            #         multiscale_discr_loss = hinge_discr_loss(multiscale_fake_logits, multiscale_real_logits)
            #         multiscale_discr_losses.append(multiscale_discr_loss)
            # else:
            #     multiscale_discr_losses.append(self.zero)

            # gradient penalty
            if apply_gradient_penalty:
                # gradient_penalty_loss = gradient_penalty(real, real_logits)
                gradient_penalty_loss = gradient_penalty(inputs, real_logits)
            else:
                gradient_penalty_loss = self.zero

            total_loss = discr_loss + self.grad_penalty_loss_weight * gradient_penalty_loss
            # self.grad_penalty_loss_weight * gradient_penalty_loss + \
            # sum(multiscale_discr_losses) * self.multiscale_adversarial_loss_weight

            log = {
                "{}/total_disc_loss".format(split): total_loss.detach(),
                "{}/discr_loss".format(split): discr_loss.detach(),
                "{}/grad_penalty_loss".format(split): gradient_penalty_loss.detach(),
                # "{}/multiscale_discr_loss".format(split): sum(multiscale_discr_losses).detach(),
                "{}/logits_real".format(split): real_logits.detach().mean(),
                "{}/logits_fake".format(split): fake_logits.detach().mean(),
            }
            return total_loss, log
