"""
Partially ported from https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
"""

from typing import Dict, Union

import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

from ...modules.diffusionmodules.sampling_utils import (
    get_ancestral_step,
    linear_multistep_coeff,
    to_d,
    to_neg_log_sigma,
    to_sigma,
)
from ...util import append_dims, default, instantiate_from_config
from ...util import SeededNoise

from .guiders import DynamicCFG

DEFAULT_GUIDER = {"target": "sgm.modules.diffusionmodules.guiders.IdentityGuider"}


class BaseDiffusionSampler:
    def __init__(
        self,
        discretization_config: Union[Dict, ListConfig, OmegaConf],
        num_steps: Union[int, None] = None,
        guider_config: Union[Dict, ListConfig, OmegaConf, None] = None,
        verbose: bool = False,
        device: str = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = instantiate_from_config(discretization_config)
        self.guider = instantiate_from_config(
            default(
                guider_config,
                DEFAULT_GUIDER,
            )
        )
        self.verbose = verbose
        self.device = device

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        sigmas = self.discretization(self.num_steps if num_steps is None else num_steps, device=self.device)
        uc = default(uc, cond)

        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        num_sigmas = len(sigmas)

        s_in = x.new_ones([x.shape[0]]).float()

        return x, s_in, sigmas, num_sigmas, cond, uc

    def denoise(self, x, denoiser, sigma, cond, uc):
        denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc))
        denoised = self.guider(denoised, sigma)
        return denoised

    def get_sigma_gen(self, num_sigmas):
        sigma_generator = range(num_sigmas - 1)
        if self.verbose:
            print("#" * 30, " Sampling setting ", "#" * 30)
            print(f"Sampler: {self.__class__.__name__}")
            print(f"Discretization: {self.discretization.__class__.__name__}")
            print(f"Guider: {self.guider.__class__.__name__}")
            sigma_generator = tqdm(
                sigma_generator,
                total=num_sigmas,
                desc=f"Sampling with {self.__class__.__name__} for {num_sigmas} steps",
            )
        return sigma_generator


class SingleStepDiffusionSampler(BaseDiffusionSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc, *args, **kwargs):
        raise NotImplementedError

    def euler_step(self, x, d, dt):
        return x + dt * d


class EDMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_churn=0.0, s_tmin=0.0, s_tmax=float("inf"), s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_churn = s_churn
        self.s_tmin = s_tmin
        self.s_tmax = s_tmax
        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, gamma=0.0):
        sigma_hat = sigma * (gamma + 1.0)
        if gamma > 0:
            eps = torch.randn_like(x) * self.s_noise
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5

        denoised = self.denoise(x, denoiser, sigma_hat, cond, uc)
        d = to_d(x, sigma_hat, denoised)
        dt = append_dims(next_sigma - sigma_hat, x.ndim)

        euler_step = self.euler_step(x, d, dt)
        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class DDIMSampler(SingleStepDiffusionSampler):
    def __init__(self, s_noise=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.s_noise = s_noise

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, s_noise=0.0):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        d = to_d(x, sigma, denoised)
        dt = append_dims(next_sigma * (1 - s_noise**2) ** 0.5 - sigma, x.ndim)

        euler_step = x + dt * d + s_noise * append_dims(next_sigma, x.ndim) * torch.randn_like(x)

        x = self.possible_correction_step(euler_step, x, d, dt, next_sigma, denoiser, cond, uc)
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                self.s_noise,
            )

        return x


class AncestralSampler(SingleStepDiffusionSampler):
    def __init__(self, eta=1.0, s_noise=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.eta = eta
        self.s_noise = s_noise
        self.noise_sampler = lambda x: torch.randn_like(x)

    def ancestral_euler_step(self, x, denoised, sigma, sigma_down):
        d = to_d(x, sigma, denoised)
        dt = append_dims(sigma_down - sigma, x.ndim)

        return self.euler_step(x, d, dt)

    def ancestral_step(self, x, sigma, next_sigma, sigma_up):
        x = torch.where(
            append_dims(next_sigma, x.ndim) > 0.0,
            x + self.noise_sampler(x) * self.s_noise * append_dims(sigma_up, x.ndim),
            x,
        )
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
            )

        return x


class LinearMultistepSampler(BaseDiffusionSampler):
    def __init__(
        self,
        order=4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.order = order

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        ds = []
        sigmas_cpu = sigmas.detach().cpu().numpy()
        for i in self.get_sigma_gen(num_sigmas):
            sigma = s_in * sigmas[i]
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma, cond, uc), **kwargs)
            denoised = self.guider(denoised, sigma)
            d = to_d(x, sigma, denoised)
            ds.append(d)
            if len(ds) > self.order:
                ds.pop(0)
            cur_order = min(i + 1, self.order)
            coeffs = [linear_multistep_coeff(cur_order, sigmas_cpu, i, j) for j in range(cur_order)]
            x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))

        return x


class EulerEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        return euler_step


class HeunEDMSampler(EDMSampler):
    def possible_correction_step(self, euler_step, x, d, dt, next_sigma, denoiser, cond, uc):
        if torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            return euler_step
        else:
            denoised = self.denoise(euler_step, denoiser, next_sigma, cond, uc)
            d_new = to_d(euler_step, next_sigma, denoised)
            d_prime = (d + d_new) / 2.0

            # apply correction if noise level is not 0
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x + d_prime * dt, euler_step)
            return x


class EulerAncestralSampler(AncestralSampler):
    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x = self.ancestral_euler_step(x, denoised, sigma, sigma_down)
        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)

        return x


class DPMPP2SAncestralSampler(AncestralSampler):
    def get_variables(self, sigma, sigma_down):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, sigma_down)]
        h = t_next - t
        s = t + 0.5 * h
        return h, s, t, t_next

    def get_mult(self, h, s, t, t_next):
        mult1 = to_sigma(s) / to_sigma(t)
        mult2 = (-0.5 * h).expm1()
        mult3 = to_sigma(t_next) / to_sigma(t)
        mult4 = (-h).expm1()

        return mult1, mult2, mult3, mult4

    def sampler_step(self, sigma, next_sigma, denoiser, x, cond, uc=None, **kwargs):
        sigma_down, sigma_up = get_ancestral_step(sigma, next_sigma, eta=self.eta)
        denoised = self.denoise(x, denoiser, sigma, cond, uc)
        x_euler = self.ancestral_euler_step(x, denoised, sigma, sigma_down)

        if torch.sum(sigma_down) < 1e-14:
            # Save a network evaluation if all noise levels are 0
            x = x_euler
        else:
            h, s, t, t_next = self.get_variables(sigma, sigma_down)
            mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, s, t, t_next)]

            x2 = mult[0] * x - mult[1] * denoised
            denoised2 = self.denoise(x2, denoiser, to_sigma(s), cond, uc)
            x_dpmpp2s = mult[2] * x - mult[3] * denoised2

            # apply correction if noise level is not 0
            x = torch.where(append_dims(sigma_down, x.ndim) > 0.0, x_dpmpp2s, x_euler)

        x = self.ancestral_step(x, sigma, next_sigma, sigma_up)
        return x


class DPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t)
        mult2 = (-h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            # apply correction if noise level is not 0 and not first step
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x


class SDEDPMPP2MSampler(BaseDiffusionSampler):
    def get_variables(self, sigma, next_sigma, previous_sigma=None):
        t, t_next = [to_neg_log_sigma(s) for s in (sigma, next_sigma)]
        h = t_next - t

        if previous_sigma is not None:
            h_last = t - to_neg_log_sigma(previous_sigma)
            r = h_last / h
            return h, r, t, t_next
        else:
            return h, None, t, t_next

    def get_mult(self, h, r, t, t_next, previous_sigma):
        mult1 = to_sigma(t_next) / to_sigma(t) * (-h).exp()
        mult2 = (-2 * h).expm1()

        if previous_sigma is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_sigma,
        sigma,
        next_sigma,
        denoiser,
        x,
        cond,
        uc=None,
    ):
        denoised = self.denoise(x, denoiser, sigma, cond, uc)

        h, r, t, t_next = self.get_variables(sigma, next_sigma, previous_sigma)
        mult = [append_dims(mult, x.ndim) for mult in self.get_mult(h, r, t, t_next, previous_sigma)]
        mult_noise = append_dims(next_sigma * (1 - (-2 * h).exp()) ** 0.5, x.ndim)

        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        if old_denoised is None or torch.sum(next_sigma) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            # apply correction if noise level is not 0 and not first step
            x = torch.where(append_dims(next_sigma, x.ndim) > 0.0, x_advanced, x_standard)

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(x, cond, uc, num_steps)

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * sigmas[i - 1],
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
            )

        return x


class SdeditEDMSampler(EulerEDMSampler):
    def __init__(self, edit_ratio=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.edit_ratio = edit_ratio

    def __call__(self, denoiser, image, randn, cond, uc=None, num_steps=None, edit_ratio=None):
        randn_unit = randn.clone()
        randn, s_in, sigmas, num_sigmas, cond, uc = self.prepare_sampling_loop(randn, cond, uc, num_steps)

        if num_steps is None:
            num_steps = self.num_steps
        if edit_ratio is None:
            edit_ratio = self.edit_ratio
        x = None

        for i in self.get_sigma_gen(num_sigmas):
            if i / num_steps < edit_ratio:
                continue
            if x is None:
                x = image + randn_unit * append_dims(s_in * sigmas[i], len(randn_unit.shape))

            gamma = (
                min(self.s_churn / (num_sigmas - 1), 2**0.5 - 1) if self.s_tmin <= sigmas[i] <= self.s_tmax else 0.0
            )
            x = self.sampler_step(
                s_in * sigmas[i],
                s_in * sigmas[i + 1],
                denoiser,
                x,
                cond,
                uc,
                gamma,
            )

        return x


class VideoDDIMSampler(BaseDiffusionSampler):
    def __init__(self, fixed_frames=0, sdedit=False, **kwargs):
        super().__init__(**kwargs)
        self.fixed_frames = fixed_frames
        self.sdedit = sdedit

    def prepare_sampling_loop(self, x, cond, uc=None, num_steps=None):
        alpha_cumprod_sqrt, timesteps = self.discretization(
            self.num_steps if num_steps is None else num_steps,
            device=self.device,
            return_idx=True,
            do_append_zero=False,
        )
        alpha_cumprod_sqrt = torch.cat([alpha_cumprod_sqrt, alpha_cumprod_sqrt.new_ones([1])])
        timesteps = torch.cat([torch.tensor(list(timesteps)).new_zeros([1]) - 1, torch.tensor(list(timesteps))])

        uc = default(uc, cond)

        num_sigmas = len(alpha_cumprod_sqrt)

        s_in = x.new_ones([x.shape[0]])

        return x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps

    def denoise(self, x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep=None, idx=None, scale=None, scale_emb=None):
        additional_model_inputs = {}

        if isinstance(scale, torch.Tensor) == False and scale == 1:
            additional_model_inputs["idx"] = x.new_ones([x.shape[0]]) * timestep
            if scale_emb is not None:
                additional_model_inputs["scale_emb"] = scale_emb
            denoised = denoiser(x, alpha_cumprod_sqrt, cond, **additional_model_inputs).to(torch.float32)
        else:
            additional_model_inputs["idx"] = torch.cat([x.new_ones([x.shape[0]]) * timestep] * 2)
            denoised = denoiser(
                *self.guider.prepare_inputs(x, alpha_cumprod_sqrt, cond, uc), **additional_model_inputs
            ).to(torch.float32)
            if isinstance(self.guider, DynamicCFG):
                denoised = self.guider(
                    denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, step_index=self.num_steps - timestep, scale=scale
                )
            else:
                denoised = self.guider(denoised, (1 - alpha_cumprod_sqrt**2) ** 0.5, scale=scale)
        return denoised

    def sampler_step(
        self,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
    ):
        denoised = self.denoise(
            x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb
        ).to(torch.float32)

        a_t = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        b_t = next_alpha_cumprod_sqrt - alpha_cumprod_sqrt * a_t

        x = append_dims(a_t, x.ndim) * x + append_dims(b_t, x.ndim) * denoised
        return x

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        for i in self.get_sigma_gen(num_sigmas):
            x = self.sampler_step(
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
                scale=scale,
                scale_emb=scale_emb,
            )

        return x


class VPSDEDPMPP2MSampler(VideoDDIMSampler):
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt**2
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5 * (-h).exp()
        mult2 = (-2 * h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
        scale=None,
        scale_emb=None,
    ):
        denoised = self.denoise(
            x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx, scale=scale, scale_emb=scale_emb
        ).to(torch.float32)
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]
        mult_noise = append_dims((1 - next_alpha_cumprod_sqrt**2) ** 0.5 * (1 - (-2 * h).exp()) ** 0.5, x.ndim)

        x_standard = mult[0] * x - mult[1] * denoised + mult_noise * torch.randn_like(x)
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d + mult_noise * torch.randn_like(x)

            x = x_advanced

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, scale_emb=None):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        if self.fixed_frames > 0:
            prefix_frames = x[:, : self.fixed_frames]
        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            if self.fixed_frames > 0:
                if self.sdedit:
                    rd = torch.randn_like(prefix_frames)
                    noised_prefix_frames = alpha_cumprod_sqrt[i] * prefix_frames + rd * append_dims(
                        s_in * (1 - alpha_cumprod_sqrt[i] ** 2) ** 0.5, len(prefix_frames.shape)
                    )
                    x = torch.cat([noised_prefix_frames, x[:, self.fixed_frames :]], dim=1)
                else:
                    x = torch.cat([prefix_frames, x[:, self.fixed_frames :]], dim=1)
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
                scale=scale,
                scale_emb=scale_emb,
            )

        if self.fixed_frames > 0:
            x = torch.cat([prefix_frames, x[:, self.fixed_frames :]], dim=1)

        return x


class VPODEDPMPP2MSampler(VideoDDIMSampler):
    def get_variables(self, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt=None):
        alpha_cumprod = alpha_cumprod_sqrt**2
        lamb = ((alpha_cumprod / (1 - alpha_cumprod)) ** 0.5).log()
        next_alpha_cumprod = next_alpha_cumprod_sqrt**2
        lamb_next = ((next_alpha_cumprod / (1 - next_alpha_cumprod)) ** 0.5).log()
        h = lamb_next - lamb

        if previous_alpha_cumprod_sqrt is not None:
            previous_alpha_cumprod = previous_alpha_cumprod_sqrt**2
            lamb_previous = ((previous_alpha_cumprod / (1 - previous_alpha_cumprod)) ** 0.5).log()
            h_last = lamb - lamb_previous
            r = h_last / h
            return h, r, lamb, lamb_next
        else:
            return h, None, lamb, lamb_next

    def get_mult(self, h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt):
        mult1 = ((1 - next_alpha_cumprod_sqrt**2) / (1 - alpha_cumprod_sqrt**2)) ** 0.5
        mult2 = (-h).expm1() * next_alpha_cumprod_sqrt

        if previous_alpha_cumprod_sqrt is not None:
            mult3 = 1 + 1 / (2 * r)
            mult4 = 1 / (2 * r)
            return mult1, mult2, mult3, mult4
        else:
            return mult1, mult2

    def sampler_step(
        self,
        old_denoised,
        previous_alpha_cumprod_sqrt,
        alpha_cumprod_sqrt,
        next_alpha_cumprod_sqrt,
        denoiser,
        x,
        cond,
        uc=None,
        idx=None,
        timestep=None,
    ):
        denoised = self.denoise(x, denoiser, alpha_cumprod_sqrt, cond, uc, timestep, idx).to(torch.float32)
        if idx == 1:
            return denoised, denoised

        h, r, lamb, lamb_next = self.get_variables(
            alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt
        )
        mult = [
            append_dims(mult, x.ndim)
            for mult in self.get_mult(h, r, alpha_cumprod_sqrt, next_alpha_cumprod_sqrt, previous_alpha_cumprod_sqrt)
        ]

        x_standard = mult[0] * x - mult[1] * denoised
        if old_denoised is None or torch.sum(next_alpha_cumprod_sqrt) < 1e-14:
            # Save a network evaluation if all noise levels are 0 or on the first step
            return x_standard, denoised
        else:
            denoised_d = mult[2] * denoised - mult[3] * old_denoised
            x_advanced = mult[0] * x - mult[1] * denoised_d

            x = x_advanced

        return x, denoised

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, scale=None, **kwargs):
        x, s_in, alpha_cumprod_sqrt, num_sigmas, cond, uc, timesteps = self.prepare_sampling_loop(
            x, cond, uc, num_steps
        )

        old_denoised = None
        for i in self.get_sigma_gen(num_sigmas):
            x, old_denoised = self.sampler_step(
                old_denoised,
                None if i == 0 else s_in * alpha_cumprod_sqrt[i - 1],
                s_in * alpha_cumprod_sqrt[i],
                s_in * alpha_cumprod_sqrt[i + 1],
                denoiser,
                x,
                cond,
                uc=uc,
                idx=self.num_steps - i,
                timestep=timesteps[-(i + 1)],
            )

        return x
