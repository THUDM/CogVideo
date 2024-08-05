import torch
from scipy import integrate

from ...util import append_dims
from einops import rearrange


class NoDynamicThresholding:
    def __call__(self, uncond, cond, scale):
        scale = append_dims(scale, cond.ndim) if isinstance(scale, torch.Tensor) else scale
        return uncond + scale * (cond - uncond)


class StaticThresholding:
    def __call__(self, uncond, cond, scale):
        result = uncond + scale * (cond - uncond)
        result = torch.clamp(result, min=-1.0, max=1.0)
        return result


def dynamic_threshold(x, p=0.95):
    N, T, C, H, W = x.shape
    x = rearrange(x, "n t c h w -> n c (t h w)")
    l, r = x.quantile(q=torch.tensor([1 - p, p], device=x.device), dim=-1, keepdim=True)
    s = torch.maximum(-l, r)
    threshold_mask = (s > 1).expand(-1, -1, H * W * T)
    if threshold_mask.any():
        x = torch.where(threshold_mask, x.clamp(min=-1 * s, max=s), x)
    x = rearrange(x, "n c (t h w) -> n t c h w", t=T, h=H, w=W)
    return x


def dynamic_thresholding2(x0):
    p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
    origin_dtype = x0.dtype
    x0 = x0.to(torch.float32)
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = append_dims(torch.maximum(s, torch.ones_like(s).to(s.device)), x0.dim())
    x0 = torch.clamp(x0, -s, s)  # / s
    return x0.to(origin_dtype)


def latent_dynamic_thresholding(x0):
    p = 0.9995
    origin_dtype = x0.dtype
    x0 = x0.to(torch.float32)
    s = torch.quantile(torch.abs(x0), p, dim=2)
    s = append_dims(s, x0.dim())
    x0 = torch.clamp(x0, -s, s) / s
    return x0.to(origin_dtype)


def dynamic_thresholding3(x0):
    p = 0.995  # A hyperparameter in the paper of "Imagen" [1].
    origin_dtype = x0.dtype
    x0 = x0.to(torch.float32)
    s = torch.quantile(torch.abs(x0).reshape((x0.shape[0], -1)), p, dim=1)
    s = append_dims(torch.maximum(s, torch.ones_like(s).to(s.device)), x0.dim())
    x0 = torch.clamp(x0, -s, s)  # / s
    return x0.to(origin_dtype)


class DynamicThresholding:
    def __call__(self, uncond, cond, scale):
        mean = uncond.mean()
        std = uncond.std()
        result = uncond + scale * (cond - uncond)
        result_mean, result_std = result.mean(), result.std()
        result = (result - result_mean) / result_std * std
        # result = dynamic_thresholding3(result)
        return result


class DynamicThresholdingV1:
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, uncond, cond, scale):
        result = uncond + scale * (cond - uncond)
        unscaled_result = result / self.scale_factor
        B, T, C, H, W = unscaled_result.shape
        flattened = rearrange(unscaled_result, "b t c h w -> b c (t h w)")
        means = flattened.mean(dim=2).unsqueeze(2)
        recentered = flattened - means
        magnitudes = recentered.abs().max()
        normalized = recentered / magnitudes
        thresholded = latent_dynamic_thresholding(normalized)
        denormalized = thresholded * magnitudes
        uncentered = denormalized + means
        unflattened = rearrange(uncentered, "b c (t h w) -> b t c h w", t=T, h=H, w=W)
        scaled_result = unflattened * self.scale_factor
        return scaled_result


class DynamicThresholdingV2:
    def __call__(self, uncond, cond, scale):
        B, T, C, H, W = uncond.shape
        diff = cond - uncond
        mim_target = uncond + diff * 4.0
        cfg_target = uncond + diff * 8.0

        mim_flattened = rearrange(mim_target, "b t c h w -> b c (t h w)")
        cfg_flattened = rearrange(cfg_target, "b t c h w -> b c (t h w)")
        mim_means = mim_flattened.mean(dim=2).unsqueeze(2)
        cfg_means = cfg_flattened.mean(dim=2).unsqueeze(2)
        mim_centered = mim_flattened - mim_means
        cfg_centered = cfg_flattened - cfg_means

        mim_scaleref = mim_centered.std(dim=2).unsqueeze(2)
        cfg_scaleref = cfg_centered.std(dim=2).unsqueeze(2)

        cfg_renormalized = cfg_centered / cfg_scaleref * mim_scaleref

        result = cfg_renormalized + cfg_means
        unflattened = rearrange(result, "b c (t h w) -> b t c h w", t=T, h=H, w=W)

        return unflattened


def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()
