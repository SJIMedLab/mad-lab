import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
import numpy as np

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """
    根据每个 batch 的时间步 t，从 a 中提取对应的系数，并 reshape 成 [B,1,1,1,1,...] 方便广播
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        depth_size,
        channels=1,
        timesteps=1000,
        loss_type='l2',
        betas=None,
        with_condition=False
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.depth_size = depth_size
        self.denoise_fn = denoise_fn
        self.with_condition = with_condition
        self.loss_type = loss_type

        # beta schedule
        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(timesteps)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas', to_torch(alphas)) # 添加 alpha
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # 前向过程
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))

        # DDPM 采样所需参数
        self.register_buffer('posterior_variance', to_torch(
            betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        ))
        # log 形式更稳定
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(self.posterior_variance.cpu().numpy(), 1e-20)) # clip for stability
        ))
        
        # 反向均值系数
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        ))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas.copy()) / (1. - alphas_cumprod)
        ))

    def q_sample(self, x_start, t, noise=None):
        """
        给定 x0 和时间步 t，采样 x_t
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def p_losses(self, x_start, t, class_id=None, noise=None):
        """
        计算单步 loss
        x_start: [B, C, D, H, W]
        class_id: [B] 或 None
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 模型预测噪声
        if class_id is not None:
            # 这里对应 unet.py 中的 forward 方法 class_id => y
            x_recon = self.denoise_fn(x_noisy, t, y=class_id)
        else:
            x_recon = self.denoise_fn(x_noisy, t)

        # loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(x_recon, noise)
        elif self.loss_type in ['l2', 'mse']:
            loss = F.mse_loss(x_recon, noise)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

        return loss

    def forward(self, x, class_ids=None, *args, **kwargs):
        """
        前向传播
        x: [B, C, D, H, W]
        class_ids: [B]
        """
        if x.ndim != 5:
            raise ValueError(f"Expected 5D input [B,C,D,H,W], got {x.shape}")

        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(x, t, class_id=class_ids, *args, **kwargs)
    

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        根据噪声预测 eps，计算 x0 的预测值 x_start
        公式: x_start = (x_t - sqrt(1-alpha_cumprod) * eps) / sqrt(alpha_cumprod)
        """
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    def p_mean_variance(self, x, t, class_ids=None):
        """
        计算 DDPM 反向扩散的均值和方差
        """
        # 1. 模型预测噪声 eps_pred
        if class_ids is not None:
            eps_pred = self.denoise_fn(x, t, y=class_ids)
        else:
            eps_pred = self.denoise_fn(x, t)

        # 2. 从 eps_pred 计算 x0 的预测值 pred_xstart
        pred_xstart = self._predict_xstart_from_eps(x, t, eps_pred)
        
        # 3. 裁剪 pred_xstart 到 [-1, 1] 范围 (与数据归一化范围一致)
        pred_xstart = torch.clamp(pred_xstart, -1., 1.) 

        # 4. 计算后验均值 (Posterior Mean)
        # DDPM 后验均值公式：mu_tilde(x_t, x_start) = coef1 * x_start + coef2 * x_t
        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * pred_xstart
            + extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        
        # 5. 设置后验方差 (Posterior Variance)
        # DDPM 默认使用固定的后验方差（或其裁剪版本）
        model_variance = extract(self.posterior_variance, t, x.shape)
        model_log_variance = extract(self.posterior_log_variance_clipped, t, x.shape)

        return {
            'mean': model_mean, 
            'variance': model_variance, 
            'log_variance': model_log_variance,
            'pred_xstart': pred_xstart
        }


    @torch.no_grad()
    def p_sample(self, x, t, class_ids=None):
        """
        单步采样 x_{t-1} ~ N(mean, variance)
        """
        
        # 使用 p_mean_variance 计算均值和方差
        out = self.p_mean_variance(x, t, class_ids=class_ids)
        
        # t=0 时，不需要加噪声，直接返回均值即 x_0
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        
        # 采样新噪声 z ~ N(0, I)
        noise = torch.randn_like(x)
        
        # x_{t-1} = mean + sqrt(variance) * noise * mask (t=0时mask=0)
        x_prev = out['mean'] + nonzero_mask * torch.exp(0.5 * out['log_variance']) * noise
        
        return x_prev

    @torch.no_grad()
    def p_sample_loop(self, shape, class_ids=None):
        """
        采样整个序列 x_T -> ... -> x_0
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device) # x_T ~ N(0, I)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, class_ids)
            
        # DDPM 采样返回的 x_0 范围在 [-1, 1]，需要将其 rescale 回 [0, 1] 才能用于保存和显示
        img = (img + 1) * 0.5
        img = torch.clamp(img, 0., 1.)
        
        return img

    @torch.no_grad()
    def sample(self, batch_size=2, class_ids=None):
        return self.p_sample_loop((batch_size, self.channels, self.depth_size, self.image_size, self.image_size), class_ids)
