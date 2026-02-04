# -*- coding:utf-8 -*-
#
# *Main part of the code is adopted from the following repository: https://github.com/lucidrains/denoising-diffusion-pytorch


import copy
import torch
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
import nibabel as nib
import numpy as np
from tqdm import tqdm
import time
import os
import torchvision
from metrics import compute_batch_ssim

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

try:
    from apex import amp

    APEX_AVAILABLE = True
    print("APEX: ON")
except:
    APEX_AVAILABLE = False
    print("APEX: OFF")

def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def loss_backwards(fp16, loss, optimizer, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


# trainer class

class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            dataset,
            ema_decay=0.995,
            image_size=128,
            depth_size=128,
            train_batch_size=2,
            train_lr=2e-6,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False,
            step_start_ema=2000,
            update_ema_every=10,
            save_and_sample_every=1000,
            results_folder='./results',
            with_condition=False,
            log_dir=None,
            tensorboard_writer=None
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.generated_samples = []  # 用于保存最近生成的 NIfTI 文件路径

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.depth_size = depth_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = dataset
        self.dl = cycle(
            data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, num_workers=0, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)
        self.train_lr = train_lr
        self.train_batch_size = train_batch_size
        self.with_condition = with_condition

        self.step = 0
        self.best_loss = float("inf")  # 记录最小 loss

        # assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'
        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.log_dir = log_dir
        self.writer = tensorboard_writer
        self.reset_parameters()

        # 准备验证集 DataLoader，用于评估生成质量
        self.val_loader = data.DataLoader(
            self.ds,  # 直接使用 MrvClassIDNiiDataset
            batch_size=self.batch_size,  # 与训练 batch 对齐
            shuffle=False,  # 关键：必须 False
            num_workers=0,
            pin_memory=True
        )

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            # 为了恢复训练添加保持优化器和最佳loss
            'opt': self.opt.state_dict(),
            'best_loss': self.best_loss
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    # 原恢复训练代码
    # def load(self, milestone):
    #     data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))
    #
    #     self.step = data['step']
    #     self.model.load_state_dict(data['model'])
    #     self.ema_model.load_state_dict(data['ema'])

    def load_from_path(self, path):
        # 路径直接来自命令行参数
        data = torch.load(path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

        # 加载优化器状态
        if 'opt' in data:
            self.opt.load_state_dict(data['opt'])
        else:
            warnings.warn("Optimizer state not found in checkpoint. Optimizer state will be reset.")

        # 加载最佳 loss
        if 'best_loss' in data:
            self.best_loss = data['best_loss']
        else:
            warnings.warn("Best loss not found in checkpoint. Resetting best_loss to infinity.")
            self.best_loss = float("inf")

    def train(self):

        # 判断下如果没有初始化tensorboard writer，报错
        if self.writer is None:
            raise ValueError("TensorBoard writer is not initialized. Please provide a valid SummaryWriter instance.")

        backwards = partial(loss_backwards, self.fp16)
        start_time = time.time()

        while self.step < self.train_num_steps:
            accumulated_loss = []
            # 梯度累积循环
            tqdm_iter = tqdm(range(self.gradient_accumulate_every), desc=f"Step {self.step}")
            for i in tqdm_iter:
                data_batch = next(self.dl)
                imgs = data_batch['image'].cuda()

                # 输入图像范围打印（调试用）
                # print(f"DEBUG: Input imgs VAR: {imgs.var().item():.6f}, MIN: {imgs.min().item():.3f}, MAX: {imgs.max().item():.3f}")

                labels = data_batch['class_id'].cuda() if self.with_condition else None

                # U-Net 主体输出范围打印（调试用）
                with torch.no_grad():
                    inner_unet = getattr(self.model, 'model', None) \
                                 or getattr(self.model, 'unet', None) \
                                 or getattr(self.model, 'denoise_fn', None) \
                                 or getattr(self.model, 'net', None)

                    if inner_unet is not None:
                        # print("[Warn] Could not find internal UNet in diffusion model.")
                        try:
                            # 构造一个随机时间步
                            t = torch.randint(0, 1000, (imgs.shape[0],), device=imgs.device).long()
                            # 调用真正的 forward
                            u_out = inner_unet(imgs, t, y=labels)
                            # print(
                            #     f"[Real U-Net output] range: {u_out.min().item():.3f} ~ {u_out.max().item():.3f}, mean={u_out.mean().item():.3f}")
                            # 将值添加到 TensorBoard
                            self.writer.add_scalar("u_net_output_min", u_out.min().item(), self.step)
                            self.writer.add_scalar("u_net_output_max", u_out.max().item(), self.step)
                            self.writer.add_scalar("u_net_output_mean", u_out.mean().item(), self.step)
                        except Exception as e:
                            # print("[Error calling UNet]:", e)
                            pass
                # 调试结束
                # GaussianDiffusion forward 返回 loss
                loss = self.model(imgs, class_ids=labels)
                loss = loss / self.gradient_accumulate_every  # 梯度累积缩放
                backwards(loss, self.opt)
                accumulated_loss.append(loss.item())

                # 更新进度条
                tqdm_iter.set_postfix({"loss": f"{loss.item():.6f}", "accum": f"{i+1}/{self.gradient_accumulate_every}"})

            # 梯度更新
            self.opt.step()
            self.opt.zero_grad()

            # 平均 loss
            average_loss = np.mean(accumulated_loss)
            self.writer.add_scalar("training_loss", average_loss, self.step)

            # EMA 更新
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # 保存模型与生成样本
            if self.step != 0 and self.step % self.save_and_sample_every == 0:

                # 添加评估代码
                metrics = self.evaluate_generation(
                    val_loader=self.val_loader,
                    step=self.step,
                    num_batches=1
                )

                for k, v in metrics.items():
                    self.writer.add_scalar(f"eval/{k}", v, self.step)


                milestone = self.step // self.save_and_sample_every

                # 生成样本一个
                batches = num_to_groups(1, self.batch_size)

                all_images_list = []
                label_str_list = []

                for n in batches:
                    if self.with_condition:
                        cond = self.ds.sample_conditions(batch_size=n)  # [B, ...] 或 [B, C, H, W, D]
                        labels = cond.argmax(dim=1) if cond.ndim > 1 else cond
                        imgs = self.ema_model.sample(batch_size=n, class_ids=cond)
                    else:
                        labels = torch.zeros(n, dtype=torch.long).cuda()
                        imgs = self.ema_model.sample(batch_size=n)

                    all_images_list.append(imgs)
                    label_str_list.extend([str(l.item()) for l in labels])

                all_images = torch.cat(all_images_list, dim=0)

                # 1030 修复bug 这个会导致 轴位 和 矢状位 颠倒
                # all_images = all_images.transpose(4, 2)  # [B, C, D, H, W] -> [B, D, H, C, W] 适配 nifti
                # sampleImage = all_images.cpu().numpy().reshape([self.image_size, self.image_size, self.depth_size])

                # 直接获取 [0, 0] 索引的数据，它的 shape 应该已经是 [image_size, image_size, depth_size]
                sampleImage = all_images[0, 0].cpu().numpy()

                # 三个方向中间切片
                mid_x, mid_y, mid_z = (
                    self.image_size // 2,
                    self.image_size // 2,
                    self.depth_size // 2
                )

                # 取三个方向的中间切片
                axial_slice = sampleImage[:, :, mid_z]  # 轴状（横切）
                coronal_slice = sampleImage[:, mid_y, :]  # 冠状（前后）
                sagittal_slice = sampleImage[mid_x, :, :]  # 矢状（左右）

                # 将切片归一化到 [0,1] 以便 TensorBoard 显示
                def normalize_slice(slice_2d):
                    slice_min, slice_max = slice_2d.min(), slice_2d.max()
                    if slice_max > slice_min:
                        return (slice_2d - slice_min) / (slice_max - slice_min)
                    return slice_2d

                axial_slice = normalize_slice(axial_slice)
                coronal_slice = normalize_slice(coronal_slice)
                sagittal_slice = normalize_slice(sagittal_slice)

                # 添加到 TensorBoard
                self.writer.add_image(f"sample_axial", axial_slice, self.step,
                                      dataformats="HW")
                self.writer.add_image(f"sample_coronal", coronal_slice, self.step,
                                      dataformats="HW")
                self.writer.add_image(f"sample_sagittal", sagittal_slice, self.step,
                                      dataformats="HW")

                nifti_img = nib.Nifti1Image(sampleImage, affine=np.eye(4))

                # 类别信息拼接进文件名
                label_str = "_".join(label_str_list)
                save_path = self.results_folder / f'sample-{milestone}-cls-{label_str}.nii.gz'
                nib.save(nifti_img, str(save_path))

                # 记录生成样本路径
                # 维护最近 3 个样本
                self.generated_samples.append(save_path)
                if len(self.generated_samples) > 3:
                    old_file = self.generated_samples.pop(0)
                    if os.path.exists(old_file):
                        os.remove(old_file)

                # 保存 last 权重
                self.save('last')
                # 保存 best 权重
                if average_loss < self.best_loss:
                    self.best_loss = average_loss
                    self.save('best')

            self.step += 1

        print('training completed')
        end_time = time.time()
        execution_time = (end_time - start_time) / 3600
        self.writer.add_hparams(
            {
                "lr": self.train_lr,
                "batchsize": self.train_batch_size,
                "image_size": self.image_size,
                "depth_size": self.depth_size,
                "execution_time (hour)": execution_time
            },
            {"last_loss": average_loss}
        )
        self.writer.close()

    @torch.no_grad()
    def log_generated_images(self, gt, samples, step, max_images=8):
        """
        gt, samples: [B, 1, H, W], range [-1, 1]
        """
        B = min(gt.shape[0], max_images)

        # 显示用，映射到 [0,1]
        gt_vis = (gt[:B] + 1) / 2
        sample_vis = (samples[:B] + 1) / 2

        # 上：GT，下：Generated
        grid = torch.cat([gt_vis, sample_vis], dim=0)
        grid_img = torchvision.utils.make_grid(grid, nrow=B)

        # ===== TensorBoard 图像 =====
        self.writer.add_image(
            "generation/GT_top__Generated_bottom",
            grid_img,
            global_step=step
        )

        # ===== 同时保存 PNG =====
        save_dir = os.path.join(self.log_dir, "samples")
        os.makedirs(save_dir, exist_ok=True)

        torchvision.utils.save_image(
            grid_img,
            os.path.join(save_dir, f"step_{step}.png")
        )

    @torch.no_grad()
    def evaluate_generation(self, val_loader, step, num_batches=1):
        self.ema_model.eval()

        ssim_scores = []

        for i, batch in enumerate(val_loader):
            if i >= num_batches:
                break

            gt = batch["image"].cuda(non_blocking=True)

            # ===== 使用 EMA diffusion 采样 =====
            samples = self.ema_model.sample(
                batch_size=gt.shape[0],
                class_ids=batch["class_id"].cuda() if self.with_condition else None
            )

            # ===== SSIM =====
            ssim_val = compute_batch_ssim(samples, gt)
            ssim_scores.append(ssim_val)

            # ===== 可视化 =====
            if i == 0:
                self.log_generated_images(gt, samples, step)

        self.ema_model.train()
        return {
            "SSIM": float(np.mean(ssim_scores))
        }
