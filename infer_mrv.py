# -*- coding:utf-8 -*-

import argparse
import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from diffusion_model.gaussian_diffusion import GaussianDiffusion
from diffusion_model.unet import create_model

'''

DDPM的Unet不支持训练后修改尺寸
--input_size, --depth_size, --num_channels, --num_res_blocks, --timesteps 必须与你用于生成 --weight_file 的训练配置完全一致。

python infer_mrv.py -w ./models/sjmrv/model-last.pt --cls 1

'''

def num_to_groups(num, divisor):
    """
    Helper function from trainer.py to split total samples into batches.
    """
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--outputs_folder', type=str, default="./outputs_mrv")
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--depth_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=64)  # 最小64
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--batchsize', type=int, default=1, help="Batch size for sampling (to fit in GPU memory)")
    parser.add_argument('--num_samples', type=int, default=1, help="Total number of samples to generate")
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('-w', '--weight_file', type=str, default="./models/model-last.pt",
                        help="Path to the .pt model weight file")
    parser.add_argument('--cls', type=int, default=1, help="The class ID to generate")
    parser.add_argument('--device', type=str, default=None, help="cpu or cuda device id (e.g. cuda:0). default auto")
    parser.add_argument('--seed', type=int, default=None, help="Set for reproducible results")
    args = parser.parse_args()

    # --- 1. 设置 ---

    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # 设置设备
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # 创建输出目录
    output_path = Path(args.outputs_folder)
    output_path.mkdir(exist_ok=True, parents=True)

    # --- 2. 构建模型 (必须与训练时完全一致) ---

    # 这些通道在 train_and.py 中是固定的
    in_channels = 1
    out_channels = 1

    # 1. 创建 U-Net
    model = create_model(args.input_size,
                         args.num_channels,
                         args.num_res_blocks,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         class_cond=True)  # 必须为 True，因为 train_and.py 中设置了

    # 2. 创建 Diffusion Wrapper
    diffusion = GaussianDiffusion(
        model,
        image_size=args.input_size,
        depth_size=args.depth_size,
        timesteps=args.timesteps,
        loss_type='l2',  # loss_type 不影响推理，但保持一致
        with_condition=True,  # 必须为 True
        channels=out_channels
    ).to(device)

    # --- 3. 加载权重 ---

    print(f"Loading weights from: {args.weight_file}")
    try:
        data = torch.load(args.weight_file, map_location=device, weights_only=False)

        # 强烈推荐使用 EMA 权重进行推理
        if 'ema' in data:
            print("Loading EMA weights...")
            diffusion.load_state_dict(data['ema'])
        elif 'model' in data:
            print("Warning: EMA weights not found. Loading model weights...")
            diffusion.load_state_dict(data['model'])
        else:
            # 兼容直接保存 state_dict 的情况
            print("Warning: Assuming file is a raw state_dict. Loading...")
            diffusion.load_state_dict(data)

    except Exception as e:
        print(f"Error loading weight file: {e}")
        print("Please ensure the model architecture parameters (input_size, num_channels, etc.) match the checkpoint.")
        return

    diffusion.eval()

    # --- 4. 采样 ---

    # 将总样本数分成多个批次
    batches = num_to_groups(args.num_samples, args.batchsize)

    all_samples_list = []
    print(f"Generating {args.num_samples} samples (class {args.cls}) in {len(batches)} batches...")

    for batch_size in tqdm(batches):
        # 准备类别标签
        # [B]
        class_ids = torch.tensor([args.cls] * batch_size, device=device).long()

        with torch.no_grad():
            # 调用 diffusion.sample
            # imgs [B, C, X, Y, Z]
            imgs = diffusion.sample(batch_size=batch_size, class_ids=class_ids)

            # 收集到 cpu，避免 GPU 内存累积
            all_samples_list.append(imgs.cpu())

    # 将所有批次合并
    all_samples_tensor = torch.cat(all_samples_list, dim=0)

    # --- 5. 保存 NIfTI 文件 ---

    print(f"Saving {all_samples_tensor.shape[0]} samples to {output_path}...")

    for i, sample_tensor in enumerate(all_samples_tensor):
        # sample_tensor shape is [C, X, Y, Z]
        # 我们假设 C=1，并取出 [X, Y, Z]
        if sample_tensor.shape[0] != 1:
            print(
                f"Warning: Sample {i} has unexpected channel count: {sample_tensor.shape[0]}. Saving only first channel.")

        # 获取 [X, Y, Z]
        sample_data = sample_tensor[0].numpy()

        # # 添加过滤代码
        # sample_data[sample_data < 0.001] = 0.0
        # sample_data = sample_data.astype(np.float32)

        # NIfTI 期望 [X, Y, Z] 顺序，这与我们的模型输出一致
        nifti_img = nib.Nifti1Image(sample_data, affine=np.eye(4))

        # 定义保存路径
        save_name = f'sample_{i + 1:03d}_cls{args.cls}.nii.gz'
        save_path = output_path / save_name

        nib.save(nifti_img, str(save_path))

    print(f"Inference complete. All files saved in {output_path}.")


if __name__ == '__main__':
    main()