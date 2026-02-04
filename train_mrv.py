
import argparse
import datetime
import json
import os

from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, ToTensor, Lambda

import torch
from torch.utils.tensorboard import SummaryWriter

from datasets_mrv import MrvClassIDNiiDataset
from diffusion_model.gaussian_diffusion import GaussianDiffusion
from diffusion_model.trainer import Trainer
from diffusion_model.unet import create_model

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# -

'''
静脉窦数据集训练参数调整说明：

# 数据参数
input_size = 128           # 从 128 保持
depth_size = 128           # ⚠️ 从 16 提升到 128（关键改变）

# 训练参数
train_lr = 1e-4            # 从 1e-5 提高 10倍
batchsize = 2              # 保持 2（因为 depth 增大了）
timesteps = 500            # 从 1000 降到 500
num_channels = 48          # 从 64 降到 48（因为 3D 变大了）
num_res_blocks = 2         # 保持 2

# 训练流程
gradient_accumulate_every = 2   # 保持 2（等效 batch=4）
save_and_sample_every = 500     # 从 1000 改为 500
epochs = 100000                 # 从 50000 提升到 100000

# autodl 执行命令示例：
python train_mrv.py -i ./dataset/sjmrv1128/ -s /root/autodl-tmp/results --input_size 128 --depth_size 128 --num_channels 64 --num_res_blocks 2 --train_lr 1e-4 --batchsize 2 --epochs 100000 --timesteps 500 --gradient_accumulate_every 2 --save_and_sample_every 500

# 大机器小样本训练命令示例：
python train_mrv.py -i ./dataset/and1028_62 -s ./results --input_size 64 --depth_size 64 --num_channels 64 --num_res_blocks 2 --train_lr 1e-4 --batchsize 2 --epochs 100000 --timesteps 500 --gradient_accumulate_every 2 --save_and_sample_every 500

'''

def create_log_dir():
    now = datetime.datetime.now().strftime("%y-%m-%dT%H%M%S")
    log_dir = os.path.join("./logs", now)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir, now

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str, default="./dataset/sjmrv1128/")
    parser.add_argument('-s', '--results_folder', type=str, default="./results")
    parser.add_argument('--input_size', type=int, default=128)
    parser.add_argument('--depth_size', type=int, default=128)
    parser.add_argument('--num_channels', type=int, default=64)  # 最小64
    parser.add_argument('--num_res_blocks', type=int, default=2)
    parser.add_argument('--train_lr', type=float, default=1e-5)
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--gradient_accumulate_every', type=int, default=2)
    parser.add_argument('--save_and_sample_every', type=int, default=1000)
    parser.add_argument('--load_path', type=str, default=None,
                        help="The absolute path to a checkpoint file (e.g., /path/to/results/YYMMDD/model-last.pt) to load for resuming training or fine-tuning.")

    args = parser.parse_args()

    input_folder = args.input_folder
    results_folder = args.results_folder
    input_size = args.input_size
    depth_size = args.depth_size
    num_channels = args.num_channels
    num_res_blocks = args.num_res_blocks
    gradient_accumulate_every = args.gradient_accumulate_every
    save_and_sample_every = args.save_and_sample_every
    train_lr = args.train_lr

    # 生成日志目录
    log_dir, now_str = create_log_dir()

    # 补全结果保存目录
    results_folder = os.path.join(results_folder, now_str)
    os.makedirs(results_folder, exist_ok=True)

    print(f'log saved in: {log_dir}')
    print(f'results saved in: {results_folder}')

    # 初始化tensorboard writer
    writer = SummaryWriter(log_dir=log_dir)

    # 将当前的配置参数保存到日志目录中
    writer.add_text('train config', json.dumps(vars(args), indent=4))

    transform = Compose([
        Lambda(lambda t: torch.tensor(t).float()),
        # Lambda(lambda t: (t * 2) - 1),
    ])

    dataset = MrvClassIDNiiDataset(image_folder=input_folder,
                                input_size=input_size,
                                depth_size=depth_size,
                                transform=transform,
                                multiple=8)
    print(f'Dataset size: {len(dataset)}')

    in_channels = 1
    out_channels = 1

    model = create_model(input_size,
                         num_channels,
                         num_res_blocks,
                         in_channels=in_channels,
                         out_channels=out_channels,
                         class_cond=True).cuda()

    diffusion = GaussianDiffusion(
        model,
        image_size=input_size,
        depth_size=depth_size,
        timesteps=args.timesteps,  # number of steps
        loss_type='l2',  # L1 or L2
        with_condition=True,
        channels=out_channels
    ).cuda()

    trainer = Trainer(
        diffusion,
        dataset,
        image_size=input_size,
        depth_size=depth_size,
        train_batch_size=args.batchsize,
        train_lr=train_lr,
        train_num_steps=args.epochs,  # total training steps
        gradient_accumulate_every=gradient_accumulate_every,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        fp16=False,  # True, turn on mixed precision training with apex
        with_condition=True,
        save_and_sample_every=save_and_sample_every,
        log_dir=log_dir,
        tensorboard_writer=writer,
        results_folder=results_folder
    )

    # === 恢复/加载逻辑：直接使用路径 ===
    if args.load_path is not None:
        try:
            print(f"Attempting to load model from: {args.load_path}...")
            # 调用修改后的加载方法
            trainer.load_from_path(args.load_path)
            print(f"Resuming/Starting from step {trainer.step}. Best loss: {trainer.best_loss:.6f}")

            # 如果是恢复训练，确保新的结果文件夹和日志文件夹与旧的检查点在同一父目录下，
            # 除非你希望在一个新的日志目录下继续。
            # (当前代码逻辑是创建新的log_dir和results_folder，这对于微调是合适的，对于严格恢复需要额外逻辑。)

        except FileNotFoundError:
            print(f"Checkpoint file not found at {args.load_path}. Starting from scratch.")
        except Exception as e:
            print(f"Error loading checkpoint from {args.load_path}: {e}. Starting from scratch.")
    # ==================================

    trainer.train()


if __name__ == '__main__':
    main()