import os
from glob import glob
import torch
import nibabel as nib
import numpy as np
import torchio as tio
from torch.utils.data import Dataset

class MrvClassIDNiiDataset(Dataset):
    def __init__(self,
                 image_folder,
                 input_size=128,
                 depth_size=128,
                 multiple=8,
                 transform=None,
                 ):
        """
        自定义数据集类，用于加载NIfTI格式的医学图像数据。
        """
        self.image_folder = image_folder
        self.input_size = input_size
        self.depth_size = depth_size

        self.multiple = multiple
        self.input_files = glob(os.path.join(image_folder, '*.nii.gz'))

        self.transform = transform

        # 预存所有 class_id
        self.class_ids = []
        for f in self.input_files:
            base_name = os.path.basename(f)
            class_id = 0 if base_name.startswith('F_') else 1
            self.class_ids.append(class_id)

    def __len__(self):
        return len(self.input_files)

    @staticmethod
    def pad_to_multiple(tensor, multiple=8):
        _, d, h, w = tensor.shape
        pad_d = (multiple - d % multiple) % multiple
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        padding = (0, pad_w, 0, pad_h, 0, pad_d)
        return torch.nn.functional.pad(tensor, padding, mode='constant', value=0)


    def read_image(self, file_path, pass_scaler=False):
        # 1. 读取 nii.gz
        img = nib.load(file_path).get_fdata().astype(np.float32)

        if not pass_scaler:
            # 归一化到 [-1, 1] 以匹配 DDPM 期望范围
            valid_pixels = img[img != 0]
            if len(valid_pixels) > 0:
                # 使用非零区域的最小值和最大值进行归一化
                img_min = valid_pixels.min()
                img_max = valid_pixels.max()
                # 先归一化到 [0, 1]
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min + 1e-8)
                # 再映射到 [-1, 1]
                img = img * 2.0 - 1.0
            else:
                # 如果全是零，保持为 -1
                img = img - 1.0

        if img.shape != (self.input_size, self.input_size, self.depth_size):
            tio_img = tio.ScalarImage(tensor=img[np.newaxis, ...])
            resize_op = tio.Resize((self.input_size, self.input_size, self.depth_size))
            img = np.asarray(resize_op(tio_img))[0]

        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = self.pad_to_multiple(img, self.multiple)

        return img.to(torch.float32)

    def __getitem__(self, idx):
        input_file = self.input_files[idx]
        base_name = os.path.basename(input_file)
        class_id = 0 if base_name.startswith('F_') else 1

        img = self.read_image(input_file)

        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,
            "class_id": torch.tensor(class_id, dtype=torch.long)
        }

    def sample_conditions(self, batch_size):
        indices = np.random.randint(0, len(self), size=batch_size)
        sampled_ids = [self.class_ids[i] for i in indices]
        return torch.tensor(sampled_ids, dtype=torch.long).cuda()
