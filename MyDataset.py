import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import os


class MyDataset(Dataset):

    def __init__(self, data_list:list, T:int=160, transform=None, method:str='train', fs:int=30):
        self.data_list = data_list # list of .h5 file paths for training
        self.T = T # video clip length
        self.transform = transform
        self.method = method
        self.fs = fs

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_path = self.data_list[idx]
        file_name = os.path.splitext(os.path.basename(file_path))[0]  # 例如 P1_1

        with h5py.File(self.data_list[idx], 'r') as f:
            img_length = min(f['imgs'].shape[0], f['bvp'].shape[0])

            if self.method == 'train':
                # 训练阶段：随机采样一个窗口
                idx_start = np.random.choice(img_length - self.T)
                idx_end = idx_start + self.T

                bvp = f['bvp'][idx_start:idx_end].astype('float32')
                img_seq = f['imgs'][idx_start:idx_end]  # (T, H, W, C)

                img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')  # (C, T, H, W)

                img_seq = self.TN(img_seq, axis=1)

                if self.transform is not None:
                    img_seq, bvp = self.transform(img_seq, bvp, self.fs)

                return img_seq, bvp, file_name

            elif self.method == 'val':
                # 验证阶段：将整个视频切成多个不重叠的窗口，并拼接处理
                num_segments = img_length // self.T
                img_segs = []
                bvp_segs = []

                for i in range(num_segments):
                    idx_start = i * self.T
                    idx_end = idx_start + self.T

                    img_seg = f['imgs'][idx_start:idx_end]  # (T, H, W, C)
                    bvp_seg = f['bvp'][idx_start:idx_end].astype('float32')

                    img_seg = np.transpose(img_seg, (3, 0, 1, 2)).astype('float32')  # (C, T, H, W)
                    img_seg = self.TN(img_seg, axis=1)

                    if self.transform is not None:
                        img_seg, bvp_seg = self.transform(img_seg, bvp_seg, self.fs)

                    img_segs.append(np.expand_dims(img_seg, axis=0))  # (1, C, T, H, W)
                    bvp_segs.append(np.expand_dims(bvp_seg, axis=0))  # (1, T)

                # 拼接所有窗口：假设你后续模型支持处理多段输入
                img_seq = np.concatenate(img_segs, axis=0)  # 在 T 维度拼接，形状为 (B, C, T, H, W)
                bvp = np.concatenate(bvp_segs, axis=0)  # (B, T)

                return img_seq, bvp, file_name

    @staticmethod
    def TN(x, axis, eps=1e-6, chunk=0):
        if not chunk:
            chunk = x.shape[axis]
        ishape = x.shape
        x = np.reshape(x, (*x.shape[:axis], -1, chunk, *x.shape[axis + 1:]))
        mean = np.mean(x, axis=axis + 1, keepdims=True)
        tshape = [1] * len(x.shape)
        tshape[axis + 1] = chunk
        t = np.reshape(np.linspace(0, 1, chunk), tshape)
        n = np.sum((t - 0.5) * (x - mean), axis=axis + 1, keepdims=True)
        d = np.sum((t - 0.5) ** 2, axis=axis + 1, keepdims=True)
        i = mean - n / d * 0.5
        trend = n / d * t + i
        x -= trend
        mean = 0
        std = (np.mean((x - mean) ** 2, axis=axis + 1, keepdims=True) + eps) ** 0.5
        x = (x - mean) / std
        r = np.reshape(x, ishape)
        return r


    @staticmethod
    def collate_fn(batch):
        """
        自定义 collate_fn 处理 batch 数据
        batch: List[Tuple[np.array, np.array]] -> (img_seq, bvp)

        返回：
        - images: Tensor, shape (batch_size, C, T, H, W)
        - bvps: Tensor, shape (batch_size, T)
        """
        images, bvps, filenames = zip(*batch)  # 解包 batch

        # 将 numpy array 转换成 PyTorch Tensor，并堆叠成 batch
        images = torch.tensor(np.stack(images), dtype=torch.float32)  # (batch_size, C, T, H, W)
        bvps = torch.tensor(np.stack(bvps), dtype=torch.float32)  # (batch_size, T)

        return images, bvps, filenames