import h5py
import torch
from torch.utils.data import Dataset
import numpy as np


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
        with h5py.File(self.data_list[idx], 'r') as f:

            img_length = np.min([f['imgs'].shape[0], f['bvp'].shape[0]])

            if self.method in ('train', 'val'):
                idx_start = np.random.choice(img_length-self.T)
                idx_end = idx_start+self.T
            else :
                idx_start = 0
                idx_end = img_length

            bvp = f['bvp'][idx_start:idx_end].astype('float32')

            img_seq = f['imgs'][idx_start:idx_end]  # PURE(T, H, W, C), (300, 128, 128, 3)

            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32') # PURE(T, C, H, W) #shape: [3, 300, 128, 128]

            if self.transform is not None:
                img_seq, bvp = self.transform(img_seq, bvp, self.fs)   # [C, T, H, W]

            # img_seq = np.transpose(img_seq, (1, 0, 2, 3)).astype('float32')  #PURE [C, T, H, W]

            # rhythmFormer[C,T,H,W]
            # PhysNet[C,T,H,W]

        return img_seq, bvp


    @staticmethod
    def collate_fn(batch):
        """
        自定义 collate_fn 处理 batch 数据
        batch: List[Tuple[np.array, np.array]] -> (img_seq, bvp)

        返回：
        - images: Tensor, shape (batch_size, C, T, H, W)
        - bvps: Tensor, shape (batch_size, T)
        """
        images, bvps = zip(*batch)  # 解包 batch

        # 将 numpy array 转换成 PyTorch Tensor，并堆叠成 batch
        images = torch.tensor(np.stack(images), dtype=torch.float32)  # (batch_size, C, T, H, W)
        bvps = torch.tensor(np.stack(bvps), dtype=torch.float32)  # (batch_size, T)

        return images, bvps