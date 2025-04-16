import numpy as np
import torch
import torch.nn.functional as F
from post_process import calculate_hr

class VideoTransform:
    """ 自定义视频数据增强管道 """
    def __init__(self, transforms):
        # 自动绑定实例方法
        self.transforms = [getattr(self, t) if isinstance(t, str) else t for t in transforms]

    def __call__(self, clip, bvp, fs):
        """ 逐步应用数据增强，传入 fs 参数 """
        for transform in self.transforms:
            clip, bvp = transform(clip, bvp, fs)  # 确保传递 fs 参数
        return clip, bvp

    def augment_gaussian_noise(self, clip, bvp, fs):
        """ 高斯噪声增强 """
        return clip + np.random.normal(0, 2, clip.shape), bvp

    def augment_illumination_noise(self, clip, bvp, fs):
        """ 亮度扰动 """
        return clip + np.random.normal(0, 10), bvp

    def augment_time_reversal(self, clip, bvp, fs):
        """ 时间翻转 """
        return np.flip(clip, 1) if np.random.rand() > 0.5 else clip, bvp

    def augment_horizontal_flip(self, clip, bvp, fs):
        """ 水平翻转 """

        return np.flip(clip, 3) if np.random.rand() > 0.5 else clip, bvp

    def augment_random_resized_crop(self, clip, bvp, fs, crop_scale_lims=[0.5, 1]):
        """ 随机裁剪并恢复原尺寸 """
        C, T, H, W = clip.shape
        crop_scale = np.random.uniform(*crop_scale_lims)
        crop_size = int(crop_scale * H)
        x1 = np.random.randint(0, H - crop_size + 1)
        cropped_clip = clip[:, :, x1:x1+crop_size, x1:x1+crop_size]
        return self.resize_clip(cropped_clip, H), bvp

    def augmentation_time_adapt(self, clip, bvp, fs, diff_flag=False):

        C, D, H, W = clip.shape  # 只处理单个样本
        clip = np.transpose(clip, (1, 0, 2, 3)) #C, T, H, W

        if D < 4:  # 避免时间长度太短导致索引错误
            return clip, bvp

        clip_aug = np.zeros((D, C, H, W), dtype=np.float32)
        bvp_aug = np.zeros(D, dtype=np.float32)

        rand1 = np.random.random()

        if rand1 < 0.5:
            gt_hr_fft, _ = calculate_hr(bvp, bvp, diff_flag=diff_flag, fs=fs)

            if gt_hr_fft > 90:
                rand3 = np.random.randint(0, max(1, D // 2 - 1))  # 确保合法范围
                even_indices = np.arange(0, D, 2)
                odd_indices = even_indices + 1

                clip_aug[even_indices] = clip[rand3 + even_indices // 2]
                bvp_aug[even_indices] = bvp[rand3 + even_indices // 2]

                clip_aug[odd_indices] = (clip[rand3 + odd_indices // 2] +
                                         clip[rand3 + (odd_indices // 2) + 1]) / 2
                bvp_aug[odd_indices] = (bvp[rand3 + odd_indices // 2] +
                                        bvp[rand3 + (odd_indices // 2) + 1]) / 2
            elif gt_hr_fft < 75:
                clip_aug[:D // 2] = clip[::2]
                bvp_aug[:D // 2] = bvp[::2]

                clip_aug[D // 2:] = clip_aug[:D // 2]
                bvp_aug[D // 2:] = bvp_aug[:D // 2]
            else:
                clip_aug = clip
                bvp_aug = bvp
        else:
            clip_aug = clip
            bvp_aug = bvp

        clip_aug = np.transpose(clip_aug, (1, 0, 2, 3)) #C, D, H, W
        return clip_aug, bvp_aug

    def resize_clip(self, clip, length):
        T = clip.shape[1]
        clip = torch.from_numpy(np.ascontiguousarray(clip[np.newaxis]))
        clip = F.interpolate(clip, (T, length, length), mode='trilinear', align_corners=False)
        return clip[0].numpy()

AUGMENTATION_MAPPING = {
    "G": "augment_gaussian_noise",
    "R": "augment_time_reversal",
    "H": "augment_horizontal_flip",
    "I": "augment_illumination_noise",
    "C": "augment_random_resized_crop",
    "T": "augmentation_time_adapt"
}

def get_transforms_from_args(aug_string):
    return [AUGMENTATION_MAPPING[key] for key in aug_string if key in AUGMENTATION_MAPPING]

if __name__ == '__main__':
    fake_video = np.random.rand(3, 120, 64, 64).astype(np.float32)
    fake_bvp = np.random.rand(120,).astype(np.float32)

    args = type('', (), {})()  # 创建一个空对象模拟 args
    args.aug = "HGT"

    selected_transforms = get_transforms_from_args(args.aug)
    data_transform = VideoTransform(selected_transforms)

    transformed_video, transformed_bvp = data_transform(fake_video, fake_bvp, fs=30)

    print("Original shape:", fake_video.shape)
    print("Transformed shape:", transformed_video.shape)
