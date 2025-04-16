import os
import glob
import h5py
import torch
import numpy as np
import cv2
from facenet_pytorch import MTCNN


def to_rgb(img):
    """ 确保图像为 RGB 格式 """
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


class MMPDLoader:
    def __init__(self, data_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.data_path = data_path
        self.detector = MTCNN(keep_all=False, device=device)

    def get_raw_data(self):
        """ 获取 MMPD 数据集中所有 .mat 文件路径 """
        data_dirs = sorted(glob.glob(self.data_path + os.sep + 'subject*'))
        dirs = []
        for data_dir in data_dirs:
            subject = int(os.path.split(data_dir)[-1][7:])  # 提取 subject ID
            mat_dirs = os.listdir(data_dir)
            for mat_dir in mat_dirs:
                index = mat_dir.split('_')[-1].split('.')[0]
                dirs.append({'index': index,
                             'path': os.path.join(data_dir, mat_dir),
                             'subject': subject})
        return dirs

    def read_mat(self, mat_file):
        """ 读取 v7.3 .mat 文件并提取所有相关信息 """
        with h5py.File(mat_file, 'r') as mat:
            frames = np.array(mat['video'])  # 读取视频数据
            bvps = np.array(mat['GT_ppg']).T.reshape(-1)  # BVP 信号

            # 读取其他元数据（HDF5 数据格式是存储为嵌套数组的）
            def read_scalar(mat, key):
                return np.array(mat[key])[0, 0] if key in mat else None

            light = read_scalar(mat, 'light')
            motion = read_scalar(mat, 'motion')
            exercise = read_scalar(mat, 'exercise')
            skin_color = read_scalar(mat, 'skin_color')
            gender = read_scalar(mat, 'gender')
            glasser = read_scalar(mat, 'glasser')
            hair_cover = read_scalar(mat, 'hair_cover')
            makeup = read_scalar(mat, 'makeup')

        processed_frames = frames
        # processed_frames = self.process_frames(frames)

        return {
            'frames': processed_frames,
            'bvps': bvps,
            'light': light,
            'motion': motion,
            'exercise': exercise,
            'skin_color': skin_color,
            'gender': gender,
            'glasser': glasser,
            'hair_cover': hair_cover,
            'makeup': makeup
        }

    def process_frames(self, frames):
        """ 使用 MTCNN 检测人脸并调整尺寸至 (128, 128, 3)，人脸框固定 """
        print("---------------人脸检测！---------------")
        processed_frames = []
        face_box = None  # 用于固定人脸框

        for frame in frames:
            rgb_frame = to_rgb(frame)

            if face_box is None:
                face_boxes = self.detector(rgb_frame)
                if face_boxes is not None and len(face_boxes) > 0:
                    face_box = face_boxes[0]  # 只取第一张人脸的坐标

            if face_box is not None:
                x, y, w, h = face_box
                x = max(0, int(x - 0.1 * w))
                y = max(0, int(y - 0.1 * h))
                w = int(1.2 * w)
                h = int(1.2 * h)

                x2 = min(rgb_frame.shape[1], x + w)
                y2 = min(rgb_frame.shape[0], y + h)

                face = rgb_frame[y:y2, x:x2]
                face_resized = cv2.resize(face, (128, 128))
            else:
                face_resized = cv2.resize(rgb_frame, (128, 128))  # 若未检测到人脸，则缩放整帧

            processed_frames.append(face_resized)

        return np.array(processed_frames)


# 示例：如何使用 MMPDLoader
if __name__ == "__main__":
    data_path = "E:/datasets/mini_MMPD"  # 你的数据集路径
    loader = MMPDLoader(data_path=data_path)

    # 获取所有 .mat 文件
    mat_files = loader.get_raw_data()
    print("找到", len(mat_files), "个 .mat 文件")

    # 读取一个示例 .mat 文件
    if mat_files:
        for i in range(min(len(mat_files), 660)):  # 避免超出索引范围
            sample_file = mat_files[i]['path']
            data = loader.read_mat(sample_file)
            print(f"成功读取第 {i+1} 个文件")
