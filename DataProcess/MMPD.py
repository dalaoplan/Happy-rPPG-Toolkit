import os
import glob
import torch
import numpy as np
import scipy.io as sio
import cv2
from facenet_pytorch import MTCNN
import h5py
from tqdm import tqdm  # 导入 tqdm 进度条库


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
                             'path': data_dir + os.sep + mat_dir,
                             'subject': subject})
        return dirs

    def read_mat(self, mat_file):
        """ 读取 .mat 文件并提取所有相关信息 """

        mat = sio.loadmat(mat_file)
        frames = np.array(mat['video'])  # 维度: (1800, 80, 60, 3)
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
        bvps = np.array(mat['GT_ppg']).T.reshape(-1)

        light = mat['light'][0][0]
        motion = mat['motion'][0][0]
        exercise = mat['exercise'][0][0]
        skin_color = mat['skin_color'][0][0]
        gender = mat['gender'][0][0]
        glasser = mat['glasser'][0][0]
        hair_cover = mat['hair_cover'][0][0]
        makeup = mat['makeup'][0][0]

        # processed_frames = frames
        processed_frames = self.process_frames(frames)

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
        """ 记录所有帧的检测框，计算最终裁剪区域，并统一处理所有帧 """
        # print("--------------- 开始人脸检测 ---------------")
        face_boxes_list = []  # 记录所有帧的检测框

        # 遍历所有帧，收集人脸框
        for idx, frame in enumerate(frames):
            # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
            face_boxes, _ = self.detector.detect(frame)

            if face_boxes is not None and len(face_boxes) > 0:
                x1, y1, x2, y2 = map(int, face_boxes[0])  # 取第一张人脸
                face_boxes_list.append((x1, y1, x2, y2))
                # print(f"第 {idx} 帧：检测到人脸，坐标：{x1, y1, x2, y2}")

        # 确保至少检测到一张人脸，否则直接返回缩放帧
        if not face_boxes_list:
            print("未检测到任何人脸，将缩放整帧处理！")
            return np.array([cv2.resize(frame, (128, 128)) for frame in frames])

        # 计算最小外接矩形
        min_x1 = min(box[0] for box in face_boxes_list)
        min_y1 = min(box[1] for box in face_boxes_list)
        max_x2 = max(box[2] for box in face_boxes_list)
        max_y2 = max(box[3] for box in face_boxes_list)

        # 计算宽高并扩展 10%
        w, h = max_x2 - min_x1, max_y2 - min_y1
        min_x1 = max(0, int(min_x1 - 0.1 * w))
        min_y1 = max(0, int(min_y1 - 0.1 * h))
        max_x2 = min(frames[0].shape[1], int(max_x2 + 0.1 * w))
        max_y2 = min(frames[0].shape[0], int(max_y2 + 0.1 * h))

        # print(f"最终裁剪区域：({min_x1}, {min_y1}, {max_x2}, {max_y2})")

        # 处理所有帧
        processed_frames = []
        for frame in frames:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face = rgb_frame[min_y1:max_y2, min_x1:max_x2]
            face_resized = cv2.resize(face, (128, 128))
            processed_frames.append(face_resized)

        # print("--------------- 人脸检测完成 ---------------")
        return np.array(processed_frames)


def save_to_h5(data, save_path):
    """ 保存数据到 .h5 文件 """
    with h5py.File(save_path, "w") as f:
        total_num_frame = len(data['frames'])
        store_size = data['frames'].shape[1]

        # 存储BVP信号
        f.create_dataset('bvp', data=data['bvps'], dtype='float64', chunks=(1,),
                         compression="gzip", compression_opts=4)

        # 存储人脸视频
        f.create_dataset('imgs', data=data['frames'], dtype='uint8',
                         chunks=(1, store_size, store_size, 3),
                         compression="gzip", compression_opts=4)

        # 存储其他元数据
        metadata_keys = ['light', 'motion', 'exercise', 'skin_color', 'hair_cover']

        for key in metadata_keys:
            if key in data:
                f.create_dataset(key, data=data[key], dtype='S1')  # 存储为 int 类型


        f.close()


# 示例：如何使用 MMPDLoader
if __name__ == "__main__":


    data_path = "E:/datasets/mini_MMPD"
    loader = MMPDLoader(data_path=data_path)

    output_path = f'E:/datasets/MMPD'


    err_file_path = f"err.txt"
    if os.path.exists(err_file_path):
        with open(err_file_path, "r", encoding="utf-8") as f:
            err_files = set(f.read().splitlines())


    # 获取所有 .mat 文件
    mat_files = loader.get_raw_data()
    print("找到", len(mat_files), "个 .mat 文件")
    print(f"其中有{len(err_files)}个无法读取文件")

    for i in tqdm(range(len(mat_files)), desc="Processing", unit="file"):

        sample_file = mat_files[i]['path']
        if sample_file not in err_files:  # 确保未在错误列表中
            data = loader.read_mat(sample_file)

            save_path = os.path.join(output_path, f"p{mat_files[i]['subject']}_{mat_files[i]['index']}.h5")

            save_to_h5(data, save_path)

            # video = data['frames']
            # # # 保存为 AVI 视频
            # output_path = f"output/video{i}.avi"
            # fps = 30
            # height, width = 128, 128
            # fourcc = cv2.VideoWriter_fourcc(*'XVID')
            # video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            #
            # for frame in video:
            #     video_writer.write(frame)
            #
            # video_writer.release()
            # print(f"视频已保存至 {output_path}")




            # print("示例 .mat 文件内容:")
            # print("视频帧维度:", data['frames'].shape)  # (1800, 128, 128, 3)
            # print("PPG 信号长度:", len(data['bvps']))
            # print("光照条件:", data['light'])
            # print("运动状态:", data['motion'])
            # print("运动后状态:", data['exercise'])
            # print("肤色等级:", data['skin_color'])
            # print("性别:", data['gender'])
            # print("是否戴眼镜:", data['glasser'])
            # print("是否有头发遮挡:", data['hair_cover'])
            # print("是否化妆:", data['makeup'])




