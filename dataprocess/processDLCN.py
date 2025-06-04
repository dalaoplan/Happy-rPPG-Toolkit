import os
import h5py
import pandas as pd
from face_detection import face_detection
import numpy as np

def resample_to_same_length(video_data, bvp_data):
    """
    通过线性插值将 BVP 信号扩展到目标长度
    :param bvp_data: 原始 BVP 数据（1D NumPy 数组）
    :param target_length: 目标长度（视频帧数）
    :return: 经过插值的 BVP 信号
    """
    video_len = video_data.shape[0]
    bvp_len = len(bvp_data)
    if bvp_len < video_len:
        bvp_resampled = np.interp(
            np.linspace(1, bvp_len, video_len),  # 目标插值点
            np.linspace(1, bvp_len, bvp_len),  # 原始数据点
            bvp_data  # 原始 BVP 信号
        )
        return video_data, bvp_resampled
    elif video_len < bvp_len:
        video_resampled = np.interp(
            np.linspace(1, video_len, bvp_len),  # 目标插值点
            np.linspace(1, video_len, video_len),  # 原始数据点
            video_data  # 原始 BVP 信号
        )
        return video_resampled, bvp_len
    else :
        return video_data, bvp_data



def read_csv_data(csv_path):
    """ 读取 CSV 文件，返回 NumPy 数组 """
    if not os.path.exists(csv_path):
        return None
    data = pd.read_csv(csv_path).values
    return data.astype(np.float32)  # 确保数据为 float32


def save_to_h5(faces, bvp_data, spo2_data,  hr_data,  save_path):
    """ 保存数据到 .h5 文件 """
    with h5py.File(save_path, "w") as f:
        total_num_frame = faces.shape[0]
        store_size =faces.shape[2]

        # 存储bvp信号
        f.create_dataset('bvp', data=bvp_data, dtype='float64', chunks=(1,),
                         compression="gzip", compression_opts=4)

        # 存储人脸视频
        f.create_dataset('imgs', data=faces, dtype='uint8',
                         chunks=(1, store_size, store_size, 3),
                         compression="gzip", compression_opts=4)

        # 存储hr信号
        f.create_dataset('hr', data=hr_data, dtype='float64', chunks=(1,),
                         compression="gzip", compression_opts=4)

        # 存储spo2信号
        f.create_dataset('spo2', data=spo2_data, dtype='float64', chunks=(1,),
                         compression="gzip", compression_opts=4)

        f.close()



def process_dataset(root_dir, save_dir):
    """
    遍历 root_dir 中的所有子文件夹，提取人脸视频和对应的 BVP、SPO2、HR 数据，
    并将每个子文件夹的数据保存为一个 .h5 文件，命名格式为 P1_1.h5, P1_2.h5,...

    :param root_dir: 数据集的根目录，包含 P001 到 P00N 的文件夹
    :param save_dir: 处理后 .h5 文件的保存目录
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 遍历所有 P001 到 P00N 文件夹
    for participant_folder in sorted(os.listdir(root_dir)):
        participant_path = os.path.join(root_dir, participant_folder)
        if not os.path.isdir(participant_path) or not participant_folder.startswith("p"):
            continue  # 只处理以 "P" 开头的目录

        participant_id = int(participant_folder[1:])  # 解析 P001 的编号为 1

        # 遍历 V01 到 V20 文件夹
        for video_folder in sorted(os.listdir(participant_path)):
            video_path = os.path.join(participant_path, video_folder)
            if not os.path.isdir(video_path) or not video_folder.startswith("v"):
                continue  # 只处理以 "V" 开头的目录

            video_id = int(video_folder[1:])  # 解析 V01 的编号为 1

            # 生成正确的 .h5 文件名，例如 P1_1.h5, P1_2.h5
            h5_filename = f"P{participant_id}_{video_id}.h5"
            h5_filepath = os.path.join(save_dir, h5_filename)

            # 获取视频和 CSV 文件路径
            video_file = os.path.join(video_path, "video_ZIP_MJPG.avi")  # Todo 注意这里的要求保存的视频文件名称为.avi
            bvp_file = os.path.join(video_path, "BVP.csv")
            spo2_file = os.path.join(video_path, "SPO2.csv")
            hr_file = os.path.join(video_path, "HR.csv")

            print("处理视频{}中！！！".format(video_file))
            # 运行人脸检测
            faces = face_detection(video_file)

            # 读取 CSV 数据

            bvp_data = read_csv_data(bvp_file)[:, 1]
            spo2_data = read_csv_data(spo2_file)[:, 1]
            hr_data = read_csv_data(hr_file)[:, 1]

            faces, bvp_data = resample_to_same_length(faces, bvp_data)

            # 保存到 .h5
            save_to_h5(faces, bvp_data, spo2_data, hr_data, h5_filepath)

            print(f"已处理成文件{h5_filename}！！！")




if __name__ == '__main__':

    root_directory = r"E:\DLCN_RawData"
    output_directory = r"E:\DLCN"
    process_dataset(root_directory, output_directory)