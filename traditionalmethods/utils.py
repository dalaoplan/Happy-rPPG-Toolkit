import numpy as np
from scipy import sparse
import os
from collections import defaultdict



def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)

def get_dataset_info(dataset_name):
    """
    根据数据集名称返回该数据集的采样率 fs 和视频长度 video_len（单位为秒）

    参数:
        dataset_name (str): 数据集名称，比如 'PURE', 'UBFC-rPPG', 'DCLN', 等等

    返回:
        fs (int or float): 帧率 (frames per second)
        video_len (int or float): 视频总时长（单位为秒）
    """

    dataset_info = {
        "PURE": {"fs": 30, "video_len": 60},
        "UBFCrPPG": {"fs": 30, "video_len": 60},
        "UBFCPhys": {"fs": 35, "video_len": 180},
        "COHFACE": {"fs": 20, "video_len": 60},
        "DLCN": {"fs": 30, "video_len": 60},
        "MMSE-HR": {"fs": 25, "video_len": 60},
        "VIPL-HR": {"fs": 30, "video_len": 30},
        "MMPD": {"fs": 30, "video_len": 60},
        # 根据需要补充更多数据集
    }

    if dataset_name not in dataset_info:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    return dataset_info[dataset_name]["fs"], dataset_info[dataset_name]["video_len"]


def read_split_data(test_dataset_name: str = "UBFCrPPG", group: str = 'R'):


    test_data_root = f"E:\datasets_h5\{test_dataset_name}"

    assert os.path.exists(test_data_root), "dataset root: {} does not exist.".format(test_data_root)



    # 映射模4结果到场景名
    scene_mapping = {
        1: 'FIFP',
        2: 'VIFP',
        3: 'FIVP',
        0: 'VIVP'  # 注意：idx % 4 == 0 表示原始 idx 是 4 或 8
    }
    # 用于存储每个场景对应的文件列表
    scene_data = defaultdict(list)
    rest_files = []
    exercise_files = []

    test_files_paths = []
    if test_dataset_name == "DLCN" and group != "Raw":
        # 遍历文件夹中所有 .h5 文件
        for file in os.listdir(test_data_root):
            if file.endswith('.h5') and file.startswith('P'):
                name = file.split('.')[0]  # eg. 'P1_3'
                _, idx = name.split('_')
                idx = int(idx)

                full_path = os.path.join(test_data_root, file)

                scene_id = idx % 4
                scene_name = scene_mapping[scene_id]
                scene_data[scene_name].append(full_path)

                # 判断属于哪种状态
                if 1 <= idx <= 4:
                    rest_files.append(full_path)
                elif 5 <= idx <= 8:
                    exercise_files.append(full_path)

        if group == 'R':
            test_files_paths = rest_files
            print(f"静止状态样本个数: {len(test_files_paths)}")

        elif group == 'E':
            test_files_paths = exercise_files
            print(f"\n运动状态样本个数: {len(test_files_paths)}")

        elif group == 'FIFP':
            test_files_paths = scene_data[group]
            print(f"\n光强固定且位置固定: {len(test_files_paths)}")
        elif group == 'VIFP':
            test_files_paths = scene_data[group]
            print(f"\n光强变化且位置固定: {len(test_files_paths)}")
        elif group == 'FIVP':
            test_files_paths = scene_data[group]
            print(f"\n光强固定且位置变化: {len(test_files_paths)}")
        elif group == 'VIVP':
            test_files_paths = scene_data[group]
            print(f"\n光强变化且位置变化: {len(test_files_paths)}")

    else:
        # test_files_paths = [f for f in os.listdir(test_data_root) if f.endswith(".h5")]
        test_files_paths = sorted([os.path.join(test_data_root, f) for f in os.listdir(test_data_root)])
        print(f"预测的样本个数共有：{len(test_files_paths)}")


    # print(test_files_paths[:5])
    return test_files_paths



if __name__ == '__main__':
    test = read_split_data(test_dataset_name='DLCN', group='VIFP')
    print(test[:10])

