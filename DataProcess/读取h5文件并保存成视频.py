import h5py
import os
import cv2
import numpy as np
from visual import plot_wave_psd


def save_video_from_h5(h5_file, output_folder, fps=30):
    """从 H5 文件中读取 imgs 数据，并保存为视频"""
    try:
        with h5py.File(h5_file, 'r') as f:
            if 'imgs' not in f:
                print(f"❌ 文件 {h5_file} 中没有 'imgs' 数据集")
                return

            imgs = f['imgs'][:]

            if imgs.ndim != 4 or imgs.shape[-1] not in [1, 3]:
                print(f"❌ 'imgs' 数据形状 {imgs.shape} 不是标准的视频格式")
                return

            height, width, channels = imgs.shape[1:]
            if channels == 1:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                is_color = False
            else:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                is_color = True

            video_name = os.path.splitext(os.path.basename(h5_file))[0] + ".avi"
            output_path = os.path.join(output_folder, video_name)

            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)

            for img in imgs:
                frame = img.astype(np.uint8)
                if channels == 1:
                    frame = frame.squeeze()
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 修正颜色顺序
                out.write(frame)

            out.release()
            print(f"✅ 视频已保存: {output_path}")
    except Exception as e:
        print(f"❌ 处理 H5 文件失败: {h5_file}, 错误信息: {e}")

def visualize_bvp_from_h5(h5_file, output_folder, fps=30):
    """从 H5 文件中读取 bvp 数据，并进行可视化"""
    try:
        with h5py.File(h5_file, 'r') as f:
            if 'bvp' not in f:
                print(f"❌ 文件 {h5_file} 中没有 'bvp' 数据集")
                return

            bvp = f['bvp'][:]
            fig_path = os.path.join(output_folder, os.path.splitext(os.path.basename(h5_file))[0] + ".jpg")
            plot_wave_psd(pred=bvp,label=bvp, fps=fps, fig_path=fig_path)
            print(f"✅ BVP 可视化已保存: {fig_path}.jpg")

    except Exception as e:
        print(f"❌ 可视化失败: {h5_file}, 错误信息: {e}")


# 指定 H5 文件夹和输出视频文件夹
h5_folder = r"F:\processdata0413"
video_output_folder = r"F:\video0413"
wave_output_folder = r"F:\wave0413"

os.makedirs(video_output_folder, exist_ok=True)
os.makedirs(wave_output_folder, exist_ok=True)

# 获取所有 H5 文件
h5_files = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith(".h5")]

# 依次处理 H5 文件
for h5_file in h5_files:
    save_video_from_h5(h5_file, video_output_folder)
    visualize_bvp_from_h5(h5_file, wave_output_folder)
