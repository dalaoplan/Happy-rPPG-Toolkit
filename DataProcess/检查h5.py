import h5py
import os

def check_h5_file(h5_file):
    """检查 H5 文件是否能正常打开并读取数据"""
    try:
        with h5py.File(h5_file, 'r') as f:
            print(f"✅ 成功打开文件: {h5_file}")
            print(f"📂 数据集列表: {list(f.keys())}")

            # 检查 bvp（信号数据）
            if 'bvp' in f:
                print(f"📈 bvp 数据集形状: {f['bvp'].shape}, 类型: {f['bvp'].dtype}")

            # 检查 imgs（视频帧）
            if 'imgs' in f:
                print(f"🖼️ imgs 数据集形状: {f['imgs'].shape}, 类型: {f['imgs'].dtype}")

            # 检查其他信息
            for key in f.keys():
                if key not in ['bvp', 'imgs']:
                    print(f"📊 {key} 数据: {f[key][()]}")  # 读取数据

    except Exception as e:
        print(f"❌ 读取 H5 文件失败: {h5_file}, 错误信息: {e}")

# 指定存储 H5 文件的路径
h5_folder = r"E:\\DLCN"  # 你的 H5 存储目录

# 获取所有 H5 文件
h5_files = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith(".h5")]

# 依次检查 H5 文件
for h5_file in h5_files[:]:  # 只检查前 5 个文件（可调整）
    check_h5_file(h5_file)
