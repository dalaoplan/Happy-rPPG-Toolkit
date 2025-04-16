import os
from collections import defaultdict


data_dir = 'E:\\DLCN'  # 替换为你的实际路径

# """
# 两个状态的数据容器
rest_files = []
exercise_files = []

# 遍历文件夹中所有 .h5 文件
for file in os.listdir(data_dir):
    if file.endswith('.h5') and file.startswith('P'):
        name = file.split('.')[0]  # eg. 'P1_3'
        _, idx = name.split('_')
        idx = int(idx)

        full_path = os.path.join(data_dir, file)

        # 判断属于哪种状态
        if 1 <= idx <= 4:
            rest_files.append(full_path)
        elif 5 <= idx <= 8:
            exercise_files.append(full_path)

# 打印划分结果
print(f"静止状态文件数: {len(rest_files)}")
for f in rest_files:
    print("  ", f)

print(f"\n运动状态文件数: {len(exercise_files)}")
for f in exercise_files:
    print("  ", f)
# """



# 用于存储每个场景对应的文件列表
scene_data = defaultdict(list)

# 映射模4结果到场景名
scene_mapping = {
    1: 'S1_FI&FP',
    2: 'S2_VI&FP',
    3: 'S3_FI&VP',
    0: 'S4_VI&VP'  # 注意：idx % 4 == 0 表示原始 idx 是 4 或 8
}

# 遍历所有 .h5 文件
for file in os.listdir(data_dir):
    if file.endswith('.h5') and file.startswith('P'):
        name = file.split('.')[0]  # P1_3
        _, idx = name.split('_')
        idx = int(idx)

        scene_id = idx % 4
        scene_name = scene_mapping[scene_id]

        full_path = os.path.join(data_dir, file)
        scene_data[scene_name].append(full_path)

# 打印结果
for scene, files in scene_data.items():
    print(f"{scene}: {len(files)} files")
    # for f in files:
    #     print(f"  {f}")

