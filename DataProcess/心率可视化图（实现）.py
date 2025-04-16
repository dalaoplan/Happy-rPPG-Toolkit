import os
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from post_process import calculate_hr, get_hr
# 模拟的数据目录结构
data_folder = 'E:\\DLCN'  # 注意：Windows 下路径需要双反斜杠

# 映射场景与状态
scene_mapping = {1: 'S1_FI&FP', 2: 'S2_VI&FP',
                 3: 'S3_FI&VP', 4: 'S4_VI&VP'}
state_mapping = {0: 'Rest', 1: 'Exercise'}

# 收集数据
records = []
hr_min = float('inf')
hr_max = float('-inf')
segment_len = 30 * 10
HR = []
# 遍历所有.h5文件
for filename in os.listdir(data_folder):
    if filename.endswith('.h5'):
        path = os.path.join(data_folder, filename)
        name = filename.split('.')[0]  # 如 P1_1 ~ P1_8
        pid, idx = name.split('_')
        idx = int(idx)
        scene_id = ((idx - 1) % 4) + 1
        state_id = 0 if idx <= 4 else 1

        with h5py.File(path, 'r') as f:
            if 'bvp' in f:
                bvp_data = f['bvp'][:]
                if len(bvp_data) == 0:
                    print(f"[跳过空数据] {filename}")
                    continue

                bvp_len = len(bvp_data)
                seg_num = bvp_len // segment_len
                # print("bvp_len:",bvp_len)
                for i in range(0, seg_num):
                    bvp_seg = bvp_data[i*segment_len:(i+1)*segment_len]
                    # print('bvp_seg', bvp_seg)
                    hr = get_hr(bvp_seg)
                    HR.append(hr)
                    records.append({
                        'Subject': pid,
                        'Scenario': scene_mapping[scene_id],
                        'State': state_mapping[state_id],
                        'HR': hr,
                        'Segment': f'{i//segment_len + 1}'
                    })

                min_hr = min(HR)
                max_hr = max(HR)

                # 打印每个样本的心率范围
                print(f"{filename}: min_hr={min_hr:.2f}, max_hr={max_hr:.2f}")

                # 更新全局心率范围
                hr_min = min(hr_min, min_hr)
                hr_max = max(hr_max, max_hr)





# 输出全局最大最小心率
print(f"\n[统计结果] 最小心率：{hr_min:.2f}, 最大心率：{hr_max:.2f}")

# 保存统计结果到文本文件
with open('hr_summary.txt', 'w', encoding='utf-8') as f:
    f.write(f"最小心率：{hr_min:.2f}\n")
    f.write(f"最大心率：{hr_max:.2f}\n")

# 转换为 DataFrame
df = pd.DataFrame(records)

ordered_scenarios = ['S1_FI&FP', 'S2_VI&FP', 'S3_FI&VP', 'S4_VI&VP']

# 可视化 - 小提琴图
font_size = 14
font_weight = 'medium'

plt.figure(figsize=(12, 6))
sns.violinplot(data=df, x='Scenario', y='HR', hue='State', split=True,
               inner='quart', order=ordered_scenarios)

plt.ylabel('Heart Rate (bpm)', fontsize=font_size, fontweight=font_weight)
plt.xlabel(None)
plt.xticks(fontsize=font_size, fontweight=font_weight)
plt.yticks(fontsize=font_size, fontweight=font_weight)
plt.legend(title=None, fontsize=font_size, title_fontsize=font_size, loc='best')

plt.tight_layout()
plt.savefig('./hr.pdf', bbox_inches='tight', dpi=300)
plt.show()
