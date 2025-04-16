import numpy as np
import matplotlib.pyplot as plt
from post_process import normalize, bandpass, calculate_psd
import os
import json


def plot_wave_psd(pred, label, fps = 30, fig_path:str=f"result/Plots"):

    pred = np.squeeze(pred)
    label = np.squeeze(label)

    filter_pred = bandpass(pred, fps)
    filter_label = bandpass(label, fps)

    filter_pred = normalize(filter_pred)
    filter_label = normalize(filter_label)

    unfilter_pred = normalize(pred)
    unfilter_label = normalize(label)

    psd_pred, psd_label = calculate_psd(filter_pred, filter_label, fps)

    fig, axes = plt.subplots(3, 1, figsize=(8.27, 8.27), tight_layout=True)

    font_size = 14  # 设置字号
    font_weight = 'medium'  # 设置加粗  # bold normal

    for i in range(3):
        axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=font_size, fontweight=font_weight)
        axes[i].set_ylabel(axes[i].get_ylabel(), fontsize=font_size, fontweight=font_weight)
        axes[i].tick_params(axis='both', labelsize=14, width=1)

    # wave1
    axes[0].plot(np.arange(len(unfilter_label)) / fps, unfilter_label, color='#000000', label='ground truth')
    axes[0].plot(np.arange(len(unfilter_pred)) / fps, unfilter_pred, color='#FF0000', label='rPPG', alpha=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(False)
    axes[0].set_xlim(0, len(unfilter_pred) / fps)
    axes[0].set_ylim(min(min(unfilter_pred), min(unfilter_label)), max(max(unfilter_pred), max(unfilter_label)))


    # wave2
    axes[1].plot(np.arange(len(filter_label)) / fps, filter_label, color='#000000', label='ground truth')
    axes[1].plot(np.arange(len(filter_pred)) / fps, filter_pred, color='#FF0000', label='rPPG', alpha=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(False)
    axes[1].set_xlim(0, len(filter_pred) / fps)
    axes[1].set_ylim(min(min(filter_pred), min(filter_label)), max(max(filter_pred), max(filter_label)))


    ### PSD
    axes[2].plot(psd_pred[0], psd_pred[1], label = 'rPPG', color='#FF0000')
    axes[2].plot(psd_label[0], psd_label[1], label = 'ground truth', color='#000000')
    axes[2].set_xlabel('Heart Rate (bpm)')

    axes[2].set_ylabel('Power Spectral Density')
    axes[2].grid(False)
    axes[2].legend(loc='upper right', ncol=1, fontsize=font_size)
    axes[2].set_xlim([45, 150])

    if os.path.exists(f"./result/Plots/") is False:
        os.makedirs("./result/Plots/")

    plt.savefig(f'{fig_path}.pdf', bbox_inches='tight', dpi=300)

    #plt.show()
    plt.close()

    print("Plot wave over!!!")


def plot_blandaltman(hr_pred, hr_label, fig_path:str = f"result/Plots"):
    # 添加随机噪声
    # noise = np.random.uniform(-1, 1, size=len(hr_pred))
    # hr_pred = np.array(hr_pred) + noise
    # hr_label = np.array(hr_label) + noise

    hr_pred = np.array(hr_pred)
    hr_label = np.array(hr_label)

    plt.figure(figsize=(7, 10), constrained_layout=True) #tight_layout=True,


    # 图1: 散点图
    plt.subplot(2, 1, 1)
    plt.scatter(hr_label, hr_pred, label='', color='#F0B5BF',
                alpha=1, edgecolors='black', s=100)
    plt.plot([45, 150], [45, 150], 'k--', alpha=0.7)  # y=x 参考线
    plt.xlabel('Ground Truth HR (bpm)', fontsize=14)
    plt.ylabel('Estimated HR (bpm)', fontsize=14)

    plt.xticks(fontsize=14)  # 调整 x 轴刻度字体大小
    plt.yticks(fontsize=14)  # 调整 y 轴刻度字体大小

    plt.grid(linestyle='--', linewidth=1.5, alpha=0.7)

    # 图2: Bland-Altman 图
    plt.subplot(2, 1, 2)
    mean_hr = (hr_label + hr_pred) / 2
    diff_hr = hr_label - hr_pred
    mean_diff = np.mean(diff_hr)
    std_diff = np.std(diff_hr)

    plt.scatter(mean_hr, diff_hr, label='', color='#F0B5BF',
                alpha=1, edgecolors='black', s=100)

    # 画出均值线和1.96 SD区间
    plt.axhline(mean_diff, color='black', linestyle='--')
    plt.axhline(mean_diff + 1.96 * std_diff, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(mean_diff - 1.96 * std_diff, color='gray', linestyle='--', alpha=0.7)

    # 添加数值标注
    x_min = np.min(mean_hr)
    x_max = np.max(mean_hr)

    # 计算动态的 x_left，使其略小于数据最小值 5%
    x_left = x_min - (x_max - x_min) * 0.05

    plt.text(x_left, mean_diff, 'Mean', color='black', va='bottom', ha='left', fontsize=14)
    plt.text(x_left, mean_diff + 1.96 * std_diff, 'Mean+1.96SD', color='gray', va='bottom', ha='left', fontsize=14)
    plt.text(x_left, mean_diff - 1.96 * std_diff, 'Mean-1.96SD', color='gray', va='top', ha='left', fontsize=14)

    plt.xlabel('(Ground Truth HR + Estimated HR) / 2 (bpm)', fontsize=14)
    plt.ylabel('Ground Truth HR - Estimated HR (bpm)', fontsize=14)
    # plt.legend(loc='upper right')

    plt.xticks(fontsize=14)  # 调整 x 轴刻度字体大小
    plt.yticks(fontsize=14)  # 调整 y 轴刻度字体大小

    plt.savefig(f'{fig_path}.pdf', dpi=300) # bbox_inches='tight', pad_inches=0.2

    #plt.show()
    plt.close()

    print("Plot blandaltman over!!!")


if __name__ == '__main__':

    # pre = np.random.rand(1, 300)
    # label = np.random.rand(1, 300)
    # plot_wave_psd(pre, label)


    # 示例数据
    hr_label = np.random.uniform(60, 120, 150)  # 训练集真实心率
    hr_pred = hr_label + np.random.normal(0, 2, 150)  # 训练集估计心率

    plot_blandaltman(hr_pred, hr_label)

