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

    fig, axes = plt.subplots(2, 1, figsize=(12, 4), tight_layout=True)

    font_size = 14  # 设置字号
    font_weight = 'medium'  # 设置加粗  # bold normal

    for i in range(2):
        axes[i].set_xlabel(axes[i].get_xlabel(), fontsize=font_size, fontweight=font_weight)
        axes[i].set_ylabel(axes[i].get_ylabel(), fontsize=font_size, fontweight=font_weight)
        axes[i].tick_params(axis='both', labelsize=14, width=1)

    # wave1
    axes[0].plot(np.arange(len(unfilter_label)) / fps, unfilter_label, color='#000000', label='ground truth')
    #axes[0].plot(np.arange(len(unfilter_pred)) / fps, unfilter_pred, color='#FF0000', label='rPPG', alpha=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(False)
    axes[0].set_xlim(0, len(unfilter_pred) / fps)
    axes[0].set_ylim(min(min(unfilter_pred), min(unfilter_label)), max(max(unfilter_pred), max(unfilter_label)))


    # wave2
    axes[1].plot(np.arange(len(filter_label)) / fps, filter_label, color='#000000', label='ground truth')
    #axes[1].plot(np.arange(len(filter_pred)) / fps, filter_pred, color='#FF0000', label='rPPG', alpha=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(False)
    axes[1].set_xlim(0, len(filter_pred) / fps)
    axes[1].set_ylim(min(min(filter_pred), min(filter_label)), max(max(filter_pred), max(filter_label)))



    plt.savefig(fig_path, bbox_inches='tight', dpi=300)

    #plt.show()
    plt.close()

    print("Plot wave over!!!")


if __name__ == '__main__':
    # 生成模拟 BVP 信号
    fps = 30
    t = np.linspace(0, 10, 10 * fps)  # 10 秒信号
    bvp_signal = np.sin(2 * np.pi * 1.2 * t)  # 1.2Hz (72bpm) 的正弦波模拟心率信号

    # 测试 plot_wave_psd
    plot_wave_psd(pred=bvp_signal,label=bvp_signal, fps=fps, fig_path="1.jpg")