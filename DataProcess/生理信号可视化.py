import numpy as np
import matplotlib.pyplot as plt
import h5py

def plot_bvp_hr_spo2(h5_path, save_path="bvp_hr_spo2_plot5.png"):
    with h5py.File(h5_path, 'r') as f:
        bvp = f['bvp'][:]     # shape: (1200,)  20Hz * 60s
        hr = f['hr'][:]       # shape: (60,)
        spo2 = f['spo2'][:]   # shape: (60,)

    fps_bvp = 20  # BVP 的采样频率为 20Hz
    time_bvp = np.arange(len(bvp)) / fps_bvp
    time_hr = np.linspace(0, 60, len(hr))
    time_spo2 = np.linspace(0, 60, len(spo2))

    font_size = 14
    font_weight = 'medium'

    fig, axes = plt.subplots(3, 1, figsize=(12, 6), tight_layout=True, sharex=True)

    # 设置统一字体风格
    for ax in axes:
        ax.tick_params(axis='both', labelsize=font_size, width=1)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size, fontweight=font_weight)

    # --- BVP 波形 ---
    axes[0].plot(time_bvp, bvp, color='#0072B2', linewidth=1)
    axes[0].set_ylabel('BVP', fontsize=font_size, fontweight=font_weight)
    axes[0].set_xlim(0, 60)
    # axes[0].set_title('BVP Signal (20Hz)', fontsize=font_size, fontweight=font_weight)

    # --- HR 曲线 ---
    axes[1].plot(time_hr, hr, color='#D55E00', marker='o', linestyle='-', linewidth=1)
    axes[1].set_ylabel('HR (bpm)', fontsize=font_size, fontweight=font_weight)
    # axes[1].set_title('Heart Rate (1Hz)', fontsize=font_size, fontweight=font_weight)

    # --- SpO2 曲线 ---
    axes[2].plot(time_spo2, spo2, color='#009E73', marker='s', linestyle='-', linewidth=1)
    axes[2].set_ylabel('SpO2 (%)', fontsize=font_size, fontweight=font_weight)
    axes[2].set_xlabel('Time (s)', fontsize=font_size, fontweight=font_weight)
    # axes[2].set_title('SpO₂ Signal (1Hz)', fontsize=font_size, fontweight=font_weight)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved visualization to: {save_path}")


plot_bvp_hr_spo2("F:/processdata0331/P6_2.h5")
