"""The post processing files for caluclating heart rate using FFT or peak detection.
The file also  includes helper funcs such as detrend, mag2db etc.
"""
import numpy as np
import scipy
import scipy.io
import torch
from numpy.ma.core import shape
from scipy.signal import butter, welch
from scipy.sparse import spdiags
from copy import deepcopy


def get_hr(y, sr=30, min=45, max=160):
    p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
    return p[(p>min/60)&(p<max/60)][np.argmax(q[(p>min/60)&(p<max/60)])]*60

# def get_psd(y, sr=30, min=45, max=150):
#     p, q = welch(y, sr, nfft=1e5/sr, nperseg=np.min((len(y)-1, 256)))
#     return q[(p>min/60)&(p<max/60)]

def get_psd(y, sr=30, min_bpm=45, max_bpm=160):
    """计算信号的功率谱密度，并筛选出 45 BPM 到 150 BPM 的部分"""
    p, q = welch(y, sr, nfft=int(1e5/sr), nperseg=np.min((len(y)-1, 256)))
    bpm = p * 60  # 频率 (Hz) 转换为 BPM
    mask = (bpm > min_bpm) & (bpm < max_bpm)
    return bpm[mask], q[mask]  # 只返回 BPM 在 45-150 之间的数据


def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def _detrend(input_signal, lambda_value):
    """Detrend PPG signal."""
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    detrended_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return detrended_signal

def mag2db(mag):
    """Convert magnitude to db."""
    return 20. * np.log10(mag)

def _calculate_fft_hr(ppg_signal, fs=30, low_pass=0.75, high_pass=2.6):
    """Calculate heart rate based on PPG using Fast Fourier transform (FFT)."""
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr

def _calculate_peak_hr(ppg_signal, fs):
    """Calculate heart rate based on PPG using peak detection."""
    ppg_peaks, _ = scipy.signal.find_peaks(ppg_signal)
    hr_peak = 60 / (np.mean(np.diff(ppg_peaks)) / fs)
    return hr_peak

def _calculate_SNR(pred_ppg_signal, hr_label, fs=30, low_pass=0.75, high_pass=2.6):
    """Calculate SNR as the ratio of the area under the curve of the frequency spectrum around the first and second harmonics
        of the ground truth HR frequency to the area under the curve of the remainder of the frequency spectrum, from 0.75 Hz
        to 2.6 Hz.

        Args:
            pred_ppg_signal(np.array): predicted PPG signal
            label_ppg_signal(np.array): ground truth, label PPG signal
            fs(int or float): sampling rate of the video
        Returns:
            SNR(float): Signal-to-Noise Ratio
    """
    # Get the first and second harmonics of the ground truth HR in Hz
    first_harmonic_freq = hr_label / 60
    second_harmonic_freq = 2 * first_harmonic_freq
    deviation = 6 / 60  # 6 beats/min converted to Hz (1 Hz = 60 beats/min)

    # Calculate FFT
    pred_ppg_signal = np.expand_dims(pred_ppg_signal, 0)
    N = _next_power_of_2(pred_ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(pred_ppg_signal, fs=fs, nfft=N, detrend=False)

    # Calculate the indices corresponding to the frequency ranges
    idx_harmonic1 = np.argwhere((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation)))
    idx_harmonic2 = np.argwhere((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation)))
    idx_remainder = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass) \
     & ~((f_ppg >= (first_harmonic_freq - deviation)) & (f_ppg <= (first_harmonic_freq + deviation))) \
     & ~((f_ppg >= (second_harmonic_freq - deviation)) & (f_ppg <= (second_harmonic_freq + deviation))))

    # Select the corresponding values from the periodogram
    pxx_ppg = np.squeeze(pxx_ppg)
    pxx_harmonic1 = pxx_ppg[idx_harmonic1]
    pxx_harmonic2 = pxx_ppg[idx_harmonic2]
    pxx_remainder = pxx_ppg[idx_remainder]

    # Calculate the signal power
    signal_power_hm1 = np.sum(pxx_harmonic1)
    signal_power_hm2 = np.sum(pxx_harmonic2)
    signal_power_rem = np.sum(pxx_remainder)

    # Calculate the SNR as the ratio of the areas
    if not signal_power_rem == 0: # catches divide by 0 runtime warning
        SNR = mag2db((signal_power_hm1 + signal_power_hm2) / signal_power_rem)
    else:
        SNR = 0
    return SNR

def _compute_macc(pred_signal, gt_signal):
    """Calculate maximum amplitude of cross correlation (MACC) by computing correlation at all time lags.
        Args:
            pred_ppg_signal(np.array): predicted PPG signal
            label_ppg_signal(np.array): ground truth, label PPG signal
        Returns:
            MACC(float): Maximum Amplitude of Cross-Correlation
    """
    pred = deepcopy(pred_signal)
    gt = deepcopy(gt_signal)
    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    min_len = np.min((len(pred), len(gt)))
    pred = pred[:min_len]
    gt = gt[:min_len]
    lags = np.arange(0, len(pred) - 1, 1)
    tlcc_list = []
    for lag in lags:
        cross_corr = np.abs(np.corrcoef(
            pred, np.roll(gt, lag))[0][1])
        tlcc_list.append(cross_corr)
    macc = max(tlcc_list)
    return macc


def calculate_metric_per_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    if use_bandpass:
        # bandpass filter between [0.75, 2.6] Hz
        # equals [45, 150] beats per min
        [b, a] = butter(1, [0.75 / fs * 2, 2.6 / fs * 2], btype='bandpass')
        predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
        labels = scipy.signal.filtfilt(b, a, np.double(labels))

    macc = _compute_macc(predictions, labels)

    if hr_method == 'FFT':
        hr_pred = get_hr(predictions, sr=fs)
        hr_label = get_hr(labels, sr=fs)
    elif hr_method == 'Peak':
        hr_pred = _calculate_peak_hr(predictions, fs=fs)
        hr_label = _calculate_peak_hr(labels, fs=fs)
    else:
        raise ValueError('Please use FFT or Peak to calculate your HR.')
    SNR = _calculate_SNR(predictions, hr_label, fs=fs)
    return hr_pred, hr_label, SNR, macc

def calculate_metric_batch_video(predictions, labels, fs=30, diff_flag=True, use_bandpass=True, hr_method='FFT'):
    """ 计算 batch 级别的视频 HR 和 SNR
        输入:
            predictions: (B, T) 预测的 rPPG 信号
            labels: (B, T) 真实的 BVP 信号
        输出:
            hr_pred: (B,) 预测的 HR
            hr_label: (B,) 真实的 HR
            SNR: (B,) 信噪比
    """
    B, T = predictions.shape  # 获取 batch_size 和 时间序列长度
    hr_pred_list = []
    hr_label_list = []
    SNR_list = []
    macc_list = []

    for i in range(B):  # 逐个样本处理
        pred = predictions[i]
        label = labels[i]

        if diff_flag:
            pred = _detrend(np.cumsum(pred), 100)
            label = _detrend(np.cumsum(label), 100)
        else:
            pred = _detrend(pred, 100)
            label = _detrend(label, 100)

        if use_bandpass:
            [b, a] = butter(1, [0.75 / fs * 2, 2.6 / fs * 2], btype='bandpass')
            pred = scipy.signal.filtfilt(b, a, np.double(pred))
            label = scipy.signal.filtfilt(b, a, np.double(label))

        macc = _compute_macc(predictions, labels)

        if hr_method == 'FFT':
            hr_pred = get_hr(pred, sr=fs)
            hr_label = get_hr(label, sr=fs)
        elif hr_method == 'Peak':
            hr_pred = _calculate_peak_hr(pred, fs=fs)
            hr_label = _calculate_peak_hr(label, fs=fs)
        else:
            raise ValueError('Please use FFT or Peak to calculate your HR.')

        SNR = _calculate_SNR(pred, hr_label, fs=fs)  # 逐个样本计算 SNR

        hr_pred_list.append(hr_pred)
        hr_label_list.append(hr_label)
        SNR_list.append(SNR)
        macc_list.append(macc)

    return np.array(hr_pred_list), np.array(hr_label_list), np.array(SNR_list), np.array(macc_list)


def calculate_hr(predictions, labels, fs=30, diff_flag=False):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    [b, a] = butter(1, [0.75 / fs * 2, 2.6 / fs * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))
    hr_pred = get_hr(predictions, sr=fs)
    hr_label = get_hr(labels, sr=fs)
    return hr_pred , hr_label

def calculate_psd(predictions, labels, fs=30, diff_flag=False):
    """Calculate video-level HR and SNR"""
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        predictions = _detrend(np.cumsum(predictions), 100)
        labels = _detrend(np.cumsum(labels), 100)
    else:
        predictions = _detrend(predictions, 100)
        labels = _detrend(labels, 100)
    [b, a] = butter(1, [0.75 / fs * 2, 2.6 / fs * 2], btype='bandpass')
    predictions = scipy.signal.filtfilt(b, a, np.double(predictions))
    labels = scipy.signal.filtfilt(b, a, np.double(labels))
    psd_pred = get_psd(predictions, sr=fs)
    psd_label = get_psd(labels, sr=fs)
    return psd_pred , psd_label

def bandpass(signal, fs = 30, diff_flag=False):
    if diff_flag:  # if the predictions and labels are 1st derivative of PPG signal.
        signal = _detrend(np.cumsum(signal), 100)
    else:
        signal = _detrend(signal, 100)

    [b, a] = butter(1, [0.75 / fs * 2, 2.6 / fs * 2], btype='bandpass')
    filter_signal = scipy.signal.filtfilt(b, a, np.double(signal))
    return filter_signal


def read_fold():
    lines = []
    with open("evaluation/vipl_filter_fold.txt", 'r', encoding='utf-8') as file:
        for line in file:
            lines.append(line.strip())
    return lines


def calculate_metrics(predict_hr_all, gt_hr_fft_all, SNR_all, MACC_all):

    predict_hr_all = np.asarray(predict_hr_all)
    gt_hr_fft_all = np.asarray(gt_hr_fft_all)
    SNR_all = np.asarray(SNR_all)
    MACC_all = np.asarray(MACC_all)

    num_samples = len(predict_hr_all)

    MAE_FFT = np.mean(np.abs(predict_hr_all - gt_hr_fft_all))
    MAE_std = np.std(np.abs(predict_hr_all - gt_hr_fft_all)) / np.sqrt(num_samples)
    print("FFT MAE (FFT Label): {0} +/- {1}".format(MAE_FFT, MAE_std))

    # Calculate the squared errors, then RMSE, in order to allow
    # for a more robust and intuitive standard error that won't
    # be influenced by abnormal distributions of errors.

    squared_errors = np.square(predict_hr_all - gt_hr_fft_all)
    RMSE_FFT = np.sqrt(np.mean(squared_errors))
    RMSE_std = np.sqrt(np.std(squared_errors) / np.sqrt(num_samples))
    print("FFT RMSE (FFT Label): {0} +/- {1}".format(RMSE_FFT, RMSE_std))

    MAPE_FFT = np.mean(np.abs((predict_hr_all - gt_hr_fft_all) / gt_hr_fft_all)) * 100
    MAPE_std = np.std(np.abs((predict_hr_all - gt_hr_fft_all) / gt_hr_fft_all)) / np.sqrt(
        num_samples) * 100
    print("FFT MAPE (FFT Label): {0} +/- {1}".format(MAPE_FFT, MAPE_std))

    Pearson_FFT = np.corrcoef(predict_hr_all, gt_hr_fft_all)
    correlation_coefficient = Pearson_FFT[0][1]
    Pearson_std = np.sqrt((1 - correlation_coefficient ** 2) / (num_samples - 2))
    print("FFT Pearson (FFT Label): {0} +/- {1}".format(correlation_coefficient, Pearson_std))

    SNR_FFT = np.mean(SNR_all)
    SNR_std = np.std(SNR_all) / np.sqrt(num_samples)
    print("FFT SNR (FFT Label): {0} +/- {1} (dB)".format(SNR_FFT, SNR_std))

    MACC_avg = np.mean(MACC_all)
    MACC_std = np.std(MACC_all) / np.sqrt(num_samples)
    print("FFT MACC (FFT Label): {0} +/- {1}".format(MACC_avg, MACC_std))

    metrics_dict = {
        "MAE": {"value": MAE_FFT, "std": MAE_std},
        "RMSE": {"value": RMSE_FFT, "std": RMSE_std},
        "MAPE": {"value": MAPE_FFT, "std": MAPE_std},
        "Pearson": {"value": correlation_coefficient, "std": Pearson_std},
        "SNR": {"value": SNR_FFT, "std": SNR_std},
        "MACC": {"value": MACC_avg, "std": MACC_std}
    }

    return metrics_dict




def normalize(x):
    return (x-x.mean())/x.std()


if __name__ == '__main__':
    # pre = torch.rand(1, 300)
    # label = torch.rand(1, 300)

    # accu_mae = torch.zeros(1)
    # hr_pred, hr_label, SNR, MACC = calculate_metric_batch_video(pre, label)

    # accu_mae += np.abs(hr_pred - hr_label).sum()
    # print(type(accu_mae/accu_mae.shape[0]))
    # print(hr_pred,'\n', hr_label,'\n',SNR,'\n', MACC)

    pre = np.random.rand(300)
    label = np.random.rand(300)
    # hr, psd_y, psd_x = hr_fft(pre, fs=30)  # fs是视频帧率
    # hr_true, psd_y2, psd_x2 = hr_fft(label, fs=30)  # fs是视频帧率
    psd_pred, psd_label = calculate_psd(pre, label)
    print(psd_pred[0].shape, psd_pred[1].shape)
    print(psd_pred[0], psd_pred[1])
    print(psd_label[0].shape, psd_label[1].shape)