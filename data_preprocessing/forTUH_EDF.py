import os
import numpy as np
import mne
import psutil  # 用于检查内存使用情况
from scipy.signal import iirnotch, filtfilt


"""
Extract EEG data from each EDF file and segment the data
The shape of each subsample is (5000, 19), 
5000 is the sequence length 
19 is the number of target electrode channels.
"""
def preprocess_and_save_data(folder_path, save_path, target_channels, sampling_rate=250, segment_length=5000):
    """
    Preprocess EDF files and save the processed data to a file.

    Parameters:
    - folder_path: The directory path containing the EDF files.
    - save_path: The path where the processed data will be saved.
    - target_channels: A list of target electrode channel names to retain.
    - sampling_rate: The target sampling rate, default is 250 Hz.
    - segment_length: The data length (number of sampling points) for each sample, default is 5000.
    """
    edf_files = [f for f in os.listdir(folder_path) if f.endswith('.edf')]

    for file in edf_files:
        file_path = os.path.join(folder_path, file)

        raw = mne.io.read_raw_edf(file_path, preload=True)

        # Check the original sampling rate, and if it is lower than the target sampling rate, skip the file
        if raw.info['sfreq'] < sampling_rate:
            print(f"跳过 {file}: 采样率低于 {sampling_rate} Hz")
            continue

        # Electrode channel processing
        raw.rename_channels(lambda x: x.strip().upper())
        cleaned_channels = raw.ch_names
        cleaned_target_channels = [ch.strip().upper() for ch in target_channels]
        matched_channels = {target: col for target in cleaned_target_channels for col in cleaned_channels if col.startswith(target[:-4])}
        missing_channels = [ch for ch in cleaned_target_channels if ch not in matched_channels]
        if missing_channels:
            print(f"文件 {file} 缺少以下通道: {missing_channels}")
            continue
        raw.pick_channels(list(matched_channels.values()))

        # bandpass filter（0.5-45 Hz）
        raw.filter(0.5, 45, method='iir')

        # Notch filtering removes power frequency noise（60 Hz ）
        for freq in np.arange(60, raw.info['sfreq'] / 2, 60):
            notch_freq = freq / (raw.info['sfreq'] / 2)
            b, a = iirnotch(notch_freq, Q=30)
            raw._data = filtfilt(b, a, raw._data, axis=1)

        # Downsample data to target sampling rate
        raw.resample(sampling_rate)

        # Extract data and segment it
        data = raw.get_data()
        n_samples = data.shape[1]
        segments = []

        for start_idx in range(0, n_samples, segment_length):
            end_idx = min(start_idx + segment_length, n_samples)
            segment = data[:, start_idx:end_idx]

            if segment.shape[1] < segment_length:
                break
            segment = segment.T  # (5000, 19)
            segments.append(segment)

        npy_file_path = os.path.join(save_path, f'{os.path.splitext(file)[0]}_processed.npy')
        np.save(npy_file_path, np.array(segments))
        print(f"文件 {file} 的数据已保存到 {npy_file_path}")


folder_path = r'D:\Dataset\tuh'
save_path = r'D:\Dataset\tuh_preprocess'

target_channels = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
                   'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
                   'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
                   'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
                   'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']

preprocess_and_save_data(folder_path, save_path, target_channels)