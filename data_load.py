from mne.io import concatenate_raws, read_raw_edf, read_raw_eeglab
import matplotlib.pyplot as plt
import mne
import numpy as np


raw = read_raw_edf("../Data/MDD/MDD S2  EC.edf",exclude =['EEG A2-A1','EEG 23A-23R','EEG 24A-24R'] ,preload=True)  # 原论文中的方法就排除了最后一个通道
# raw = read_raw_eeglab("../Data/MDD/MDD S1 EC.edf", preload=False)  # 原论文中的方法就排除了最后一个通道
old_chan_names = ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE',
                  'EEG T3-LE', 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE',
                  'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE',
                  'EEG Pz-LE','EEG A2-A1']  # 最后一个channel没用  参考电
old_chan_names = ['EEG Fp1-LE', 'EEG F3-LE', 'EEG C3-LE', 'EEG P3-LE', 'EEG O1-LE', 'EEG F7-LE',
                  'EEG T3-LE', 'EEG T5-LE', 'EEG Fz-LE', 'EEG Fp2-LE', 'EEG F4-LE', 'EEG C4-LE',
                  'EEG P4-LE', 'EEG O2-LE', 'EEG F8-LE', 'EEG T4-LE', 'EEG T6-LE', 'EEG Cz-LE',
                  'EEG Pz-LE']  # 最后一个channel没用
chan_nams_dict = {old_chan_names[i]:old_chan_names[i].split(' ')[1].split('-')[0] for i in range(len(old_chan_names))}
montage = mne.channels.make_standard_montage('standard_1020')
raw.rename_channels(chan_nams_dict)
# 传⼊数据的电极位置信息
raw.set_montage(montage)
print(raw)
print(raw.info)

def sample_edf(raw):
    """
    获取采样频率sfreq

    知识点:

    “采样频率，也称为采样速度或者采样率，定义了每秒从连续信号中提取并组成离散信号的采样个数，它用赫兹（Hz）来表示。
    采样频率的倒数是采样周期或者叫作采样时间，它是采样之间的时间间隔。
    通俗的讲采样频率是指计算机每秒钟采集多少个信号样本。”

    """
    sfreq = raw.info['sfreq']
    time_interval = len(raw.times)
    """
    获取索引为m到n的样本，每个样本从第k次到第h次.
    data,times=raw[m:n,k:h]

    其中data为索引为m到n的样本，每个样本从第k次到第h次.
    times是以第k次采样的时间作为开始时间，第h次采样时的时间为结束时间的时间数组。
    """
    window = 4
    data, times = raw[:, :]
    data = data.reshape(19,int(sfreq)*window, -1)
    times = times.reshape(int(sfreq)*window, -1)
    data = data.transpose(2,0,1)
    times = times.transpose(1,0)
    return data, times

# data, times = sample_edf(raw)
# print(data.shape)

# raw.plot(duration=5, n_channels = 19, clipping=None)
# raw.plot_sensors(ch_type='eeg', show_names=True)
# plt.show()
a = np.lib.stride_tricks.sliding_window_view(np.array([1, 2, 3, 4, 5, 6]), window_shape = 3)[::2]
print(a.shape)