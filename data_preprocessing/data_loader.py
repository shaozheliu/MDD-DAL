import os.path
import torch
import numpy as np
from scipy.io import loadmat
import random
from pylab import *
from numpy import *
from scipy import interpolate
from mne.io import concatenate_raws, read_raw_edf, read_raw_eeglab
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import cohen_kappa_score
import random
from torch.utils.data import Dataset, DataLoader


class DataProcess():
    '''
    load data from BCI Competition 2a and 2b
    '''
    data = []
    label = []

    '''
    data_path: the data files path
    data_files:the data files name(ex:["A01T"] or ["B0103E"])
    choose2aor2b: if choose 2a dataset set 1;else set 2
    choose2aclasses: if choose all 4 classes in 2a dataset set 1;
                     if choose left hand and right hand 2 classes set 2;
                     if choose foot and tongue 2 classes set 3.

    '''

    def __init__(self,
                 data_path,
                 data_files,
                 subject_list,
                 domain_flg = True,
                 domain_num = 1):
        self.data_path = data_path
        self.data_files = data_files
        self.subject_list = subject_list
        self.domain_flg = domain_flg
        self.domain_num = domain_num
        # 加载MDD样本的数据
        data_MDD, label_MDD, domain_label_MDD = self.import_subjecti_data_DAL(subject_list, domain_flg, "MDD", self.domain_num)
        # data_MDD, label_MDD, domain_label_MDD = self.import_subjecti_data(subject_list, domain_flg, "MDD")
        # data_H, label_H, domain_label_H = self.import_subjecti_data(subject_list, domain_flg, "H")
        data_H, label_H, domain_label_H = self.import_subjecti_data_DAL(subject_list, domain_flg, "H", self.domain_num)
        if np.array(data_MDD).shape[0] == 0:
            data = data_H
            label = label_H
            domain_label = domain_label_H
        else:
            data = np.concatenate((data_MDD, data_H), axis=0)
            label = np.concatenate((label_MDD, label_H), axis=0)
            domain_label = np.concatenate((domain_label_MDD, domain_label_H), axis=0)
        # Normalized
        # data = data.swapaxes(1, 2)
        data = np.array(data)
        norm_data  = data - data.mean(axis=0)
        norm_data =  norm_data / data.std(axis=0)

        self.data = norm_data
        self.label = label
        self.domain_label = domain_label
        print(self.data.shape)
        print(self.label.shape)
        print(self.domain_label.shape)

    def sample_edf(self, raw):
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
        data = np.lib.stride_tricks.sliding_window_view(data, window_shape=window*int(sfreq), axis=1)[:,::1*int(sfreq),:]  # 一秒的步长
        # times = times.reshape(int(sfreq) * window, -1)
        data = data.transpose(1, 0, 2)
        # times = times.transpose(1, 0)
        return data

    def import_subjecti_data(self, subject_list, domain_flg, subject_type):
        domainset = list(i[1:] for i in subject_list)
        data = []
        label = []
        domain_label = []
        fortest = []
        # 区分健康样本和非健康样本
        if subject_type == 'H':
            subject_label = 0
        else:
            subject_label = 1
        for i in range(len(subject_list)):
            domain_value = domainset.index(subject_list[i][1:])
            fortest.append(domain_value)
            data_file = self.data_path + f'{subject_type} {subject_list[i]} EC.edf'
            if os.path.exists(data_file) == False:
                continue
            if np.array(data).shape[0] == 0:
                data = read_raw_edf(data_file, exclude=['EEG A2-A1','EEG 23A-23R','EEG 24A-24R'])  # 原论文中的方法就排除了最后一个通道
                data = self.sample_edf(data)
                label = np.ones(data.shape[0], dtype=int) * subject_label
                if domain_flg == True:
                    domain_label = np.ones(data.shape[0], dtype=int) * domain_value
            else:
                data_t = read_raw_edf(data_file, exclude=['EEG A2-A1','EEG 23A-23R','EEG 24A-24R'])
                data_t = self.sample_edf(data_t)
                label_t = np.ones(data_t.shape[0], dtype=int) * subject_label
                if domain_flg == True:
                    domain_label_t = np.ones(data_t.shape[0], dtype=int) * domain_value # 这个dtype可能有问题
                data = np.concatenate((data, data_t), axis=0)
                label = np.concatenate((label, label_t), axis=0)
                domain_label = np.concatenate((domain_label, domain_label_t), axis=0)
            print(subject_list[i], "load success.")
        # domain_label = domain_label - domain_label.min()
        ret_data = data
        return ret_data, label, domain_label

    def assign_domain_labels(self, input_list, alpha):
        # 确保 alpha 的值在有效范围内
        if alpha < 1 or alpha > len(input_list):
            raise ValueError("Alpha must be between 1 and the length of the list.")

        labels = []

        if alpha == len(input_list):
            # 按照索引顺序赋值标签
            labels = list(range(len(input_list)))
        else:
            # 生成随机标签
            for _ in input_list:
                label = random.randint(0, alpha - 1)
                labels.append(label)

        return labels

    def import_subjecti_data_DAL(self, subject_list, domain_flg, subject_type, domain_num):
        domainset = list(i[1:] for i in subject_list)
        domain_flg_test = self.assign_domain_labels(domainset, domain_num)
        data = []
        label = []
        domain_label = []
        # 区分健康样本和非健康样本
        if subject_type == 'H':
            subject_label = 0
        else:
            subject_label = 1
        for i in range(len(subject_list)):
            domain_value = domain_flg_test[domainset.index(subject_list[i][1:])]
            data_file = self.data_path + f'{subject_type} {subject_list[i]} EC.edf'
            if os.path.exists(data_file) == False:
                continue
            if np.array(data).shape[0] == 0:
                data = read_raw_edf(data_file, exclude=['EEG A2-A1','EEG 23A-23R','EEG 24A-24R'])  # 原论文中的方法就排除了最后一个通道
                data = self.sample_edf(data)
                label = np.ones(data.shape[0], dtype=int) * subject_label
                if domain_flg == True:
                    domain_label = np.ones(data.shape[0], dtype=int) * domain_value
            else:
                data_t = read_raw_edf(data_file, exclude=['EEG A2-A1','EEG 23A-23R','EEG 24A-24R'])
                data_t = self.sample_edf(data_t)
                label_t = np.ones(data_t.shape[0], dtype=int) * subject_label
                if domain_flg == True:
                    domain_label_t = np.ones(data_t.shape[0], dtype=int) * domain_value # 这个dtype可能有问题
                data = np.concatenate((data, data_t), axis=0)
                label = np.concatenate((label, label_t), axis=0)
                domain_label = np.concatenate((domain_label, domain_label_t), axis=0)
            print(subject_list[i], "load success.")
        # domain_label = domain_label - domain_label.min()
        ret_data = data
        return ret_data, label, domain_label





    def import_subjecti_data_test(self, subject_list, data_files, domain_flg):
        domainset = list(set(i[1:] for i in subject_list))
        data = []
        label = []
        domain_label = []
        # 健康样本
        for i in data_files:
            if i.split(' ')[1] in subject_list:
                data_file = self.data_path + i
            else:
                continue
            # 确定domain label
            subject_label = 0 if i.split(' ')[0] == 'H' else 1
            domain_value = domainset.index(i.split(' ')[1][1:])
            if np.array(data).shape[0] == 0:
                data = read_raw_edf(data_file, exclude=['EEG A2-A1','EEG 23A-23R','EEG 24A-24R'])  # 原论文中的方法就排除了最后一个通道
                data = self.sample_edf(data)
                label = np.ones(data.shape[0], dtype=int) * subject_label
                if domain_flg == True:
                    domain_label = np.ones(data.shape[0], dtype=int) * domain_value
            else:
                data_t = read_raw_edf(data_file, exclude=['EEG A2-A1','EEG 23A-23R','EEG 24A-24R'])
                data_t = self.sample_edf(data_t)
                label_t = np.ones(data_t.shape[0], dtype=int) * subject_label
                if domain_flg == True:
                    domain_label_t = np.ones(data_t.shape[0], dtype=int) * domain_value # 这个dtype可能有问题
                data = np.concatenate((data, data_t), axis=0)
                label = np.concatenate((label, label_t), axis=0)
                domain_label = np.concatenate((domain_label, domain_label_t), axis=0)
            print(i, "load success.")
            # domain_label = domain_label - domain_label.min()
        return data, label, domain_label


class dataset_with_domain(Dataset):
    def __init__(self, X, y, y_domain, train=True):
        self.X = X
        self.y = y
        self.y_domain = y_domain
        self.train=train

    def __len__(self):
        return len(self.y)

    # def __getitem__(self, idx):
    #     rng = np.random.randint(0, high=200)
    #     if self.train:
    #         x = self.X[idx][:, rng:rng + 600]
    #     else:
    #         x = self.X[idx][:, 200: 800]
    #     return x, self.y[idx]

    def __getitem__(self, idx):
        # idx = 1000
        x = self.X[idx]
        if self.train:
#             rn = np.random.randint(0, high=500)
#             x = x[:, rn:rn+4000]
            x = x[:, 0:4000]
        else:
            x = x[:, 0:4000]
        return x, self.y[idx], self.y_domain[idx]


def get_loaders_with_domain(train_X, train_y, train_domain_y, test_X, test_y, test_domain_y, batch_size = 250):
    train_set, test_set = dataset_with_domain(train_X, train_y, train_domain_y, True), \
                                   dataset_with_domain(test_X, test_y, test_domain_y, False)
    data_loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        # pin_memory=True,
        drop_last=False,
        shuffle= True
    )
    data_loader_test = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            num_workers=0,
            # pin_memory=True,
            drop_last=False,
            shuffle=True
    )
    dataloaders = {
        'train': data_loader_train,
        'test': data_loader_test
    }
    return dataloaders


if __name__ == '__main__':
    data_2b_path = '/hy-tmp/data/MDD/'
    data_2b_files = ["H S1 EC", "H S1 EC"]
    subject_type = 'H'
    subject_lists = ['S' + str(i) for i in range(1,31)]
    # MDD_subjects = ['S' + str(i) for i in range(1, 35)]
    loo = LeaveOneOut()
    GetData = DataProcess(data_2b_path, data_2b_files, subject_lists)
    data = GetData.data
    label = GetData.label