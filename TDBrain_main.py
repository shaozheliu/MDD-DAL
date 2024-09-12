import random
import os
import sys
import re
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from sklearn.model_selection import LeaveOneOut
from data_preprocessing.data_loader import *
from models.model_factory import prepare_training, train_model_with_domain, test_evaluate, test_evaluate_2b, train_model_baseline
from torchsummary import summary
from collections import  Counter
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
# model_name = 'Mymodel_TDBrain'
model_name = 'EEGNet_TDBrain'
type = 'all'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 32
# d_model = 50
# n_heads = 10
# 全卷积特征给你提取
d_model = 62
emb_feature = 30
n_heads = 2
d_ff = 32
n_layers = 1
n_epoch = 50
# lr = 0.01
dropout = 0.4
alpha_scala = 1
domain_flg = False

import os


def process_files(folder_path):
    files_dict = {}

    # 遍历文件夹下所有文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            parts = file_name.split('.')[0].split('_')
            key = parts[1]
            if key not in files_dict:
                files_dict[key] = []
            files_dict[key].append(file_name)

    filtered_files = []

    # 筛选满足要求的文件
    for key, files in files_dict.items():
        filtered_files.append(key)

    return filtered_files


# 指定文件夹路径
data_type = 'TDBrain'
data_path = '/home/alk/Data/Resultdata0906/'
domain_num = 3
# 调用函数处理文件
subject_lists = process_files(data_path)





# 打印日志设定
outputfile = open(f'./logs/data:{data_type}_exp:{model_name}_dropout:{dropout}_attdim:{d_model}_encoder_num:{n_layers}_lr:{lr}_domainnum:{domain_num}.txt', 'w')
sys.stdout = outputfile
data_files = os.listdir(data_path)
loo = LeaveOneOut()
acc_ls = []
ka_ls = []
prec_ls = []
recall_ls = []
roc_auc_ls = []
for i, (train_idx, test_idx) in enumerate(loo.split(subject_lists)):
    print(f"---------Start Fold {i} process---------")
    print(f'train subject is: {train_idx}, test subject is: {test_idx}')
    train_subjects = []
    test_subjects = []
    for idx in train_idx[:]:
        train_subjects.append(subject_lists[idx])
    for idt in test_idx:
        test_subjects.append(subject_lists[idt])
    GetData_train = TDBrain_DataProcess(data_path, data_files, train_subjects, True, domain_num)
    X_train, y_train, y_train_domain = GetData_train.data, GetData_train.label, GetData_train.domain_label
    GetData_test = TDBrain_DataProcess(data_path, data_files, test_subjects,True, 1)
    X_test, y_test, y_test_domain = GetData_test.data, GetData_test.label, GetData_test.domain_label
    # 1155， 1024， 26
    a = Counter(y_test)
    b = Counter(y_train)
    print(a)
    print(b)
    dataloaders = get_loaders_with_domain(X_train, y_train, y_train_domain, X_test, y_test, y_test_domain, batch_size)
    train_sample = dataloaders['train'].dataset.X.shape[0]
    test_sample = dataloaders['test'].dataset.X.shape[0]
    dataset = {'dataset_sizes': {'train': train_sample, 'test': test_sample}}
    model, optimizer, lr_scheduler, criterion, device, criterion_domain = prepare_training(d_model, d_ff, n_heads,
                                                                                              n_layers, dropout,
                                                                                              lr, model_name, type,
                                                                                              emb_feature, 2, domain_num)
    # print(summary(model, input_size=[(X_train.shape[1], X_train.shape[2]), (1, alpha_scala)]))
    if domain_flg == True:
        best_model = train_model_with_domain(model, criterion, criterion_domain, optimizer, lr_scheduler, device,
                                             dataloaders, n_epoch, dataset)
    else:
        best_model = train_model_baseline(model, criterion, criterion_domain, optimizer, lr_scheduler, device,
                                             dataloaders, n_epoch, dataset)
    acc, ka, prec, recall, roc_auc = test_evaluate_2b(best_model, device, X_test, y_test, model_name)
    acc_ls.append(acc)
    ka_ls.append(ka)
    prec_ls.append(prec)
    recall_ls.append(recall)
    roc_auc_ls.append(roc_auc)

    print(f'The accuracy is: {acc_ls}, cross-subject acc is: {np.mean(acc_ls)} \n')
    print(f'The acohen_kappa_score is: {ka_ls}, cross-subject acohen_kappa_score is: {np.mean(ka_ls)} \n')
    print(f'The precision is: {prec_ls}, cross-subject precision is: {np.mean(prec_ls)} \n')
    print(f'The recall is: {recall_ls}, cross-subject recall is: {np.mean(recall_ls)} \n')
    print(f'The roc_auc is: {roc_auc_ls}, cross-subject roc is: {np.mean(roc_auc_ls)} \n')

outputfile.close()  # 关闭文件