import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import manifold, datasets
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.manifold import TSNE
import os
import sys
from functools import partial
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import tqdm
import sys, time, copy

import torch.utils.data
from models.model_factory import prepare_training, train_model_with_domain, test_evaluate, test_evaluate_2b, save_best_model
from sklearn.model_selection import LeaveOneOut
from data_preprocessing.data_loader import *
base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)


model_name = 'Mymodel'
type = 'all'
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 128
# d_model = 50
# n_heads = 10
# 全卷积特征给你提取
d_model = 244
emb_feature = 30
n_heads = 4
d_ff = 32
n_layers = 1
n_epoch = 50
# lr = 0.01
dropout = 0.1
alpha_scala = 1
domain_flg = True
domain_num = 5


def plot_embedding_2d(X, y, ax=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Define colors for y values (red for class 0, grey for class 1, green for class 2, blue for class 3)
    colors = ['red' if label == 0 else 'grey' if label == 1 else 'green' if label == 2 else 'blue' for label in y]

    # Plot scatter plot with colors based on y
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], color=colors[i])

    # Remove x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])


import numpy as np
import matplotlib.pyplot as plt


def plot_embedding_2d_domain(X, y, domain, ax=None):
    """Plot an embedding X with the class label y colored by the domain label."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Define colors for y values (5 classes)
    colors = [
        'red',  # Class 0
        'grey',  # Class 1
        'green',  # Class 2
        'blue',  # Class 3
        'orange'  # Class 4
    ]

    # Define markers for domain labels (2 types)
    markers = ['o', '^']  # Circle for domain 0, Triangle for domain 1

    # Plot scatter plot with colors based on y and different shapes based on domain
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], color=colors[y[i]], marker=markers[domain[i]], edgecolor='k')

    # Remove x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])


# 初始化模型
model, optimizer, lr_scheduler, criterion, device, criterion_domain = prepare_training(d_model, d_ff, n_heads,
                                                                                                  n_layers, dropout,
                                                                                                  lr, model_name, type,
                                                                                                  emb_feature, domain_num)
# 加载参数
model_path = '/home/alk/MDD-classification/checkpoints/Mymodel/dropout:0.1/' \
             'domain_num:5/Model:Mymodel_domainnum:5_dropout:0.1_fold:0_testsub:S13.pth'
model.load_state_dict(torch.load(model_path))
print(model)

# 加载指定的数据
data_path = '/home/alk/Data/MDD/'
data_files = os.listdir(data_path)
test_subjects = ['S13', 'S29', 'S14', 'S19', 'S17']
GetData_test = DataProcess(data_path, data_files, test_subjects,True, domain_num)
train_X, train_y, y_test_domain = GetData_test.data, GetData_test.label, GetData_test.domain_label
X_train_mean = train_X.mean(0)
X_train_var = np.sqrt(train_X.var(0))
train_X -= X_train_mean
train_X /= X_train_var


def sample_uniformly(a, b, c, total_samples):
    unique_classes, counts = np.unique(c, return_counts=True)
    num_classes = len(unique_classes)

    # 计算每个类别应抽取的样本数
    samples_per_class = total_samples // num_classes
    if samples_per_class == 0:
        raise ValueError("Total samples must be greater than the number of unique classes.")

    sampled_a = []
    sampled_b = []
    sampled_c = []

    for cls in unique_classes:
        # 找到当前类别的索引
        indices = np.where(c == cls)[0]

        # 从当前类别中随机抽样
        selected_indices = np.random.choice(indices, min(samples_per_class, len(indices)), replace=False)

        # 将抽样的结果添加到结果列表中
        sampled_a.append(a[selected_indices])
        sampled_b.append(b[selected_indices])
        sampled_c.append(c[selected_indices])

    # 将所有抽样结果合并
    sampled_a = np.vstack(sampled_a)  # 可能需要调整形状
    sampled_b = np.concatenate(sampled_b)
    sampled_c = np.concatenate(sampled_c)

    return sampled_a, sampled_b, sampled_c


# 进行均匀抽样
train_X, train_y, y_test_domain = sample_uniformly(train_X, train_y, y_test_domain, 300)

# forward
model.eval()
inputs = torch.from_numpy(train_X).type(torch.cuda.FloatTensor).to(device)
labels = torch.from_numpy(train_y).type(torch.cuda.FloatTensor).to(device)
outputs, domain_outputs = model(inputs, 0.1)
outputs = outputs.detach().cpu().numpy()
tsne2d = TSNE(n_components=2, init='pca', random_state=0)
X_raw = tsne2d.fit_transform(train_X.reshape(train_X.shape[0], -1))
X_l2g = tsne2d.fit_transform(outputs)


fig, axs = plt.subplots(1, 2, figsize=(8, 4))
# 在每个子图中绘制不同的数据
for j in range(2):
    ax = axs[j]  # 直接使用 j 作为索引
    if j == 0:
        ax.set_title('', fontweight='bold')
        ax.set_ylabel('Raw Data', fontweight='bold')
        plot_embedding_2d_domain(X_raw[:, 0:2], y_test_domain ,train_y, ax)
    else:
        ax.set_title('', fontweight='bold')
        ax.set_ylabel('MMDDAL', fontweight='bold')
        plot_embedding_2d_domain(X_l2g[:, 0:2], y_test_domain ,train_y, ax)
# 创建共用的图例
# handles, labels = axs[0,0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
print("Computing t-SNE embedding")
# 自动调整子图布局
plt.tight_layout()
# 保存每个子图为单独的图像文件
plt.savefig(f'./tSNE.png', dpi=300)
plt.show()



#
#
# # 训练函数，待调参数为神经网络隐藏层的神经元数hiddenLayer
# def train_iris(dataSetlist, model_type):
#     data_dic = {}
#     y_dic = {}
#     for dataSet in dataSetlist:
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#         # 参数
#         spatial_local_dict = CONSTANT[dataSet]['spatial_ch_group']
#         temporal_div_dict = CONSTANT[dataSet]['temporal_ch_region']
#         try:
#             d_model_dict = config[model_type][dataSet]['d_model_dict']
#         except KeyError:
#             d_model_dict = None
#         try:
#             head_dict = config[model_type][dataSet]['head_dict']
#         except KeyError:
#             head_dict = None
#         path = CONSTANT[dataSet]['raw_path']
#         clf_class = config[model_type][dataSet]['num_class']
#         domain_class = CONSTANT[dataSet]['n_subjs']
#         try:
#             d_ff = config[model_type][dataSet]['d_ff']
#         except KeyError:
#             d_ff = None
#         try:
#             n_layers = config[model_type][dataSet]['n_layers']
#         except KeyError:
#             n_layers = None
#         batch_size = config[model_type][dataSet]['batch_size']
#         lr = config[model_type][dataSet]['lr']
#         dropout = config[model_type][dataSet]['dropout']
#         num_ch = len(CONSTANT[dataSet]['sel_chs'])
#         # 加载数据
#         sel_chs = CONSTANT[dataSet]['sel_chs']
#         if dataSet == 'OpenBMI':
#             train_subs = [i for i in range(6, 7)]
#             train_subs_conv = [i for i in range(5, 6)]
#             id_ch_selected = raw.chanel_selection(sel_chs)
#             div_id = raw.channel_division(spatial_local_dict)
#             spatial_region_split = raw.region_id_seg(div_id, id_ch_selected)
#             train_X, train_y, train_domain_y = raw.load_data_batchs(path, 1, train_subs, clf_class,
#                                                                     id_ch_selected, 0.1)
#             train_X_conv, train_y, train_domain_y = raw.load_data_batchs(path, 1, train_subs_conv, clf_class,
#                                                                     id_ch_selected, 0.1)
#             model_path1 = '/home/alk/L2G-MI/checkpoints/OpenBMI/L2GNet/test/test.pth'
#             model_path2 = '/home/alk/L2G-MI/checkpoints/OpenBMI/EEGNet/val/EEGNet.pth'
#             # model_path3 = '/home/alk/L2G-MI/checkpoints/OpenBMI/DeepConvNet/val/convNet.pth'
#             model_path3 = '/home/alk/L2G-MI/checkpoints/OpenBMI/EEGNet/val/EEGNet.pth'
#             model_path4 = '/home/alk/L2G-MI/checkpoints/OpenBMI/ShallowNet/val/ShallowNet.pth'
#         elif dataSet == 'BCIIV2A':
#             train_subs = [i for i in range(9, 10)]
#             id_ch_selected = raw_bci2a.chanel_selection(sel_chs)
#             div_id = raw_bci2a.channel_division(spatial_local_dict)
#             spatial_region_split = raw_bci2a.region_id_seg(div_id, id_ch_selected)
#             train_X, train_y, train_domain_y = raw_bci2a.load_data_batchs(path, 1, train_subs, clf_class,
#                                                                 id_ch_selected, 100)
#             model_path1 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/L2GNet/test/save.pth'
#             model_path2 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/EEGNet/test/save.pth'
#             model_path3 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/DeepConvNet/save/save.pth'
#             model_path4 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/ShallowNet/save/save.pth'
#         # 数据标准化
#         X_train_mean = train_X.mean(0)
#         X_train_var = np.sqrt(train_X.var(0))
#         train_X -= X_train_mean
#         train_X /= X_train_var
#         X_train_conv_mean = train_X_conv.mean(0)
#         X_train_conv_var = np.sqrt(train_X_conv.var(0))
#         train_X_conv -= X_train_conv_mean
#         train_X_conv /= X_train_conv_var
#
#         input_data = torch.from_numpy(train_X).to(device, dtype=torch.float32)
#         input_data_conv =  torch.from_numpy(train_X_conv).to(device, dtype=torch.float32)
#         # load model
#         if dataSet == 'OpenBMI':
#             M_l2g, optimizer, lr_scheduler, criterion, device, criterion_domain = \
#                     L2GNet_prepare_training(spatial_region_split, temporal_div_dict, d_model_dict, head_dict, d_ff, n_layers,
#                                             dropout, lr,
#                                             clf_class, domain_class, num_ch)
#             M_l2g.load_state_dict(torch.load(model_path1))
#             M_l2g.eval()
#             # L2G
#             output_L2G = M_l2g.L2G(input_data)
#             output_L2G = output_L2G.detach().cpu().numpy()
#
#             # EEGNet
#             M_EEG, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(num_ch, lr, dropout, clf_class)
#             M_EEG.load_state_dict(torch.load(model_path2))
#             M_EEG.eval()
#             eeg_1 = input_data.unsqueeze(1).permute(0,1,3,2)
#             eeg_2 = M_EEG.conv1(eeg_1)
#             eeg_3 =M_EEG.batchnorm1(eeg_2)
#             eeg_4 = eeg_3.permute(0,3,1,2)
#             eeg_5 = M_EEG.padding1(eeg_4)
#             eeg_6 = M_EEG.conv2(eeg_5)
#             eeg_7 = M_EEG.pooling2(eeg_6)
#             eeg_8 = M_EEG.padding2(eeg_7)
#             eeg_9 = M_EEG.conv3(eeg_8)
#             eeg_10 = M_EEG.batchnorm3(eeg_9)
#             output_EEG = M_EEG.pooling3(eeg_10)
#             output_EEG = output_EEG.reshape(200,-1)
#             output_EEG = output_EEG.detach().cpu().numpy()
#
#             # ConvNet
#             # M_conv, optimizer, lr_scheduler, criterion, device, criterion_domain = DeepConvNet_prepare_training(num_ch, lr, dropout, clf_class)
#             # M_conv.load_state_dict(torch.load(model_path3))
#             # M_conv.eval()
#             # input_data_re = input_data.unsqueeze(1)
#             # layer_index = 5  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
#             # intermediate_model = nn.Sequential(*list(M_conv.children())[:layer_index + 1])  # 选择要提取的层及之前的层
#             # output_Conv = intermediate_model(input_data_re)
#             # output_Conv = torch.mean(output_Conv, dim=1)
#             # output_Conv = output_Conv.reshape(output_Conv.shape[0],-1)
#             # output_Conv = output_Conv.detach().cpu().numpy()
#             M_conv, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(num_ch, lr,
#                                                                                                           dropout,
#                                                                                                           clf_class)
#             M_conv.load_state_dict(torch.load(model_path2))
#             M_conv.eval()
#             eeg_1 = input_data_conv.unsqueeze(1).permute(0, 1, 3, 2)
#             eeg_2 = M_conv.conv1(eeg_1)
#             eeg_3 = M_conv.batchnorm1(eeg_2)
#             eeg_4 = eeg_3.permute(0, 3, 1, 2)
#             eeg_5 = M_conv.padding1(eeg_4)
#             eeg_6 = M_conv.conv2(eeg_5)
#             eeg_7 = M_conv.pooling2(eeg_6)
#             eeg_8 = M_conv.padding2(eeg_7)
#             eeg_9 = M_conv.conv3(eeg_8)
#             eeg_10 = M_conv.batchnorm3(eeg_9)
#             output_Conv = M_conv.pooling3(eeg_10)
#             output_Conv = output_Conv.reshape(200, -1)
#             output_Conv = output_Conv.detach().cpu().numpy()
#
#
#             # ShallowNet
#             M_shallow, optimizer, lr_scheduler, criterion, device, criterion_domain = ShallowNet_prepare_training(num_ch, lr, dropout, clf_class)
#             M_shallow.load_state_dict(torch.load(model_path4))
#             M_shallow.eval()
#             input_data_re = input_data.unsqueeze(1)
#             layer_index = 4  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
#             intermediate_model = nn.Sequential(*list(M_shallow.children())[:layer_index + 1])  # 选择要提取的层及之前的层
#             output_shallow = intermediate_model(input_data_re)
#
#             output_shallow = output_shallow.detach().cpu().numpy()
#             output_shallow = output_shallow.reshape(output_shallow.shape[0],-1)
#
#             tsne2d = TSNE(n_components=2, init='pca', random_state=0)
#             X_raw = tsne2d.fit_transform(train_X.reshape(train_X.shape[0], -1))
#             X_l2g = tsne2d.fit_transform(output_L2G)
#             X_eeg = tsne2d.fit_transform(output_EEG)
#             X_conv = tsne2d.fit_transform(output_Conv)
#             X_shallow = tsne2d.fit_transform(output_shallow)
#
#         elif dataSet == 'BCIIV2A':
#             M_l2g, optimizer, lr_scheduler, criterion, device, criterion_domain = \
#                 L2GNet_prepare_training(spatial_region_split, temporal_div_dict, d_model_dict, head_dict, d_ff,
#                                         n_layers,
#                                         dropout, lr,
#                                         clf_class, domain_class, num_ch)
#             M_l2g.load_state_dict(torch.load(model_path1))
#             M_l2g.eval()
#             # L2G
#             output_L2G = M_l2g.L2G(input_data)
#             output_L2G = output_L2G.detach().cpu().numpy()
#
#             # EEGNet
#             M_EEG, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(num_ch, lr,
#                                                                                                           dropout,
#                                                                                                           clf_class)
#             M_EEG.load_state_dict(torch.load(model_path2))
#             M_EEG.eval()
#             eeg_1 = input_data.unsqueeze(1).permute(0, 1, 3, 2)
#             eeg_2 = M_EEG.conv1(eeg_1)
#             eeg_3 = M_EEG.batchnorm1(eeg_2)
#             eeg_4 = eeg_3.permute(0, 3, 1, 2)
#             eeg_5 = M_EEG.padding1(eeg_4)
#             eeg_6 = M_EEG.conv2(eeg_5)
#             eeg_7 = M_EEG.pooling2(eeg_6)
#             eeg_8 = M_EEG.padding2(eeg_7)
#             eeg_9 = M_EEG.conv3(eeg_8)
#             eeg_10 = M_EEG.batchnorm3(eeg_9)
#             output_EEG = M_EEG.pooling3(eeg_10)
#             output_EEG = output_EEG.reshape(output_EEG.shape[0], -1)
#             output_EEG = output_EEG.detach().cpu().numpy()
#
#             # ConvNet
#             M_conv, optimizer, lr_scheduler, criterion, device, criterion_domain = DeepConvNet_prepare_training(num_ch,
#                                                                                                                 lr,
#                                                                                                                 dropout,
#                                                                                                                 clf_class)
#             M_conv.load_state_dict(torch.load(model_path3))
#             M_conv.eval()
#             input_data_re = input_data.unsqueeze(1)
#             layer_index = 5  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
#             intermediate_model = nn.Sequential(*list(M_conv.children())[:layer_index + 1])  # 选择要提取的层及之前的层
#             output_Conv = intermediate_model(input_data_re)
#             output_Conv = torch.mean(output_Conv, dim=1)
#             output_Conv = output_Conv.reshape(output_Conv.shape[0], -1)
#             output_Conv = output_Conv.detach().cpu().numpy()
#
#             # ShallowNet
#             M_shallow, optimizer, lr_scheduler, criterion, device, criterion_domain = ShallowNet_prepare_training(
#                 num_ch, lr, dropout, clf_class)
#             M_shallow.load_state_dict(torch.load(model_path4))
#             M_shallow.eval()
#             input_data_re = input_data.unsqueeze(1)
#             layer_index = 4  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
#             intermediate_model = nn.Sequential(*list(M_shallow.children())[:layer_index + 1])  # 选择要提取的层及之前的层
#             output_shallow = intermediate_model(input_data_re)
#
#             output_shallow = output_shallow.detach().cpu().numpy()
#             output_shallow = output_shallow.reshape(output_shallow.shape[0], -1)
#
#             tsne2d = TSNE(n_components=2, init='pca', random_state=0)
#             X_raw = tsne2d.fit_transform(train_X.reshape(train_X.shape[0],-1))
#             X_l2g = tsne2d.fit_transform(output_L2G)
#             X_eeg = tsne2d.fit_transform(output_EEG)
#             X_conv = tsne2d.fit_transform(output_Conv)
#             X_shallow = tsne2d.fit_transform(output_shallow)
#         data_dic[dataSet] = [X_raw, X_eeg, X_conv, X_shallow, X_l2g]
#         y_dic[dataSet] = train_y
#
#     # 创建画布和子图
#     fig, axs = plt.subplots(5, 2, figsize=(8, 15))
#     # 在每个子图中绘制不同的数据
#     for i in range(5):
#         for j in range(2):
#             ax = axs[i, j]
#             if i == 0:
#                 if j == 0:
#                     ax.set_title('BCIIV 2A', fontweight='bold')
#                     ax.set_ylabel('Raw Data', fontweight='bold')
#                     plot_embedding_2d(data_dic[dataSetlist[0]][0][:, 0:2], y_dic[dataSetlist[0]], ax)
#                 else:
#                     ax.set_title('OpenBMI', fontweight='bold')
#                     plot_embedding_2d(data_dic[dataSetlist[1]][0][:, 0:2], y_dic[dataSetlist[1]], ax)
#
#                 # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
#             elif i == 1:
#                 # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
#                 if j == 0:
#                     plot_embedding_2d(data_dic[dataSetlist[0]][1][:, 0:2], y_dic[dataSetlist[0]], ax)
#                     ax.set_ylabel('EEGNet', fontweight='bold')
#                 else:
#                     plot_embedding_2d(data_dic[dataSetlist[1]][1][:, 0:2],  y_dic[dataSetlist[1]], ax)
#             elif i == 2:
#                 # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
#                 if j == 0:
#                     ax.set_ylabel('DeepConvNet', fontweight='bold')
#                     plot_embedding_2d(data_dic[dataSetlist[0]][2][:, 0:2], y_dic[dataSetlist[0]], ax)
#                 else:
#                     plot_embedding_2d(data_dic[dataSetlist[1]][2][:, 0:2],  y_dic[dataSetlist[1]], ax)
#             elif i == 3:
#                 # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
#                 if j == 0:
#                     plot_embedding_2d(data_dic[dataSetlist[0]][3][:, 0:2], y_dic[dataSetlist[0]], ax)
#                     ax.set_ylabel('MMCNN', fontweight='bold')
#                 else:
#                     plot_embedding_2d(data_dic[dataSetlist[1]][3][:, 0:2],  y_dic[dataSetlist[1]], ax)
#             elif i == 4:
#                 # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
#                 if j == 0:
#                     ax.set_ylabel('STL2G-DG', fontweight='bold')
#                     plot_embedding_2d(data_dic[dataSetlist[0]][4][:, 0:2], y_dic[dataSetlist[0]], ax)
#                 else:
#                     plot_embedding_2d(data_dic[dataSetlist[1]][4][:, 0:2],  y_dic[dataSetlist[1]], ax)
#     # 创建共用的图例
#     # handles, labels = axs[0,0].get_legend_handles_labels()
#     # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
#     print("Computing t-SNE embedding")
#     # 自动调整子图布局
#     plt.tight_layout()
#     # 保存每个子图为单独的图像文件
#     plt.savefig(f'./experiments/tSNE.png', dpi=300)
#     plt.show()
#
#
#
#
#
# if __name__ == '__main__':
#     # sys.path.append(r"\home\alk\L2G-MI\stl2g")
#     model_type = 'L2GNet'
#     datalist = ['OpenBMI', 'OpenBMI']
#     train_iris(datalist, model_type)
