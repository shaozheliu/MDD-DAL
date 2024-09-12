# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
from matplotlib.transforms import Bbox


# names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
# x = range(len(names))
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
plt.rcParams['legend.fontsize'] = 15
# plt.rcParams['legend.handlelength'] = 16
#
# # accuracy
# # ST_DG_ACC = [0.75, 0.7175, 0.67, 0.8547619047619047, 0.7071428571428572, 0.7975, 0.7775, 0.7386363636363636, 0.745]
# # ST_ACC = [0.7025, 0.71, 0.66, 0.8738095238095238, 0.6738095238095239, 0.765, 0.825, 0.75, 0.6825]  # 0.7380687830687831
# # S_DG_ACC =  [0.71, 0.69, 0.6475, 0.8523809523809524, 0.7023809523809523, 0.765, 0.7875, 0.7636363636363637, 0.68]
# # T_DG_ACC = [0.7325, 0.7, 0.655, 0.8214285714285714, 0.7047619047619048, 0.7825, 0.79, 0.7522727272727273, 0.68]
#
# ST_DG_ACC = [0.75, 0.7175, 0.67, 0.8738095238095238, 0.7071428571428572, 0.7975, 0.825, 0.7636363636363637, 0.745]
# ST_ACC = [0.7025, 0.71, 0.66, 0.8547619047619047, 0.6738095238095239, 0.765, 0.7775, 0.75, 0.6825]  # 0.7380687830687831
# S_DG_ACC =  [0.71, 0.69, 0.6475, 0.8523809523809524, 0.7023809523809523, 0.765, 0.7875, 0.7386363636363636, 0.68]
# T_DG_ACC = [0.7325, 0.7, 0.655, 0.8214285714285714, 0.7047619047619048, 0.7825, 0.79, 0.7522727272727273, 0.68]
#
# # AUC
# # ST_DG_AUC = [0.839225, 0.7700750000000001, 0.7330249999999999, 0.9629251700680272, 0.7936281179138323, 0.8749, 0.8836750000000001, 0.8517561983471075, 0.797275]
# # ST_AUC = [0.7638750000000001, 0.7879499999999999, 0.69975, 0.9243083900226757, 0.7507936507936508, 0.8512000000000001, 0.89035, 0.822086776859504, 0.7566250000000001]  # 0.7380687830687831
# # S_DG_AUC = [0.80095, 0.773175, 0.685525, 0.9172562358276645, 0.7739909297052154, 0.843775, 0.89105, 0.8433677685950414, 0.7434999999999999]
# # T_DG_AUC = [0.80555, 0.777025, 0.6927249999999999, 0.9455782312925171, 0.7731746031746031, 0.8569249999999999, 0.9001250000000001, 0.851404958677686, 0.756675]
#
# ST_DG_AUC = [0.839225, 0.7879499999999999, 0.7330249999999999, 0.9629251700680272, 0.7936281179138323, 0.8749, 0.9001250000000001, 0.8517561983471075, 0.797275]
# ST_AUC = [0.7638750000000001, 0.7700750000000001, 0.69975, 0.9243083900226757, 0.7507936507936508, 0.8512000000000001, 0.89035, 0.822086776859504, 0.7566250000000001]  # 0.7380687830687831
# S_DG_AUC = [0.80095, 0.773175, 0.685525, 0.9172562358276645, 0.7739909297052154, 0.843775, 0.89105, 0.8433677685950414, 0.7434999999999999]
# T_DG_AUC = [0.80555, 0.777025, 0.6927249999999999, 0.9455782312925171, 0.7731746031746031, 0.8569249999999999, 0.8836750000000001, 0.851404958677686, 0.756675]
#
#
# plt.subplot(211)
# plt.plot(x, ST_ACC, color='rosybrown', linewidth=1.6, marker='o', markersize=4.0, linestyle='-', label='ST')
# plt.plot(x, S_DG_ACC, color='lightcoral', linewidth=1.6, marker='o', markersize=4.0, linestyle='-', label='S_DG')
# plt.plot(x, T_DG_ACC, color='firebrick', linewidth=1.6, marker='o', markersize=4.0,linestyle='-', label='T_DG')
# plt.plot(x, ST_DG_ACC, color='red', linewidth=1.6, marker='*',  markersize=4.0, linestyle='-', label='ST_DG')
# plt.legend( loc='upper left',ncol=2, columnspacing=1)  # 显示图例
# plt.xticks(x, names)
# plt.ylabel("Classification accuracy")  # X轴标签
# # plt.xlabel("Subject index")  # Y轴标签
# plt.subplot(212)
# plt.plot(x, ST_AUC, color='lightsteelblue', linewidth=1.6, marker='o', markersize=4.0, linestyle='-', label='ST')
# plt.plot(x, S_DG_AUC, color='cornflowerblue', linewidth=1.6, marker='o', markersize=4.0, linestyle='-', label='S_DG')
# plt.plot(x, T_DG_AUC, color='c', linewidth=1.6, marker='o', markersize=4.0, linestyle='-', label='T_DG')
# plt.plot(x, ST_DG_AUC, color='blue', linewidth=1.6, marker='*', markersize=4.0, linestyle='-', label='ST_DG')
# plt.legend( loc='upper left',ncol=2, columnspacing=1)  # 显示图例
# plt.xticks(x, names)
# plt.ylabel("Classification AUC")  # X轴标签
# plt.xlabel("Subject index")  # Y轴标签
# plt.tight_layout()

# savefig('./ablation_fig.png',dpi=1000,bbox_inches = 'tight')
# plt.show()



#------------------第二个图片
# 画第1个图：折线图
#  定义字体Palatino Linotype
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15}
plt.rc('font', **font)
labels=[ 'Accuracy', 'Recall', 'Precision']
x = np.arange(len(labels))  # the label locations

fig, ax = plt.subplots(figsize=(15,5))
total_width, n = 0.6, 3
# 每种类型的柱状图宽度
width = 0.15

plt.subplot(131)#两行两列第一个图

df = pd.DataFrame({'2':[0.9231,0.941,0.914], '3': [0.919,0.931,0.90], '4': [ 0.914, 0.900,0.892]}, index=labels)
x = np.arange(len(labels))
print (x)
plt.bar(x - width , df['2'], width=width, label="$L_t$ = 4",fc='paleturquoise')
plt.bar(x , df['3'], width=width, label="$L_t$ = 6",fc='c')
plt.bar(x + width, df['4'], width=width, label="$L_t$ = 10",fc='teal')
#固定X轴、Y轴的范围
plt.ylim(0.85, 1)
# ax.set_xlim(xmin = -5, xmax = 5)
plt.xticks(x,labels,fontsize=15)
plt.legend( loc='upper left', fontsize = 15,columnspacing=1)  # 显示图例

#
# 画第2个图
plt.subplot(132)#两行两列第二个图

df = pd.DataFrame({'2':[0.9231,0.941,0.914], '3': [0.912,0.931,0.90], '4': [ 0.897, 0.913, 0.897]}, index=labels)
x = np.arange(len(labels))

plt.bar(x-width, df['2'], width=width, label="$L_s$ = 2",fc='paleturquoise')
plt.bar(x , df['3'], width=width, label="$L_s$ = 4",fc='c')
plt.bar(x + width, df['4'], width=width, label="$L_s$ = 6",fc='teal')
plt.xticks(x,labels,fontsize=15)
plt.ylim(0.85, 1)
plt.legend( loc='upper left', columnspacing=1)  # 显示图例
#
# # 画第3个图：条形图
plt.subplot(133)#两行两列第二个图

df = pd.DataFrame({'2':[0.9231,0.9195,0.90], '3': [0.9191,0.9083,0.891], '4': [0.9153,0.900,0.887]}, index=labels)
x = np.arange(len(labels))

plt.bar(x-width, df['2'], width=width, label="$h$ = 2", fc='paleturquoise')
plt.bar(x , df['3'], width=width, label="$h$ = 6", fc='c')
plt.bar(x + width, df['4'], width=width, label="$h$ = 8" ,fc='teal')
plt.xticks(x,labels,fontsize=15)
plt.ylim(0.85, 1)
plt.legend( loc='upper left', columnspacing=1)  # 显示图例
#
# plt.subplot(224)#两行两列第二个图
#
# df = pd.DataFrame({'2':[ 0.7508,0.8348,0.80], '3': [ 0.7458,0.8227,0.80], '4': [ 0.7432, 0.8207,0.80]}, index=labels)
# x = np.arange(len(labels))
#
# plt.bar(x-width, df['2'], width=width, label="$h$ = 2",fc='lightsteelblue')
# plt.bar(x , df['3'], width=width, label="$h$ = 5",fc='cornflowerblue')
# plt.bar(x + width, df['4'], width=width, label="$h$ = 10",fc='blue')
# plt.xticks(x,labels,fontsize=15)
# plt.ylim(0.5, 0.9)
# plt.legend( loc='upper left', columnspacing=1)  # 显示图例
#
plt.tight_layout()
# plt.show()
savefig('./parameterexp_fig.png',dpi=1000,bbox_inches = 'tight')




# x = np.arange(len(labels))  # the label locations
# error_params1=dict(elinewidth=3,ecolor='crimson',capsize=4)#设置误差标记参数
# fig, ax = plt.subplots(figsize=(10,4))
# total_width, n = 0.6, 3
# 每种类型的柱状图宽度
# width = 0.18
# plt.subplot(222)#两行两列第一个图
# df = pd.DataFrame({'EEGNet':[0.5771,0.8098], 'DeepConvNet': [0.5766,0.8037], 'MMCNN': [ 0.5421, 0.7827], 'ST-DG': [ 0.5421, 0.7827]}, index=labels)
# stds_EEGNet = [0.0179, 0.0142]
# stds_DeepConvNet = [0.1, 0.1]
# stds_MMCNN = [0.1, 0.1]
# stds_ST_DG = [0.0129, 0.0138]
# x = np.arange(len(labels))
# print(x)
# plt.bar(x - (3/2)*width, df['EEGNet'],yerr=stds_EEGNet, width=width, error_kw = error_params1,label="EEGNet",fc='blue')
# plt.bar(x - (1/2)*width , df['DeepConvNet'],yerr=stds_DeepConvNet, width=width, error_kw = error_params1, label="DeepConvNet",fc='lightsteelblue')
# plt.bar(x + (1/2)*width, df['MMCNN'], yerr=stds_MMCNN, width=width, error_kw = error_params1,label="MMCNN",fc='cornflowerblue')
# plt.bar(x + (3/2)*width, df['ST-DG'], yerr=stds_ST_DG, width=width, error_kw = error_params1,label="ST-DG",fc='blue')
# plt.title('BCI-2a')
#
# plt.xticks(x,labels,fontsize=15)
# plt.legend(loc='upper left')
#
# # 画第2个图
# plt.subplot(224)#两行两列第二个图
#
# df = pd.DataFrame({'EEGNet':[0.5771,0.8098], 'DeepConvNet': [0.5766,0.8037], 'MMCNN': [ 0.5421, 0.7827], 'ST-DG': [ 0.5421, 0.7827]}, index=labels)
# x = np.arange(len(labels))
#
# plt.bar(x - (3/2)*width, df['EEGNet'],yerr=stds_EEGNet, width=width, error_kw = error_params1,label="EEGNet",fc='blue')
# plt.bar(x - (1/2)*width , df['DeepConvNet'],yerr=stds_DeepConvNet, width=width, error_kw = error_params1,label="DeepConvNet",fc='lightsteelblue')
# plt.bar(x + (1/2)*width, df['MMCNN'], yerr=stds_MMCNN, width=width, error_kw = error_params1,label="MMCNN",fc='cornflowerblue')
# plt.bar(x + (3/2)*width, df['ST-DG'], yerr=stds_ST_DG, width=width, error_kw = error_params1,label="ST-DG",fc='blue')
# plt.title('BCI-2b')
# plt.xticks(x,labels,fontsize=15)
# # plt.legend()
#
# plt.show()


