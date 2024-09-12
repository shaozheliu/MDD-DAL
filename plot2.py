# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
from matplotlib.transforms import Bbox


plt.rcParams['legend.fontsize'] = 15


#------------------第二个图片---------------
# 画第1个图：折线图
#  定义字体Palatino Linotype
font = {'family': 'serif',
        'serif': 'Times New Roman',
        'weight': 'normal',
        'size': 15}
plt.rc('font', **font)
labels=[ 'Accuracy', 'Recall', 'Precision']
x = np.arange(len(labels))  # the label locations
width = 0.15

df = pd.DataFrame({'v1':[0.91,0.922,0.907], 'v2': [0.906,0.913,0.897], 'v3': [ 0.913, 0.923,0.911], 'MDD': [ 0.923, 0.941,0.914]}, index=labels)
x = np.arange(len(labels))
# plt.bar(x - (3/2)*width , df['v1'], width=width, label="$Variant1$",fc='lightsteelblue')
# plt.bar(x - (1/2)*width , df['v2'], width=width, label="$Variant2$",fc='cornflowerblue')
# plt.bar(x + (1/2)*width, df['v3'], width=width, label="$Variant3$",fc='royalblue')
# plt.bar(x + (3/2)*width, df['MDD'], width=width, label="$MDDNet$",fc='blue')
plt.bar(x - (3/2)*width , df['v1'], width=width, label="$Variant1$",fc='paleturquoise')
plt.bar(x - (1/2)*width , df['v2'], width=width, label="$Variant2$",fc='aqua')
plt.bar(x + (1/2)*width, df['v3'], width=width, label="$Variant3$",fc='c')
plt.bar(x + (3/2)*width, df['MDD'], width=width, label="$MDDNet$",fc='teal')
#固定X轴、Y轴的范围
plt.ylim(0.85, 1)
# ax.set_xlim(xmin = -5, xmax = 5)
plt.xticks(x,labels,fontsize=15)
plt.legend( loc='upper left', fontsize = 15,columnspacing=1)  # 显示图例
plt.tight_layout()
# plt.show()
savefig('./ablation_fig.png',dpi=1000,bbox_inches = 'tight')
# plt.show()




