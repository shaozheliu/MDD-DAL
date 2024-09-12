import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置全局字体大小
plt.rcParams.update({'font.size': 16})  # 设置全局字体大小为14
rcParams['font.family'] = 'Times New Roman' # 使用字体中的无衬线体
# 数据
params = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
gara16_acc = [0.422, 0.429, 0.426, 0.429, 0.431, 0.431, 0.430,  0.435,  0.432]
gara32_acc = [0.427, 0.430, 0.426, 0.428, 0.427, 0.428, 0.429, 0.432, 0.431]

# 创建图形和轴
plt.figure(figsize=(8, 6))
plt.plot(params, gara16_acc, marker='o', markersize = 10,linewidth = 2.5,linestyle='--', color="#1E4A68", label='GaRA-16')
plt.plot(params, gara32_acc, marker='s', markersize = 10,linewidth = 2,linestyle='-', color="#904435", label='GaRA-32')

# 设置标题和标签
plt.title('PEFT results under different alpha')
plt.xlabel('alpha')
plt.ylabel('Mean Squared Error')
plt.xticks(params)  # 设置x轴的刻度
plt.ylim(0.42, 0.44)  # 设置y轴的范围

# 添加图例
plt.legend()

# 显示图表
# plt.grid()
# plt.tight_layout()
plt.savefig('./alpha-ratio.png', dpi=300, bbox_inches='tight')
plt.show()

