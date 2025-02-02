import matplotlib.pyplot as plt
import numpy as np

# 数据准备
data_points = [
    {'label': 'sail.tif – Chrominance subsampling', 'bpp': 12, 'psnr': 45.97},
    {'label': 'sail.tif – RGB-subsampling (CIF)', 'bpp': 6, 'psnr': 29.72},
    {'label': 'sail.tif – RGB-subsampling (QCIF)', 'bpp': 2, 'psnr': 23.9},
    {'label': 'sail.tif – starting algorithm', 'bpp': 8, 'psnr': 21.5},
    {'label': 'smandril.tif – starting algorithm', 'bpp': 8, 'psnr': 16},
    {'label': 'lena.tif – starting algorithm', 'bpp': 8, 'psnr': 15.5},
    {'label': 'lena.tif-VQ (Ours)', 'bpp': 1.37, 'psnr': 34.01}  # 突出显示我们的结果
]

# 创建画布
plt.figure(figsize=(12, 8), dpi=100)
ax = plt.gca()

# 绘制所有数据点
for point in data_points:
    # 特殊标记我们的结果
    if "Ours" in point['label']:
        marker = '*'
        color = 'red'
        size = 200
        edgecolor = 'darkred'
    else:
        marker = 'o'
        color = 'blue'
        size = 100
        edgecolor = 'navy'

    # 绘制散点
    sc = plt.scatter(
        point['bpp'],
        point['psnr'],
        s=size,
        c=color,
        marker=marker,
        edgecolor=edgecolor,
        linewidths=1.5,
        zorder=3  # 确保在前景
    )


# 设置坐标轴
plt.xlim(0, 14)
plt.ylim(10, 50)
plt.xlabel('Bitrate [bits per pixel]', fontsize=12, labelpad=10)
plt.ylabel('PSNR [dB]', fontsize=12, labelpad=10)
plt.title('Rate-Distortion Comparison', fontsize=14, pad=15)

# 添加网格线
plt.grid(True,
         linestyle='--',
         alpha=0.7,
         which='both',
         zorder=1)

# 创建自定义图例
legend_elements = [
    plt.Line2D([0], [0],
               marker='o',
               color='w',
               label='Baseline Methods',
               markerfacecolor='blue',
               markersize=10),
    plt.Line2D([0], [0],
               marker='*',
               color='w',
               label='Our VQ Method',
               markerfacecolor='red',
               markersize=15)
]

# 显示图例
ax.legend(handles=legend_elements,
          loc='lower right',
          fontsize=10,
          framealpha=0.9)

# 优化布局
plt.tight_layout()

# 显示图形
plt.show()