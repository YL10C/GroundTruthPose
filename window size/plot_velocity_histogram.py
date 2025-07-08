import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np

file_name = 'velocity_window_3'
# 获取所有velocity_window_*.csv文件
velocity_files = glob.glob(file_name + '.csv')

# 创建一个图形，设置大小
plt.figure(figsize=(15, 10))

# 为每个文件创建一个子图
for i, file in enumerate(velocity_files, 1):
    # 读取CSV文件
    df = pd.read_csv(file)
    
    # 删除velocity_magnitude列中的缺失值
    df = df.dropna(subset=['velocity_magnitude'])
    
    # 提取velocity_magnitude列并四舍五入为整数
    velocity_magnitude = np.round(df['velocity_magnitude']).astype(int)
    
    # 计算平均值
    mean_velocity = velocity_magnitude.mean()
    
    # 创建子图
    plt.subplot(2, 2, i)
    
    # 绘制直方图，设置范围从0到50，使用50个bins
    plt.hist(velocity_magnitude, bins=50, range=(0, 50), edgecolor='black')
    
    # 添加标题和标签
    window_size = file.split('_')[-1].split('.')[0]  # 提取窗口大小
    plt.title(f'Velocity Distribution for Window Size {window_size}\nMean: {mean_velocity:.2f}')
    plt.xlabel('Velocity Magnitude (Integer)')
    plt.ylabel('Frequency')
    
    # 添加网格线
    plt.grid(True, alpha=0.3)

# 调整子图之间的间距
plt.tight_layout()

# 保存图片
plt.savefig(file_name + '.png', dpi=300, bbox_inches='tight')
plt.close()

print("Histogram has been saved as 'velocity_histograms.png'") 