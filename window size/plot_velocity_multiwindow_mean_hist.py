import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

file_name = 'velocity_multiwindow_mean[7, 9, 11, 13, 15]'

# 读取数据
df = pd.read_csv('window size/' + file_name + '.csv')

# 删除mean_velocity_magnitude列中的缺失值
df = df.dropna(subset=['velocity_magnitude'])

# 提取mean_velocity_magnitude列并四舍五入为整数
velocity_magnitude = np.round(df['velocity_magnitude']).astype(int)

# 计算平均值
mean_velocity = velocity_magnitude.mean()

# 创建一个图形，设置大小
plt.figure(figsize=(10, 7))

# 绘制直方图，设置范围从0到50，使用50个bins
plt.hist(velocity_magnitude, bins=50, range=(0, 50), edgecolor='black')

# 添加标题和标签
plt.title(f'Velocity Distribution (Multiwindow Mean)\nMean: {mean_velocity:.2f}')
plt.xlabel('Mean Velocity Magnitude (Integer)')
plt.ylabel('Frequency')

# 添加网格线
plt.grid(True, alpha=0.3)

# 保存图片
plt.savefig('window size/' + file_name + '.png', dpi=300, bbox_inches='tight')
plt.close()

print("Histogram has been saved as 'window size/" + file_name + ".png'") 