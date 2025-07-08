import pandas as pd
import numpy as np

# 你可以在这里自定义窗口大小
window_sizes = [7, 9, 11, 13]

# 读取所有窗口的velocity_magnitude列
velocity_cols = []
for w in window_sizes:
    df = pd.read_csv(f'window size/velocity_window_{w}.csv', usecols=['velocity_magnitude'])
    velocity_cols.append(df['velocity_magnitude'])

# 拼成一个DataFrame，每一列是一个窗口的velocity
velocities = pd.concat(velocity_cols, axis=1)
velocities.columns = [f'velocity_{w}' for w in window_sizes]

# 计算多窗口中位数（忽略NaN）
velocities['velocity_magnitude'] = velocities.median(axis=1, skipna=True)

# 如果需要frame_idx，可以从任意一个csv读取
frame_idx = pd.read_csv(f'window size/velocity_window_{window_sizes[0]}.csv', usecols=['frame_idx'])

# 合并结果
result = pd.concat([frame_idx, velocities['velocity_magnitude']], axis=1)

file_name = 'velocity_multiwindow_mean' + str(window_sizes)
# 保存结果
result.to_csv('window size/' + file_name + '.csv', index=False)
print('Saved to window size/' + file_name + '.csv') 