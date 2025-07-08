import pandas as pd
import numpy as np
from typing import List, Tuple

def find_continuous_segments(df: pd.DataFrame) -> List[Tuple[int, int]]:
    segments = []
    start_idx = 0
    for i in range(1, len(df)):
        if df['frame_idx'].iloc[i] != df['frame_idx'].iloc[i-1] + 1:
            segments.append((start_idx, i-1))
            start_idx = i
    segments.append((start_idx, len(df)-1))
    return segments

def mirror_pad_series(series: pd.Series, half_window: int) -> pd.Series:
    total_length = len(series) + 2 * half_window
    padded_series = pd.Series(index=range(total_length), dtype=float)
    padded_series.iloc[half_window:half_window + len(series)] = series.values
    for i in range(half_window):
        padded_series.iloc[half_window - i - 1] = padded_series.iloc[half_window + i + 1]
    for i in range(half_window):
        padded_series.iloc[half_window + len(series) + i] = padded_series.iloc[half_window + len(series) - i - 2]
    return padded_series

# 你可以在这里自定义窗口大小
window_sizes = [7, 9, 11, 13]

# 读取数据
source_df = pd.read_csv('Training_Data/source_data.csv')

joint_num = 18
coords = ['x', 'y', 'z']

# 先准备所有关节点的坐标列名
joint_cols = [f'{i}_{c}' for i in range(joint_num) for c in coords]

# 结果DataFrame初始化
result = source_df[['person', 'frame_idx'] + joint_cols].copy()

# 计算velocity, unit velocity, speed
velocity_cols = []
unit_velocity_cols = []
speed_cols = []

segments = find_continuous_segments(source_df)

for j in range(joint_num):
    vx_all, vy_all, vz_all = [], [], []
    for c in coords:
        col = f'{j}_{c}'
        # 新增：将0和-1都视为无效，替换为NaN
        coord_series = source_df[col].replace([0, -1], np.nan)
        # 针对每个窗口分别计算velocity
        v_windows = []
        for w in window_sizes:
            half_window = (w - 1) // 2
            v = pd.Series(index=source_df.index, dtype=float)
            for start_idx, end_idx in segments:
                segment_data = coord_series.iloc[start_idx:end_idx+1]
                padded_data = mirror_pad_series(segment_data, half_window)
                for i in range(len(segment_data)):
                    future_pos = padded_data.iloc[half_window + i + half_window]
                    past_pos = padded_data.iloc[half_window + i - half_window]
                    v.iloc[start_idx + i] = (future_pos - past_pos) / (w - 1)
            v_windows.append(v)
        v_median = pd.concat(v_windows, axis=1).median(axis=1, skipna=True)
        result[f'{j}_v{c}'] = v_median
        velocity_cols.append(f'{j}_v{c}')
        if c == 'x':
            vx_all = v_median
        elif c == 'y':
            vy_all = v_median
        else:
            vz_all = v_median
    speed = np.sqrt(vx_all**2 + vy_all**2 + vz_all**2)
    result[f'{j}_s'] = speed
    speed_cols.append(f'{j}_s')
    for c, v in zip(coords, [vx_all, vy_all, vz_all]):
        uv = v / (speed + 1e-8)
        result[f'{j}_uv{c}'] = uv
        unit_velocity_cols.append(f'{j}_uv{c}')

# 添加label
result['label'] = source_df['label']

# 按指定顺序排列列
final_cols = (
    ['person', 'frame_idx'] +
    joint_cols +
    velocity_cols +
    unit_velocity_cols +
    speed_cols +
    ['label']
)
result = result[final_cols]

# 保存
result.to_csv('Training_Data/joint_velocity_multiwindow.csv', index=False)
print('Saved to Training_Data/joint_velocity_multiwindow.csv') 