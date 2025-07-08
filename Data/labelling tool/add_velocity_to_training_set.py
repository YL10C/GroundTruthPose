import pandas as pd
import numpy as np

# === 加载原始训练集 ===
df = pd.read_csv("training_set.csv")

# === 提取结构信息 ===
joint_coords_cols = [col for col in df.columns if '_x' in col or '_y' in col or '_z' in col]
label_col = 'label'
frame_col = 'frame_idx'

# === 准备 velocity 存储 ===
velocity_rows = []

# === 按段处理，每段 frame_idx 应从 0 连续递增 ===
# 方法：按连续 frame_idx 分组
df['_segment_id'] = (df[frame_col] != df[frame_col].shift(1) + 1).cumsum()

for _, group in df.groupby('_segment_id'):
    positions = group[joint_coords_cols].values  # shape: (num_frames, 54)
    labels = group[label_col].values
    frame_ids = group[frame_col].values

    # === 镜像填充五帧 ===
    padded = np.pad(positions, ((5, 5), (0, 0)), mode='reflect')  # shape: (n+10, 54)

    # === 中心差分计算 velocity ===
    velocities = (padded[10:] - padded[:-10]) / 10  # shape: (n, 54)

    # === 合并回 frame_idx + velocity + position + label
    for i in range(len(group)):
        row = [frame_ids[i]] + list(velocities[i]) + list(positions[i]) + [labels[i]]
        velocity_rows.append(row)

# === 构建列名 ===
columns = ['frame_idx']
for i in range(18):
    columns += [f'{i}_vx', f'{i}_vy', f'{i}_vz']
for i in range(18):
    columns += [f'{i}_x', f'{i}_y', f'{i}_z']
columns += ['label']

# === 保存新 CSV ===
df_out = pd.DataFrame(velocity_rows, columns=columns)
df_out.to_csv("training_set_with_velocity.csv", index=False)
print(f"✅ Velocity 计算完成，共 {len(df_out)} 帧，保存为 training_set_with_velocity.csv")
