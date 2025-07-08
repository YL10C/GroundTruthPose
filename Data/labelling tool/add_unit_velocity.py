import pandas as pd
import numpy as np
import os

# 加载原始CSV文件
file_path = "training_set_with_velocity.csv"
df = pd.read_csv(file_path)

# 提取帧索引和标签
frame_idx_col = ['frame_idx']
label_col = ['label']

# 提取速度列（假设从第2列开始，到第2+54=56列为止）
velocity_cols = df.columns[1:55]
position_cols = df.columns[55:-1]  # 不包括最后一个 label 列

# 准备新的列名列表
unit_velocity_cols = []
unit_velocity_data = []

# 遍历每一帧数据
for index, row in df.iterrows():
    unit_row = []
    for i in range(18):  # 18 个点
        vx = row[f"{i}_vx"]
        vy = row[f"{i}_vy"]
        vz = row[f"{i}_vz"]
        norm = np.sqrt(vx**2 + vy**2 + vz**2)
        if norm != 0:
            unit_vx = vx / norm
            unit_vy = vy / norm
            unit_vz = vz / norm
        else:
            unit_vx = unit_vy = unit_vz = 0.0
        unit_row.extend([unit_vx, unit_vy, unit_vz])
    unit_velocity_data.append(unit_row)

# 构建unit velocity的列名
for i in range(18):
    unit_velocity_cols.extend([f"{i}_uvx", f"{i}_uvy", f"{i}_uvz"])

# 创建 DataFrame
unit_velocity_df = pd.DataFrame(unit_velocity_data, columns=unit_velocity_cols)

# 拼接原始数据和unit velocity
df_with_unit_velocity = pd.concat([df, unit_velocity_df], axis=1)

# 重新排列列的顺序
# 1. 帧序号
frame_cols = ['frame_idx']
# 2. 坐标列
position_cols = [col for col in df.columns if any(col.endswith(f'_{i}') for i in ['x', 'y', 'z'])]
# 3. 速度列
velocity_cols = [col for col in df.columns if any(col.endswith(f'_{i}') for i in ['vx', 'vy', 'vz'])]
# 4. 单位速度列
unit_velocity_cols = [col for col in unit_velocity_df.columns]
# 5. 标签列
label_cols = ['label']

# 按顺序组合所有列
ordered_cols = frame_cols + position_cols + velocity_cols + unit_velocity_cols + label_cols
df_with_unit_velocity = df_with_unit_velocity[ordered_cols]

# 保存结果
output_path = "training_set_with_velocity_and_unit.csv"
df_with_unit_velocity.to_csv(output_path, index=False)

