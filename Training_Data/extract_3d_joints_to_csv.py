"""
该脚本用于根据标签文件（merged_labels.csv）批量提取3D关节点数据，并将其整理为CSV格式。

主要流程：
1. 读取标签文件，按文件和时间排序。
2. 遍历每一条标签，根据对应的pkl文件和时间段，提取人体第5个人（假定为目标人物）的18个3D关节点坐标。
3. 每一帧数据展开为54维（18个关节点 × 3坐标），并附加frame_idx和label。
4. 所有数据合并后保存为velocity_selected.csv，便于后续特征工程或机器学习使用。

注意：
- 需要保证标签文件和pkl数据文件路径正确。
- 仅提取第5个人的关节点数据，且数据完整时才会保存。
"""
import os
import pickle
import pandas as pd

# === 加载标签文件 ===
label_csv_path = "Training_Data\merged_labels.csv"
df = pd.read_csv(label_csv_path)

# 排序，确保按文件 & 时间顺序处理
df = df.sort_values(by=["filepath", "hour", "minute", "second"]).reset_index(drop=True)

# === 初始化 ===
all_rows = []
pkl_cache = {}

# 时间转换辅助函数
def time_to_seconds(h, m, s):
    return h * 3600 + m * 60 + s

# === 主循环 ===
prev_path = None
prev_time = None
frame_idx = 0

for idx, row in df.iterrows():
    filepath = row['filepath']
    hour = int(row['hour'])
    minute = int(row['minute'])
    second = int(row['second'])
    label = int(row['label'])
    seg = second // 10

    # 新增：判断person
    if 'p1' in filepath:
        person = 'p1'
    elif 'p2' in filepath:
        person = 'p2'
    else:
        person = 'unknown'

    current_time_sec = time_to_seconds(hour, minute, second)

    # 判断是否属于同一连续段
    is_new_segment = False
    if filepath != prev_path:
        is_new_segment = True
    elif prev_time is not None and current_time_sec != prev_time + 10:
        is_new_segment = True

    if is_new_segment:
        frame_idx = 0  # 重置局部 frame index

    prev_time = current_time_sec
    prev_path = filepath

    # 加载 pkl 文件（缓存）
    if filepath not in pkl_cache:
        if not os.path.exists(filepath):
            print(f"❌ 文件不存在：{filepath}")
            continue
        try:
            with open(filepath, 'rb') as f:
                pkl_cache[filepath] = pickle.load(f)
        except Exception as e:
            print(f"❌ 读取失败：{filepath}，错误：{e}")
            continue
    data = pkl_cache[filepath]

    # 提取该时间段的数据
    try:
        data_seg = data[hour][minute][seg]
    except Exception:
        print(f"⚠ 无数据段：{filepath} {hour}:{minute}:{second}")
        continue

    for frame in data_seg:
        try:
            poses = frame.get('poses', [])
            if len(poses) < 5 or not poses[4]:
                continue
            joints_3d = poses[4][0]
            if not isinstance(joints_3d, list) or len(joints_3d) != 18:
                continue

            flat_coords = []
            valid = True
            for joint in joints_3d:
                if not isinstance(joint, list) or len(joint) != 3:
                    valid = False
                    break
                flat_coords.extend(joint)
            if not valid:
                continue

            row_out = [person, frame_idx] + flat_coords + [label]
            all_rows.append(row_out)
            frame_idx += 1
        except Exception as e:
            print(f"⚠ 提取帧失败：{e}")
            continue

# === 构建列名 ===
columns = ['person', 'frame_idx']
for i in range(18):
    columns += [f'{i}_x', f'{i}_y', f'{i}_z']
columns += ['label']

# === 保存为 CSV ===
df_out = pd.DataFrame(all_rows, columns=columns)
output_path = "Training_Data\source_data.csv"
df_out.to_csv(output_path, index=False)
print(f"✅ 提取完成，共 {len(df_out)} 帧，保存为 {output_path}")
