import pandas as pd
import numpy as np
from typing import List, Tuple

def calculate_main_node_position(df: pd.DataFrame, joint_indices: List[int]) -> pd.DataFrame:
    """
    计算主节点位置（多个关节点的平均位置）
    
    Args:
        df: 包含关节点坐标的DataFrame
        joint_indices: 需要计算平均位置的关节点索引列表
    
    Returns:
        包含主节点位置的DataFrame
    """
    result_df = pd.DataFrame()
    result_df['frame_idx'] = df['frame_idx']
    
    # 计算每个轴的平均位置
    for axis in ['x', 'y', 'z']:
        # 获取所有指定关节点的该轴坐标
        cols = [f'{idx}_{axis}' for idx in joint_indices]
        # 过滤掉值为-1的关节点
        valid_coords = df[cols].replace(-1, np.nan)
        # 计算平均值（忽略NaN值）
        result_df[f'main_node_{axis}'] = valid_coords.mean(axis=1)
    
    # 将(0,0,0)和(-0,-0,-0)替换为NaN
    mask_invalid = (
        (result_df['main_node_x'].abs() < 1e-8) &
        (result_df['main_node_y'].abs() < 1e-8) &
        (result_df['main_node_z'].abs() < 1e-8)
    ) | (
        (result_df['main_node_x'] == -0) &
        (result_df['main_node_y'] == -0) &
        (result_df['main_node_z'] == -0)
    )
    result_df.loc[mask_invalid, ['main_node_x', 'main_node_y', 'main_node_z']] = np.nan
    
    return result_df

def find_continuous_segments(df: pd.DataFrame) -> List[Tuple[int, int]]:
    """
    找出连续的时间段
    
    Args:
        df: 包含frame_idx的DataFrame
    
    Returns:
        连续时间段的列表，每个元素为(start_idx, end_idx)
    """
    segments = []
    start_idx = 0
    
    for i in range(1, len(df)):
        if df['frame_idx'].iloc[i] != df['frame_idx'].iloc[i-1] + 1:
            segments.append((start_idx, i-1))
            start_idx = i
    
    # 添加最后一个时间段
    segments.append((start_idx, len(df)-1))
    return segments

def mirror_pad_series(series: pd.Series, half_window: int, start_idx: int, end_idx: int) -> pd.Series:
    """
    对连续时间段进行镜像填充
    
    Args:
        series: 需要填充的Series
        half_window: 窗口的半长
        start_idx: 时间段的起始索引
        end_idx: 时间段的结束索引
    
    Returns:
        填充后的Series
    """
    # 创建新的Series，包含原始数据和填充空间
    total_length = len(series) + 2 * half_window
    padded_series = pd.Series(index=range(total_length), dtype=float)
    
    # 复制原始数据
    padded_series.iloc[half_window:half_window + len(series)] = series.values
    
    # 对开始部分进行镜像填充
    for i in range(half_window):
        padded_series.iloc[half_window - i - 1] = padded_series.iloc[half_window + i + 1]
    
    # 对结束部分进行镜像填充
    for i in range(half_window):
        padded_series.iloc[half_window + len(series) + i] = padded_series.iloc[half_window + len(series) - i - 2]
    
    return padded_series

def calculate_velocity(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    计算主节点的速度
    
    Args:
        df: 包含主节点位置的DataFrame
        window_size: 计算速度的窗口大小（帧数）
    
    Returns:
        包含速度信息的DataFrame
    """
    result_df = pd.DataFrame()
    result_df['frame_idx'] = df['frame_idx']
    
    # 计算窗口的半长
    half_window = (window_size - 1) // 2
    
    # 找出连续的时间段
    segments = find_continuous_segments(df)
    
    # 计算主节点的速度（使用欧几里得距离）
    for axis in ['x', 'y', 'z']:
        col_name = f'main_node_{axis}'
        velocity = pd.Series(index=df.index, dtype=float)
        
        # 对每个连续时间段分别处理
        for start_idx, end_idx in segments:
            # 获取当前时间段的数据
            segment_data = df[col_name].iloc[start_idx:end_idx+1]
            
            # 进行镜像填充
            padded_data = mirror_pad_series(segment_data, half_window, start_idx, end_idx)
            
            # 计算速度
            for i in range(len(segment_data)):
                # 在填充后的数据中，原始数据的起始位置是 half_window
                future_pos = padded_data.iloc[half_window + i + half_window]
                past_pos = padded_data.iloc[half_window + i - half_window]
                velocity.iloc[start_idx + i] = (future_pos - past_pos) / (window_size - 1)
        
        result_df[f'velocity_{axis}'] = velocity
    
    # 计算合速度（欧几里得距离）
    result_df['velocity_magnitude'] = np.sqrt(
        result_df['velocity_x']**2 + 
        result_df['velocity_y']**2 + 
        result_df['velocity_z']**2
    )
    
    return result_df

def main():
    # 读取数据
    df = pd.read_csv('velocity_selected.csv')
    
    # 需要计算平均位置的关节点
    joint_indices = [0, 14, 15, 16, 17]
    
    # 计算主节点位置
    print("Calculating main node position...")
    main_node_df = calculate_main_node_position(df, joint_indices)
    
    # 不同的窗口大小
    window_sizes = [3, 5, 7, 9, 11, 13, 15, 17, 19, 51, 101]
    
    # 对每个窗口大小计算速度
    for window_size in window_sizes:
        print(f"Calculating velocity with window size {window_size}...")
        velocity_df = calculate_velocity(main_node_df, window_size)
        
        # 保存结果
        output_filename = f'velocity_window_{window_size}.csv'
        velocity_df.to_csv(output_filename, index=False)
        print(f"Saved results to {output_filename}")

if __name__ == "__main__":
    main() 