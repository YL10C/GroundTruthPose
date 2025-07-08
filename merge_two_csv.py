import pandas as pd
import os

# 配置文件名（可根据需要修改）
file_a = 'p2_living_art_2025_2100_2200(1).csv'  # 先加载的csv
file_b = 'merged_labels.csv'  # 后加载的csv，冲突时以此为准

# 你可以将上面两个文件名替换为实际文件名
# 或者用命令行参数/交互式输入实现更灵活的用法

def merge_csvs(file_a, file_b, output_file='merged_labels.csv'):
    # 读取两个csv
    df_a = pd.read_csv(file_a, dtype={'filepath': str, 'hour': int, 'minute': int, 'second': int, 'label': int})
    df_b = pd.read_csv(file_b, dtype={'filepath': str, 'hour': int, 'minute': int, 'second': int, 'label': int})

    # 生成主键
    for df in [df_a, df_b]:
        df['key'] = df['filepath'].astype(str) + '_' + df['hour'].astype(str) + '_' + df['minute'].astype(str) + '_' + df['second'].astype(str)

    # 以file_a为基础，更新/添加file_b中的内容
    df_a = df_a.set_index('key')
    df_b = df_b.set_index('key')

    # 用file_b的内容更新file_a
    df_a.update(df_b)

    # 合并所有key
    merged = pd.concat([df_a, df_b[~df_b.index.isin(df_a.index)]], axis=0)
    merged = merged.reset_index(drop=True)

    # 去除key列，按时间排序
    merged = merged.drop(columns=['key', 'label1', 'label2', 'label3'], errors='ignore')
    merged = merged.sort_values(['filepath', 'hour', 'minute', 'second']).reset_index(drop=True)

    merged.to_csv(output_file, index=False)
    print(f'Merged CSV saved to: {output_file}')
    print(f'Total frames in merged file: {len(merged)}')

if __name__ == '__main__':
    # 默认合并同目录下csv1.csv和csv2.csv
    merge_csvs(file_a, file_b) 