import pandas as pd
import os

# 文件路径
base_dir = os.path.dirname(__file__)
file1 = os.path.join(base_dir, 'test_label_LC.csv')
file2 = os.path.join(base_dir, 'p2_living_art_2025_2100_2200.csv')
file3 = os.path.join(base_dir, 'label_compare.csv')

# 读取数据
cols = ['filepath', 'hour', 'minute', 'second', 'label']
df1 = pd.read_csv(file1, dtype={'filepath': str, 'hour': int, 'minute': int, 'second': int, 'label': int})
df2 = pd.read_csv(file2, dtype={'filepath': str, 'hour': int, 'minute': int, 'second': int, 'label': int})
df3 = pd.read_csv(file3, dtype={'filepath': str, 'hour': int, 'minute': int, 'second': int, 'label': int})

# 统一主键格式（去除路径差异，只保留文件名）
def normalize_path(path):
    return os.path.basename(path)

df1['filekey'] = df1['filepath'].apply(normalize_path)
df2['filekey'] = df2['filepath'].apply(normalize_path)
df3['filekey'] = df3['filepath'].apply(normalize_path)

# 生成主键
for df in [df1, df2, df3]:
    df['key'] = df['filekey'].astype(str) + '_' + df['hour'].astype(str) + '_' + df['minute'].astype(str) + '_' + df['second'].astype(str)

# 只保留主键和label
s1 = df1.set_index('key')['label']
s2 = df2.set_index('key')['label']
s3 = df3.set_index('key')['label']

# 找到三者都标注的帧
common_keys = set(s1.index) & set(s2.index) & set(s3.index)
common_keys = list(common_keys)  # 修复pandas索引问题

# 合并label
labels_df = pd.DataFrame({
    'label1': s1.loc[common_keys],
    'label2': s2.loc[common_keys],
    'label3': s3.loc[common_keys],
})

# 判断三者是否一致
labels_df['all_equal'] = (labels_df['label1'] == labels_df['label2']) & (labels_df['label2'] == labels_df['label3'])

# 统计
n_total = len(labels_df)
n_diff = (~labels_df['all_equal']).sum()
ratio = n_diff / n_total if n_total > 0 else 0

print(f'Number of frames labeled by all three: {n_total}')
print(f'Number of frames with inconsistent labels: {n_diff}')
print(f'Inconsistency ratio: {ratio:.2%}')

# 可选：输出不一致的详细信息
if n_diff > 0:
    print('\nInconsistent frames:')
    print(labels_df[~labels_df['all_equal']].reset_index()[['key', 'label1', 'label2', 'label3']])

# 投票机制函数
def voting_mechanism(label1, label2, label3):
    """
    投票机制：选择标注最多的标签
    如果票数相同，按优先级选择：test_label_LC > label_compare > p2_living_art_2025_2100_2200
    """
    from collections import Counter
    
    # 收集所有非NaN的标签
    labels = [l for l in [label1, label2, label3] if pd.notna(l)]
    
    if not labels:
        return None
    
    # 统计每个标签的出现次数
    label_counts = Counter(labels)
    
    # 找到出现次数最多的标签
    max_count = max(label_counts.values())
    most_common_labels = [label for label, count in label_counts.items() if count == max_count]
    
    # 如果只有一个最多票数的标签，直接返回
    if len(most_common_labels) == 1:
        return most_common_labels[0]
    
    # 如果有多个标签票数相同，按优先级选择
    priority_order = [label1, label2, label3]  # test_label_LC, p2_living_art, label_compare
    
    for label in priority_order:
        if pd.notna(label) and label in most_common_labels:
            return label
    
    # 如果都找不到，返回第一个最多票数的标签
    return most_common_labels[0]

# 生成包含所有被标注过的帧的DataFrame
def generate_consolidated_csv():
    """
    生成包含所有被标注过的帧的CSV文件
    优先级：test_label_LC > label_compare > p2_living_art_2025_2100_2200
    """
    # 获取所有唯一的key
    all_keys = set(s1.index) | set(s2.index) | set(s3.index)
    
    # 创建合并的DataFrame
    consolidated_data = []
    
    for key in all_keys:
        # 获取每个文件对该帧的标注
        label1 = s1.get(key, None)
        label2 = s2.get(key, None)
        label3 = s3.get(key, None)
        
        # 使用投票机制确定最终标签
        final_label = voting_mechanism(label1, label2, label3)
        
        # 解析key获取时间信息
        parts = key.split('_')
        filename = parts[0]
        hour = int(parts[1])
        minute = int(parts[2])
        second = int(parts[3])
        
        # 使用label_compare.csv的filepath格式作为标准
        standard_filepath = f"D:\\OneDrive\\文档\\Yilin\\Edinburgh\\MSc project\\codebase\\Data\\p2\\user002 living art\\{filename}"
        
        consolidated_data.append({
            'filepath': standard_filepath,
            'hour': hour,
            'minute': minute,
            'second': second,
            'label': final_label,
            'label1': label1,
            'label2': label2,
            'label3': label3
        })
    
    # 创建DataFrame并排序
    df_consolidated = pd.DataFrame(consolidated_data)
    df_consolidated = df_consolidated.sort_values(['filepath', 'hour', 'minute', 'second']).reset_index(drop=True)

    # 统计标注来源
    labeled_by_1 = ((df_consolidated['label1'].notna()) & 
                   (df_consolidated['label2'].isna()) & 
                   (df_consolidated['label3'].isna())).sum()
    labeled_by_2 = ((df_consolidated['label1'].notna() + df_consolidated['label2'].notna() + 
                    df_consolidated['label3'].notna()) == 2).sum()
    labeled_by_3 = ((df_consolidated['label1'].notna()) & 
                   (df_consolidated['label2'].notna()) & 
                   (df_consolidated['label3'].notna())).sum()
    
    print(f'Frames labeled by 1 annotator: {labeled_by_1}')
    print(f'Frames labeled by 2 annotators: {labeled_by_2}')
    print(f'Frames labeled by 3 annotators: {labeled_by_3}')

    # 只保留五列
    df_consolidated = df_consolidated[['filepath', 'hour', 'minute', 'second', 'label']]
    
    # 保存到CSV文件
    output_file = os.path.join(base_dir, 'consolidated_labels.csv')
    df_consolidated.to_csv(output_file, index=False)
    
    print(f'\nConsolidated CSV saved to: {output_file}')
    print(f'Total frames in consolidated file: {len(df_consolidated)}')
    
    return df_consolidated

# 执行合并
consolidated_df = generate_consolidated_csv() 