import pandas as pd
import numpy as np

"""
用法：直接修改main函数中的参数
- input_file: 输入csv文件路径
- output_file: 输出增强后csv文件路径  
- max_num: 每类目标样本数
- noise_scale: 噪声系数，默认0.1
"""

def augment_class(df, label, max_num, feature_cols, noise_scale=0.1, random_state=42):
    np.random.seed(random_state)
    class_df = df[df['label'] == label]
    n = len(class_df)
    if n == 0:
        return pd.DataFrame([])
    if n >= max_num:
        # 过采样类别，随机采样max_num个
        return class_df.sample(n=max_num, random_state=random_state)
    # 少数类，增强
    stds = class_df[feature_cols].std()
    samples = []
    for i in range(n):
        row = class_df.iloc[i]
        samples.append(row)
        # 需要生成的增强样本数
        num_aug = int(np.ceil((max_num - n) / n))
        for _ in range(num_aug):
            noise = np.random.randn(len(feature_cols)) * stds.values * noise_scale
            new_features = row[feature_cols].values + noise
            new_row = row.copy()
            new_row[feature_cols] = new_features
            samples.append(new_row)
    # 合并并采样max_num个
    aug_df = pd.DataFrame(samples)
    return aug_df.sample(n=max_num, random_state=random_state)

def main():
    # 在这里修改参数
    input_file = "Training_Data/p1&p2/p1&p2_data_train.csv"
    output_file = "Training_Data/p1&p2/p1&p2_data_train_augmented.csv"
    max_num = 5760  # 每类目标样本数
    noise_scale = 0.1  # 噪声系数

    df = pd.read_csv(input_file)
    feature_cols = [col for col in df.columns if col not in ['person','frame_idx','label']]
    labels = df['label'].unique()
    new_dfs = []
    for label in labels:
        aug_df = augment_class(df, label, max_num, feature_cols, noise_scale)
        new_dfs.append(aug_df)
    result = pd.concat(new_dfs, ignore_index=True)
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱
    result.to_csv(output_file, index=False)
    print(f"增强完成，保存到 {output_file} ，总样本数：{len(result)}")

if __name__ == '__main__':
    main() 