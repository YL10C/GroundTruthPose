#!/usr/bin/env python3
"""
脚本功能：将CSV文件分割成训练集、验证集和测试集
分割方式：每10个样本中，第9个作为验证集，第10个作为测试集，其余作为训练集
输入：CSV文件（例如p1_data.csv）
输出：train_data.csv, val_data.csv, test_data.csv
"""

import pandas as pd
import os
import math

# ==================== 参数定义 ====================
# 输入文件路径
INPUT_FILE = "Training_Data/p1&p2/p1&p2_data.csv"

# 输出目录
OUTPUT_DIR = "Training_Data/p1&p2"

# ==================== 主函数 ====================

def split_csv_train_val_test(input_file, output_dir='.'):
    """
    将CSV文件分割成训练集、验证集和测试集
    分割方式：每10个样本中，第9个作为验证集，第10个作为测试集，其余作为训练集
    
    Args:
        input_file (str): 输入CSV文件路径
        output_dir (str): 输出目录，默认为当前目录
    
    Returns:
        bool: 是否成功分割
    """
    
    try:
        # 读取CSV文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        print(f"总数据行数: {len(df)}")
        
        # 初始化空的DataFrame
        train_data = []
        val_data = []
        test_data = []
        
        # 按照每10个样本进行分割
        total_rows = len(df)
        num_groups = math.ceil(total_rows / 10)
        
        print(f"将数据分成 {num_groups} 组，每组最多10个样本")
        
        for i in range(num_groups):
            start_idx = i * 10
            end_idx = min((i + 1) * 10, total_rows)
            
            # 获取当前组的数据
            group_data = df.iloc[start_idx:end_idx]
            group_size = len(group_data)
            
            if group_size >= 9:
                # 如果组内有9个或更多样本
                # 前8个作为训练集
                train_data.append(group_data.iloc[:8])
                # 第9个作为验证集
                val_data.append(group_data.iloc[8:9])
                # 第10个作为测试集
                if group_size >= 10:
                    test_data.append(group_data.iloc[9:10])
            elif group_size == 8:
                # 如果组内有8个样本
                # 前7个作为训练集
                train_data.append(group_data.iloc[:7])
                # 第8个作为验证集
                val_data.append(group_data.iloc[7:8])
            elif group_size == 7:
                # 如果组内有7个样本
                # 前6个作为训练集
                train_data.append(group_data.iloc[:6])
                # 第7个作为验证集
                val_data.append(group_data.iloc[6:7])
            elif group_size == 6:
                # 如果组内有6个样本
                # 前5个作为训练集
                train_data.append(group_data.iloc[:5])
                # 第6个作为验证集
                val_data.append(group_data.iloc[5:6])
            elif group_size == 5:
                # 如果组内有5个样本
                # 前4个作为训练集
                train_data.append(group_data.iloc[:4])
                # 第5个作为验证集
                val_data.append(group_data.iloc[4:5])
            elif group_size == 4:
                # 如果组内有4个样本
                # 前3个作为训练集
                train_data.append(group_data.iloc[:3])
                # 第4个作为验证集
                val_data.append(group_data.iloc[3:4])
            elif group_size == 3:
                # 如果组内有3个样本
                # 前2个作为训练集
                train_data.append(group_data.iloc[:2])
                # 第3个作为验证集
                val_data.append(group_data.iloc[2:3])
            elif group_size == 2:
                # 如果组内有2个样本
                # 第1个作为训练集
                train_data.append(group_data.iloc[:1])
                # 第2个作为验证集
                val_data.append(group_data.iloc[1:2])
            else:
                # 如果组内只有1个样本，作为训练集
                train_data.append(group_data)
        
        # 合并所有数据
        train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
        val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
        test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
        
        # 创建输出文件名
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        train_file = os.path.join(output_dir, f"{base_name}_train.csv")
        val_file = os.path.join(output_dir, f"{base_name}_val.csv")
        test_file = os.path.join(output_dir, f"{base_name}_test.csv")
        
        # 保存文件
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        # 打印统计信息
        print(f"\n分割完成！")
        print(f"训练集: {len(train_df)} 行 -> {train_file}")
        print(f"验证集: {len(val_df)} 行 -> {val_file}")
        print(f"测试集: {len(test_df)} 行 -> {test_file}")
        
        # 计算比例
        total_split = len(train_df) + len(val_df) + len(test_df)
        if total_split > 0:
            train_ratio = len(train_df) / total_split * 100
            val_ratio = len(val_df) / total_split * 100
            test_ratio = len(test_df) / total_split * 100
            print(f"\n数据分布:")
            print(f"训练集: {train_ratio:.1f}%")
            print(f"验证集: {val_ratio:.1f}%")
            print(f"测试集: {test_ratio:.1f}%")
        
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return False
    except Exception as e:
        print(f"错误：{str(e)}")
        return False

def main():
    """主函数"""
    
    print("=" * 60)
    print("CSV文件训练集/验证集/测试集分割脚本")
    print("=" * 60)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("分割方式: 每10个样本中，第9个作为验证集，第10个作为测试集")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：文件 {INPUT_FILE} 不存在")
        return
    
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")
    
    # 执行分割
    success = split_csv_train_val_test(INPUT_FILE, OUTPUT_DIR)
    
    if success:
        print("\n" + "=" * 60)
        print("分割完成！")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("分割失败！")
        print("=" * 60)

if __name__ == "__main__":
    main() 