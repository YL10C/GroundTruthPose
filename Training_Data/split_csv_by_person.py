#!/usr/bin/env python3
"""
脚本功能：根据person列将CSV文件分成两个文件
输入：包含person列的CSV文件
输出：p1_data.csv 和 p2_data.csv
"""

import pandas as pd
import os

# ==================== 参数定义 ====================
# 输入文件路径
INPUT_FILE = "Training_Data\p1&p2_data.csv"

# 输出目录
OUTPUT_DIR = "Training_Data"

# ==================== 主函数 ====================

def split_csv_by_person(input_file, output_dir='.'):
    """
    根据person列将CSV文件分成两个文件
    
    Args:
        input_file (str): 输入CSV文件路径
        output_dir (str): 输出目录，默认为当前目录
    """
    
    try:
        # 读取CSV文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        # 检查是否存在person列
        if 'person' not in df.columns:
            print("错误：CSV文件中没有找到'person'列")
            return False
        
        # 获取唯一的person值
        unique_persons = df['person'].unique()
        print(f"发现的人员: {unique_persons}")
        
        # 为每个person创建单独的文件
        for person in unique_persons:
            # 过滤数据
            person_data = df[df['person'] == person]
            
            # 创建输出文件名
            output_file = os.path.join(output_dir, f"{person}_data.csv")
            
            # 保存到文件
            person_data.to_csv(output_file, index=False)
            print(f"已保存 {person} 的数据到: {output_file}")
            print(f"  - 数据行数: {len(person_data)}")
        
        return True
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return False
    except Exception as e:
        print(f"错误：{str(e)}")
        return False

def main():
    """主函数"""
    
    print("=" * 50)
    print("CSV文件分割脚本")
    print("=" * 50)
    print(f"输入文件: {INPUT_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 50)
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误：文件 {INPUT_FILE} 不存在")
        return
    
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"已创建输出目录: {OUTPUT_DIR}")
    
    # 执行分割
    success = split_csv_by_person(INPUT_FILE, OUTPUT_DIR)
    
    if success:
        print("\n" + "=" * 50)
        print("分割完成！")
        print("=" * 50)
    else:
        print("\n" + "=" * 50)
        print("分割失败！")
        print("=" * 50)

if __name__ == "__main__":
    main() 