import pandas as pd

# 加载 Excel 文件
excel_path = "Data\labelling tool\P1 mindful meal.xlsx"
df_excel = pd.read_excel(excel_path, header=None)  # 没有表头

# 设置正确列名
df_excel.columns = ['filepath', 'hour', 'minute', 'second', 'label']

# 加载现有 label.csv，如果有的话
csv_path = "label.csv"
try:
    df_csv = pd.read_csv(csv_path)
    df_combined = pd.concat([df_csv, df_excel], ignore_index=True)
except FileNotFoundError:
    df_combined = df_excel  # 如果没有原始 label.csv 就用 Excel 替代

# 保存回 CSV
df_combined.to_csv(csv_path, index=False)
print("✅ Excel 内容已成功合并到 label.csv 中。")