import pickle
import pprint  # 用于更清晰打印嵌套结构

pkl_path = r"D:\OneDrive\文档\Yilin\Edinburgh\MSc project\codebase\Data\p1\user_inter_001_kitchen_mindful_meal\20250309.pkl"  # 👈 替换为你的 pkl 文件路径

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 打印顶层结构（如 24 小时）
print(f"Top-level length (hours): {len(data)}")

# 打印某个具体时间段的数据结构
example = data[9][27][2]  # 👈 你可以换成其他 hour, min, seg

print("Example frame structure:")
pprint.pprint(example[10])  # 打印第0帧
