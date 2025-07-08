import pandas as pd
import matplotlib.pyplot as plt
import os

# === 加载数据 ===
base_dir = os.path.dirname(os.path.abspath(__file__))
results_df = pd.read_csv(os.path.join(base_dir, "feature_combination_results.csv"))

# === 创建图表 ===
plt.figure(figsize=(12, 6))

# === 绘制训练集和验证集的得分 ===
plt.plot(results_df['n_features'], results_df['train_score'], 
         label='Training Score', color='blue', linewidth=2)
plt.plot(results_df['n_features'], results_df['val_score'], 
         label='Validation Score', color='red', linewidth=2)

# === 找到验证集最高点 ===
max_val_idx = results_df['val_score'].idxmax()
max_val_score = results_df.loc[max_val_idx, 'val_score']
max_val_features = results_df.loc[max_val_idx, 'n_features']

# === 标记最高点 ===
plt.plot(max_val_features, max_val_score, 'ro', markersize=8)
plt.annotate(f'Max: {max_val_score:.4f}\nFeatures: {max_val_features}',
             xy=(max_val_features, max_val_score),
             xytext=(max_val_features + 5, max_val_score - 0.02),
             arrowprops=dict(facecolor='black', shrink=0.05))

# === 设置图表属性 ===
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Model Performance vs Number of Features', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)

# === 设置Y轴范围 ===
plt.ylim(0.88, 1.0)

# === 保存图表 ===
plt.savefig(os.path.join(base_dir, 'feature_performance.png'), dpi=300, bbox_inches='tight')
print("Plot has been saved as 'feature_performance.png'") 