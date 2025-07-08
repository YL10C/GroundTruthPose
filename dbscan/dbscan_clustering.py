import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, accuracy_score, classification_report
import seaborn as sns
import os

# 创建输出目录
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# 读取数据
print("Loading data...")
data = pd.read_csv('training_set_with_velocity_and_unit.csv')

# 分离特征和标签
selected_features = ['1_y', '2_x', 'frame_idx', '0_y', '5_z', '10_y']
X = data[selected_features]
true_labels = data['label']

print(f"Data shape: {data.shape}")
print(f"Number of features: {len(selected_features)}")
print(f"Number of samples: {len(data)}")

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# DBSCAN参数
eps = 0.3  # 邻域半径，减小以形成更紧凑的簇
min_samples = 5  # 最小样本数，减小以适应较小的数据集

# 执行DBSCAN聚类
print("Performing DBSCAN clustering...")
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
cluster_labels = dbscan.fit_predict(X_scaled)

# 计算聚类数量（不包括噪声点，标签为-1的点）
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')
print(f'Percentage of noise points: {(n_noise/len(data))*100:.2f}%')

# 计算轮廓系数（不包括噪声点）
if n_clusters > 1:
    mask = cluster_labels != -1
    if sum(mask) > 1:  # 确保有足够的非噪声点
        silhouette_avg = silhouette_score(X_scaled[mask], cluster_labels[mask])
        print(f'Silhouette Score: {silhouette_avg:.3f}')

# 为每个簇分配类别标签
cluster_to_class = {}
for cluster in set(cluster_labels):
    if cluster == -1:  # 跳过噪声点
        continue
    
    # 获取该簇的所有点的索引
    cluster_indices = np.where(cluster_labels == cluster)[0]
    
    # 获取这些点的真实标签
    cluster_true_labels = true_labels.iloc[cluster_indices]
    
    # 找出最常见的标签
    most_common_label = cluster_true_labels.mode().iloc[0]
    cluster_to_class[cluster] = most_common_label

print("\nCluster to class mapping:")
for cluster, class_label in cluster_to_class.items():
    print(f"Cluster {cluster} -> Class {class_label}")

# 计算预测的类别标签
predicted_labels = np.array([cluster_to_class.get(label, -1) for label in cluster_labels])

# 计算分类准确率（不包括噪声点）
mask = predicted_labels != -1
if sum(mask) > 0:
    accuracy = accuracy_score(true_labels[mask], predicted_labels[mask])
    print(f"\nClassification Accuracy (excluding noise points): {accuracy:.3f}")
    
    # 生成详细的分类报告
    print("\nClassification Report:")
    print(classification_report(true_labels[mask], predicted_labels[mask]))

# 可视化结果
plt.figure(figsize=(12, 8))

# 使用前两个特征进行可视化
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.title('DBSCAN Clustering Results')
plt.xlabel(selected_features[0])
plt.ylabel(selected_features[1])
plt.colorbar(label='Cluster')
plt.savefig(os.path.join(output_dir, 'dbscan_clusters.png'))
plt.close()

# 保存聚类结果
results_df = pd.DataFrame({
    'cluster': cluster_labels,
    'predicted_class': predicted_labels,
    'true_class': true_labels
})
results_df.to_csv(os.path.join(output_dir, 'dbscan_clusters.csv'), index=False)

# 分析每个簇的特征分布
plt.figure(figsize=(15, 10))
for i, feature in enumerate(selected_features):
    plt.subplot(2, 3, i+1)
    # 创建用于箱线图的数据框
    plot_data = pd.DataFrame({
        'cluster': cluster_labels,
        feature: X[feature].values  # 使用.values来获取numpy数组
    })
    sns.boxplot(x='cluster', y=feature, data=plot_data)
    plt.title(f'{feature} Distribution by Cluster')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cluster_distributions.png'))
plt.close()

# 保存评估结果
with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
    f.write(f"DBSCAN Clustering Evaluation Results\n")
    f.write(f"===================================\n\n")
    f.write(f"Number of clusters: {n_clusters}\n")
    f.write(f"Number of noise points: {n_noise}\n")
    f.write(f"Percentage of noise points: {(n_noise/len(data))*100:.2f}%\n")
    if n_clusters > 1:
        f.write(f"Silhouette Score: {silhouette_avg:.3f}\n")
    if sum(mask) > 0:
        f.write(f"\nClassification Accuracy (excluding noise points): {accuracy:.3f}\n")
        f.write("\nClassification Report:\n")
        f.write(classification_report(true_labels[mask], predicted_labels[mask]))