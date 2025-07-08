import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
import os

# === 参数设置 ===
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "..", "training_set_with_velocity_and_unit.csv")
max_features_to_test = 163  # 最多测试多少个特征

# === 加载数据 ===
print("Loading data...")
df = pd.read_csv(file_path)

# === 加载特征重要性排名 ===
print("Loading feature importance rankings...")
feature_ranking = pd.read_csv(os.path.join(base_dir, "feature_importance_ranking.csv"))
top_features = feature_ranking['feature'].tolist()

# === 准备数据 ===
X = df[top_features]  # 使用排序后的特征
y = df['label']

# === 标准化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 划分训练集和验证集 ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 初始化分类器 ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# === 存储结果 ===
results = []

print("\nStarting feature combination testing...")
start_time = time.time()

# === 测试不同数量的特征 ===
for n_features in range(1, min(max_features_to_test + 1, len(top_features) + 1)):
    print(f"\nTesting with top {n_features} features...")
    
    # 选择前n个特征
    selected_features = top_features[:n_features]
    feature_indices = [top_features.index(f) for f in selected_features]
    
    # 准备数据
    X_train_sel = X_train[:, feature_indices]
    X_val_sel = X_val[:, feature_indices]
    
    # 训练模型
    clf.fit(X_train_sel, y_train)
    
    # 评估模型
    train_score = accuracy_score(y_train, clf.predict(X_train_sel))
    val_score = accuracy_score(y_val, clf.predict(X_val_sel))
    
    # 记录结果
    results.append({
        'n_features': n_features,
        'train_score': train_score,
        'val_score': val_score
    })
    
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Validation accuracy: {val_score:.4f}")

end_time = time.time()
print(f"\nTotal testing time: {end_time - start_time:.2f} seconds")

# === 保存结果 ===
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(base_dir, "feature_combination_results.csv"), index=False)
print("\nResults have been saved to 'feature_combination_results.csv'") 