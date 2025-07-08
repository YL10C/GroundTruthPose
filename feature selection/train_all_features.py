import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# === 参数设置 ===
file_path = "training_set_with_velocity_and_unit.csv"

# === 加载数据 ===
df = pd.read_csv(file_path)

# === 特征列名 ===
position_cols = [f"{i}_{c}" for i in range(18) for c in ['x', 'y', 'z']]
velocity_cols = [f"{i}_v{c}" for i in range(18) for c in ['x', 'y', 'z']]
unit_velocity_cols = [f"{i}_uv{c}" for i in range(18) for c in ['x', 'y', 'z']]
all_features = ['frame_idx'] + position_cols + velocity_cols + unit_velocity_cols

X = df[all_features]
y = df['label']

# === 标准化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 划分训练集和验证集 ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 初始化分类器 ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Starting training with all features...")
start_time = time.time()

# === 训练模型 ===
clf.fit(X_train, y_train)

# === 评估模型 ===
train_score = accuracy_score(y_train, clf.predict(X_train))
val_score = accuracy_score(y_val, clf.predict(X_val))

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")
print(f"Training accuracy: {train_score:.4f}")
print(f"Validation accuracy: {val_score:.4f}")

# === 特征重要性分析 ===
feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': clf.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nTop 10 most important features:")
print(feature_importance.head(10))

# === 保存特征重要性到CSV文件 ===
feature_importance.to_csv("feature_importance_ranking.csv", index=False)
print("\nFeature importance rankings have been saved to 'feature_importance_ranking.csv'")