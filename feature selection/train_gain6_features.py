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

# === 选用的特征（来自feature_gain_trace.csv前6个） ===
gain6_features = ['1_y', '2_x', 'frame_idx', '0_y', '5_z', '10_y']

# === 加载数据 ===
df = pd.read_csv(file_path)
X = df[gain6_features]
y = df['label']

# === 标准化 ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === 划分训练集和验证集 ===
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === 初始化分类器 ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

print("Training with 6 gain-selected features...")
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