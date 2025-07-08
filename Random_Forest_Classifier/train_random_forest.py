import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# === 路径设置 ===
feature_name = "p1"
file = "p2"
train_file = f"Training_Data/{file}/{file}_data_train_augmented.csv"
feature_file = f"Training_Data/{file}/feature_gain_trace_{file}.csv"
model_save_path = f"Random_Forest_Classifier/random_forest_{file}_{feature_name}.joblib"

# === 读取特征列表 ===
feature_df = pd.read_csv(feature_file)
selected_features = feature_df['feature'].tolist()

# === 读取训练数据 ===
df_train = pd.read_csv(train_file)
X_train = df_train[selected_features]
y_train = df_train['label']

# === 训练模型 ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

# === 保存模型 ===
joblib.dump(clf, model_save_path)
print(f"模型已保存到 {model_save_path}") 