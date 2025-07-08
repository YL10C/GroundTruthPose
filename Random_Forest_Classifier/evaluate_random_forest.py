import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import os

# === 路径设置 ===
model_name = "p2_p1"
file = "p1"
model_path = f"Random_Forest_Classifier/random_forest_{model_name}.joblib"
test_file = f"Training_Data/{file}/{file}_data_test.csv"
feature_file = f"Training_Data/{model_name}/feature_gain_trace_{model_name}.csv"

# === 读取特征列表 ===
feature_df = pd.read_csv(feature_file)
selected_features = feature_df['feature'].tolist()

# === 读取测试数据 ===
df_test = pd.read_csv(test_file)
X_test = df_test[selected_features]
y_test = df_test['label']

# === 加载模型 ===
clf = joblib.load(model_path)

# === 预测 ===
y_pred = clf.predict(X_test)

# === 评估指标 ===
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

# 分类报告（含各类别F1、精确率、召回率）
print(f"\nClassification Report (per class) for {model_name} predict {file}:")
print(classification_report(y_test, y_pred, digits=4, zero_division=0))
