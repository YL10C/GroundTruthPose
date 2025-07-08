import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time

# === 参数设置 ===
file = "p1&p2"
train_file = f"Training_Data/{file}/{file}_data_train_augmented.csv"
val_file = f"Training_Data/{file}/{file}_data_val.csv"
n_features_to_select = 30  # 最多选多少个特征
early_stop_on_negative_gain = True  # 是否在出现负增益时提前停止

# === 加载数据 ===
df_train = pd.read_csv(train_file)
df_val = pd.read_csv(val_file)

# === 特征列名 ===
all_columns = df_train.columns.tolist()
# 去掉person, frame_idx, 所有_x和_z结尾的特征
feature_cols = [col for col in all_columns if col not in ['person', 'frame_idx', 'label'] and not (col.endswith('_x') or col.endswith('_z'))]

X_train = df_train[feature_cols]
y_train = df_train['label']
X_val = df_val[feature_cols]
y_val = df_val['label']

# === 标准化 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# === 初始化 ===
clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

selected_features = []
remaining_features = list(feature_cols)
feature_gains = []
last_score = 0

print("Starting Forward Feature Selection...\n")
start_time = time.time()

for step in range(n_features_to_select):
    best_gain = -np.inf
    best_feature = None
    best_score = None

    for feature in remaining_features:
        current_feature_set = selected_features + [feature]
        idxs = [feature_cols.index(f) for f in current_feature_set]
        X_train_sel = X_train_scaled[:, idxs]
        X_val_sel = X_val_scaled[:, idxs]

        clf.fit(X_train_sel, y_train)
        score = accuracy_score(y_val, clf.predict(X_val_sel))
        gain = score - last_score

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_score = score

    if best_feature is None or (early_stop_on_negative_gain and best_gain < 0):
        print(f"\n🛑 Early stopping at step {step+1}: no positive gain (gain = {best_gain:.4f})")
        break

    # 更新记录
    selected_features.append(best_feature)
    remaining_features.remove(best_feature)
    feature_gains.append({
        "step": step + 1,
        "feature": best_feature,
        "score": best_score,
        "gain": best_gain
    })
    last_score = best_score

    print(f"✅ Step {step+1}: Added '{best_feature}', Score: {best_score:.4f}, Gain: {best_gain:.4f}")

end_time = time.time()
print(f"\nTotal time: {end_time - start_time:.2f} seconds")

# === 保存结果 ===
result_df = pd.DataFrame(feature_gains)
result_df.to_csv(f"Training_Data/{file}/feature_gain_trace_{file}.csv", index=False)
print(f"\nResults saved to 'feature_gain_trace_{file}.csv'")
