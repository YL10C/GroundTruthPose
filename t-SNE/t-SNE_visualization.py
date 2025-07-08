import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# === 加载 CSV ===
file_path = "training_set_with_velocity_and_unit.csv"
df = pd.read_csv(file_path)

# === 选择所有特征列（除了label）===
feature_cols = [col for col in df.columns if col != 'label']
X = df[feature_cols].values

# === 标签列（用于画图）===
labels = df['label'].values

# === t-SNE 降维到 2D ===
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
X_embedded = tsne.fit_transform(X)

# === 可视化 ===
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=labels, palette='tab10', s=20)
plt.title("t-SNE on 163 features(perplexity=30)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="Label", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('tsne_visualization(perplexity=30).png', dpi=300, bbox_inches='tight')
plt.show()
