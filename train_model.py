import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import os

data_path = r'c:\Users\admin\Desktop\project2\data\transactions.csv'

if not os.path.exists(data_path):
    raise FileNotFoundError(f"Файл не найден: {data_path}")

df = pd.read_csv(data_path)


df = df.rename(columns={
    'transaction_dollar_amount': 'amount',
    'Long': 'long',
    'Lat': 'lat'
})

features = ['amount', 'long', 'lat']
X = df[features].copy()


X = X.dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

X['cluster'] = clusters

joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("✅ Модель и scaler сохранены!")

plt.figure(figsize=(15, 6))

# --- Scatter ---
plt.subplot(1, 2, 1)
sns.scatterplot(
    data=X,
    x='long',
    y='amount',
    hue='cluster',
    palette='viridis'
)
plt.title('Кластеры: Сумма и Долгота')


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['cluster'] = clusters

plt.subplot(1, 2, 2)
sns.scatterplot(
    data=pca_df,
    x='PCA1',
    y='PCA2',
    hue='cluster',
    palette='magma'
)
plt.title('PCA визуализация кластеров')

plt.tight_layout()
plt.savefig('cluster_plots.png')
print("📊 Графики сохранены в cluster_plots.png")

plt.show()