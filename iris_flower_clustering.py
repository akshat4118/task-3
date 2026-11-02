import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

sns.set_theme(style="whitegrid")
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
X_scaled = StandardScaler().fit_transform(X)
clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
plot_df = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Cluster": clusters,
                        "True Species": [iris.target_names[i] for i in y]})
print("Cluster counts:\n", plot_df["Cluster"].value_counts().sort_index())
print(f"Silhouette score: {silhouette_score(X_scaled, clusters):.4f}")
print(f"Adjusted Rand Index: {adjusted_rand_score(y, clusters):.4f}")
print(f"PCA variance explained by 2 components: {pca.explained_variance_ratio_.sum():.4f}")
print("\nTrue species vs clusters:\n", pd.crosstab(plot_df["True Species"], plot_df["Cluster"]))
for hue, title, filename in [
    ("Cluster", "K-Means Clusters on Iris Dataset (PCA 2D)", "kmeans_clusters_pca.png"),
    ("True Species", "True Iris Species in PCA 2D Space", "true_species_pca.png")
]:
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue=hue, palette="Set2" if hue=="Cluster" else "Set1", s=75)
    plt.title(title); plt.tight_layout(); plt.savefig(filename, dpi=150); plt.show()
