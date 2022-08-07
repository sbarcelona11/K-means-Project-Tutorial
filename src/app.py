
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import decomposition
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/k-means-project-tutorial/main/housing.csv')
df_raw.head()
df_raw.info()
df_raw.describe()

df_interin = df_raw.copy()
df_interin = df_interin[['Latitude', 'Longitude', 'MedInc']]

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_interin)
df_scaled

sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df_scaled)
    sum_of_squared_distances.append(km.inertia_)

plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('Clusters (k)')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method Optimal k')
plt.show()

rango_n_clusters = [2, 3, 4, 5, 6, 7, 8,9,10]
silhouette_avg = []
for num_clusters in rango_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(df_scaled)
    cluster_labels = kmeans.labels_
    silhouette_avg.append(silhouette_score(df_scaled, cluster_labels))


plt.plot(rango_n_clusters,silhouette_avg,'bx-')
plt.xlabel('Clusters (k)')
plt.ylabel('Silhouette score')
plt.title('Silhouette to optimal k')
plt.show()

kmeans = KMeans(init="random",n_clusters=2, random_state=42, n_init=10, max_iter=300)
kmeans.fit(df_scaled)

df_inverse_scaled = scaler.inverse_transform(df_scaled)

df = pd.DataFrame(df_inverse_scaled ,columns=['Latitude','Longitude','MedInc'])
df['Cluster'] = kmeans.labels_
df

df['Cluster'] = pd.Categorical(df_2.Cluster)

sns.relplot(x="Longitude", y="Latitude", hue="Cluster", data=df, height=6)
plt.show()

sns.relplot(x="MedInc", y="Latitude", hue="Cluster", data=df, height=6)
plt.show()

sns.relplot(x="MedInc", y="Longitude", hue="Cluster", data=df, height=6)
plt.show()

sns.relplot(x='Latitude', y='Longitude', data=df, kind='scatter', size = 'MedInc', hue='Cluster')
plt.show()

pd.plotting.parallel_coordinates(df, 'Cluster')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(df['Latitude'])
y = np.array(df['Longitude'])
z = np.array(df['MedInc'])
ax.scatter(x,y,z, c=df["Cluster"], s=40)
plt.show()

X = df[['Latitude', 'Longitude']]
pca = decomposition.PCA(n_components=2)
pca.fit(X)
X_prime = pca.transform(X)
print(pca.explained_variance_ratio_)

df_pca = pd.DataFrame(data = X_prime, columns = ['pca_1', 'pca_2'])
df['pca_1'] = df_pca['pca_1']

sns.relplot(x="MedInc", y="pca_1", hue="Cluster", data=df, height=6)
plt.show()