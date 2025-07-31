
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = pd.DataFrame({
    'CustomerID': range(1, 11),
    'AnnualIncome': [15, 16, 17, 18, 20, 80, 85, 86, 87, 90], 
    'SpendingScore': [39, 81, 6, 77, 40, 76, 94, 3, 72, 12]
})

print("\nFirst 5 rows of dataset:")
print(data.head())


X = data[['AnnualIncome', 'SpendingScore']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Find Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


optimal_clusters = 3 
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClustered Data:")
print(data)

plt.figure(figsize=(6, 4))
sns.scatterplot(data=data, x='AnnualIncome', y='SpendingScore', hue='Cluster', palette='Set1', s=100)
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend(title='Cluster')
plt.show()
