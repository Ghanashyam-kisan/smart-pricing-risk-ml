from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def find_optimal_clusters(X_scaled):

    inertias = []

    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)

    plt.plot(range(2, 10), inertias, marker='o')
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.show()


def perform_clustering(X_scaled, n_clusters=3):

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    clusters = kmeans.fit_predict(X_scaled)

    return kmeans, clusters
