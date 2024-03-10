import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
import umap
from mpl_toolkits.mplot3d import Axes3D
warnings.filterwarnings("ignore", category=Warning)

##cols = ['goal', 'go_out', 'sports', 'tvsports', 'sports',
##        'dining', 'museums', 'art', 'hiking', 'gaming',
##        'clubbing', 'reading', 'tv', 'theater', 'movies',
##        'concerts', 'music','shopping',  'yoga']
cols = ['goal', 'sports', 'yoga', 'museums', 'art']
n_cluster = 8
def read_data(fn):
    return pd.read_csv(fn)

def preprocess_data(df):
    return df[cols].dropna(how='any')

def count_elbow(df):
    inertia = []
    n = 30
    for k in range(1, n):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n), inertia, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.savefig("elbow.png")
    plt.clf()
def count_k_means(df):
    kmeans = KMeans(n_cluster)
    kmeans.fit(df)
    return kmeans.labels_
def count_dbscan(df):
    dbscan = DBSCAN(eps=0.1, min_samples=5) 
    return dbscan.fit_predict(df)
def count_pca(df):
    pca = PCA(n_components=2)
    return pca.fit_transform(df)
def count_umap(df):
    return umap.UMAP(n_components=2).fit_transform(df)
def visual(df, f_cluster, f_resize, title):
    df_resize = f_resize(df)
    labels = f_cluster(df)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(df[cols[0]],df[cols[1]], df[cols[2]], c=labels, marker='o')
    plt.legend(*sc.legend_elements())
    plt.savefig('before_' + title + ".png")
    plt.clf()
    
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df_resize[:, 0], y=df_resize[:, 1], hue=labels, palette='viridis')
    plt.title('after_' + title)
    plt.legend()
    plt.savefig('after_' + title + ".png")
    plt.clf()
df = read_data("Speed Dating Data.csv")
df = preprocess_data(df)

count_elbow(df)
visual(df, count_k_means, count_pca, "k_means_pca")
visual(df, count_dbscan, count_pca, "dbscan_pca")
visual(df, count_dbscan, count_umap, "dbscan_umap")
visual(df, count_k_means, count_umap, "k_means_umap")
