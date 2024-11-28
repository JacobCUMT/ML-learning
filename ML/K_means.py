import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成一个简单的数据集
# 生成100个样本点，分成3个聚类中心
data, labels_true = make_blobs(n_samples=100, centers=3, cluster_std=1.0, random_state=42)

# K均值聚类实现
class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4):
        """
        初始化K均值算法的参数
        :param n_clusters: 聚类的数量
        :param max_iter: 最大迭代次数
        :param tol: 判断收敛的阈值
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """
        拟合数据，执行K均值聚类算法
        :param X: 输入数据，形状为 (n_samples, n_features)
        """
        # 随机初始化聚类中心
        n_samples, n_features = X.shape
        np.random.seed(42)
        self.centers = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # 计算每个点到各个聚类中心的距离
            distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
            # 分配每个点到最近的聚类中心
            self.labels = np.argmin(distances, axis=1)

            # 更新聚类中心
            new_centers = np.array([X[self.labels == j].mean(axis=0) for j in range(self.n_clusters)])
            
            # 如果聚类中心的变化小于阈值，认为算法收敛
            if np.all(np.abs(new_centers - self.centers) < self.tol):
                break
            
            self.centers = new_centers

    def predict(self, X):
        """
        对新数据进行预测，分配到最近的聚类中心
        :param X: 输入数据
        :return: 每个数据点的聚类标签
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2)
        return np.argmin(distances, axis=1)

# 创建K均值模型实例
kmeans = KMeans(n_clusters=3)
# 训练模型
kmeans.fit(data)

# 可视化结果
plt.figure(figsize=(8, 6))
# 绘制样本点，并按照聚类结果着色
for i in range(3):
    plt.scatter(data[kmeans.labels == i, 0], data[kmeans.labels == i, 1], label=f"Cluster {i}")
# 绘制聚类中心
plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], s=200, c='red', marker='X', label="Centers")
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()
