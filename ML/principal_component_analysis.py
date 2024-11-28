import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import ace_tools_open as tools


iris = load_iris()
data = iris.data  # 使用 Iris 数据集的特征
target = iris.target  # 使用目标标签
columns = iris.feature_names

# 对数据进行标准化处理，使每个特征的均值为0，标准差为1
data_std = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(data_std.T)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 对特征值和特征向量按特征值大小进行排序
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# 将数据投影到前两个主成分上
pc1, pc2 = eigenvectors[:, 0], eigenvectors[:, 1]
data_pca = np.dot(data_std, np.column_stack((pc1, pc2)))

# 转换为 DataFrame 便于可视化
pca_df = pd.DataFrame(data_pca, columns=["PC1", "PC2"])
pca_df['Target'] = target

# 可视化 PCA 结果
plt.figure(figsize=(8, 6))
for label, color in zip(np.unique(target), ['red', 'green', 'blue']):
    plt.scatter(
        pca_df.loc[pca_df['Target'] == label, 'PC1'],
        pca_df.loc[pca_df['Target'] == label, 'PC2'],
        label=iris.target_names[label],
        alpha=0.7
    )
plt.title("PCA of Iris Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()

tools.display_dataframe_to_user("PCA Transformed Data", pca_df)
