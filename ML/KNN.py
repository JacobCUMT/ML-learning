import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成二维数据集
def generate_data():
    np.random.seed(42)  # 固定随机种子，方便复现
    # 类别 0 数据
    class_0 = np.random.randn(50, 2) + [2, 2]
    # 类别 1 数据
    class_1 = np.random.randn(50, 2) + [6, 6]
    # 合并数据和标签
    X = np.vstack((class_0, class_1))
    y = np.array([0] * 50 + [1] * 50)
    return X, y

# K 近邻算法实现
class KNN:
    def __init__(self, k=3):
        """
        初始化 KNN 模型
        :param k: 近邻数量
        """
        self.k = k

    def fit(self, X, y):
        """
        训练模型（对于 KNN 来说，只需存储训练数据）
        :param X: 训练特征
        :param y: 训练标签
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        对新数据进行预测
        :param X: 待预测数据
        :return: 预测标签
        """
        predictions = []
        for x in X:
            # 计算每个点与训练集中点的欧氏距离
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # 找到最近的 k 个点的索引
            k_indices = distances.argsort()[:self.k]
            # 找到这 k 个点的标签
            k_nearest_labels = self.y_train[k_indices]
            # 多数投票决定分类结果
            prediction = np.bincount(k_nearest_labels).argmax()
            predictions.append(prediction)
        return np.array(predictions)

# 数据可视化函数
def plot_decision_boundary(X, y, model, resolution=0.01):
    """
    绘制决策边界
    :param X: 特征数据
    :param y: 标签数据
    :param model: 已训练的模型
    :param resolution: 网格精度
    """
    # 设置网格范围
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    
    # 预测网格中的每个点
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Decision Boundary')
    plt.show()

# 主程序
if __name__ == "__main__":
    # 生成数据集
    X, y = generate_data()

    # 数据集划分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 初始化 KNN 模型，设定 k=3
    knn = KNN(k=3)

    # 训练模型
    knn.fit(X_train, y_train)

    # 预测测试集
    y_pred = knn.predict(X_test)

    # 输出模型准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"KNN 模型准确率: {accuracy:.2f}")

    # 绘制决策边界
    plot_decision_boundary(X, y, knn)
