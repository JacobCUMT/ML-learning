import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 生成数据集
def generate_data():
    # 创建一个线性可分的二分类数据集
    X, y = make_classification(
        n_samples=100,  # 样本数
        n_features=2,   # 特征数
        n_classes=2,    # 类别数
        n_informative=2,  # 有效特征数
        n_redundant=0,    # 冗余特征数
        class_sep=2.0,    # 类别分离度
        random_state=42
    )
    y = np.where(y == 0, -1, 1)  # 将标签转换为 -1 和 1
    return X, y

# 定义支持向量机 (SVM) 类
class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate          # 学习率
        self.lambda_param = lambda_param  # 正则化参数
        self.n_iters = n_iters           # 迭代次数
        self.w = None                   # 权重向量
        self.b = None                   # 偏置

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)   # 初始化权重为0
        self.b = 0                      # 初始化偏置为0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 检查样本是否满足分类条件
                condition = y[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # 样本被正确分类，应用正则化更新
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # 样本被错误分类，更新权重和偏置
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.lr * y[idx]

    def predict(self, X):
        # 根据符号预测标签
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# 数据可视化函数
def visualize_svm(X, y, model):
    plt.figure(figsize=(8, 6))
    
    # 绘制数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.8, label="Data points")
    
    # 绘制决策边界
    x0_1 = np.amin(X[:, 0])  # x 的最小值
    x0_2 = np.amax(X[:, 0])  # x 的最大值
    x1 = - (model.w[0] * x0_1 + model.b) / model.w[1]
    x2 = - (model.w[0] * x0_2 + model.b) / model.w[1]
    plt.plot([x0_1, x0_2], [x1, x2], 'k', label="Decision boundary")

    # 绘制支持向量边界
    x1_margin = - (model.w[0] * x0_1 + model.b - 1) / model.w[1]
    x2_margin = - (model.w[0] * x0_2 + model.b - 1) / model.w[1]
    plt.plot([x0_1, x0_2], [x1_margin, x2_margin], 'k--', label="Support boundary (+1)")

    x1_margin_neg = - (model.w[0] * x0_1 + model.b + 1) / model.w[1]
    x2_margin_neg = - (model.w[0] * x0_2 + model.b + 1) / model.w[1]
    plt.plot([x0_1, x0_2], [x1_margin_neg, x2_margin_neg], 'k--', label="Support boundary (-1)")

    plt.legend()
    plt.title("SVM Visualization")
    plt.show()

# 主程序
X, y = generate_data()

# 创建并训练 SVM 模型
svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm.fit(X, y)

# 预测并计算准确率
predictions = svm.predict(X)
accuracy = np.mean(predictions == y)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# 可视化分析
visualize_svm(X, y, svm)
