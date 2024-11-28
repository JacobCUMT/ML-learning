import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载 Iris 数据集
data = load_iris()
X, y = data.data, data.target  # X是特征，y是标签
feature_names = data.feature_names
class_names = data.target_names

# 数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 手写实现朴素贝叶斯模型
class NaiveBayes:
    def __init__(self):
        self.class_priors = {}  # 存储每个类别的先验概率
        self.mean = {}  # 存储每个类别每个特征的均值
        self.var = {}  # 存储每个类别每个特征的方差

    def fit(self, X, y):
        """
        训练朴素贝叶斯模型
        :param X: 输入特征
        :param y: 标签
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)  # 获取所有类别
        for cls in self.classes:
            # 筛选出属于当前类别的样本
            X_c = X[y == cls]
            # 计算先验概率 P(y)
            self.class_priors[cls] = X_c.shape[0] / n_samples
            # 计算每个特征的均值和方差
            self.mean[cls] = X_c.mean(axis=0)
            self.var[cls] = X_c.var(axis=0)

    def gaussian_pdf(self, x, mean, var):
        """
        计算高斯分布的概率密度函数
        :param x: 输入值
        :param mean: 均值
        :param var: 方差
        :return: 概率密度值
        """
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent

    def predict(self, X):
        """
        对输入样本进行预测
        :param X: 输入特征
        :return: 预测类别
        """
        y_pred = []
        for x in X:
            # 计算每个类别的后验概率 P(y|x) ~ P(x|y) * P(y)
            posteriors = {}
            for cls in self.classes:
                prior = np.log(self.class_priors[cls])  # 使用对数防止数值下溢
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.mean[cls], self.var[cls])))
                posteriors[cls] = prior + likelihood
            # 选择后验概率最大的类别
            y_pred.append(max(posteriors, key=posteriors.get))
        return np.array(y_pred)

# 创建朴素贝叶斯实例并训练模型
nb = NaiveBayes()
nb.fit(X_train, y_train)

# 对测试集进行预测
y_pred = nb.predict(X_test)

# 计算模型准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 混淆矩阵可视化
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Naive Bayes Confusion Matrix')
plt.show()

# 数据可视化：特征的类别分布
for i in range(X.shape[1]):
    plt.figure(figsize=(6, 4))
    for cls in np.unique(y):
        sns.kdeplot(X_train[y_train == cls, i], label=f"{class_names[cls]}")
    plt.xlabel(feature_names[i])
    plt.ylabel('Density')
    plt.title(f'{feature_names[i]} Distribution by Class')
    plt.legend()
    plt.show()
