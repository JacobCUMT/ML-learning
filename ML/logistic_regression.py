import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 使用 sklearn 的 make_classification 方法生成二分类数据集
from sklearn.datasets import make_classification

# 生成一个包含 1000 个样本的二维特征数据集
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                           n_redundant=0, random_state=42, n_clusters_per_class=1)

# 将数据集分为训练集 (70%) 和测试集 (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 对特征进行标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 使用训练集数据拟合并转换
X_test = scaler.transform(X_test)       # 使用相同的标准化参数转换测试集

# 初始化并训练逻辑回归模型
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# 使用训练好的模型在测试集上进行预测
y_pred = log_reg.predict(X_test)

# 模型评估
accuracy = accuracy_score(y_test, y_pred)  # 计算测试集的准确率
conf_matrix = confusion_matrix(y_test, y_pred)  # 混淆矩阵
class_report = classification_report(y_test, y_pred)  # 分类报告

# 可视化：绘制决策边界
plt.figure(figsize=(8, 6))

# 创建用于绘制决策边界的网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 使用模型预测网格点的分类
Z = log_reg.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)

# 绘制原始数据点
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 可视化：绘制混淆矩阵的热力图
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# 输出模型性能评估结果
print(f"准确率 (Accuracy): {accuracy}")
print("\n分类报告 (Classification Report):")
print(class_report)
