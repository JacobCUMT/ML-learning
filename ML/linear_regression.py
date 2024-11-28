import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 手写线性回归模型
class LinearRegression:
    def __init__(self):
        self.weights = None
        self.losses = []  # 用于存储每次迭代的损失值
    
    def fit(self, X, y, lr=0.01, epochs=1000):
        # 初始化权重
        self.weights = np.zeros(X.shape[1])
        
        # 梯度下降
        for epoch in range(epochs):
            # 计算预测值
            y_pred = np.dot(X, self.weights)
            
            # 计算误差
            error = y_pred - y
            
            # 计算梯度
            gradient = np.dot(X.T, error) / len(y)
            
            # 更新权重
            self.weights -= lr * gradient
            
            # 记录损失
            loss = mean_squared_error(y, y_pred)
            self.losses.append(loss)
            
            # 每100次迭代打印损失
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """
        预测
        :param X: 输入特征，形状为 (样本数, 特征数)
        :return: 预测值，形状为 (样本数,)
        """
        return np.dot(X, self.weights)

# 生成模拟回归数据集
X, y = make_regression(n_samples=1000, n_features=5, noise=10, random_state=42)

# 数据标准化处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 添加偏置项
X = np.hstack((np.ones((X.shape[0], 1)), X))

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train, lr=0.01, epochs=1000)

# 在测试集上预测并评估
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on Test Set: {mse:.4f}")

# 可视化分析

# 1. 绘制训练损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(len(model.losses)), model.losses, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Mean Squared Error")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# 2. 实际值与预测值对比
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# 3. 残差分析
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=20, edgecolor='k', alpha=0.7)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Distribution of Residuals")
plt.show()