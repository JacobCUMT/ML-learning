import numpy as np
import matplotlib.pyplot as plt


def generate_data(n_samples=100, noise=0.1):
    np.random.seed(42)
    X = np.random.uniform(-3, 3, size=n_samples)
    y = 0.5 * X**3 - 2 * X**2 + X + np.random.normal(0, noise, size=n_samples)
    return X, y


class PolynomialRegression:
    def __init__(self, degree, learning_rate=0.001, n_iterations=5000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coefficients = None

    def _expand_features(self, X):
        """将特征扩展到多项式维度"""
        return np.vstack([X**i for i in range(self.degree + 1)]).T

    def fit(self, X, y):
        """训练模型"""
        # 扩展特征
        X_poly = self._expand_features(X)
        n_samples, n_features = X_poly.shape
        
        # 初始化参数
        self.coefficients = np.random.uniform(-0.1, 0.1, size=n_features)
        
        # 梯度下降
        for iteration in range(self.n_iterations):
            predictions = X_poly.dot(self.coefficients)
            errors = predictions - y
            gradients = (2 / n_samples) * X_poly.T.dot(errors)
            self.coefficients -= self.learning_rate * gradients
            if iteration % 500 == 0:
                loss = np.mean(errors**2)
                print(f"Iteration {iteration}: Loss = {loss:.4f}")

    def predict(self, X):
        """预测新数据"""
        X_poly = self._expand_features(X)
        return X_poly.dot(self.coefficients)

# 数据集生成
X, y = generate_data(n_samples=100, noise=1.0)

# 训练多项式回归模型
model = PolynomialRegression(degree=3, learning_rate=0.001, n_iterations=5000)
model.fit(X, y)

# 预测
X_test = np.linspace(-3, 3, 100)
y_pred = model.predict(X_test)

# 可视化结果
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X_test, y_pred, color='red', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Polynomial Regression')
plt.legend()
plt.show()
