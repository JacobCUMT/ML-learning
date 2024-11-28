import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 定义一个函数计算熵
def entropy(y):
    """计算标签的熵"""
    counts = Counter(y)  # 统计每个类别的数量
    probabilities = [count / len(y) for count in counts.values()]  # 每个类别的概率
    return -sum(p * np.log2(p) for p in probabilities if p > 0)

# 定义一个函数计算信息增益
def information_gain(X_column, y, threshold):
    """计算指定特征列和分裂点的分裂信息增益"""
    # 分裂数据
    left_mask = X_column <= threshold
    right_mask = X_column > threshold
    if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
        return 0

    # 计算分裂后的熵
    left_entropy = entropy(y[left_mask])
    right_entropy = entropy(y[right_mask])
    n = len(y)
    n_left = np.sum(left_mask)
    n_right = np.sum(right_mask)

    # 信息增益公式
    weighted_entropy = (n_left / n) * left_entropy + (n_right / n) * right_entropy
    return entropy(y) - weighted_entropy

# 决策树节点定义
class DecisionTreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # 分裂特征
        self.threshold = threshold    # 分裂阈值
        self.left = left              # 左子树
        self.right = right            # 右子树
        self.value = value            # 叶子节点的类别

# 决策树类定义
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth  # 树的最大深度
        self.root = None            # 根节点

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # 停止条件：达到最大深度或只有一个类
        if depth == self.max_depth or n_labels == 1:
            most_common_label = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=most_common_label)

        # 寻找最佳分裂点
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature in range(n_features):
            X_column = X[:, feature]
            thresholds = np.unique(X_column)  # 获取所有唯一分裂点

            for threshold in thresholds:
                gain = information_gain(X_column, y, threshold)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        # 如果没有信息增益，返回叶子节点
        if best_gain == 0:
            most_common_label = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=most_common_label)

        # 分裂数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        # 构建子树
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        return DecisionTreeNode(feature=best_feature, threshold=best_threshold,
                                 left=left_child, right=right_child)

    def fit(self, X, y):
        """训练模型"""
        self.root = self._build_tree(X, y)

    def _predict(self, x, node):
        """递归预测单个样本"""
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)

    def predict(self, X):
        """预测多个样本"""
        return np.array([self._predict(x, self.root) for x in X])

    def visualize(self, node=None, depth=0):
        """可视化决策树"""
        if node is None:
            node = self.root
        if node.value is not None:
            print(f"{'|   ' * depth}Leaf: Class={node.value}")
        else:
            print(f"{'|   ' * depth}Feature {node.feature}, Threshold {node.threshold}")
            self.visualize(node.left, depth + 1)
            self.visualize(node.right, depth + 1)

# 使用自定义决策树进行训练和测试
if __name__ == "__main__":
    # 加载 Iris 数据集
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建并训练决策树
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)

    # 测试模型
    y_pred = clf.predict(X_test)

    # 计算准确率
    accuracy = np.mean(y_pred == y_test)
    print(f"自定义决策树的准确率: {accuracy:.2f}\n")

    # 可视化决策树结构
    print("决策树结构:")
    clf.visualize()
