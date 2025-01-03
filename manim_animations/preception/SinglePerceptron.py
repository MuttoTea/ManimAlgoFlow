from functools import wraps
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

def clear_previous_decorator(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.clear_previous()
        result = method(self, *args, **kwargs)
        if isinstance(result, VGroup):
            self.sections.append(result)
        return result
    return wrapper


class SinglePerceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 训练过程
        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # 更新权重和偏置
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted


class IrisDataProcessor:
    def __init__(self):
        # 加载鸢尾花数据集
        iris = load_iris()
        self.iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        self.iris_data["species"] = iris.target
        self.iris_data["species"] = self.iris_data["species"].map(
            {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
        )

        # 去掉 sepal length 为 4.5 的数据
        self.iris_data = self.iris_data[self.iris_data["sepal length (cm)"] != 4.5]

    def get_data(self):
        # 分离山鸢尾和变色鸢尾的数据
        setosa_data = (
            self.iris_data[self.iris_data["species"] == "Iris-setosa"]
            .head(50)
            .reset_index(drop=True)
        )
        versicolor_data = (
            self.iris_data[self.iris_data["species"] == "Iris-versicolor"]
            .head(50)
            .reset_index(drop=True)
        )

        # 删除特定列
        example_setosa = setosa_data.drop(
            ["petal length (cm)", "petal width (cm)"], axis=1
        )
        example_versicolor = versicolor_data.drop(
            ["petal length (cm)", "petal width (cm)"], axis=1
        )

        # 提取坐标并确保为 float 类型
        X = (
            pd.concat([example_setosa, example_versicolor]).values[:, :-1].astype(float)
        )  # 特征
        y = np.concatenate(
            [np.zeros(len(example_setosa)), np.ones(len(example_versicolor))]
        ).astype(
            float
        )  # 标签

        return X, y
