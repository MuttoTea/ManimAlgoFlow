from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

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
