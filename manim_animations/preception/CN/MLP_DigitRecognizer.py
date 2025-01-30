
import numpy as np  
from sklearn.datasets import fetch_openml  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from perceptron_models import SinglePerceptron, MultiLayerPerceptron

def main_single_perceptron():  
    # 加载MNIST数据集  
    print("加载MNIST数据集...")  
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)  
    X, y = mnist["data"], mnist["target"]  

    # 筛选数字0和1  
    print("筛选数字0和1...")  
    mask = (y == '0') | (y == '1')  
    X, y = X[mask], y[mask]  

    # 将标签转换为二进制  
    y = np.where(y == '0', 0, 1)  

    # 数据归一化  
    X = X / 255.0  

    # 分割训练集和测试集  
    print("分割训练集和测试集...")  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    # 初始化并训练感知机  
    print("训练单层感知机模型...")  
    perceptron = SinglePerceptron(learning_rate=0.01, n_iterations=1000)  
    perceptron.fit(X_train, y_train)  

    # 预测  
    print("进行预测...")  
    y_pred = perceptron.predict(X_test)  

    # 评估  
    accuracy = accuracy_score(y_test, y_pred)  
    print(f"单层感知机模型准确率: {accuracy * 100:.2f}%")  

def main_mlp():  
    # 加载MNIST数据集  
    print("加载MNIST数据集...")  
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)  
    X, y = mnist["data"], mnist["target"].astype(int)  

    # 数据归一化  
    X = X / 255.0  

    # 分割训练集和测试集  
    print("分割训练集和测试集...")  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

    # 初始化多层感知机  
    input_size = X_train.shape[1]  # 784  
    hidden_sizes = [128, 64]        # 可根据需要调整  
    output_size = 10                # 0-9  
    learning_rate = 0.01  
    n_epochs = 100  

    print("初始化多层感知机模型...")  
    mlp = MultiLayerPerceptron(input_size, hidden_sizes, output_size, learning_rate, n_epochs)  

    # 训练模型  
    print("训练多层感知机模型...")  
    mlp.fit(X_train, y_train)  

    # 预测  
    print("进行预测...")  
    y_pred = mlp.predict(X_test)  

    # 评估  
    accuracy = accuracy_score(y_test, y_pred)  
    print(f"多层感知机模型准确率: {accuracy * 100:.2f}%")  

if __name__ == "__main__":  
    # 训练和评估单层感知机  
    main_single_perceptron()  
    
    # 训练和评估多层感知机  
    main_mlp()