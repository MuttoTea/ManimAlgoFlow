from functools import wraps
import numpy as np
import pandas as pd


# def clear_previous_decorator(method):
#     @wraps(method)
#     def wrapper(self, *args, **kwargs):
#         self.clear_previous()
#         result = method(self, *args, **kwargs)
#         if isinstance(result, VGroup):
#             self.sections.append(result)
#         return result
#     return wrapper


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


class MultiLayerPerceptron:  
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, n_epochs=100):  
        """  
        初始化多层感知机  

        :param input_size: 输入层神经元数量  
        :param hidden_sizes: 隐藏层神经元数量列表  
        :param output_size: 输出层神经元数量  
        :param learning_rate: 学习率  
        :param n_epochs: 训练轮数  
        """  
        self.learning_rate = learning_rate  
        self.n_epochs = n_epochs  
        self.layers = []  
        self.weights = []  
        self.biases = []  

        layer_sizes = [input_size] + hidden_sizes + [output_size]  
        # 初始化权重和偏置  
        for i in range(len(layer_sizes) - 1):  
            weight = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2. / layer_sizes[i])  
            bias = np.zeros((1, layer_sizes[i+1]))  
            self.weights.append(weight)  
            self.biases.append(bias)  

    def relu(self, x):  
        return np.maximum(0, x)  

    def relu_derivative(self, x):  
        return (x > 0).astype(float)  

    def softmax(self, x):  
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 稳定性考虑  
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)  

    def cross_entropy_loss(self, y_true, y_pred):  
        m = y_true.shape[0]  
        # 防止log(0)  
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)  
        log_likelihood = -np.log(y_pred[range(m), y_true])  
        loss = np.sum(log_likelihood) / m  
        return loss  

    def predict_proba(self, X):  
        a = X  
        for i in range(len(self.weights) - 1):  
            z = np.dot(a, self.weights[i]) + self.biases[i]  
            a = self.relu(z)  
        z = np.dot(a, self.weights[-1]) + self.biases[-1]  
        a = self.softmax(z)  
        return a  

    def predict(self, X):  
        proba = self.predict_proba(X)  
        return np.argmax(proba, axis=1)  

    def fit(self, X, y):  
        """  
        训练多层感知机  

        :param X: 输入特征，形状为 (样本数, 特征数)  
        :param y: 标签，形状为 (样本数,) 且为整数类标签  
        """  
        m = X.shape[0]  

        # 将标签转换为one-hot编码  
        y_one_hot = np.zeros((m, np.max(y) + 1))  
        y_one_hot[np.arange(m), y] = 1  

        for epoch in range(self.n_epochs):  
            # 前向传播  
            activations = [X]  
            zs = []  
            a = X  
            for i in range(len(self.weights) - 1):  
                z = np.dot(a, self.weights[i]) + self.biases[i]  
                zs.append(z)  
                a = self.relu(z)  
                activations.append(a)  
            # 最后一层使用softmax  
            z = np.dot(a, self.weights[-1]) + self.biases[-1]  
            zs.append(z)  
            a = self.softmax(z)  
            activations.append(a)  

            # 损失计算  
            loss = self.cross_entropy_loss(y, a)  

            # 反向传播  
            delta = a  
            delta[range(m), y] -= 1  # dL/dz for softmax & cross-entropy  
            delta /= m  

            grad_w = []  
            grad_b = []  

            # 最后一层权重和偏置梯度  
            dw = np.dot(activations[-2].T, delta)  
            db = np.sum(delta, axis=0, keepdims=True)  
            grad_w.insert(0, dw)  
            grad_b.insert(0, db)  

            # 反向传播到隐藏层  
            for i in range(len(self.weights) - 2, -1, -1):  
                delta = np.dot(delta, self.weights[i+1].T) * self.relu_derivative(zs[i])  
                dw = np.dot(activations[i].T, delta)  
                db = np.sum(delta, axis=0, keepdims=True)  
                grad_w.insert(0, dw)  
                grad_b.insert(0, db)  

            # 更新权重和偏置  
            for i in range(len(self.weights)):  
                self.weights[i] -= self.learning_rate * grad_w[i]  
                self.biases[i] -= self.learning_rate * grad_b[i]  

            if (epoch + 1) % 10 == 0 or epoch == 0:  
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")  