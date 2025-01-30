"""  
Summary:  
This code implements a Single Layer Perceptron model and a class for processing the Iris dataset (IrisDataProcessor). The main features include:  

1. Single Layer Perceptron:  
   - The `SinglePerceptron` class implements the basic functionality of a single-layer perceptron, including initialization, training (fit), activation function, and prediction methods.  
   - It uses the perceptron algorithm for binary classification tasks.  

2. Iris Dataset Processing:  
   - The `IrisDataProcessor` class is used to load the Iris dataset and preprocess the data.  
   - It includes mapping the species of Iris to their names, removing specific data, and separating data for specific species to return features and labels.  

3. Decorator:  
   - The `clear_previous_decorator` decorator is used to clear previous results before calling a method and to add the result to the `sections` list upon method return.  

This code is suitable for learning and practicing machine learning and data analysis, particularly in classification tasks.  
"""  

from functools import wraps  
import numpy as np  
import pandas as pd  
from sklearn.datasets import load_iris  
from manim import VGroup  

def clear_previous_decorator(method):  
    @wraps(method)  
    def wrapper(self, *args, **kwargs):  
        # Clear previous results before executing the method  
        self.clear_previous()  
        result = method(self, *args, **kwargs)  
        # If the result is a VGroup, append it to the sections list  
        if isinstance(result, VGroup):  
            self.sections.append(result)  
        return result  
    return wrapper  


class SingleLayerPerceptron:  
    def __init__(self, learning_rate=0.01, n_iterations=1000):  
        self.learning_rate = learning_rate  # Learning rate for weight updates  
        self.n_iterations = n_iterations      # Number of iterations for training  
        self.weights = None                   # Weights of the perceptron  
        self.bias = None                      # Bias term  

    def fit(self, X, y):  
        n_samples, n_features = X.shape  
        # Initialize weights and bias  
        self.weights = np.zeros(n_features)  
        self.bias = 0  

        # Training process  
        for _ in range(self.n_iterations):  
            for idx, x_i in enumerate(X):  
                # Calculate the linear output  
                linear_output = np.dot(x_i, self.weights) + self.bias  
                # Apply the activation function  
                y_predicted = self.activation_function(linear_output)  

                # Update weights and bias  
                update = self.learning_rate * (y[idx] - y_predicted)  
                self.weights += update * x_i  
                self.bias += update  

    def activation_function(self, x):  
        # Activation function: step function  
        return np.where(x >= 0, 1, 0)  

    def predict(self, X):  
        # Predict the class labels for the input data  
        linear_output = np.dot(X, self.weights) + self.bias  
        y_predicted = self.activation_function(linear_output)  
        return y_predicted  


class IrisDataProcessor:  
    def __init__(self):  
        # Load the Iris dataset  
        iris = load_iris()  
        self.iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)  
        self.iris_data["species"] = iris.target  
        # Map species numbers to their names  
        self.iris_data["species"] = self.iris_data["species"].map(  
            {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}  
        )  

        # Remove data where sepal length is 4.5  
        self.iris_data = self.iris_data[self.iris_data["sepal length (cm)"] != 4.5]  

    def get_data(self):  
        # Separate data for Iris-setosa and Iris-versicolor  
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

        # Drop specific columns  
        example_setosa = setosa_data.drop(  
            ["petal length (cm)", "petal width (cm)"], axis=1  
        )  
        example_versicolor = versicolor_data.drop(  
            ["petal length (cm)", "petal width (cm)"], axis=1  
        )  

        # Extract features and ensure they are of float type  
        X = (  
            pd.concat([example_setosa, example_versicolor]).values[:, :-1].astype(float)  
        )  # Features  
        y = np.concatenate(  
            [np.zeros(len(example_setosa)), np.ones(len(example_versicolor))]  
        ).astype(  
            float  
        )  # Labels  

        return X, y