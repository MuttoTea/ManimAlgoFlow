"""  
Summary:  
This code implements the definition and training of Single Perceptron and Multi-Layer Perceptron models. The main functionalities include:  

1. Single Perceptron:  
   - The `SinglePerceptron` class implements a basic single-layer perceptron, including methods for initialization, training (fit), activation function, and prediction.  
   - It uses the perceptron algorithm for binary classification tasks.  

2. Multi-Layer Perceptron:  
   - The `MultiLayerPerceptron` class implements a multi-layer perceptron that supports the configuration of hidden layers and includes forward propagation and backpropagation algorithms.  
   - It utilizes the ReLU activation function and a Softmax output layer, supporting multi-class classification tasks.  
   - It includes the calculation of the cross-entropy loss function and a weight update mechanism.  

This code is suitable for learning and practicing machine learning and deep learning, particularly in classification tasks.  
"""  

from functools import wraps  
import numpy as np  
import pandas as pd  


class SingleLayerPerceptron:  
    def __init__(self, learning_rate=0.01, n_iterations=1000):  
        self.learning_rate = learning_rate  # Learning rate for weight updates  
        self.n_iterations = n_iterations  # Number of training iterations  
        self.weights = None  # Weights of the perceptron  
        self.bias = None  # Bias term  

    def fit(self, features, labels):  
        n_samples, n_features = features.shape  # Get the number of samples and features  
        # Initialize weights and bias  
        self.weights = np.zeros(n_features)  
        self.bias = 0  

        # Training process  
        for _ in range(self.n_iterations):  
            for index, sample in enumerate(features):  
                linear_output = np.dot(sample, self.weights) + self.bias  # Calculate linear output  
                predicted_label = self.activation_function(linear_output)  # Apply activation function  

                # Update weights and bias  
                update = self.learning_rate * (labels[index] - predicted_label)  
                self.weights += update * sample  
                self.bias += update  

    def activation_function(self, x):  
        """Activation function: Step function."""  
        return np.where(x >= 0, 1, 0)  

    def predict(self, features):  
        """Predict the class labels for the given features."""  
        linear_output = np.dot(features, self.weights) + self.bias  
        predicted_labels = self.activation_function(linear_output)  
        return predicted_labels  


class MultiLayerPerceptron:  
    def __init__(self, input_size, hidden_layer_sizes, output_size, learning_rate=0.01, n_epochs=100):  
        """  
        Initialize the Multi-Layer Perceptron.  

        :param input_size: Number of neurons in the input layer  
        :param hidden_layer_sizes: List of numbers of neurons in hidden layers  
        :param output_size: Number of neurons in the output layer  
        :param learning_rate: Learning rate for weight updates  
        :param n_epochs: Number of training epochs  
        """  
        self.learning_rate = learning_rate  # Learning rate  
        self.n_epochs = n_epochs  # Number of training epochs  
        self.layers = []  # List to hold layer configurations  
        self.weights = []  # List to hold weights for each layer  
        self.biases = []  # List to hold biases for each layer  

        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]  # Define layer sizes  
        # Initialize weights and biases  
        for i in range(len(layer_sizes) - 1):  
            weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2. / layer_sizes[i])  # He initialization  
            bias = np.zeros((1, layer_sizes[i + 1]))  # Initialize biases to zero  
            self.weights.append(weight)  
            self.biases.append(bias)  

    def relu(self, x):  
        """ReLU activation function."""  
        return np.maximum(0, x)  

    def relu_derivative(self, x):  
        """Derivative of the ReLU activation function."""  
        return (x > 0).astype(float)  

    def softmax(self, x):  
        """Softmax activation function for multi-class classification."""  
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability improvement  
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)  

    def cross_entropy_loss(self, true_labels, predicted_probs):  
        """Calculate the cross-entropy loss."""  
        m = true_labels.shape[0]  # Number of samples  
        # Prevent log(0) by clipping predictions  
        predicted_probs = np.clip(predicted_probs, 1e-12, 1. - 1e-12)  
        log_likelihood = -np.log(predicted_probs[range(m), true_labels])  
        loss = np.sum(log_likelihood) / m  # Average loss  
        return loss  

    def predict_proba(self, features):  
        """Predict class probabilities for the given features."""  
        activation = features  
        for i in range(len(self.weights) - 1):  
            z = np.dot(activation, self.weights[i]) + self.biases[i]  # Linear transformation  
            activation = self.relu(z)  # Apply ReLU activation  
        z = np.dot(activation, self.weights[-1]) + self.biases[-1]  # Last layer linear transformation  
        activation = self.softmax(z)  # Apply softmax activation  
        return activation  

    def predict(self, features):  
        """Predict the class labels for the given features."""  
        probabilities = self.predict_proba(features)  
        return np.argmax(probabilities, axis=1)  # Return the index of the highest probability  

    def fit(self, features, labels):  
        """  
        Train the Multi-Layer Perceptron.  

        :param features: Input features, shape (number of samples, number of features)  
        :param labels: Labels, shape (number of samples,) and integer class labels  
        """  
        m = features.shape[0]  # Number of samples  

        # Convert labels to one-hot encoding  
        one_hot_labels = np.zeros((m, np.max(labels) + 1))  
        one_hot_labels[np.arange(m), labels] = 1  

        for epoch in range(self.n_epochs):  
            # Forward propagation  
            activations = [features]  # Store activations for each layer  
            zs = []  # Store linear outputs for each layer  
            activation = features  
            for i in range(len(self.weights) - 1):  
                z = np.dot(activation, self.weights[i]) + self.biases[i]  # Linear transformation  
                zs.append(z)  # Store linear output  
                activation = self.relu(z)  # Apply ReLU activation  
                activations.append(activation)  # Store activation  

            # Last layer uses softmax  
            z = np.dot(activation, self.weights[-1]) + self.biases[-1]  
            zs.append(z)  
            activation = self.softmax(z)  
            activations.append(activation)  

            # Calculate loss  
            loss = self.cross_entropy_loss(labels, activation)  

            # Backward propagation  
            delta = activation  
            delta[range(m), labels] -= 1  # Gradient of loss w.r.t. z for softmax & cross-entropy  
            delta /= m  # Average gradient  

            grad_weights = []  # List to hold gradients for weights  
            grad_biases = []  # List to hold gradients for biases  

            # Gradient for the last layer weights and biases  
            dw = np.dot(activations[-2].T, delta)  # Gradient for weights  
            db = np.sum(delta, axis=0, keepdims=True)  # Gradient for biases  
            grad_weights.insert(0, dw)  
            grad_biases.insert(0, db)  

            # Backpropagate through hidden layers  
            for i in range(len(self.weights) - 2, -1, -1):  
                delta = np.dot(delta, self.weights[i + 1].T) * self.relu_derivative(zs[i])  # Gradient for hidden layers  
                dw = np.dot(activations[i].T, delta)  # Gradient for weights  
                db = np.sum(delta, axis=0, keepdims=True)  # Gradient for biases  
                grad_weights.insert(0, dw)  
                grad_biases.insert(0, db)  

            # Update weights and biases  
            for i in range(len(self.weights)):  
                self.weights[i] -= self.learning_rate * grad_weights[i]  # Update weights  
                self.biases[i] -= self.learning_rate * grad_biases[i]  # Update biases  

            # Print loss every 10 epochs  
            if (epoch + 1) % 10 == 0 or epoch == 0:  
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {loss:.4f}")