"""  
Summary:  
This code implements the loading, preprocessing, and training/evaluation of two different perceptron models on the MNIST dataset. The main functionalities include:  

1. Data Loading: Uses the `fetch_openml` function to load the MNIST dataset and filter samples for digits 0 and 1.  
2. Data Preprocessing:  
   - Converts labels to binary (0 and 1).  
   - Normalizes feature data.  
   - Splits the dataset into training and testing sets.  
3. Model Training and Evaluation:  
   - Trains a Single Layer Perceptron (`SinglePerceptron`) and outputs the model's accuracy.  
   - Trains a Multi-Layer Perceptron (`MultiLayerPerceptron`) and outputs the model's accuracy.  
"""  

import numpy as np  
from sklearn.datasets import fetch_openml  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from perceptron_models import SinglePerceptron, MultiLayerPerceptron  

def main_single_layer_perceptron():  
    """Load the MNIST dataset, preprocess it, and train a Single Layer Perceptron."""  
    
    # Load the MNIST dataset  
    print("Loading MNIST dataset...")  
    mnist_data = fetch_openml('mnist_784', version=1, as_frame=False)  
    features, labels = mnist_data["data"], mnist_data["target"]  

    # Filter for digits 0 and 1  
    print("Filtering for digits 0 and 1...")  
    filter_mask = (labels == '0') | (labels == '1')  
    features, labels = features[filter_mask], labels[filter_mask]  

    # Convert labels to binary  
    labels = np.where(labels == '0', 0, 1)  

    # Normalize feature data  
    features = features / 255.0  

    # Split the dataset into training and testing sets  
    print("Splitting the dataset into training and testing sets...")  
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)  

    # Initialize and train the Single Layer Perceptron  
    print("Training Single Layer Perceptron model...")  
    single_layer_perceptron = SinglePerceptron(learning_rate=0.01, n_iterations=1000)  
    single_layer_perceptron.fit(X_train, y_train)  

    # Make predictions  
    print("Making predictions...")  
    y_pred = single_layer_perceptron.predict(X_test)  

    # Evaluate the model  
    accuracy = accuracy_score(y_test, y_pred)  
    print(f"Single Layer Perceptron model accuracy: {accuracy * 100:.2f}%")  

def main_multi_layer_perceptron():  
    """Load the MNIST dataset, preprocess it, and train a Multi-Layer Perceptron."""  
    
    # Load the MNIST dataset  
    print("Loading MNIST dataset...")  
    mnist_data = fetch_openml('mnist_784', version=1, as_frame=False)  
    features, labels = mnist_data["data"], mnist_data["target"].astype(int)  

    # Normalize feature data  
    features = features / 255.0  

    # Split the dataset into training and testing sets  
    print("Splitting the dataset into training and testing sets...")  
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)  

    # Initialize the Multi-Layer Perceptron  
    input_size = X_train.shape[1]  # 784 features  
    hidden_layer_sizes = [128, 64]  # Adjustable hidden layer sizes  
    output_size = 10  # Digits 0-9  
    learning_rate = 0.01  
    n_epochs = 100  

    print("Initializing Multi-Layer Perceptron model...")  
    multi_layer_perceptron = MultiLayerPerceptron(input_size, hidden_layer_sizes, output_size, learning_rate, n_epochs)  

    # Train the model  
    print("Training Multi-Layer Perceptron model...")  
    multi_layer_perceptron.fit(X_train, y_train)  

    # Make predictions  
    print("Making predictions...")  
    y_pred = multi_layer_perceptron.predict(X_test)  

    # Evaluate the model  
    accuracy = accuracy_score(y_test, y_pred)  
    print(f"Multi-Layer Perceptron model accuracy: {accuracy * 100:.2f}%")  

if __name__ == "__main__":  
    # Train and evaluate the Single Layer Perceptron  
    main_single_layer_perceptron()  
    
    # Train and evaluate the Multi-Layer Perceptron  
    main_multi_layer_perceptron()