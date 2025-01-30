"""  
Summary:  
This code implements the functionality to load, preprocess the MNIST dataset, and train and evaluate a Single Layer Perceptron model. The main features include:  

1. Data Loading: Uses the `fetch_openml` function to load the MNIST dataset and filter samples for digits 0 and 1.  
2. Data Preprocessing:  
   - Converts labels to binary (0 and 1).  
   - Normalizes feature data.  
   - Splits the dataset into training and testing sets.  
3. Model Training and Evaluation:  
   - Trains the Single Layer Perceptron (`SingleLayerPerceptron`) and predicts, outputting the model's accuracy.  

This code is suitable for learning and practicing machine learning and deep learning, particularly in image classification tasks.  
"""  

import numpy as np  
from sklearn.datasets import fetch_openml  
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score  
from perceptron_models import SingleLayerPerceptron 

def main():  
    # Load the MNIST dataset  
    print("Loading MNIST dataset...")  
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)  
    features, labels = mnist["data"], mnist["target"]  

    # Filter for digits 0 and 1  
    print("Filtering for digits 0 and 1...")  
    mask = (labels == '0') | (labels == '1')  
    features, labels = features[mask], labels[mask]  

    # Convert labels to binary  
    labels = np.where(labels == '0', 0, 1)  

    # Normalize feature data  
    features = features / 255.0  

    # Split the dataset into training and testing sets  
    print("Splitting the dataset into training and testing sets...")  
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)  

    # Initialize and train the perceptron model  
    print("Training the perceptron model...")  
    perceptron_model = SingleLayerPerceptron(learning_rate=0.01, n_iterations=1000)  
    perceptron_model.fit(features_train, labels_train)  

    # Make predictions  
    print("Making predictions...")  
    labels_predicted = perceptron_model.predict(features_test)  

    # Evaluate the model  
    accuracy = accuracy_score(labels_test, labels_predicted)  
    print(f"Model accuracy: {accuracy * 100:.2f}%")  

if __name__ == "__main__":  
    main()