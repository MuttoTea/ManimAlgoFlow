"""  
Summary:  
This code defines an `IrisDataProcessor` class for processing the Iris dataset. The main functionalities include:  

1. Loading the Dataset: Uses the `load_iris` function to load the Iris dataset and converts it into a Pandas DataFrame format.  
2. Data Preprocessing:  
   - Maps target labels to corresponding Iris species names (Iris-setosa, Iris-versicolor, Iris-virginica).  
   - Removes data points where the sepal length is 4.5 cm.  
3. Data Extraction: Provides a `get_data` method that separates the data for Iris-setosa and Iris-versicolor, drops specific columns, and returns the features and labels.  

Note: Only the data for Iris-setosa and Iris-versicolor is used in this processing. Additionally, data points with a sepal length of 4.5 cm are removed for better demonstration of data separation.  
"""  

from sklearn.datasets import load_iris  
import pandas as pd  
import numpy as np  

class IrisDataProcessor:  
    def __init__(self):  
        """Initialize the IrisDataProcessor class and load the Iris dataset.  

        This constructor loads the Iris dataset using the `load_iris` function from  
        sklearn, converts it into a Pandas DataFrame, and performs initial data  
        preprocessing steps, including mapping species labels to their names and  
        removing specific data points.  
        """  
        # Load the Iris dataset  
        iris = load_iris()  
        
        # Create a DataFrame from the Iris dataset features  
        self.iris_dataframe = pd.DataFrame(data=iris.data, columns=iris.feature_names)  
        
        # Add a new column for species, mapping numerical labels to species names  
        self.iris_dataframe["species"] = iris.target  
        self.iris_dataframe["species"] = self.iris_dataframe["species"].map(  
            {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}  
        )  

        # Remove rows where the sepal length is equal to 4.5 cm  
        self.iris_dataframe = self.iris_dataframe[self.iris_dataframe["sepal length (cm)"] != 4.5]  

    def get_data(self):  
        """Extract and return features and labels for Iris-setosa and Iris-versicolor.  

        This method separates the data for Iris-setosa and Iris-versicolor species,  
        limits the number of samples to 50 for each species, drops specific columns,  
        and prepares the feature matrix (X) and label vector (y).  

        Returns:  
            X (numpy.ndarray): A 2D array containing the features (sepal length and sepal width).  
            y (numpy.ndarray): A 1D array containing the labels (0 for Iris-setosa, 1 for Iris-versicolor).  
        """  
        # Filter and select the first 50 samples of Iris-setosa  
        setosa_data = (  
            self.iris_dataframe[self.iris_dataframe["species"] == "Iris-setosa"]  
            .head(50)  
            .reset_index(drop=True)  # Reset index for consistency  
        )  
        
        # Filter and select the first 50 samples of Iris-versicolor  
        versicolor_data = (  
            self.iris_dataframe[self.iris_dataframe["species"] == "Iris-versicolor"]  
            .head(50)  
            .reset_index(drop=True)  # Reset index for consistency  
        )  

        # Drop the petal length and petal width columns from the data  
        setosa_features = setosa_data.drop(  
            ["petal length (cm)", "petal width (cm)"], axis=1  
        )  
        versicolor_features = versicolor_data.drop(  
            ["petal length (cm)", "petal width (cm)"], axis=1  
        )  

        # Concatenate the features of both species and ensure they are of float type  
        X = (  
            pd.concat([setosa_features, versicolor_features]).values[:, :-1].astype(float)  
        )  # Features: sepal length and sepal width  
        
        # Create labels: 0 for Iris-setosa and 1 for Iris-versicolor  
        y = np.concatenate(  
            [np.zeros(len(setosa_features)), np.ones(len(versicolor_features))]  
        ).astype(float)  # Labels  

        return X, y  # Return the feature matrix and label vector