Sure! Here’s the English version of your document:

---

<div align="left" style="position: relative;">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="right" width="35%" style="margin: -20px 0 0 20px;">
<h1>Perceptron</h1>

<p align="left">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/SymPy-3B5526.svg?style=default&logo=SymPy&logoColor=white" alt="SymPy">
	<img src="https://img.shields.io/badge/NumPy-013243.svg?style=default&logo=NumPy&logoColor=white" alt="NumPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/pandas-150458.svg?style=default&logo=pandas&logoColor=white" alt="pandas">
    <img src="https://img.shields.io/badge/Manim-1F4BB4.svg?style=default&logo=Manim&logoColor=white" alt="Manim">
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?style=default&logo=PyTorch&logoColor=white" alt="PyTorch">
    <img src="https://img.shields.io/badge/SciPy-8CA0E2.svg?style=default&logo=SciPy&logoColor=white" alt="SciPy">
    <img src="https://img.shields.io/badge/SciKit-Learn-FF5E26.svg?style=default&logo=SciKit-Learn&logoColor=white" alt="SciKit-Learn">
    <img src="https://img.shields.io/badge/GPLv3.0-221E1F.svg?style=default&logo=GPLv3&logoColor=white" alt="GPLv3.0">
</p>
</div>
<br clear="right"/>

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
	- [Direct Installation](#direct-installation)
	- [Create a Virtual Environment](#create-a-virtual-environment)
	- [Other Considerations](#other-considerations)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

This is an animation demonstration project about Perceptrons and Multi-Layer Perceptrons (MLP), aimed at visualizing complex algorithms through vivid animations, making them easier to understand. The project combines powerful libraries such as TensorFlow, Matplotlib, and [Manim](https://github.com/manimCommunity/manim) to provide detailed animations and related algorithm code, facilitating learning and communication. As a beginner, I hope this project can inspire your interest in algorithms and help you make progress in your learning journey!

## Project Structure
```sh
└── perception/
    ├── CustomClasses.py
    ├── CustomFunction.py
    ├── DataProcessor.py
    ├── Introduction.py
    ├── IrisData.py
    ├── media
    │   └── videos
    ├── MLP_DigitRecognizer.py
    ├── MlpVisualizationForMnist.py
    ├── MultilayerPerceptron.py
    ├── Perceptron_ClassificationLogic.py
    ├── Perceptron_ClassPrediction.py
    ├── Perceptron_FindWeights.py
    ├── Perceptron_HandwritingRecognition.py
    ├── Perceptron_HighDimension.py
    ├── Perceptron_ModelBuilding.py
    ├── perceptron_models.py
    ├── Perceptron_ParamAdjustment.py
    ├── Perceptron_SeparationLine.py
    ├── Perceptron_XORProblem.py
    ├── README.md
    ├── requirements.txt
    ├── SinglePerceptron.py
    └── SinglePerceptronDigitRecognizer.py
```

## Installation 

### Direct Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/MuttoTea/ManimAlgoFlow.git
   ```
   This command will clone the ManimAlgoFlow repository to your local machine.

2. **Install Dependencies**
   - Use the `requirements.txt` file to install the required libraries:
   ```sh
   pip install -r requirements.txt
   ```
   Alternatively, you can manually install specific versions of the libraries:
   ```sh
   pip install manim==0.19.0 matplotlib==3.10.0 numpy==2.2.2 pandas==2.2.3 scikit_learn==1.6.1 sympy==1.13.1 tensorflow==2.18.0 tensorflow_intel==2.18.0 torch==2.5.1
   ```

### Create a Virtual Environment

To avoid conflicts between libraries, it is recommended to use a virtual environment. Here are the steps to create and use a virtual environment:

1. **Install Anaconda**
   - If you haven't installed Anaconda yet, please visit the [Anaconda website](https://www.anaconda.com/products/distribution) to download and install the version suitable for your operating system.

2. **Create a Virtual Environment**
   - Open your terminal or Anaconda Prompt and enter the following command to create a virtual environment named `ManimProject` with Python version 3.11:
   ```sh
   conda create -n ManimProject python=3.11
   ```

3. **Activate the Virtual Environment**
   - After creating the environment, activate it:
   ```sh
   conda activate ManimProject
   ```

4. **Install Dependencies**
   - In the activated virtual environment, use the `requirements.txt` file to install the required libraries:
   ```sh
   conda install --file requirements.txt
   ```
   Note: If the `requirements.txt` file contains libraries that need to be installed via `pip`, you may need to use `pip` for those.

5. **Check Installation Success**
   - You can check the installed libraries with the following command:
   ```sh
   conda list
   ```
   If the list shows libraries like `manim`, `matplotlib`, `numpy`, `pandas`, `scikit-learn`, `sympy`, `tensorflow`, `torch`, etc., the installation was successful.

### Other Considerations

- **Updating Libraries**: If you need to update a library, you can use `pip install --upgrade <library_name>` or `conda update <library_name>`.
- **Exiting the Virtual Environment**: After completing your work, you can exit the virtual environment with:
  ```sh
  conda deactivate
  ```
- If you have any questions about using Anaconda, please refer to the [Anaconda official documentation](https://docs.anaconda.com/anaconda/).
- If you have any questions about using Manim, please refer to the [Manim official documentation](https://docs.manim.community/en/stable/).

## Usage

After installation, you can run the Python scripts in the project to view the animation demonstrations of the perceptron. For example, you can run `SinglePerceptron.py` to see the demonstration of a single-layer perceptron.

## Contributing

If you have any suggestions or would like to contribute code, please feel free to submit a Pull Request or create an Issue.

## License

This project is licensed under the GPL license. Please refer to the LICENSE file for more information.

## Contact

If you have any questions or suggestions, please contact me via GitHub Issues or email at mzuwtd262@gmail.com.
