<div align="left" style="position: relative;">
<img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="right" width="35%" style="margin: -20px 0 0 20px;">
<h1>感知机</h1>

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

## 目录
- [简介](#简介)
- [项目结构](#项目结构)
- [安装](#安装)
- [使用](#使用)
- [贡献](#贡献)
- [许可证](#许可证)
- [联系方式](#联系方式)

## 简介

这是一个关于感知机（Perceptron）和多层感知机（MLP）的动画演示项目，旨在通过生动的动画将复杂的算法可视化，使其变得简单易懂。项目结合了 TensorFlow、Matplotlib 和 [Manim](https://github.com/manimCommunity/manim) 等强大的库，提供详细的动画演示和相关的算法代码，方便大家进行学习和交流。作为一个初学者，我希望这个项目能激发你对算法的兴趣，并帮助你在学习的过程中取得进步！

## 项目结构
```sh
└── preception/
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

## 安装 

### 直接安装

1. **克隆仓库**
   ```sh
   git clone https://github.com/MuttoTea/ManimAlgoFlow.git
   ```
   这条命令会将 ManimAlgoFlow 的代码库克隆到本地。

2. **安装依赖**
   - 使用 `requirements.txt` 文件安装所需的库：
   ```sh
   pip install -r requirements.txt
   ```
   或者，您可以手动安装特定版本的库：
   ```sh
   pip install manim==0.19.0 matplotlib==3.10.0 numpy==2.2.2 pandas==2.2.3 scikit_learn==1.6.1 sympy==1.13.1 tensorflow==2.18.0 tensorflow_intel==2.18.0 torch==2.5.1
   ```

### 创建虚拟环境

为了避免库之间的冲突，建议使用虚拟环境。以下是创建和使用虚拟环境的步骤：

1. **安装 Anaconda**
   - 如果您还没有安装 Anaconda，请访问 [Anaconda官网](https://www.anaconda.com/products/distribution) 下载并安装适合您操作系统的版本。

2. **创建虚拟环境**
   - 打开终端或 Anaconda Prompt，输入以下命令创建一个名为 `ManimProject` 的虚拟环境，并指定 Python 版本为 3.11：
   ```sh
   conda create -n ManimProject python=3.11
   ```

3. **激活虚拟环境**
   - 创建完成后，激活该虚拟环境：
   ```sh
   conda activate ManimProject
   ```

4. **安装依赖**
   - 在激活的虚拟环境中，使用 `requirements.txt` 文件安装所需的库：
   ```sh
   conda install --file requirements.txt
   ```
   注意：如果 `requirements.txt` 文件中包含 `pip` 安装的库，您可能需要使用 `pip` 安装这些库。

5. **检查安装是否成功**
   - 您可以通过以下命令查看已安装的库：
   ```sh
   conda list
   ```
   如果列表中显示 `manim`、`matplotlib`、`numpy`、`pandas`、`scikit-learn`、`sympy`、`tensorflow`、`torch` 等库，则表示安装成功。

### 其他注意事项

- **更新库**：如果需要更新某个库，可以使用 `pip install --upgrade <库名>` 或 `conda update <库名>`。
- **退出虚拟环境**：完成工作后，可以通过以下命令退出虚拟环境：
  ```sh
  conda deactivate
  ```
- 如果关于 Anaconda 的使用有任何问题，请参考 [Anaconda 官方文档](https://docs.anaconda.com/anaconda/)。
- 如果关于 Manim 的使用有任何问题，请参考 [Manim 官方文档](https://docs.manim.community/en/stable/)。

## 使用

在安装完成后，您可以运行项目中的 Python 脚本来查看感知机的动画演示。例如，您可以运行 `SinglePerceptron.py` 来查看单层感知机的演示。

## 贡献

如果你有任何建议或想要贡献代码，请随时提交 Pull Request 或创建 Issue。

## 许可证

本项目遵循 GPL 许可证。请查看 LICENSE 文件了解更多信息。

## 联系方式

如有任何问题或建议，请通过 GitHub Issues 或电子邮件 mzuwtd262@gmail.com 与我联系。