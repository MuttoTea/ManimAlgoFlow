from manim import *  
import importlib  
from concurrent.futures import ThreadPoolExecutor  
import os
# 文件名列表  
file_names = [  
    "Introduction",  
    "IrisData",  
    "Perceptron_ClassPrediction",  
    "Perceptron_ClassificationLogic",  
    "Perceptron_SeparationLine",  
    "Perceptron_FindWeights",  
    "Perceptron_ParamAdjustment",  
    "Perceptron_HighDimension",  
    "Perceptron_ModelBuilding",  
    "Perceptron_HandwritingRecognition",  
    "Perceptron_XORProblem",  
    "MultilayerPerceptron",  
    "MLP",  
]  

# 获取所有场景  
scenes = []  
for file in file_names:  
    module = importlib.import_module(file)  
    # 假设每个模块中都有一个名为 `scenes` 的列表，包含所有场景类  
    scenes.extend([getattr(module, scene_name) for scene_name in dir(module) if isinstance(getattr(module, scene_name), type) and issubclass(getattr(module, scene_name), Scene)])  

# 渲染单个场景的函数  
def render_scene(scene_class):  
    config.pixel_height = 2160
    config.pixel_width = 3840
    config.frame_rate = 60
    config.renderer= "opengl"
    
    # 设置导出文件夹和文件名  
    output_folder = "output"  # 你想要的输出文件夹  
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）  
    
    # 使用场景类的名称作为文件名  
    scene_name = scene_class.__name__  
    config.output_file = os.path.join(output_folder, scene_name)  # 设置输出文件路径  
    
    scene = scene_class()  
    scene.render()  

# 多线程渲染所有场景  
if __name__ == "__main__":  
    with ThreadPoolExecutor() as executor:  
        executor.map(render_scene, scenes) 