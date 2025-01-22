from manim import *
import tensorflow as tf


def load_mnist_image(label_target=1, index=0):  
    """  
    加载 MNIST 数据集并选择指定标签的图片。  

    参数:  
        label_target (int): 目标标签（0-9）。  
        index (int): 选择的图片在目标标签中的索引。  

    返回:  
        image (np.ndarray): 选中的图片数组。  
        label (int): 图片的标签。  
    """  
    # 加载 MNIST 数据集  
    mnist = tf.keras.datasets.mnist  
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  
    images = np.concatenate((x_train, x_test), axis=0)  
    labels = np.concatenate((y_train, y_test), axis=0)  
    
    # 找到所有目标标签的索引  
    target_indices = np.where(labels == label_target)[0]  
    
    if index >= len(target_indices):  
        raise IndexError(f"标签为 {label_target} 的图片不足 {index + 1} 张。")  
    
    selected_index = target_indices[index]  
    image = images[selected_index]  
    label = labels[selected_index]  
    
    print(f'选择的图片索引: {selected_index}')  
    print(f'标签: {label}')  
    
    return image, label  

def create_pixel_group(image, pixel_size=0.2, spacing=0.05):  
    """  
    根据输入的图片数组创建一个像素组。  

    参数:  
        image (np.ndarray): 28x28 的灰度图片数组。  
        pixel_size (float): 每个像素方块的边长。  
        spacing (float): 方块之间的间距。  

    返回:  
        pixel_group (VGroup): 包含所有像素方块的组。  
    """  
    pixel_group = VGroup()  
    
    for i in range(28):  
        for j in range(28):  
            # 获取像素值（0-255），转换为灰度颜色并反转  
            pixel_value = image[i, j]  
            gray = pixel_value / 255  # 0为黑色，1为白色  
            
            # 创建一个方块代表像素  
            square = Square(  
                side_length=pixel_size,  
                fill_color=WHITE,  
                fill_opacity=gray,  # 根据灰度设置不透明度  
                stroke_width=0.2,  
                stroke_color=WHITE  
            )  
            square.shift(  
                (j - 14) * (pixel_size + spacing) * RIGHT +  
                (14 - i) * (pixel_size + spacing) * UP  
            )  
            pixel_group.add(square)  
    
    return pixel_group  