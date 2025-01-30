"""  
摘要：  
该代码使用 TensorFlow 和 Manim 库创建多个动画场景，展示手写数字识别的过程，特别是通过感知机进行二分类的示例。主要内容包括：  

1. 手写数字可视化：加载 MNIST 数据集中的手写数字图片，并将其转换为像素方块的形式进行展示。通过动画展示灰度值的变化和图像的尺寸标注。  

2. 简化像素行：将手写数字的像素行简化为一个长行，并展示其矩阵形式。  

3. 感知机结构：展示感知机的输入层和输出层，演示如何将输入数据（手写数字的像素）传递到感知机中进行分类。  

4. 二分类示例：从 MNIST 数据集中选择数字 0 和 1 的图片，展示感知机如何对这些图片进行分类。  
  
"""  


import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
from manim import *
from CustomClasses import Perceptron

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


class HandwritingVisualization(Scene):
    def construct(self):
        # 加载 MNIST 数据集并选择图片  
        image_1, label_1 = load_mnist_image(label_target=1, index=0)  
        image_0, label_0 = load_mnist_image(label_target=0, index=0)  

        print(f'灰度值: {image_1}')  

        # 创建并缩放像素组  
        pixel_group_1 = create_pixel_group(image_1).scale(0.8).to_edge(LEFT, buff=1)  
        pixel_group_0 = create_pixel_group(image_0).scale(0.8).to_edge(RIGHT, buff=1)  

        # 添加像素组到场景  
        self.play(Create(pixel_group_1), Create(pixel_group_0))  
        self.wait(2)  

        # 显示标题  
        title = self.create_title("如何用感知机识别手写数字")  
        self.play(Write(title))  
        self.wait(1)  
        self.play(FadeOut(title))  
        self.wait(1)  

        # 移动像素组  
        self.play(FadeOut(pixel_group_0), pixel_group_1.animate.move_to(ORIGIN))  
        self.wait(1)  

        # 显示尺寸标注  
        image_dimensions = {"width": 28, "height": 28}  
        self.display_image_dimensions(pixel_group_1, image_dimensions)  
        self.wait(1)  

        # 替换像素为灰度值文本  
        text_group = self.create_text_group(pixel_group_1, image_1)  
        self.play(FadeIn(text_group))  
        self.wait(1)  

        # 演示特定像素灰度变化  
        self.animate_selected_pixel(pixel_group_1, image_1, text_group, row=8, col=16)  
        self.wait(1)  

    def create_title(self, text):  
        """创建标题文本"""  
        return Text(text).to_edge(UP, buff=0.5).scale(0.6)  

    def display_image_dimensions(self, pixel_group, dimensions):  
        """显示图像的宽度和高度标注"""  
        braces = {  
            "width": Brace(pixel_group, DOWN),  
            "height": Brace(pixel_group, LEFT)  
        }  
        labels = {  
            "width": braces["width"].get_tex(f"W={dimensions['width']}"),  
            "height": braces["height"].get_tex(f"H={dimensions['height']}")  
        }  
        label_groups = {  
            key: VGroup(braces[key], labels[key]) for key in braces  
        }  

        self.play(FadeIn(braces["width"]), FadeIn(labels["width"]))  
        self.wait(1)  
        self.play(FadeIn(braces["height"]), Write(labels["height"]))  
        self.wait(1)  
        self.play(FadeOut(label_groups["width"]), FadeOut(label_groups["height"]))  

    def create_text_group(self, pixel_group, image):  
        """创建灰度值文本组"""  
        text_group = VGroup()  
        for i, square in enumerate(pixel_group):  
            row, col = divmod(i, 28)  
            gray_value = image[row, col] / 255  
            formatted_gray = f"{gray_value:.1f}"  
            text = Text(formatted_gray, font_size=8).move_to(square.get_center())  
            text.set_color(WHITE if gray_value < 0.5 else BLACK)  
            text_group.add(text)  
        return text_group  

    def animate_selected_pixel(self, pixel_group, image, text_group, row, col):  
        """动画展示选定像素的灰度变化，然后将像素返回原位并消除所有灰度值文本"""  
        index = row * 28 + col  
        selected_pixel = pixel_group[index]  
        initial_gray = image[row, col] / 255  
        gray_tracker = ValueTracker(initial_gray)  

        # 创建显示灰度值的文本  
        gray_text = DecimalNumber(  
            gray_tracker.get_value(),  
            num_decimal_places=1,  
            include_sign=False,  
            font_size=8  
        ).add_updater(lambda m: m.set_value(gray_tracker.get_value()))  
        gray_text.move_to(selected_pixel.get_center() + RIGHT * 4 + DOWN * 1).scale(2.5)  

        # 移动并放大选中的像素  
        self.play(  
            selected_pixel.animate.shift(RIGHT * 4 + DOWN * 1).scale(2.5)  
        )  
        self.add(gray_text)  
        self.remove(text_group[index])  

        # 更新文本颜色  
        gray_text.add_updater(  
            lambda d: d.set_color(WHITE if gray_tracker.get_value() < 0.5 else BLACK)  
        )  

        # 动画：灰度从初始值变为1.0（白色）  
        self.play(  
            gray_tracker.animate.set_value(1.0),  
            selected_pixel.animate.set_fill(WHITE, opacity=1),  
            run_time=2,  
        )  

        # 动画：灰度从1.0变为0.0（黑色）  
        self.play(  
            gray_tracker.animate.set_value(0.0),  
            selected_pixel.animate.set_fill(BLACK, opacity=1),  
            run_time=2,  
        )  

        # 回复初始状态并返回像素到原位  
        self.play(
            selected_pixel.animate.set_fill(WHITE, opacity=initial_gray)
                                .shift(LEFT * 4 + UP * 1)
                                .scale(1 / 2.5),
        )
        # 消除所有灰度值文本  
        self.play(FadeOut(text_group))

class SimplifyLongRow(Scene):  
    def construct(self):  
        # 加载 MNIST 数据集并选择第一张标签为1的图片  
        image, label = load_mnist_image(label_target=1, index=0)  

        # 创建像素组并播放创建动画  
        pixel_group = create_pixel_group(image)  
        self.play(Create(pixel_group))  
        self.wait(1)  

        # 排列像素组并播放转换动画  
        arranged_group = self.arrange_pixel_group(pixel_group)  
        self.play(ReplacementTransform(pixel_group, arranged_group))  
        self.wait(1)  

        # 将多行像素扁平化为一个长行  
        long_row = VGroup(*[pixel for row in arranged_group for pixel in row]).arrange(RIGHT, buff=0.05)  

        # 简化长行并播放转换动画  
        simplified_group = self.simplify_long_row(long_row)  
        self.play(ReplacementTransform(arranged_group, simplified_group))  
        self.wait(1)  

        # 添加表示像素数量的下方大括号  
        brace = Brace(simplified_group, DOWN)  
        brace_text = brace.get_text("728")  
        self.play(FadeIn(brace), FadeIn(brace_text))  
        self.wait(1)  

        # 将简化后的像素组变为矩阵并播放转换动画  
        matrix = MathTex("\\begin{bmatrix} x_{1} & x_{2} & \dots & x_{728} \\end{bmatrix}").scale(0.6)  
        matrix.move_to(simplified_group.get_center())  
        self.play(  
            ReplacementTransform(simplified_group, matrix),  
            FadeOut(brace),  
            FadeOut(brace_text)  
        )  
        self.wait(1)  
        self.play(matrix.animate.shift(UP))  

    def get_row_group(self, pixel_group, row_index, num_columns=28):  
        """获取指定行的像素组。  

        参数:  
            pixel_group (VGroup): 包含所有像素方块的组。  
            row_index (int): 要获取的行索引（从0开始）。  
            num_columns (int): 每行像素的数量。  

        返回:  
            VGroup: 指定行的像素方块组。  
        """  
        total_rows = len(pixel_group) // num_columns  
        if not 0 <= row_index < total_rows:  
            raise ValueError(f"行索引应在 0 到 {total_rows - 1} 之间。")  
        
        start = row_index * num_columns  
        end = start + num_columns  
        return VGroup(*pixel_group[start:end])  

    def arrange_pixel_group(self, pixel_group, center_row=None, num_columns=28):  
        """  
        将像素按指定行排列，指定行位于中心，其余行上下排列。  

        参数：  
            pixel_group (VGroup): 包含所有像素方块的组  
            center_row (int, optional): 指定的中心行索引。默认为中间行。  
            num_columns (int): 每行像素的数量  

        返回：  
            VGroup: 排列后的像素组  
        """  
        total_rows = len(pixel_group) // num_columns  
        center_row = center_row if center_row is not None else total_rows // 2  

        if not 0 <= center_row < total_rows:  
            raise ValueError(f"中心行索引应在 0 到 {total_rows - 1} 之间。")  
        
        rows = [self.get_row_group(pixel_group, i, num_columns) for i in range(total_rows)]  
        arranged_rows = VGroup(rows[center_row].copy())  

        for offset in range(1, max(center_row + 1, total_rows - center_row)):  
            if center_row - offset >= 0:  
                arranged_rows.add(rows[center_row - offset].copy().next_to(arranged_rows[-1], DOWN, buff=0))  
            if center_row + offset < total_rows:  
                arranged_rows.add(rows[center_row + offset].copy().next_to(arranged_rows[-1], DOWN, buff=0))  
                    
        return arranged_rows.arrange(RIGHT)  

    def simplify_long_row(self, long_row, left_keep=4, right_keep=4, pixel_size=0.5, spacing=0.05):  
        """  
        简化长行，仅保留左右指定数量的像素，中间用省略号表示。  

        参数：  
            long_row (VGroup): 长行像素组  
            left_keep (int): 保留左侧像素数量  
            right_keep (int): 保留右侧像素数量  
            pixel_size (float): 像素大小  
            spacing (float): 方块间距  

        返回：  
            VGroup: 简化后的像素组  
        """  
        total_pixels = len(long_row)  
        if total_pixels <= left_keep + right_keep:  
            return long_row.copy()  
        
        left_group = VGroup(*long_row[:left_keep]).copy()  
        right_group = VGroup(*long_row[-right_keep:]).copy()  
        ellipsis = Text("...", font_size=24).scale(pixel_size)  
        
        simplified = VGroup(left_group, ellipsis, right_group).arrange(RIGHT, buff=spacing)  
        return simplified

    
class MatrixToPerceptronScene(Scene):
    def construct(self):
        """  
        在场景中展示矩阵 X_matrix，并将其条目复制/移动到感知机的输入层位置。  
        最后显示感知机（Perceptron）。  
        """  
        # 1. 创建矩阵并显示  
        def custom_element_to_mobject(element):  
            return MathTex(element, font_size=36)  # 设置字体大小为24  

        # 创建矩阵并应用自定义字体大小  
        X_matrix = Matrix(  
            [  
                ["x_{1}", "x_{2}", "\\dots", "x_{784}"]  
            ],  
            h_buff=0.8,  
            bracket_h_buff=SMALL_BUFF,  
            bracket_v_buff=SMALL_BUFF,  
            element_to_mobject=custom_element_to_mobject  # 应用自定义函数  
        ).to_edge(UP).scale(0.8)  

        self.play(Create(X_matrix))
        self.wait(1)  

        # 2. 获取矩阵条目  
        #    假设矩阵总共有 4 个条目 (x_{1}, x_{2}, \dots, x_{784})  
        X_entries = X_matrix.get_entries()  

        # 3. 创建感知机，但是暂不在屏幕上显示  
        #    show_input_labels 和 show_input_circles 均为 False 时，默认不显示任何输入层可视化对象  
        perceptron = Perceptron(  
            show_input_labels=False,  
            show_input_circles=False,  
            n_inputs=784  
        )  

        # 4. 获取感知机输入层位置。由于不显示标签和圆圈，可在 Perceptron 中进行相应处理（或直接返回 input_layer_centers）  
        positions = perceptron.get_positions()  
        # 前四个位置分别对应 x_{1}, x_{2}, \dots, x_{784}  
        input_label_positions = positions["input_layer"][:4]  

        # 打印出输入层位置
        for idx, pos in enumerate(input_label_positions, start=1):
            print(f'输入层_{idx}：{pos}')
        # 5. 创建动画：复制矩阵条目并移动到感知机输入层预定位置  
        copied_entries = VGroup()  # 用于存放复制的条目  
        animations = []  
        for entry, target_pos in zip(X_entries, input_label_positions):  
            # 复制并移动到对应位置  
            copied_entry = entry.copy()  
            copied_entries.add(copied_entry)  
            animations.append(copied_entry.animate.move_to(target_pos))  
        
        # 一次性播放所有移动动画  
        self.play(*animations, run_time=1)  
        self.wait(1)  

        # 6. 在屏幕上创建并显示感知机  
        self.play(Create(perceptron))  
        self.wait(1)  

        # 7. 移除复制过去的矩阵条目  
        self.play(FadeOut(copied_entries))
        perceptron.enable_input_circles()
        self.wait(1)  

        # 8. 移动感知到指定位置
        self.play(perceptron.animate.to_edge(RIGHT, buff=1).scale(0.6))
        self.wait(1)


class PerceptronBinaryClassification(Scene):  
    def construct(self):  
        # 加载 MNIST 数据集  
        mnist = tf.keras.datasets.mnist  
        (image_train, label_train), (image_test, label_test) = mnist.load_data()
        
        # 挑选数字 0 和数字 1 的图片  
        zero_indices = np.where(label_train == 0)[0]  # 返回所有数字为 0 的索引  
        one_indices  = np.where(label_train == 1)[0]  # 返回所有数字为 1 的索引  

        # 各挑选 2 张数字 0 和数字 1 的图片  
        zero_images = image_train[zero_indices[:2]]  
        one_images  = image_train[one_indices[:2]]  

        # 将图片转换为灰度图  
        zero_images_gray = [create_pixel_group(img) for img in zero_images]  
        one_images_gray  = [create_pixel_group(img) for img in one_images]  

        # 合并并标注标签  
        labeled_images = [(img, 0) for img in zero_images_gray] + [(img, 1) for img in one_images_gray]  

        # 随机打乱顺序  
        np.random.shuffle(labeled_images)  

        # 分离图片和标签  
        images, labels = zip(*labeled_images)  

        # 创建感知机  
        perceptron = Perceptron(  
            show_input_labels=False,  
        )  
        perceptron.to_edge(RIGHT, buff=1)  # 将感知机放置在屏幕右侧  
        self.play(Create(perceptron))  
        self.wait(1)  

        for img, label in zip(images, labels):  
            # 在感知机左边添加图片  
            pixel_group = img  
            pixel_group.next_to(perceptron.input_circles, LEFT, buff=0.5).scale(0.6)  

            # 显示感知机输出层分类结果  
            output_label = MathTex(str(label), font_size=36, color=WHITE)  
            output_label.move_to(perceptron.output_layer.get_center())  
            perceptron.disable_output_label()  
            self.play(FadeIn(pixel_group), FadeIn(output_label))  
            self.wait(1)  
            self.play(FadeOut(pixel_group), FadeOut(output_label))  