import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
from manim import *


class HandwritingVisualization(Scene):
    def construct(self):
        # 加载 MNIST 数据集并选择图片  
        image_1, label_1 = self.load_mnist_image(label_target=1, index=0)  
        image_0, label_0 = self.load_mnist_image(label_target=0, index=0)

        print(f'灰度值: {image_1}')
        # 创建像素组  
        pixel_group_1 = self.create_pixel_group(image_1)  
        pixel_group_0 = self.create_pixel_group(image_0)
        
        # 缩放像素组  
        pixel_group_1.scale(0.8).to_edge(LEFT, buff=1)
        pixel_group_0.scale(0.8).to_edge(RIGHT, buff=1)
        
        # 添加像素组到场景  
        self.play(Create(pixel_group_1), Create(pixel_group_0))  
        self.wait(2)  
        
        title = Text("如何用感知机识别手写数字").to_edge(UP, buff=1)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        self.wait(1)
        self.play(FadeOut(pixel_group_0), pixel_group_1.animate.shift(RIGHT*2))
        self.wait(1)

        # 定义图片的长和宽  
        image_dimensions = {"width": 28, "height": 28}
        # 添加表示宽度的下方大括号  
        width_brace = Brace(pixel_group_1, DOWN)  
        width_label = width_brace.get_tex(f"W={image_dimensions['width']}")  
        width_group = VGroup(width_brace, width_label)  
        self.play(FadeIn(width_brace), Write(width_label))  
        self.wait(1)  
        
        # 添加表示高度的左侧大括号  
        height_brace = Brace(pixel_group_1, LEFT)  
        height_label = height_brace.get_tex(f"H={image_dimensions['height']}")  
        height_group = VGroup(height_brace, height_label)  
        self.play(FadeIn(height_brace), Write(height_label))  
        self.wait(1)  
        
        # 添加外框和替换像素为灰度值文本同时进行  
        # frame = SurroundingRectangle(pixel_group_1, buff=0.1, color=BLUE)  
        text_group = VGroup()  
        for i, square in enumerate(pixel_group_1):  
            # if i != 8 * 28 + 17:  # 排除选中的像素  
            row = i // 28  
            col = i % 28  
            gray_value = image_1[row, col] / 255  
            
            # 格式化灰度值为一位小数  
            formatted_gray_value = f"{gray_value:.1f}"  
            
            text = Text(formatted_gray_value, font_size=8).move_to(square.get_center())  
            
            if gray_value < 0.5:  
                text.set_color(WHITE)  
            else:  
                text.set_color(BLACK)  
            
            text_group.add(text)
        
        self.play(FadeIn(text_group))
        self.wait(1)  
        
        # 选择一个特定的像素点演示灰度变化
        select_row = 8
        select_col = 17
        selected_index = select_row * 28 + select_col   
        selected_pixel = pixel_group_1[selected_index]  # 选择特定的像素点
        selected_pixel_gray_value = ValueTracker(image_1[select_row, select_col] / 255)  # 创建一个值跟踪器来跟踪灰度值
        selected_pixel_text = DecimalNumber(
            selected_pixel_gray_value.get_value(), 
            num_decimal_places=1,
            include_sign=False,
            font_size=8
        )   # 创建一个数字对象来显示灰度值

        selected_pixel_text.add_updater(lambda m: m.set_value(selected_pixel_gray_value.get_value())) # 更新灰度值
        selected_pixel_text.move_to(selected_pixel.get_center()+RIGHT*4+DOWN*1) # 将文本移动到选中的像素点上
        selected_pixel_text.scale(2.5) # 将文本放大2.5倍与方块大小一致
        self.play(selected_pixel.animate.shift(RIGHT*4 + DOWN*1).scale(2.5))

        # 用新的文字来取代这个像素原本的文字便于后续动画演示
        self.add(selected_pixel_text) # 添加文本到场景中
        self.remove(text_group[selected_index]) # 移除原来的文本
        

        # 添加文本颜色更新器，确保对比度
        selected_pixel_text.add_updater(
            lambda d: d.set_color(
                WHITE if selected_pixel_gray_value.get_value() < 0.5 else BLACK
            )
        )

        # 颜色由初始颜色（灰色）变为白色（1.0）  
        self.play(  
            selected_pixel_gray_value.animate.set_value(1.0),  
            selected_pixel.animate.set_fill(WHITE, opacity=1),  # 仅更改填充颜色  
            run_time=2,  
        )  

        # 颜色由白色（1.0）变为黑色（0.0）  
        self.play(  
            selected_pixel_gray_value.animate.set_value(0.0),  
            selected_pixel.animate.set_fill(BLACK, opacity=1),  # 仅更改填充颜色  
            run_time=2  
        )

        self.wait(1)
        self.play(selected_pixel.animate.shift(LEFT*4 + UP*1).scale(1/2.5))
        self.play(FadeOut(selected_pixel_text))

    def load_mnist_image(self, label_target=1, index=0):  
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
    
    def create_pixel_group(self, image, pixel_size=0.2, spacing=0.05):  
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


class SimplifyLongRow(Scene):  
    def construct(self):  
        # 加载 MNIST 数据集并选择图片  
        image_1 = self.load_mnist_image(label_target=1, index=0)  

        # 创建像素组  
        pixel_group_1 = self.create_pixel_group(image_1)  
        self.play(Create(pixel_group_1))  
        print("像素组创建完成")
        self.wait(1)  

        transform_pixel_group = self.arrange_pixel_group(pixel_group_1)
        self.play(ReplacementTransform(pixel_group_1, transform_pixel_group))
        print("像素组排列完成")
        self.wait(1)

        # 将多行像素扁平化为一个长行  
        long_row = VGroup(*[pixel for row in transform_pixel_group for pixel in row]).arrange(RIGHT, buff=0.05)  
        print("像素合并完成")  

        # **打印 Long Row 的信息**  
        print(f"Long Row 中的像素数量: {len(long_row)}")  # 应为784  
        for idx, pixel in enumerate(long_row):  
            if idx < 10:  # 仅打印前10个像素的位置，避免过多输出  
                print(f"Pixel {idx}: Position = {pixel.get_center()}")  
            elif idx == 10:  
                print("...")  # 省略中间部分  
                break  

        simplified_pixel_group = self.simplify_long_row(long_row)
        print("像素简化完成")
        self.play(ReplacementTransform(transform_pixel_group, simplified_pixel_group))
        self.wait(1)

        # 添加表示像素个数的下方大括号
        pixel_num_brace = Brace(simplified_pixel_group, DOWN)
        pixel_num_text = pixel_num_brace.get_text("728")
        self.play(FadeIn(pixel_num_brace), FadeIn(pixel_num_text))
        self.wait(1)

        # 将简化后的像素组变为矩阵
        matrix = MathTex("\\begin{bmatrix} x_{1} & x_{2} & \dots  & x_{728} \\end{bmatrix}").scale(0.6)
        matrix.move_to(simplified_pixel_group.get_center())
        self.play(ReplacementTransform(simplified_pixel_group, matrix), FadeOut(pixel_num_brace), FadeOut(pixel_num_text))
        self.wait(1)
        self.play(matrix.animate.shift(UP))

        # # 添加感知机
        # perceptron = Perceptron(input_label_opacity=0.0)
        # perceptron.next_to(matrix, DOWN, buff=1.5)
        # self.play(Create(perceptron))
        # self.wait(1)

    def load_mnist_image(self, label_target=1, index=0):  
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

    def create_pixel_group(self, image, pixel_size=0.2, spacing=0.05):  
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
    
    def get_row_group(self, pixel_group, row_index, num_columns=28):
        """获取指定行的像素组。

        参数:
            pixel_group (VGroup): 包含所有像素方块的组。
            row_index (int): 要获取的行索引（从0开始）。
            num_columns (int): 每行像素的数量。

        返回:
            row_group (VGroup): 包含指定行的像素方块的组。
        """
        if row_index < 0 or row_index >= (len(pixel_group) // num_columns):
            raise ValueError(f"行索引应在 0 到 {len(pixel_group) // num_columns - 1} 之间。")
        
        start = row_index * num_columns
        end = start + num_columns
        selected_row = VGroup(*pixel_group[start:end])

        return selected_row
    
    def arrange_pixel_group(self, pixel_group, specified_row_index=None, num_columns=28):  
        """  
        将像素以指定行为中心行，其余行按顺序排列在该行的上下两侧。  

        参数：  
            pixel_group (VGroup): 包含所有像素方块的组  
            specified_row_index (int, optional): 指定的中心行索引。如果未指定，默认为中间行。  
            num_columns (int): 每行像素的数量  

        返回：  
            arranged_group (VGroup): 包含所有像素方块的组，已按指定行排列  
        """  
        arranged_group = VGroup()  

        total_rows = len(pixel_group) // num_columns  

        # 如果未指定中心行，默认为中间行  
        if specified_row_index is None:  
            specified_row_index = total_rows // 2  

        if specified_row_index < 0 or specified_row_index >= total_rows:  
            raise ValueError(f"指定的行索引应在 0 到 {total_rows - 1} 之间。")  
        
        # 获取所有行  
        rows = [self.get_row_group(pixel_group, i, num_columns) for i in range(total_rows)]  

        # 生成行的排列顺序：指定行，向上依次，向下依次  
        arranged_rows = VGroup()  
        arranged_rows.add(rows[specified_row_index].copy())  

        for offset in range(1, max(specified_row_index + 1, total_rows - specified_row_index)):  
            if specified_row_index - offset >= 0:  
                arranged_rows.add(rows[specified_row_index - offset].copy().next_to(arranged_rows[-1], DOWN, buff=0))  
            if specified_row_index + offset < total_rows:  
                arranged_rows.add(rows[specified_row_index + offset].copy().next_to(arranged_rows[-1], DOWN, buff=0))  
        
        # 重新排列所有行，使指定行位于顶部，其他行向下排列  
        arranged_group = arranged_rows.arrange(RIGHT)  

        return arranged_group  

    def simplify_long_row(self, long_row, left_pixels=4, right_pixels=4, pixel_size=0.5, spacing=0.05):  
        """  
        简化长行，左右两侧保留指定像素个数，其余用省略号代替。  

        参数：  
            long_row (VGroup): 长行像素方块组  
            left_pixels (int): 左侧保留的像素个数  
            right_pixels (int): 右侧保留的像素个数  
            pixel_size (float): 像素方块的大小  
            spacing (float): 方块之间的间距  

        返回：  
            simplified_row (VGroup): 简化后的像素方块组  
        """  
        # 检查参数是否合法  
        if left_pixels < 0 or right_pixels < 0 or not isinstance(left_pixels, int) or not isinstance(right_pixels, int):  
            raise ValueError("left_pixels 和 right_pixels 必须是非负整数")  
        
        num_pixels = len(long_row)  
        if num_pixels <= left_pixels + right_pixels:  
            return long_row.copy()  
        
        # 保留左右两侧的像素  
        left_group = VGroup(*long_row[:left_pixels]).copy()  
        right_group = VGroup(*long_row[-right_pixels:]).copy()  

        # 创建省略号，并根据像素大小调整缩放比例  
        ellipsis = Text("...", font_size=24).scale(pixel_size)  
        
        # 创建简化后的行  
        simplified_row = VGroup(left_group, ellipsis, right_group)  
        simplified_row.arrange(RIGHT, buff=spacing)  

        return simplified_row  
    
class MatrixToPerceptronScene(Scene):
    def construct(self):
        # 创建矩阵对象
        X_matrix = (
            Matrix(
                [["x_{1}", "x_{2}", "\dots", "x_{4}"]],
                h_buff=0.8,
                bracket_h_buff=SMALL_BUFF,
                bracket_v_buff=SMALL_BUFF,
            )
            .to_edge(UP)
            .scale(0.8)
        )
        self.play(Create(X_matrix))
        self.wait(1)

        # 获取矩阵条目，用于Transformations
        # Entry 0: MathTex('x_{1}')
        # Entry 1: MathTex('x_{2}')
        # Entry 2: MathTex('\\dots')
        # Entry 3: MathTex('x_{4}')
        X_entries = X_matrix.get_entries()

        # 打印出矩阵条目及对应索引
        for entry, index in zip(X_entries, range(len(X_entries))):
            print(f"Entry {index}: {entry}")
        
        # ==============================  
        # 1. 参数设置（和原先逻辑保持一致）  
        # ==============================  
        receive_center = ORIGIN  
        n_inputs = 4  # 可自行修改  
        input_line_color = "#6ab7ff"  
        input_line_tip_length = 0.2  
        input_line_opacity = 0.6  
        input_label_opacity = 0  
        receive_color = "#ff9e9e"  
        output_line_color = GREEN  
        output_line_tip_length = 0.3  
        output_line_opacity = 0.6  
        output_label_opacity = 1  

        # ==============================  
        # 2. 创建“接收层”（感知机主体）  
        # ==============================  
        receive_layer = Circle(radius=0.5, fill_opacity=1, color=receive_color)  
        receive_layer.move_to(receive_center)  

        # ==============================  
        # 3. 创建输出层及标签  
        # ==============================  
        output_layer = Circle(radius=0.2)  
        output_layer.next_to(receive_layer, RIGHT, buff=2.5)  
        output_label = MathTex("y").scale(0.8)  
        output_label.move_to(output_layer.get_center()).set_opacity(output_label_opacity)  

        # ==============================  
        # 4. 创建输入层及标签  
        #    （根据 n_inputs 判断是否显示省略号）  
        # ==============================  
        if n_inputs <= 3:  
            # 输入数量不超过3，全部显示  
            input_circles = VGroup(  
                *[  
                    Circle(radius=0.2, fill_opacity=1, color=input_line_color)  
                    for _ in range(n_inputs)  
                ]  
            )  
            input_circles.arrange(DOWN, buff=0.7)  
            input_circles.next_to(receive_layer, LEFT, buff=2.5)  

            input_labels = VGroup(  
                *[  
                    MathTex(f"x_{{{i+1}}}").scale(0.8)  
                    .move_to(neuron.get_center())  
                    .set_opacity(input_label_opacity)  
                    for i, neuron in enumerate(input_circles)  
                ]  
            )  

        else:  
            # 输入数量超过3，显示 x1, x2, ..., x_n  
            input_circles = VGroup(  
                Circle(radius=0.2, fill_opacity=1, color=input_line_color),  
                Circle(radius=0.2, fill_opacity=1, color=input_line_color),  
                Tex("...").scale(0.8),  
                Circle(radius=0.2, fill_opacity=1, color=input_line_color),  
            )  
            input_circles.arrange(DOWN, buff=0.7)  
            input_circles.next_to(receive_layer, LEFT, buff=2.5)  

            input_labels = VGroup(  
                MathTex("x_{1}").scale(0.8)  
                    .move_to(input_circles[0].get_center())  
                    .set_opacity(input_label_opacity),  
                MathTex("x_{2}").scale(0.8)  
                    .move_to(input_circles[1].get_center())  
                    .set_opacity(input_label_opacity),  
                MathTex("...").scale(0.8)  
                    .move_to(input_circles[2].get_center())  
                    .set_opacity(input_label_opacity),  
                MathTex(f"x_{{{n_inputs}}}").scale(0.8)  
                    .move_to(input_circles[3].get_center())  
                    .set_opacity(input_label_opacity),  
            )  

        # ==============================  
        # 5. 创建从输入层到接收层的连接箭头  
        # ==============================  
        connections = VGroup()  
        for obj in input_circles:  
            # 只有 Circle 才画箭头，“...”跳过  
            if isinstance(obj, Circle):  
                arrow = Arrow(  
                    start=obj.get_right(),  
                    end=receive_layer.get_center(),  
                    color=input_line_color,  
                    stroke_opacity=input_line_opacity,  
                    tip_length=input_line_tip_length,  
                    buff=0.6,  
                )  
                connections.add(arrow)  

        # ==============================  
        # 6. 创建从接收层到输出层的箭头  
        # ==============================  
        output_arrow = Arrow(  
            start=receive_layer.get_center(),  
            end=output_layer.get_center(),  
            color=output_line_color,  
            stroke_opacity=output_line_opacity,  
            tip_length=output_line_tip_length,  
            buff=0.6,  
        )  
        
        perceptron_group = VGroup(input_labels, receive_layer, output_label, connections, output_arrow)
        perceptron_group.scale(0.6)

        # 接下来，让矩阵中每个 MathTex 条目移动到对应输入标签那里  
        # 1. 如果要“复制”并移动，可使用 Transform(entry.copy(), label)  
        # 2. 如果直接移动现有的 entry，则使用 entry.animate.move_to(...)  
        # 下面以复制并变换示例：  
        animations = []  
        for entry, label in zip(X_entries, input_labels):  
            # 创建移动动画  
            animations.append(entry.copy().animate.move_to(label.get_center()))  
            # 可选：调整动画持续时间  
            # self.play(entry.animate.move_to(label.get_center()), run_time=1)  
        
        # 一次性播放所有动画  
        self.play(*animations, run_time=2)  
        self.wait(1)  

        self.play(Create(perceptron_group))
        self.wait(1)