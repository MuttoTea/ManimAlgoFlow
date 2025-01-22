from manim import *
import tensorflow as tf
# from CustomFunction import load_mnist_image, create_pixel_group
import matplotlib.pyplot as plt
import numpy as np

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

# # 加载MNIST数据集  
# mnist = tf.keras.datasets.mnist  
# (x_train, y_train), (x_test, y_test) = mnist.load_data()  

# # 每种手写数字各选一张  
# selected_images = []  
# selected_labels = []  

# for digit in range(10):  
#     try:  
#         image, label = load_mnist_image(label_target=digit, index=0)  
#         selected_images.append(image)  
#         selected_labels.append(label)  
#     except IndexError as e:  
#         print(e)  

# # 显示选中的图像  
# fig, axes = plt.subplots(1, 10, figsize=(15, 3))  
# for i in range(10):  
#     axes[i].imshow(selected_images[i], cmap='gray')  
#     axes[i].set_title(f"Digit {selected_labels[i]}")  
#     axes[i].axis('off')  

# plt.tight_layout()  
# plt.show()  

class MNISTExample(Scene):  
    def construct(self):  
        selected_images = []  
        selected_labels = []  
        
        # 每种手写数字各选一张  
        for digit in range(10):  
            try:  
                image, label = load_mnist_image(label_target=digit, index=0)  
                selected_images.append(image)  
                selected_labels.append(label)  
            except IndexError as e:  
                print(e)  
        
        # 为每个选中的图片创建像素组  
        pixel_groups = [create_pixel_group(img) for img in selected_images] 
        
        # 排列像素组在屏幕上  
        arranged_pixel_groups = VGroup(*pixel_groups).arrange_in_grid(rows=2, cols=5, buff=1).scale(0.3) 
        
        self.play(Create(arranged_pixel_groups)) 
        self.wait(0.5) 
        self.play(FadeOut(arranged_pixel_groups))
        self.wait(2)  

        # 创建多层感知机
        MLP = self.initialize_network()
        # 将字典转换为VGroup
        # 获取 MLP 组合中的所有元素  
        input_circles = MLP["input_circles"]  
        hidden_circles_1 = MLP["hidden_circles_1"]  
        hidden_circles_2 = MLP["hidden_circles_2"]  
        hidden_circles_3 = MLP["hidden_circles_3"]  
        output_circles = MLP["output_circles"]  
        connection_in_h1 = MLP["connection_in_h1"]  
        connection_h1_h2 = MLP["connection_h1_h2"]  
        connection_h2_h3 = MLP["connection_h2_h3"]  
        connection_h3_o = MLP["connection_h3_o"]  
        MLP_Group = VGroup(input_circles, hidden_circles_1, hidden_circles_2, hidden_circles_3, output_circles, connection_in_h1, connection_h1_h2, connection_h2_h3, connection_h3_o)
        # 打印 MLP_Group 的所有元素
        for element in MLP_Group:
            print(element)
        # 创建多层感知机
        self.play(Create(MLP_Group))
        self.wait(0.5)
        self.play(MLP_Group.animate.to_edge(RIGHT, buff=1))
        self.wait(0.5)

        for index, (img, label) in enumerate(zip(selected_images, selected_labels)):  
            # 获取激活值
            label = int(label)
            activation_values = self.activation(label, MLP, seed=label)  # 使用 label 作为数字

            input_circles_animation = set_activation(input_circles, activation_values["input"])  
            h1_circles_animation = set_activation(hidden_circles_1, activation_values["hidden_1"])  
            h2_circles_animation = set_activation(hidden_circles_2, activation_values["hidden_2"])  
            h3_circles_animation = set_activation(hidden_circles_3, activation_values["hidden_3"])
            output_circles_animation = set_activation(output_circles, activation_values["output"])
            
            img_group = create_pixel_group(img)
            img_group.scale(0.6).to_edge(LEFT, buff=1)
            # 播放动画
            self.play(FadeIn(img_group), *input_circles_animation, *h1_circles_animation, *h2_circles_animation, *h3_circles_animation, *output_circles_animation)
            self.play(FadeOut(img_group))
            self.wait(0.5)
        self.wait(2)

    # 设置MLP激活函数  
    def activation(self, number, MLP, seed=None):  
        # 设置随机种子  
        if seed is not None:  
            np.random.seed(seed)  

        input_circles = MLP["input_circles"]  
        hidden_circles_1 = MLP["hidden_circles_1"]  
        hidden_circles_2 = MLP["hidden_circles_2"]  
        hidden_circles_3 = MLP["hidden_circles_3"]  
        output_circles = MLP["output_circles"]  

        # 找到输入层中省略号的位置  
        non_circle_index = [  
            index for index, obj in enumerate(input_circles) if not isinstance(obj, Circle)  
        ]  
        ellipsis_index = non_circle_index[0] if non_circle_index else None  # 确保有省略号存在  

        # 生成随机激活值  
        activation_input_values = np.random.random(len(input_circles))  
        if ellipsis_index is not None:  
            activation_input_values[ellipsis_index] = -1  # 省略号激活值设为-1  

        activation_hidden_1_values = np.random.rand(len(hidden_circles_1))  
        activation_hidden_2_values = np.random.rand(len(hidden_circles_2))  
        activation_hidden_3_values = np.random.rand(len(hidden_circles_3))  

        # 根据 number 将对应的输出层的圆圈的激活值设为 1，其余在 0~0.8 之间随机生成   
        activation_output_values = np.random.rand(len(output_circles)) * 0.8  # 生成 0~0.8 之间的随机值  
        activation_output_values[number] = 1  # 将对应的输出层的圆圈激活值设为 1  

        return {  
            "input": activation_input_values,  
            "hidden_1": activation_hidden_1_values,  
            "hidden_2": activation_hidden_2_values,  
            "hidden_3": activation_hidden_3_values,  
            "output": activation_output_values  
        }

    def initialize_network(self):  
        # 生成各层的圆圈和省略号  
        input_circles, Icircles = generate_circles_with_vertical_ellipsis(  
            n=20,  # 示例：20个圆圈  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        input_circles.to_edge(LEFT, buff=3)  

        hidden_circles_1, Hcircles_1 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_1.next_to(input_circles, RIGHT*6)  

        hidden_circles_2, Hcircles_2 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_2.next_to(hidden_circles_1, RIGHT*6)  

        hidden_circles_3, Hcircles_3 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_3.next_to(hidden_circles_2, RIGHT*6)  

        output_circles, Ocircles = generate_circles_with_vertical_ellipsis(  
            n=10,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        output_circles.next_to(hidden_circles_3, RIGHT*6)  

        # 添加连接  
        connection_in_h1 = add_full_connections_between_groups(Icircles, Hcircles_1, connection_color=BLUE)  
        connection_h1_h2 = add_full_connections_between_groups(Hcircles_1, Hcircles_2, connection_color=BLUE)  
        connection_h2_h3 = add_full_connections_between_groups(Hcircles_2, Hcircles_3, connection_color=BLUE)  
        connection_h3_o = add_full_connections_between_groups(Hcircles_3, Ocircles, connection_color=BLUE)  

        # 创建 VGroup 包含所有连接线和圆圈  
        MLP_Group_VGroup = VGroup(  
            input_circles, hidden_circles_1, hidden_circles_2, hidden_circles_3,  
            output_circles, connection_h1_h2, connection_h2_h3, connection_h3_o,  
            connection_in_h1  
        )  
        MLP_Group_VGroup.scale(0.8)  
        # 将输出层圆圈添加对应的手写数字结果  
        numbers = VGroup()  
        for i, circle in enumerate(output_circles):  
            number = MathTex(str(i), font_size=24, color=BLUE).next_to(circle, RIGHT)  
            numbers.add(number)  

        # 组合所有元素  
        MLP_Group = {  
            "input_circles": input_circles,  
            "Icircles": Icircles,  
            "hidden_circles_1": hidden_circles_1,  
            "Hcircles_1": Hcircles_1,  
            "hidden_circles_2": hidden_circles_2,  
            "Hcircles_2": Hcircles_2,  
            "hidden_circles_3": hidden_circles_3,  
            "Hcircles_3": Hcircles_3,  
            "output_circles": output_circles,  
            "Ocircles": Ocircles,  
            "connection_in_h1": connection_in_h1,  
            "connection_h1_h2": connection_h1_h2,  
            "connection_h2_h3": connection_h2_h3,  
            "connection_h3_o": connection_h3_o,
            "numbers": numbers
        }  

        return MLP_Group  
    
# 根据激活值设置颜色和透明度  
def set_activation(circles, activation_values):  
    animations = []  
    for circle, value in zip(circles, activation_values):  
        if value == -1:  
            continue  # 省略号跳过  
        color = interpolate_color(BLACK, WHITE, value)  
        opacity = value  
        animations.append(circle.animate.set_fill(color, opacity=opacity))  
    return animations  

def generate_circles_with_vertical_ellipsis(  
    n,   
    radius=0.3,   
    spacing=0.8,   
    color=WHITE,  
    ellipsis_dots=3,  
    ellipsis_dot_radius=None,  
    ellipsis_buff=0.1,  
    ellipsis_color=None,  
    stroke_width=1.5  # 设置圆环厚度  
):  
    """  
    生成一列圆圈，超过16个时上下各八个，中间用垂直排列的省略号代替。  

    参数：  
    - n (int): 圆圈总数  
    - radius (float): 圆圈半径  
    - spacing (float): 圆圈之间的垂直间距  
    - color (str/Color): 圆圈和省略号的颜色  
    - ellipsis_dots (int): 省略号中点的数量，默认3  
    - ellipsis_dot_radius (float): 省略号中每个点的半径，默认radius/6  
    - ellipsis_buff (float): 省略号中点之间的垂直间距，默认0.1  
    - ellipsis_color (str/Color): 省略号中点的颜色，默认与圆圈颜色相同  
    - stroke_width (float): 圆圈边框的厚度，默认1.5  

    返回:  
    - elements (VGroup): 包含圆圈和省略号的组合  
    - circles (List[Circle]): 所有圆圈的列表  
    """  
    elements = VGroup()  
    circles = []  # 存储所有圆圈的列表  

    # 设置省略号点的半径和颜色  
    if ellipsis_dot_radius is None:  
        ellipsis_dot_radius = radius / 6  
    if ellipsis_color is None:  
        ellipsis_color = color  

    if n <= 16:  
        # 全部显示圆圈  
        for i in range(n):  
            circle = Circle(  
                radius=radius,   
                color=color,  
                stroke_width=stroke_width  # 设置圆环厚度  
            )  
            circle.number = i + 1  # 添加编号属性  
            circles.append(circle)  
            elements.add(circle)  
    else:  
        # 上八个圆圈  
        for i in range(8):  
            circle = Circle(  
                radius=radius,   
                color=color,  
                stroke_width=stroke_width  # 设置圆环厚度  
            )  
            circle.number = i + 1  # 添加编号属性  
            circles.append(circle)  
            elements.add(circle)  
        
        # 添加垂直排列的省略号  
        ellipsis = VGroup()  
        for _ in range(ellipsis_dots):  
            dot = Dot(  
                radius=ellipsis_dot_radius,   
                color=ellipsis_color  
            )  
            ellipsis.add(dot)  
        ellipsis.arrange(DOWN, buff=ellipsis_buff)  
        elements.add(ellipsis)  
        
        # 下八个圆圈  
        for i in range(n - 8, n):  
            circle = Circle(  
                radius=radius,   
                color=color,  
                stroke_width=stroke_width  # 设置圆环厚度  
            )  
            circle.number = i + 1  # 添加编号属性  
            circles.append(circle)  
            elements.add(circle)  

    # 设置圆圈之间的间距  
    elements.arrange(DOWN, buff=spacing / 4)  
    return elements, circles  


def add_full_connections_between_groups(  
    group1_circles,   
    group2_circles,   
    connection_color=GRAY,   
    stroke_width=0.5,   
    buff=0.1  
):  
    """  
    在两组圆圈之间添加全连接。  

    参数和返回值同上。  

    """  
    lines = VGroup()  
    for circle1 in group1_circles:  
        for circle2 in group2_circles:  
            line = Line(  
                circle1.get_right() + RIGHT * buff,  
                circle2.get_left() + LEFT * buff,  
                color=connection_color,  
                stroke_width=stroke_width  
            )  
            lines.add(line)  
    return lines


        

class NeuralNetworkAnimation(Scene):  
    def construct(self):  
        # 初始化网络并获取网络组  
        MLP_Group = self.initialize_network()  
        
        # 执行数据传播动画  
        self.data_propagation_animation(MLP_Group)  
        
        self.wait(2)  

    def initialize_network(self):  
        # 生成各层的圆圈和省略号  
        input_circles, Icircles = generate_circles_with_vertical_ellipsis(  
            n=20,  # 示例：20个圆圈  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        input_circles.to_edge(LEFT, buff=3)  

        hidden_circles_1, Hcircles_1 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_1.next_to(input_circles, RIGHT*6)  

        hidden_circles_2, Hcircles_2 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_2.next_to(hidden_circles_1, RIGHT*6)  

        hidden_circles_3, Hcircles_3 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_3.next_to(hidden_circles_2, RIGHT*6)  

        output_circles, Ocircles = generate_circles_with_vertical_ellipsis(  
            n=10,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        output_circles.next_to(hidden_circles_3, RIGHT*6)  

        # 添加连接  
        connection_in_h1 = add_full_connections_between_groups(Icircles, Hcircles_1, connection_color=BLUE)  
        connection_h1_h2 = add_full_connections_between_groups(Hcircles_1, Hcircles_2, connection_color=BLUE)  
        connection_h2_h3 = add_full_connections_between_groups(Hcircles_2, Hcircles_3, connection_color=BLUE)  
        connection_h3_o = add_full_connections_between_groups(Hcircles_3, Ocircles, connection_color=BLUE)  

        # 组合所有元素  
        MLP_Group = {  
            "input_circles": input_circles,  
            "Icircles": Icircles,  
            "hidden_circles_1": hidden_circles_1,  
            "Hcircles_1": Hcircles_1,  
            "hidden_circles_2": hidden_circles_2,  
            "Hcircles_2": Hcircles_2,  
            "hidden_circles_3": hidden_circles_3,  
            "Hcircles_3": Hcircles_3,  
            "output_circles": output_circles,  
            "Ocircles": Ocircles,  
            "connection_in_h1": connection_in_h1,  
            "connection_h1_h2": connection_h1_h2,  
            "connection_h2_h3": connection_h2_h3,  
            "connection_h3_o": connection_h3_o  
        }  

        # 创建 VGroup 包含所有连接线和圆圈  
        MLP_Group_VGroup = VGroup(  
            input_circles, hidden_circles_1, hidden_circles_2, hidden_circles_3,  
            output_circles, connection_h1_h2, connection_h2_h3, connection_h3_o,  
            connection_in_h1  
        )  
        MLP_Group_VGroup.scale(0.8)  

        # 为输入层添加大括号显示总数, 总共784个像素点  
        input_brace = Brace(input_circles, LEFT, stroke_width=0.4)  
        input_brace_text = input_brace.get_text("784").scale(0.6)  
        input_brace.add(input_brace_text)  

        # 将输出层圆圈添加对应的手写数字结果  
        numbers = VGroup()  
        for i, circle in enumerate(output_circles):  
            number = MathTex(str(i), font_size=24, color=BLUE).next_to(circle, RIGHT)  
            numbers.add(number)  

        # 动画展示  
        self.play(Create(input_circles))  
        self.play(FadeIn(input_brace))  
        self.wait(0.5)  
        self.play(FadeOut(input_brace))  
        self.wait(0.2)  
        self.play(FadeIn(hidden_circles_1), FadeIn(connection_in_h1))
        self.wait(0.2)
        self.play(FadeIn(hidden_circles_2), FadeIn(connection_h1_h2))
        self.wait(0.2)
        self.play(FadeIn(hidden_circles_3), FadeIn(connection_h2_h3))
        self.wait(0.2)
        self.play(FadeIn(output_circles), FadeIn(connection_h3_o))
        self.wait(0.5)
        self.play(FadeIn(numbers))  
        self.wait(1)  

        return MLP_Group  

    def data_propagation_animation(self, MLP_Group):  
        """  
        数据传播动画  
        """  
        # 获取 MLP 组合中的所有元素  
        input_circles = MLP_Group["input_circles"]  
        hidden_circles_1 = MLP_Group["hidden_circles_1"]  
        hidden_circles_2 = MLP_Group["hidden_circles_2"]  
        hidden_circles_3 = MLP_Group["hidden_circles_3"]  
        output_circles = MLP_Group["output_circles"]  
        connection_in_h1 = MLP_Group["connection_in_h1"]  
        connection_h1_h2 = MLP_Group["connection_h1_h2"]  
        connection_h2_h3 = MLP_Group["connection_h2_h3"]  
        connection_h3_o = MLP_Group["connection_h3_o"]  

        # 找到输入层中省略号的位置  
        non_circle_index = [  
            index for index, obj in enumerate(input_circles) if not isinstance(obj, Circle)  
        ]  
        ellipsis_index = non_circle_index[0]  

        # 生成随机激活值  
        activation_input_values = [np.random.random() for _ in range(len(input_circles))]  
        activation_input_values[ellipsis_index] = -1  # 省略号激活值为-1  

        activation_hidden_1_values = np.random.rand(len(hidden_circles_1))  
        activation_hidden_2_values = np.random.rand(len(hidden_circles_2))  
        activation_hidden_3_values = np.random.rand(len(hidden_circles_3))  
        activation_output_values = np.random.rand(len(output_circles))  

        # 定义连接脉冲动画  
        def play_pulse(connection):  
            # 每条线段都添加脉冲，同时播放
            wave_animations = []
            for line in connection:
                sublines, wave_animation = animate_line_wave(line, wave_type="pulse",stroke_width=0.3)
                wave_animations.append(wave_animation)
            return wave_animations
        # 分步骤播放动画  
        # 1. 输入层激活  
        self.play(*set_activation(input_circles, activation_input_values), run_time=1)  

        # 2. 输入层到隐藏层1的连接脉冲  
        I_h1_pulse=play_pulse(connection_in_h1)  
        self.play(*I_h1_pulse, run_time=1)

        # 3. 隐藏层1激活  
        self.play(*set_activation(hidden_circles_1, activation_hidden_1_values), run_time=1)  

        # 4. 隐藏层1到隐藏层2的连接脉冲  
        H1_H2_pulse=play_pulse(connection_h1_h2)
        self.play(*H1_H2_pulse, run_time=1)
  
        # 5. 隐藏层2激活  
        self.play(*set_activation(hidden_circles_2, activation_hidden_2_values), run_time=1)  

        # 6. 隐藏层2到隐藏层3的连接脉冲  
        H2_H3_pulse=play_pulse(connection_h2_h3)
        self.play(*H2_H3_pulse, run_time=1)

        # 7. 隐藏层3激活  
        self.play(*set_activation(hidden_circles_3, activation_hidden_3_values), run_time=1)  

        # 8. 隐藏层3到输出层的连接脉冲  
        H3_O_pulse=play_pulse(connection_h3_o)
        self.play(*H3_O_pulse, run_time=1)
 
        # 9. 输出层激活  
        self.play(*set_activation(output_circles, activation_output_values), run_time=1)  

        # 在 MLP 上下个添加一个箭头分别表示正向传播和反向传播
        # 获取圆圈的中心位置  
        forward_arrow_start = hidden_circles_1[0].get_center() + 3 * LEFT
        forward_arrow_end = hidden_circles_3[0].get_center() + 3 * RIGHT

        # 创建正向箭头  
        forward_arrow = Arrow(forward_arrow_start, forward_arrow_end, color=RED, buff=1).to_edge(UP, buff=0.75)  

        # 反向箭头的起点和终点  
        backward_arrow_start = hidden_circles_3[0].get_center() + 3 * RIGHT
        backward_arrow_end = hidden_circles_1[0].get_center() + 3 * LEFT

        # 创建反向箭头  
        backward_arrow = Arrow(backward_arrow_start, backward_arrow_end, color=RED, buff=1).to_edge(DOWN, buff=0.75)  

        # 播放动画  
        self.play(Create(backward_arrow), Create(forward_arrow), run_time=1)  
        self.wait(1)
        
        # 去掉箭头以及将圆圈恢复到初始状态
        self.play(
            FadeOut(backward_arrow), 
            FadeOut(forward_arrow), 
            *set_activation(input_circles, [0]*len(input_circles)),
            *set_activation(hidden_circles_1, [0]*len(hidden_circles_1)),
            *set_activation(hidden_circles_2, [0]*len(hidden_circles_2)),
            *set_activation(hidden_circles_3, [0]*len(hidden_circles_3)),
            run_time=1
        )
        self.wait(1)

def split_line_into_sublines(line, num_subsegments=100, overlap=0.01, stroke_width=2):  
    """  
    将线段分割成多个子线段，并增加重叠以减少分界线的明显性。  
    """  
    sublines = VGroup(*[  
        Line(  
            line.point_from_proportion(i / num_subsegments),  
            line.point_from_proportion((i + 1) / num_subsegments) + overlap * RIGHT,  
            color=BLUE,  
            stroke_width=stroke_width  
        ) for i in range(num_subsegments)  
    ])  
    return sublines  

def continuous_color_wave(mob, alpha):  
    """  
    定义持续颜色波动的函数。  
    颜色从蓝色逐渐变化到黄色，再回到蓝色，循环往复。  
    """  
    phase = alpha * TAU  
    gradient_colors = color_gradient([BLUE, YELLOW, BLUE], 100)  
    num_subsegments = len(mob)  
    
    for i in range(num_subsegments):  
        t = i / num_subsegments  
        offset = (t * TAU + phase) % TAU  
        color_value = (np.sin(offset) + 1) / 2  # 范围 [0,1]  
        color_index = int(color_value * (len(gradient_colors) - 1))  
        current_color = gradient_colors[color_index]  
        mob[i].set_color(current_color)  

def single_pulse_wave(mob, alpha):  
        """  
        让脉冲中心从 -pulse_width 跑到 1 + pulse_width，  
        确保动画结束时脉冲尾部也能到达线段末端。  
        """  
        pulse_width = 0.05  
        num_subsegments = len(mob)  

        # 将 alpha 的 [0, 1] 映射到脉冲中心的 [-pulse_width, 1 + pulse_width]  
        current_pulse = -pulse_width + alpha * (1 + pulse_width - (-pulse_width))  
        # 或者直接： current_pulse = (alpha * (1 + 2 * pulse_width)) - pulse_width  

        for i in range(num_subsegments):  
            t = i / num_subsegments  
            if current_pulse - pulse_width < t < current_pulse + pulse_width:  
                mob[i].set_color(YELLOW)  
            else:  
                mob[i].set_color(BLUE)    

def animate_line_wave(line, wave_type="continuous",   
                        num_subsegments=100, overlap=0.01, stroke_width=2):  
    """  
    接受一条Line，将其分段并根据 wave_type 参数播放相应动画。  
    wave_type 支持 "continuous"（持续波动）或 "pulse"（脉冲波动）。  
    """  
    # 1. 拆分线段   
    sublines = split_line_into_sublines(  
        line, num_subsegments, overlap, stroke_width  
    )  
    
    # 2. 根据类型选择动画更新函数  
    if wave_type == "continuous":  
        wave_func = continuous_color_wave  
    elif wave_type == "pulse":  
        wave_func = single_pulse_wave  
    else:  
        raise ValueError("wave_type 只能是 'continuous' 或 'pulse'。")  
    
    # 3. 创建动画（UpdateFromAlphaFunc 会在播放时连续调用 wave_func）  
    line_wave_animation = UpdateFromAlphaFunc(  
        sublines,   
        lambda m, alpha: wave_func(m, alpha)  
    )  
    
    return sublines, line_wave_animation  


if __name__ == '__main__':
    # 设置渲染配置
    # 8k
    # config.pixel_height = 7680
    # config.pixel_width = 15360
    # 1080p
    config.pixel_height = 1080
    config.pixel_width = 1920
    config.frame_rate = 30
    # config.renderer = 'opengl'
    scene = MNISTExample()
    scene.render(preview=True)
