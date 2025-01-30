"""  
摘要：  
该代码使用 Manim 和 TensorFlow 库创建一个动画场景，展示手写数字识别的多层感知机（MLP）工作原理。主要内容包括：  

1. MNIST 数据加载：从 MNIST 数据集中加载手写数字图像，并为每个数字选择一张图片。  
2. 像素组可视化：将选中的手写数字图像转换为像素组，并在屏幕上排列展示。  
3. 多层感知机构建：创建一个多层感知机的可视化表示，包括输入层、隐藏层和输出层。  
4. 激活值动画：通过动画展示输入数据如何在多层感知机中传播，展示每层的激活值变化。  
5. 连接动画：在数据传播过程中，展示各层之间的连接和激活状态的变化。  
 
"""  


from manim import *
import tensorflow as tf
from CustomFunction import *
import matplotlib.pyplot as plt
import numpy as np


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