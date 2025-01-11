from manim import *


class AbstractScene(Scene):
    def construct(self):
        z_equation = MathTex(r"z = \mathbf{w}^T \mathbf{x} + b").to_edge(UP).scale(0.8)

        plane_matrix_equation = MathTex(r"\mathbf{w}^T \mathbf{x} + b = 0")
        plane_matrix_equation.next_to(z_equation, DOWN, buff=1.5).scale(0.8)

        loss_function_new = MathTex(
            r"L = \max\left(0,\,-y\bigl(\mathbf{w}^T \mathbf{x} + b\bigr)\right)"
        )
        loss_function_new.next_to(z_equation, DOWN, buff=3).scale(0.8)

        gradient_update_w = MathTex(
            r"\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}"
        )
        gradient_update_b = MathTex(
            r"b \leftarrow b - \eta \frac{\partial L}{\partial b}"
        )

        gradient_updates = VGroup(gradient_update_w, gradient_update_b)
        gradient_updates.arrange(RIGHT, buff=0.5).scale(0.8)
        gradient_updates.next_to(z_equation, DOWN, buff=4.5)

        self.play(
            Write(z_equation),
            Write(plane_matrix_equation),
            Write(loss_function_new),
            Write(gradient_update_w),
            Write(gradient_update_b),
        )

        Group1 = VGroup(
            z_equation,
            plane_matrix_equation,
            loss_function_new,
            gradient_update_w,
            gradient_update_b,
        )
        # 设置颜色
        input_color = "#6ab7ff"
        receive_color = "#ff9e9e"

        # 创建神经元
        input_layer = VGroup(
            *[Circle(radius=0.2, fill_opacity=1, color=input_color) for _ in range(4)]
        )
        input_layer.arrange(DOWN, buff=0.7)

        receive_layer = Circle(radius=0.5, fill_opacity=1, color=receive_color).move_to(
            ORIGIN + UP
        )

        # 输出层
        output_layer = Circle(radius=0.2)

        # 排列层
        layers = VGroup(input_layer, receive_layer, output_layer)
        layers.arrange(RIGHT, buff=2.5)

        # 添加连接线
        connections = VGroup()
        for i, input_neuron in enumerate(input_layer):
            connection = Arrow(
                start=input_neuron.get_center(),
                end=receive_layer.get_center(),
                stroke_opacity=0.3,
                tip_length=0.2,
                buff=0.6,
            )
            connections.add(connection)

        # 添加输入和输出标签
        input_labels = VGroup(
            *[
                MathTex(f"x_{i + 1}").scale(0.8).move_to(input_neuron.get_center())
                for i, input_neuron in enumerate(input_layer)
            ]
        )

        theta = np.deg2rad(0)
        direction = np.array([np.cos(theta), np.sin(theta), 0])
        start_point = receive_layer.get_center()
        end_point = output_layer.get_center()
        # 创建箭头，方向由start到end，颜色为绿色
        arrow = Arrow(
            start=start_point,
            end=end_point,
            color=GREEN,
            stroke_opacity=0.6,
            tip_length=0.3,
            buff=0.6,
        )
        output_label_y = MathTex("y").scale(0.8).move_to(output_layer.get_center())

        self.play(FadeOut(Group1))
        self.play(Create(receive_layer))
        self.play(Create(connections))
        self.play(Write(input_labels))
        self.play(Create(arrow))
        self.play(Write(output_label_y))
        self.wait(2)

        # 创建新的输入层，包括省略号
        input_layer_n = VGroup(
            Circle(radius=0.2, fill_opacity=1, color=input_color),
            Circle(radius=0.2, fill_opacity=1, color=input_color),
            Tex("...").scale(0.8),  # 使用省略号替换第三个神经元
            Circle(radius=0.2, fill_opacity=1, color=input_color),
        )
        input_layer_n.arrange(DOWN, buff=0.7)
        input_layer_n.next_to(receive_layer, LEFT, buff=2.5)

        # 创建新的标签
        labels = VGroup()
        labels.add(MathTex("x_{1}").move_to(input_layer_n[0].get_center()).scale(0.8))
        labels.add(MathTex("x_{2}").move_to(input_layer_n[1].get_center()).scale(0.8))
        labels.add(MathTex("...").move_to(input_layer_n[2].get_center()).scale(0.8))
        labels.add(MathTex("x_{n}").move_to(input_layer_n[3].get_center()).scale(0.8))

        # 创建新的连接线，仅为圆形神经元添加箭头
        connections_n = VGroup()
        for input_neuron in input_layer_n:
            if isinstance(input_neuron, Circle):
                connection = Arrow(
                    start=input_neuron.get_right(),  # 使用 get_right() 使箭头从神经元右侧发出
                    end=receive_layer.get_left(),  # 使用 get_left() 连接到接收层左侧
                    stroke_opacity=0.3,
                    tip_length=0.2,
                    buff=0.6,
                    color=WHITE,
                )
                connections_n.add(connection)

        # 动画替换
        self.play(
            ReplacementTransform(
                Group(connections, input_labels), Group(connections_n, labels)
            )
        )
        self.wait(2)

        W_matrix = (
            MathTex(
                r"\begin{bmatrix} w_{1} & w_{1} & \dots  & w_{1} \end{bmatrix}^{T} "
            )
            .scale(0.8)
            .next_to(receive_layer, DOWN, buff=1.5)
        )
        X_matrix = (
            MathTex(r"\begin{bmatrix} x_{1} & x_{2} & \dots & x_{n} \end{bmatrix}^{T} ")
            .scale(0.8)
            .next_to(W_matrix, DOWN, buff=1)
        )
        self.play(Write(W_matrix))
        self.play(Write(X_matrix))
        self.wait(2)
