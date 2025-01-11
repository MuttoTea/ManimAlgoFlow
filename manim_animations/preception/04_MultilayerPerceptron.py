from manim import *
from CustomClasses import XorGate


class XorCoordinateSystem(Scene):
    def construct(self):
        title = Text("异或问题").set_color(RED).to_edge(UP, buff=0.5)

        # 创建一个 XorGate，放在原点
        input_color = GREEN
        output_color = BLUE
        xor_gate = XorGate(
            center=ORIGIN - RIGHT,
            arcs_color=RED,  # 弧线颜色
            input_line_color=input_color,  # 输入线颜色
            output_line_color=output_color,  # 输出线颜色
            fill_color=RED,  # 填充颜色
            fill_opacity=0.3,  # 填充透明度
        ).scale(0.8)

        input_label_x1 = (
            MathTex("x_1")
            .move_to(xor_gate.input_point_up.get_center() - RIGHT * 0.5)
            .set_color(input_color)
        )
        input_label_x2 = (
            MathTex("x_2")
            .move_to(xor_gate.input_point_down.get_center() - RIGHT * 0.5)
            .set_color(input_color)
        )
        input_label_y = (
            MathTex("y")
            .move_to(xor_gate.output_point.get_center() + RIGHT * 0.5)
            .set_color(output_color)
        )

        Group_1 = VGroup(xor_gate, input_label_x1, input_label_x2, input_label_y)
        Group_1.scale(0.9).to_edge(LEFT, buff=1)

        table_data = [
            ["X_{1}", "X_{2}", "X_{1} \\oplus X_{2}"],  # 标题行
            [1, 1, 0],
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
        ]

        table = Table(
            table_data,
            include_outer_lines=True,
            element_to_mobject=lambda x: (
                MathTex(x) if isinstance(x, str) else MathTex(str(x))
            ),
        )

        table.scale(0.7)  # 将表格缩放至适当大小
        table.to_edge(RIGHT, buff=1.0)  # 将表格移动到屏幕右侧

        y_1_cells = []
        y_0_cells = []

        for row in range(2, len(table_data) + 1):
            cell_y = table.get_entries((row, 3))
            y_value = table_data[row - 1][2]
            if y_value == 1:
                y_1_cells.append(cell_y)
            else:
                y_0_cells.append(cell_y)

        # 创建对应颜色的点
        dots_y1 = [
            Dot(color=RED, radius=0.15).move_to(cell.get_center()).scale(0.7)
            for cell in y_1_cells
        ]
        dots_y0 = [
            Dot(color=YELLOW, radius=0.15).move_to(cell.get_center()).scale(0.7)
            for cell in y_0_cells
        ]

        # 直接替换 y=1 的单元格为红点
        for cell, dot in zip(y_1_cells, dots_y1):
            cell.remove(*cell.submobjects)  # 移除原有内容
            cell.add(dot)  # 添加红色点

        # 直接替换 y=0 的单元格为黄点
        for cell, dot in zip(y_0_cells, dots_y0):
            cell.remove(*cell.submobjects)  # 移除原有内容
            cell.add(dot)

        # 创建并保存对应颜色的点为独立变量
        dot_y0_row2 = (
            Dot(color=YELLOW, radius=0.15).move_to(y_0_cells[0].get_center()).scale(0.7)
        )
        dot_y0_row3 = (
            Dot(color=YELLOW, radius=0.15).move_to(y_0_cells[1].get_center()).scale(0.7)
        )
        dot_y1_row4 = (
            Dot(color=RED, radius=0.15).move_to(y_1_cells[0].get_center()).scale(0.7)
        )
        dot_y1_row5 = (
            Dot(color=RED, radius=0.15).move_to(y_1_cells[1].get_center()).scale(0.7)
        )

        self.play(
            FadeIn(title),
            FadeIn(Group_1),
            FadeIn(table),
            FadeIn(dot_y0_row2),
            FadeIn(dot_y0_row3),
            FadeIn(dot_y1_row4),
            FadeIn(dot_y1_row5),
        )
        self.wait(1)

        self.play(FadeOut(title), FadeOut(Group_1), FadeOut(table))

        axes = (
            Axes(
                x_range=[-0.5, 1.5, 1],
                y_range=[-0.5, 1.5, 1],
                axis_config={"include_numbers": True},
                tips=True,
            )
            .to_edge(DOWN, buff=1)
            .scale(0.6)
        )

        axes_labels = axes.get_axis_labels(x_label="x_1", y_label="x_2")
        self.play(FadeIn(axes), FadeIn(axes_labels))
        self.wait(1)

        # ── 10. 定义点的新位置并移动到坐标轴上 ──────────────────────────────
        # 定义每个点对应的 (x1, x2) 值
        points_positions = {
            dot_y0_row2: (1, 1),
            dot_y0_row3: (0, 0),
            dot_y1_row4: (0, 1),
            dot_y1_row5: (1, 0),
        }

        # 将所有点移动到坐标轴上的目标位置
        self.play(
            *[
                dot.animate.move_to(axes.c2p(x1, x2))
                for dot, (x1, x2) in points_positions.items()
            ],
            run_time=2,
        )
        self.wait(1)

        # 绘制函数图像

        graph1 = axes.plot(
            lambda x: self.func1(x), x_range=[-0.5, 1.5, 0.01], color=BLUE
        )
        graph2 = axes.plot(lambda x: self.func2(x), color=BLUE)
        graph3 = axes.plot(lambda x: self.func3(x), color=BLUE)
        graph4 = axes.plot(
            lambda x: self.func4(x), x_range=[-0.5, 1.5, 0.01], color=BLUE
        )

        self.play(Create(graph1))
        self.wait(0.5)
        self.play(FadeOut(graph1))

        self.play(Create(graph2))
        self.wait(0.5)
        self.play(FadeOut(graph2))

        self.play(Create(graph3))
        self.wait(0.5)
        self.play(FadeOut(graph3))

        self.play(Create(graph1))
        self.wait(1)
        self.play(Create(graph4))
        self.wait(1)

        Group_2 = VGroup(
            axes,
            graph1,
            graph4,
            dot_y0_row2,
            dot_y0_row3,
            dot_y1_row4,
            dot_y1_row5,
            axes_labels,
        )
        self.play(Group_2.animate.shift(UP * 2.5).scale(0.7))
        self.wait(1)

        # 设置颜色
        input_color = "#6ab7ff"
        receive_color = "#ff9e9e"

        ## 创建神经元1
        # 输入层
        input_layer_1 = VGroup(
            *[Circle(radius=0.2, fill_opacity=1, color=input_color) for _ in range(2)]
        )
        input_layer_1.arrange(DOWN, buff=1.5)

        # 接受层
        receive_layer_1 = Circle(
            radius=0.5, fill_opacity=1, color=receive_color
        ).move_to(ORIGIN + UP)

        # 输出层
        output_layer_1 = Circle(radius=0.2)

        # 排列层
        layers_1 = VGroup(input_layer_1, receive_layer_1, output_layer_1)
        layers_1.arrange(RIGHT, buff=2.5)

        # 添加连接线
        connections_1 = VGroup()
        for i, input_neuron in enumerate(input_layer_1):
            connection = Arrow(
                start=input_neuron.get_center(),
                end=receive_layer_1.get_center(),
                stroke_opacity=0.3,
                tip_length=0.2,
                buff=0.6,
            )
            connections_1.add(connection)

        # 添加输入和输出标签
        input_labels_1 = VGroup(
            *[
                MathTex(f"x_{i + 1}").move_to(input_neuron.get_center())
                for i, input_neuron in enumerate(input_layer_1)
            ]
        )

        theta = np.deg2rad(0)
        direction = np.array([np.cos(theta), np.sin(theta), 0])
        start_point = receive_layer_1.get_center()
        end_point = output_layer_1.get_center()
        # 创建箭头，方向由start到end，颜色为绿色
        arrow_1 = Arrow(
            start=start_point,
            end=end_point,
            color=GREEN,
            stroke_opacity=0.6,
            tip_length=0.3,
            buff=0.6,
        )
        output_label_y_1 = MathTex("y").move_to(output_layer_1.get_center())
        p1 = VGroup(
            input_labels_1,
            connections_1,
            receive_layer_1,
            arrow_1,
            receive_layer_1,
            output_label_y_1,
        )

        ## 创建神经元2
        # 输入层
        input_layer_2 = VGroup(
            *[Circle(radius=0.2, fill_opacity=1, color=input_color) for _ in range(2)]
        )
        input_layer_2.arrange(DOWN, buff=1.5)

        # 接受层
        receive_layer_2 = Circle(
            radius=0.5, fill_opacity=1, color=receive_color
        ).move_to(ORIGIN + UP)

        # 输出层
        output_layer_2 = Circle(radius=0.2)

        # 排列层
        layers_2 = VGroup(input_layer_2, receive_layer_2, output_layer_2)
        layers_2.arrange(RIGHT, buff=2.5)

        # 添加连接线
        connections_2 = VGroup()
        for i, input_neuron in enumerate(input_layer_2):
            connection = Arrow(
                start=input_neuron.get_center(),
                end=receive_layer_2.get_center(),
                stroke_opacity=0.3,
                tip_length=0.2,
                buff=0.6,
            )
            connections_2.add(connection)

        # 添加输入和输出标签
        input_labels_2 = VGroup(
            *[
                MathTex(f"x_{i + 1}").move_to(input_neuron.get_center())
                for i, input_neuron in enumerate(input_layer_2)
            ]
        )
        theta = np.deg2rad(0)
        direction = np.array([np.cos(theta), np.sin(theta), 0])
        start_point = receive_layer_2.get_center()
        end_point = output_layer_2.get_center()
        # 创建箭头，方向由start到end，颜色为绿色
        arrow_2 = Arrow(
            start=start_point,
            end=end_point,
            color=GREEN,
            stroke_opacity=0.6,
            tip_length=0.3,
            buff=0.6,
        )
        output_label_y_2 = MathTex("y").move_to(output_layer_2.get_center())
        p2 = VGroup(
            input_labels_2,
            connections_2,
            receive_layer_2,
            arrow_2,
            receive_layer_2,
            output_label_y_2,
        )

        # 创建神经元3
        # 输入层
        input_layer_3 = VGroup(
            *[Circle(radius=0.2, fill_opacity=1, color=input_color) for _ in range(2)]
        )
        input_layer_3.arrange(DOWN, buff=1.5)

        # 接受层
        receive_layer_3 = Circle(
            radius=0.5, fill_opacity=1, color=receive_color
        ).move_to(ORIGIN + UP)

        # 输出层
        output_layer_3 = Circle(radius=0.2)

        # 排列层
        layers_3 = VGroup(input_layer_3, receive_layer_3, output_layer_3)

        layers_3.arrange(RIGHT, buff=2.5)

        # 添加连接线
        connections_3 = VGroup()
        for i, input_neuron in enumerate(input_layer_3):
            connection = Arrow(
                start=input_neuron.get_center(),
                end=receive_layer_3.get_center(),
                stroke_opacity=0.3,
                tip_length=0.2,
                buff=0.6,
            )
            connections_3.add(connection)

        # 添加输入和输出标签
        input_labels_3 = VGroup(
            *[
                MathTex(f"x_{i + 1}").move_to(input_neuron.get_center())
                for i, input_neuron in enumerate(input_layer_3)
            ]
        )
        theta = np.deg2rad(0)
        direction = np.array([np.cos(theta), np.sin(theta), 0])
        start_point = receive_layer_3.get_center()
        end_point = output_layer_3.get_center()
        # 创建箭头，方向由start到end，颜色为绿色
        arrow_3 = Arrow(
            start=start_point,
            end=end_point,
            color=GREEN,
            stroke_opacity=0.6,
            tip_length=0.3,
            buff=0.6,
        )
        output_label_y_3 = MathTex("y").move_to(output_layer_3.get_center())
        p3 = VGroup(receive_layer_3, arrow_3, receive_layer_3, output_label_y_3)

        p1.scale(0.6).to_edge(DOWN, buff=2.5).to_edge(LEFT, buff=3.5)
        p2.scale(0.6).next_to(p1, DOWN, buff=0.5)
        p3.scale(0.6).to_edge(DOWN, buff=2)
        self.play(Create(p1), Create(p2))
        self.wait(1)

        start_point_new_1 = receive_layer_1.get_center()
        end_point_new_1 = receive_layer_3.get_center()
        arrow_new_1 = Arrow(
            start=start_point_new_1,
            end=end_point_new_1,
            color=YELLOW,
            stroke_opacity=0.6,
            tip_length=0.2,
            buff=0.3,
        )

        start_point_new_2 = receive_layer_2.get_center()
        end_point_new_2 = receive_layer_3.get_center()
        arrow_new_2 = Arrow(
            start=start_point_new_2,
            end=end_point_new_2,
            color=YELLOW,
            stroke_opacity=0.6,
            tip_length=0.2,
            buff=0.3,
        )

        x1 = (
            MathTex("x_{1}").move_to(receive_layer_1.get_center() + LEFT * 2).scale(0.6)
        )
        x2 = (
            MathTex("x_{2}").move_to(receive_layer_2.get_center() + LEFT * 2).scale(0.6)
        )

        input_arrow_1_1 = Arrow(
            start=x1.get_center(),
            end=receive_layer_1.get_center(),
            color=input_color,
            stroke_opacity=0.6,
            tip_length=0.2,
            buff=0.3,
        )
        input_arrow_1_2 = Arrow(
            start=x1.get_center(),
            end=receive_layer_2.get_center(),
            color=input_color,
            stroke_opacity=0.6,
            tip_length=0.2,
            buff=0.3,
        )
        input_arrow_2_1 = Arrow(
            start=x2.get_center(),
            end=receive_layer_1.get_center(),
            color=input_color,
            stroke_opacity=0.6,
            tip_length=0.2,
            buff=0.3,
        )
        input_arrow_2_2 = Arrow(
            start=x2.get_center(),
            end=receive_layer_2.get_center(),
            color=input_color,
            stroke_opacity=0.6,
            tip_length=0.2,
            buff=0.3,
        )

        self.play(
            Transform(
                Group(arrow_1, arrow_2, output_label_y_1, output_label_y_2),
                Group(arrow_new_1, arrow_new_2),
            ),
            FadeIn(p3),
        )
        self.wait(1)

        self.play(
            Transform(
                Group(connections_1, connections_2),
                Group(
                    input_arrow_1_1, input_arrow_1_2, input_arrow_2_1, input_arrow_2_2
                ),
            ),
            Transform(Group(input_labels_1, input_labels_2), Group(x1, x2)),
        )
        self.wait(1)

    def func1(self, x):
        return x + 0.5

    def func2(self, x):
        return x - 0.5

    def func3(self, x):
        return -x + 1.5

    def func4(self, x):
        return x - 0.5
