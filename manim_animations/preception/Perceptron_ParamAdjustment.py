"""  
摘要：  
该代码使用 Manim 库创建多个动画场景，展示数学和机器学习中的一些概念，包括：  

1. **UnivariateQuadratic**：展示一元二次函数的图像，演示如何通过求导找到函数的最小值，并标注最小值点。  

2. **QuadraticSurfaceVisualization**：在三维空间中可视化一个复杂的二次曲面，展示其形状和特性。  

3. **GradientDescentOneVariable**：演示一维梯度下降法，包括可变步长和固定步长的实现，帮助观众理解梯度下降的过程。  

4. **ComplexTVariable**：展示一个复杂函数的图像及其导数，演示梯度下降的过程。  

5. **GradientDescentTwoVariable**：在三维空间中可视化一个复杂函数的表面，并展示在该表面上进行的梯度下降过程。  
  
"""  

from manim import *
import torch
import math
import sympy as sp



class UnivariateQuadratic(Scene):
    def construct(self):
        # 绘制坐标轴
        axes = Axes(
            x_range=[-3, 7, 1],
            y_range=[-2, 8, 1],
            axis_config={"color": BLUE},
        )

        # 绘制函数图像
        graph = axes.plot(lambda x: (x - 2) ** 2 + 1, color=YELLOW)
        graph_label = axes.get_graph_label(graph, label="y=(x-2)^2+1")

        self.play(Create(axes))
        self.play(Create(graph))

        # 提出问题
        question = Text("当x为几时，函数能取得最小值？").to_edge(UP).scale(0.6)
        self.play(Write(question))
        self.wait(1)
        self.play(FadeOut(question))

        # 方法一：通过求导的方式找到极值/最值并求出此时的变量
        derivative = (
            MathTex(r"\frac{\mathrm{d} y}{\mathrm{d} x} ")
            .to_edge(RIGHT, buff=2)
            .shift(UP * 0.5)
        )
        equal_zero = MathTex(r"= 0").next_to(derivative, RIGHT, buff=0.15)

        # 显示导数
        self.play(FadeIn(derivative))
        self.wait(1)

        # 令导数等于0
        self.play(FadeIn(equal_zero))
        self.wait(1)

        # 标注最小值点
        x_min = 2
        y_min = (x_min - 2) ** 2 + 1  # y_min =1

        # 创建点
        point = axes.coords_to_point(x_min, 0)
        dot = Dot(point, color=RED)
        self.play(FadeIn(dot))

        # 添加点标签
        point_label = MathTex(r"x_{min}").next_to(dot, DR, buff=0.3)
        self.play(Write(point_label))
        self.wait(1)

        # 创建虚线
        vertical_line = DashedLine(
            start=axes.coords_to_point(x_min, 0),
            end=axes.coords_to_point(x_min, y_min),
            color=GREY,
        )
        horizontal_line = DashedLine(
            start=axes.coords_to_point(0, y_min),
            end=axes.coords_to_point(x_min, y_min),
            color=GREY,
        )
        self.play(Create(vertical_line), Create(horizontal_line))
        self.wait(2)

        # 最终展示
        self.wait(2)

        formular1 = (
            MathTex(
                r"L = \max(0, -y(w \cdot x + b))",
                substrings_to_isolate=["L", "=", r"\max", "w", "x", "b"],
            )
            .scale(0.8)
            .to_edge(UP)
        )

        # # 打印公式中每个部分的索引
        # for i, part in enumerate(formular1):
        #     print(f"{i}: {part}")

        part1 = formular1[6]  # w
        part2 = formular1[8]  # x
        part3 = formular1[10]  # b

        self.play(Write(formular1))
        self.wait(0.5)
        self.play(
            part1.animate.set_color(RED),
            part2.animate.set_color(RED),
            part3.animate.set_color((RED)),
        )
        self.play(
            FadeOut(formular1),
            FadeOut(vertical_line),
            FadeOut(horizontal_line),
            FadeOut(dot),
            FadeOut(axes),
            FadeOut(graph),
            FadeOut(point_label),
            FadeOut(derivative),
            FadeOut(equal_zero),
        )


class QuadraticSurfaceVisualization(ThreeDScene):
    def construct(self):
        def f(x, y):
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            term1 = 1.5 * (1 - x) ** 2 * torch.exp(-0.5 * x**2 - (y + 2) ** 2)
            term2 = -2 * (x / 10 - x**3 / 2 - y**3) * torch.exp(-0.5 * (x**2 + y**2))
            term3 = -0.1 * torch.exp(-0.5 * (x + 2) ** 2 - 0.5 * y**2)
            return (term1 + term2 + term3) * 3

        # 创建 3D 坐标轴
        axes = ThreeDAxes(
            x_range=[-5, 5, 1],
            y_range=[-5, 5, 1],
            z_range=[-10, 10, 2],
            axis_config={"color": BLUE},
        )

        # 创建表面
        resolution = 20  # 网格分辨率
        surface = Surface(
            lambda u, v: axes.c2p(u, v, f(u, v)),
            u_range=[-5, 5],
            v_range=[-5, 5],
            resolution=(resolution, resolution),
            fill_color=YELLOW,
            fill_opacity=1,
            checkerboard_colors=[RED, RED_E],
        )

        # 设置表面样式
        surface.set_style(fill_opacity=0.8, stroke_width=0.8)
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)

        # 添加坐标轴和表面到场景
        self.play(Create(axes))
        self.play(Create(surface))

        # 开始环境摄像机旋转
        self.begin_ambient_camera_rotation(rate=PI / 24)  # 设置旋转速度

        # 旋转一段时间
        self.wait(8)

        # 停止摄像机旋转
        self.stop_ambient_camera_rotation()

        # 等待动画结束
        self.wait(2)


class GradientDescentOneVariable(Scene):
    def construct(self):
        # 设置选项: "T" 为可变步长, "F" 为固定步长, "A" 为先演示可变步长再演示固定步长
        option = "F"  # 可根据需要更改为 "T", "F" 或 "A"

        # 绘制坐标轴
        axes = Axes(
            x_range=[-3, 7, 1],
            y_range=[-2, 8, 1],
            axis_config={"color": BLUE},
        )

        # 绘制函数图像
        graph = axes.plot(lambda x: self.func(x), color=YELLOW)
        graph_label = axes.get_graph_label(graph, label="y=(x-2)^2+1")

        self.play(Create(axes), Create(graph), Write(graph_label))

        # 根据选项演示不同的梯度下降方法
        if option == "A":
            # 演示可变步长梯度下降
            # title_var_lr = Text("可变步长梯度下降法", font_size=24).to_edge(UP)
            # self.play(Write(title_var_lr))
            self.gradient_descent(
                axes=axes,
                initial_x=4,
                learning_rate=None,  # None 表示可变步长
                iterations=7,
                color=RED,
            )
            self.wait(1)
            # self.play(FadeOut(title_var_lr))

            # 演示固定步长梯度下降
            # title_fixed_lr = Text("固定步长梯度下降法", font_size=24).to_edge(UP)
            # self.play(Write(title_fixed_lr))
            self.gradient_descent(
                axes=axes,
                initial_x=4,
                learning_rate=0.5,  # 固定步长
                iterations=4,
                color=BLUE,
            )
            self.wait(1)
            # self.play(FadeOut(title_fixed_lr))

        elif option == "T":
            # 仅演示可变步长梯度下降
            # title_var_lr = Text("可变步长梯度下降法", font_size=24).to_edge(UP)
            # self.play(Write(title_var_lr))
            self.gradient_descent(
                axes=axes,
                initial_x=4,
                learning_rate=None,  # None 表示可变步长
                iterations=7,
                color=RED,
            )
            self.wait(1)
            # self.play(FadeOut(title_var_lr))

        elif option == "F":
            # 仅演示固定步长梯度下降
            # title_fixed_lr = Text("固定步长梯度下降法", font_size=24).to_edge(UP)
            # self.play(Write(title_fixed_lr))
            self.gradient_descent(
                axes=axes,
                initial_x=4,
                learning_rate=0.5,  # 固定步长
                iterations=4,
                color=BLUE,
            )
            self.wait(1)
            # self.play(FadeOut(title_fixed_lr))

        else:
            self.play(Write(Text("无效的选项选择！", font_size=24)))

        self.wait(2)

    def func(self, x):
        """定义函数 y = (x - 2)^2 + 1"""
        return (x - 2) ** 2 + 1

    def gradient(self, x):
        """计算梯度 dy/dx = 2*(x - 2)"""
        return 2 * (x - 2)

    def create_tangent_line(self, axes, x):
        """根据x值创建切线对象"""
        y = self.func(x)
        m = self.gradient(x)
        b = y - m * x
        return axes.plot(lambda x_val: m * x_val + b, x_range=[-3, 7], color=GREEN)

    def gradient_descent(self, axes, initial_x, learning_rate, iterations, color):
        """
        执行梯度下降动画
        :param axes: 坐标轴对象
        :param initial_x: 初始x值
        :param learning_rate: 学习率, 如果为None则使用可变步长
        :param iterations: 迭代次数
        :param color: 点的颜色
        """
        point = Dot(color=color).move_to(axes.c2p(initial_x, self.func(initial_x)))
        self.play(FadeIn(point))

        # 计算并绘制初始切线
        current_x = initial_x
        tangent_line = self.create_tangent_line(axes, current_x)
        self.play(Create(tangent_line))

        for i in range(iterations):
            self.wait(0.5)

            # 计算梯度
            grad = self.gradient(current_x)

            # 计算步长
            if learning_rate is not None:
                lr = learning_rate  # 固定步长
                new_x = current_x - lr
            else:
                lr = self.variable_learning_rate(i)  # 可变步长
                new_x = current_x - lr * grad
            # 更新x值

            new_y = self.func(new_x)

            # 计算并绘制新的切线
            new_tangent_line = self.create_tangent_line(axes, new_x)

            # 更新点和切线
            new_point = Dot(color=color).move_to(axes.c2p(new_x, new_y))
            self.play(
                Transform(point, new_point), Transform(tangent_line, new_tangent_line)
            )
            self.wait(0.5)

            # 更新当前值
            current_x = new_x

    def variable_learning_rate(self, iteration):
        """
        可变步长策略: 学习率逐渐减小
        比如 lr = 0.5 / (iteration + 1)
        """
        return 0.5 / (iteration + 1)


class ComplexTVariable(ThreeDScene):
    def ComplexFunction(self, x):
        """定义函数： y = (x^2)/20 - cos(x) + sin(2x)/2"""
        return (x**2) / 20 - math.cos(x) + math.sin(2 * x) / 2

    def get_derivative_function(self):
        """使用 SymPy 计算 ComplexFunction 的导数，并返回一个可调用的数值函数"""
        x = sp.symbols("x")
        # 定义符号表达式
        f = (x**2) / 20 - sp.cos(x) + sp.sin(2 * x) / 2
        # 计算导数
        f_prime = sp.diff(f, x)
        # 打印函数和导数（可选）
        # print("f(x) =", f)
        # print("f'(x) =", f_prime)
        # 将符号表达式转换为数值函数
        f_prime_lambdified = sp.lambdify(x, f_prime, modules=["numpy"])
        return f_prime_lambdified

    def get_function_range(self, x_min, x_max, num_samples=200):
        """动态计算函数的 y 值范围"""
        x_values = np.linspace(x_min, x_max, num_samples)
        y_values = [self.ComplexFunction(x) for x in x_values]
        y_min, y_max = min(y_values), max(y_values)
        return y_min, y_max

    def create_tangent_line(self, axes, x, derivative_func, length=4):
        """
        根据x值创建较短的切线对象。

        参数：
        - axes: 坐标轴对象。
        - x: 当前点的x坐标。
        - derivative_func: 导数函数。
        - length: 切线的总长度（默认4）。
        """
        y = self.ComplexFunction(x)
        m = derivative_func(x)  # 斜率
        b = y - m * x  # 截距

        # 定义切线的范围，确保不超出坐标轴的x范围
        half_length = length / 2
        x_min, x_max = axes.x_range[:2]
        x_start = max(x - half_length, x_min)
        x_end = min(x + half_length, x_max)

        return axes.plot(
            lambda x_val: m * x_val + b, x_range=[x_start, x_end], color=GREEN
        )

    def construct(self):
        # 定义 x 范围
        x_min, x_max = -10, 10

        # 自动计算 y 范围
        y_min, y_max = self.get_function_range(x_min, x_max)

        # 定义动态坐标轴
        axes = Axes(
            x_range=[x_min, x_max, 5],  # x 刻度间隔
            y_range=[y_min - 5, y_max + 5, 5],  # y 刻度间隔，增加缓冲区
            axis_config={"color": BLUE},
        ).add_coordinates()

        # 绘制函数
        graph = axes.plot(lambda x: self.ComplexFunction(x), color=YELLOW)
        graph_label = axes.get_graph_label(
            graph, label="f(x) = \\frac{x^2}{20} - \\cos(x) + \\frac{\\sin(2x)}{2}"
        )

        # 动画展示坐标轴和函数图形
        self.play(Create(axes), Create(graph), Write(graph_label))
        self.wait(1)

        # 获取导数函数
        derivative_func = self.get_derivative_function()

        # 梯度下降参数
        initial_x = 4.2  # 初始点（确保在 x 范围内）
        learning_rate = 0.15  # 学习率
        iterations = 10  # 迭代次数

        # 创建初始点
        initial_y = self.ComplexFunction(initial_x)
        point = Dot(color=GREEN).move_to(axes.c2p(initial_x, initial_y))
        self.play(FadeIn(point))
        self.wait(0.5)

        # 创建初始切线
        tangent_line = self.create_tangent_line(axes, initial_x, derivative_func)
        self.play(Create(tangent_line))
        self.wait(0.5)

        current_x = initial_x

        for i in range(iterations):
            # 计算梯度（导数）
            grad = derivative_func(current_x)

            # 计算步长
            new_x = current_x + learning_rate
            new_y = self.ComplexFunction(new_x)

            # 创建新的切线
            new_tangent_line = self.create_tangent_line(axes, new_x, derivative_func)

            # 创建新的点
            new_point = Dot(color=RED).move_to(axes.c2p(new_x, new_y))

            # 动画：移动点、更新切线
            self.play(
                Transform(point, new_point), Transform(tangent_line, new_tangent_line)
            )

            # 标注当前迭代的 x 值
            # label = MathTex(f"x = {new_x:.2f}").next_to(new_point, UP)
            # self.play(FadeIn(label))
            # self.wait(0.5)
            # self.play(FadeOut(label))

            # 更新当前 x 值
            current_x = new_x

        self.wait(2)


class GradientDescentTwoVariable(ThreeDScene):
    def construct(self):
        def f(x, y):
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            term1 = 1.5 * (1 - x) ** 2 * torch.exp(-0.5 * x**2 - (y + 2) ** 2)
            term2 = -2 * (x / 10 - x**3 / 2 - y**3) * torch.exp(-0.5 * (x**2 + y**2))
            term3 = -0.1 * torch.exp(-0.5 * (x + 2) ** 2 - 0.5 * y**2)
            return (term1 + term2 + term3) * 3

        # 创建三维坐标轴
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-10, 10, 2],
            axis_config={"color": BLUE},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="z")

        # 创建表面
        resolution = 20  # 网格分辨率
        surface = Surface(
            lambda u, v: axes.c2p(u, v, f(u, v)),
            u_range=[-5, 5],
            v_range=[-5, 5],
            resolution=(resolution, resolution),
            fill_color=YELLOW,
            fill_opacity=1,
            checkerboard_colors=[RED, RED_E],
        )

        # 设置表面样式
        surface.set_style(fill_opacity=0.8, stroke_width=0.8)
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)

        # 缩小整个坐标轴及其标签
        axes.scale(0.5)
        surface.scale(0.5)

        # 添加坐标轴和表面到场景
        self.play(Create(axes), Write(axes_labels))
        self.play(Create(surface))
        self.wait(1)

        # 存储点的坐标数据
        selected_points = {
            1: np.array([-0.8, -2.3, 5.847186]),
            2: np.array([-0.445482, -2.384923, 3.047376]),
            3: np.array([-0.178311, -2.390921, 0.649276]),
            4: np.array([0.272616, -2.238446, -3.136957]),
            5: np.array([0.554393, -1.826272, -5.143542]),
        }

        # 点的列表（确保按顺序排列）
        point_list = list(selected_points.values())

        # 设置三维视角
        self.set_camera_orientation(phi=60 * DEGREES, theta=-30 * DEGREES)

        for i, point in enumerate(point_list):
            # (a) 将原始点（数据坐标）转换为 Manim 坐标系中的点
            manim_point = axes.coords_to_point(*point)
            # (b) 添加三维点——这里 Dot3D 参数改为 manim_point
            dot = Dot3D(manim_point, color=BLUE, radius=0.05)  # 缩小点的半径

            # 如果有前一个点，则连接线段
            if i > 0:
                prev_manim_point = axes.coords_to_point(*point_list[i - 1])
                line = Line3D(prev_manim_point, manim_point, color=GREEN)
                self.play(Create(line), run_time=0.5)
                self.wait(0.5)
            # 绘制当前点
            self.play(FadeIn(dot), run_time=0.2)

        self.wait(2)