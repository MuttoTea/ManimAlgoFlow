from manim import *


class JudegeZ(Scene):
    def construct(self):
        # 定义字体以支持中文显示
        chinese_font = "SimHei"  # 确保系统中已安装该字体

        # 创建第一个条件行: Z ≥ A -> 1 -> 山鸢尾
        condition1 = MathTex("Z", "\geq", "A").set_color(YELLOW)
        arrow1 = MathTex("\\rightarrow").set_color(WHITE)
        output1 = MathTex("1").set_color(GREEN)
        arrow2 = MathTex("\\rightarrow").set_color(WHITE)
        label1 = Text("山鸢尾", font=chinese_font).set_color(BLUE)

        # 创建第二个条件行: Z < A -> -1 -> 变色鸢尾
        condition2 = MathTex("Z", "<", "A").set_color(YELLOW)
        arrow3 = MathTex("\\rightarrow").set_color(WHITE)
        output2 = MathTex("-1").set_color(RED)
        arrow4 = MathTex("\\rightarrow").set_color(WHITE)
        label2 = Text("变色鸢尾", font=chinese_font).set_color(BLUE)

        # 组合第一个条件行
        line1 = VGroup(condition1, arrow1, output1, arrow2, label1).arrange(RIGHT, buff=0.5)

        # 组合第二个条件行
        line2 = VGroup(condition2, arrow3, output2, arrow4, label2).arrange(RIGHT, buff=0.5)

        # 将两行垂直排列，并指定对齐边缘为左侧
        lines = VGroup(line1, line2).arrange(DOWN, buff=1, aligned_edge=LEFT)

        # 将组合好的内容移动到场景中心
        lines.move_to(ORIGIN)

        # 逐一显示每一行的各个元素，保持原有显示顺序
        self.play(Write(condition1), Write(condition2))
        self.play(Write(arrow1), Write(output1))
        self.play(Write(arrow3), Write(output2))
        self.play(Write(arrow2), Write(label1))
        self.play(Write(arrow4), Write(label2))
        self.wait(2)


class fun_sign(Scene):
    def construct(self):
        # 定义函数表达式
        expression = MathTex(
            r"f(x) = \begin{cases} 1 & x > 0 \\ 0 & x = 0 \\ -1 & x < 0 \end{cases}"
        )
        # 将表达式放在屏幕中央
        self.play(Write(expression))
        self.wait(1)

        # 同时上移并缩小表达式
        self.play(
            expression.animate.scale(0.8).to_edge(UP),
            run_time=2
        )
        self.wait(1)

        # 创建坐标轴
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"color": BLUE},
        )

        # 添加坐标轴标签
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")

        # 定义 sign 函数
        def sign_function(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0

                # 生成函数图像，指定断点

        graph = axes.plot(sign_function, color=WHITE, discontinuities=[0])

        # 创建不连续点的标记
        # (0, 0) 实心圆
        closed_circle = Dot(axes.c2p(0, 0), color=WHITE)
        closed_circle.set_fill(WHITE, opacity=1)

        # (0, 1) 和 (0, -1) 空心圆
        open_circle_pos = Dot(axes.c2p(0, 1), color=WHITE)
        open_circle_pos.set_fill(BLACK, opacity=1)
        open_circle_pos.set_stroke(WHITE, width=2)

        open_circle_neg = Dot(axes.c2p(0, -1), color=WHITE)
        open_circle_neg.set_fill(BLACK, opacity=1)
        open_circle_neg.set_stroke(WHITE, width=2)

        # 将所有坐标系相关对象组合在一起
        coord_group = VGroup(axes, axes_labels, graph, closed_circle, open_circle_pos, open_circle_neg)
        coord_group.scale(0.6).move_to(DOWN)

        # 显示坐标轴和标签
        self.play(Create(axes), Write(axes_labels))
        self.wait(1)

        # 显示函数图像
        self.play(Create(graph))

        # 添加不连续点标记
        self.play(
            FadeIn(closed_circle),
            FadeIn(open_circle_pos),
            FadeIn(open_circle_neg),
            run_time=1
        )
        self.wait(3)

