from manim import *


class NetInput(Scene):
    def construct(self):
        # 创建文本和数学公式对象
        sepal_length_text = Text("花萼长度", color=RED)
        sepal_width_text = Text("花萼宽度", color=RED)
        sepal_length_var = MathTex("x_{1}", color=YELLOW)
        sepal_width_var = MathTex("x_{2}", color=YELLOW)
        weight1_var = MathTex("w_{1}", color=GREEN)
        weight2_var = MathTex("w_{2}", color=GREEN)

        # 创建每一列的VGroup并垂直排列
        label_column = VGroup(sepal_length_text, sepal_width_text).arrange(
            DOWN, buff=0.5
        )
        x_column = VGroup(sepal_length_var, sepal_width_var).arrange(DOWN, buff=0.5)
        w_column = VGroup(weight1_var, weight2_var).arrange(DOWN, buff=0.5)

        # 将三列水平排列
        columns = VGroup(label_column, x_column, w_column).arrange(RIGHT, buff=1.5)

        # 将整个组移动到场景中心（可选）
        columns.move_to(ORIGIN)

        # 创建动画组，每列间隔一秒渲染
        animations = [
            AnimationGroup(Write(label_column), lag_ratio=0),
            AnimationGroup(Write(x_column), lag_ratio=0),
            AnimationGroup(Write(w_column), lag_ratio=0),
        ]

        # 使用 LaggedStart 来控制每列之间的时间间隔
        self.play(LaggedStart(*animations, lag_ratio=1))
        self.wait(2)

        # 1. 淡出 label_column
        self.play(FadeOut(label_column))
        self.wait(1)

        # 2. 创建乘号对象并放置在对应的x变量旁边
        multiply1 = MathTex("\\times").set_color(WHITE)
        multiply2 = MathTex("\\times").set_color(WHITE)

        # 3. 定位乘号在x变量旁边但不显示
        multiply1.next_to(sepal_length_var, RIGHT, buff=0.2)
        multiply2.next_to(sepal_width_var, RIGHT, buff=0.2)
        multiply_group = VGroup(multiply1, multiply2)

        # 4. 将 w_column 中的每个权重变量移动到对应乘号的右侧
        self.play(
            weight1_var.animate.next_to(multiply1, RIGHT, buff=0.2),
            weight2_var.animate.next_to(multiply2, RIGHT, buff=0.2),
        )

        # 5. 显示乘号
        self.play(Write(multiply_group))
        self.wait(2)

        # 6. 创建加号和等号对象
        plus = MathTex("+").set_color(WHITE)
        equals = MathTex("=").set_color(WHITE)
        z = MathTex("Z").set_color(RED).scale(1.2)
        b = MathTex("b").set_color(BLUE)

        # 7. 创建 term1 和 term2 的组合
        term1 = VGroup(sepal_length_var, multiply1, weight1_var)
        term2 = VGroup(sepal_width_var, multiply2, weight2_var)

        # 8. 移动 term1 到 term2 的左侧
        self.play(term1.animate.next_to(term2, LEFT, buff=0.7))
        self.wait(0.5)

        # 9. 创建并定位加号，但不显示
        plus.next_to(term1, RIGHT, buff=0.2)
        # 同时为 term2 设置位置
        term2.next_to(plus, RIGHT, buff=0.2)

        # 10. 显示加号
        self.play(Write(plus))
        self.wait(1)

        # 11. 移动等号到 term1 的左侧位置
        equals.next_to(term1, LEFT, buff=0.5)
        z.next_to(equals, LEFT, buff=0.5)

        # 12. 移动等号和Z到指定位置
        self.play(
            term1.animate.next_to(term2, LEFT, buff=0.7),
            equals.animate.move_to(equals.get_center()),  # 保持原位
            z.animate.next_to(equals, LEFT, buff=0.5),
        )
        self.wait(0.5)

        # 13. 显示等号和Z
        self.play(Write(equals), Write(z))
        self.wait(1)

        # 14. 创建并定位第二个加号
        second_plus = MathTex("+").set_color(WHITE)
        second_plus.next_to(term2, RIGHT, buff=0.3)
        b.next_to(second_plus, RIGHT, buff=0.4)

        # 15. 显示第二个加号和b
        self.play(Write(second_plus), Write(b))
        self.wait(1)
        self.play(FadeOut(Group(term1, term2, plus, equals, z, second_plus, b)))
        self.wait(1)
