"""  
摘要：  
该代码使用 Manim 库创建一个动画场景，展示感知器（Perceptron）模型的权重计算和分类过程。  
具体步骤包括：  
1. 数据处理：使用 IrisDataProcessor 获取鸢尾花数据集的特征和标签。  
2. 设置权重和偏置：定义感知器的权重和偏置，并计算直线的斜率和截距。  
3. 创建坐标轴：使用 Axes 类创建坐标系，设置 x 轴和 y 轴的范围和标签。  
4. 数学公式展示：创建数学公式，表示感知器的线性方程，并逐步替换权重和偏置的值。  
5. 数据点可视化：筛选不同类别的鸢尾花数据点，并在坐标系中绘制散点图。  
6. 绘制超平面：根据计算出的斜率和截距绘制超平面（决策边界）。  
7. 选择特定数据点：选择特定的 Setosa 和 Versicolor 数据点，并为它们创建标签。  
8. 动画效果：通过动画展示数据点的选择、替换数学公式中的变量、绘制超平面等过程。  
9. 分类结果展示：计算每个数据点的分类结果，并根据结果显示相应的文本（如“分类正确”或“分类错误”）。  
10. 最终结果：显示损失/代价的计算结果，强调感知器模型的输出。  
"""  

from manim import *
from DataProcessor import IrisDataProcessor


class PerceptronFindWeights(Scene):
    def construct(self):
        # 数据处理
        data_processor = IrisDataProcessor()
        X, y = data_processor.get_data()

        # 获取直线信息
        w1 = -0.4
        w2 = 0.9
        b = -0.3
        slope = -w1 / w2
        intercept = -b / w2

        # 创建坐标轴
        axes = Axes(
            x_range=[4, 7.5, 0.5],  # x轴范围
            y_range=[1, 5, 1],  # y轴范围
            axis_config={"color": BLUE, "include_numbers": True},
        )

        # 定义变量及其对应的数值
        replacements = {"w_1": "-0.4", "w_2": "0.9", "b": "-0.3"}

        # 创建初始数学公式，并孤立需要替换的变量
        equation = MathTex(
            r"w_1 \times x_1 + w_2 \times x_2 + b = 0",
            font_size=46,
            substrings_to_isolate=["w_1", "x_1", "w_2", "x_2", "+ b", "0"],
        ).to_edge(UP)

        # 创建替换后的方程
        equation_with_values = MathTex(
            r"-0.4 \times x_1 + 0.9 \times x_2 - 0.3 = 0", font_size=42
        ).to_edge(UP)

        self.add(equation)
        self.wait(1)

        # 添加标签
        x_label = axes.get_x_axis_label(Text("花萼长度 (cm)").scale(0.6))
        y_label = axes.get_y_axis_label(Text("花萼宽度 (cm)").scale(0.6))

        # 使用 numpy 筛选数据
        setosa_indices = np.where(y == 0)[0]
        versicolor_indices = np.where(y == 1)[0]

        setosa_points = [(X[i, 0], X[i, 1]) for i in setosa_indices]
        versicolor_points = [(X[i, 0], X[i, 1]) for i in versicolor_indices]

        # 创建散点
        setosa_dots = [Dot(axes.c2p(x, y), color=BLUE) for x, y in setosa_points]
        versicolor_dots = [
            Dot(axes.c2p(x, y), color=ORANGE) for x, y in versicolor_points
        ]

        # 绘制超平面
        hyperplane = axes.plot(
            lambda x: slope * x + intercept, color=WHITE, x_range=[4, 7]
        )

        # 创建一个 VGroup，包括轴、标签、散点和直线
        group = VGroup(
            axes, x_label, y_label, *setosa_dots, *versicolor_dots, hyperplane
        )

        # 对整个坐标系集合进行缩放
        group.scale(0.8)

        # 添加根据位置改变后整体添加到场景中
        self.play(Create(axes), Write(x_label), Write(y_label))
        self.play(*[Create(dot) for dot in setosa_dots + versicolor_dots])

        # 遍历每个变量，创建替换文本并播放动画
        for var, value in replacements.items():
            # 获取要替换的部分
            var_obj = equation.get_part_by_tex(var)

            if var_obj is None:
                # 如果无法找到对应的部分，跳过当前循环
                print(f"未找到要替换的部分: {var}")
                continue

                # 创建数值文本对象，保持与替换部分相同的位置和大小
            value_tex = MathTex(value, font_size=38).move_to(var_obj)

            # 根据不同的变量调整替换后文本的位置
            if var == "w_1":
                value_tex.shift(LEFT * 0.06, UP * 0.06)  # 根据需要调整偏移量
            elif var == "w_2":
                value_tex.shift(UP * 0.06, RIGHT * 0.05)
            # elif var == "+ b":
            #     value_tex.shift(LEFT * 0.2, UP * 0.1)

            # 使用 ReplacementTransform 进行替换动画
            self.play(ReplacementTransform(var_obj, value_tex))

        self.wait(2)

        self.play(Create(hyperplane))
        self.wait(1)

        # 定义要选择的特定数据点
        setosa_selected = (5, 3.6)
        versicolor_selected_1 = (6, 2.2)
        versicolor_selected_2 = (5.9, 3.2)

        # 创建 Dot 和 Label for Setosa
        setosa_dot = Dot(axes.c2p(*setosa_selected), color=BLUE)
        setosa_label = MathTex(r"(5, 3.6)", font_size=24).next_to(
            setosa_dot, UP + RIGHT
        )

        # 创建 Dot 和 Label for Versicolor 1
        versicolor_dot_1 = Dot(axes.c2p(*versicolor_selected_1), color=ORANGE)
        versicolor_label_1 = MathTex(r"(6, 2.2)", font_size=24).next_to(
            versicolor_dot_1, UP + RIGHT
        )

        # 创建 Dot 和 Label for Versicolor 2
        versicolor_dot_2 = Dot(axes.c2p(*versicolor_selected_2), color=ORANGE)
        versicolor_label_2 = MathTex(r"(5.9, 3.2)", font_size=24).next_to(
            versicolor_dot_2, UP + RIGHT
        )

        # 创建 VGroups
        all_dots = VGroup(*setosa_dots, *versicolor_dots)
        selected_dots = VGroup(setosa_dot, versicolor_dot_1, versicolor_dot_2)
        selected_labels = VGroup(setosa_label, versicolor_label_1, versicolor_label_2)

        # 将选中的点和标签添加到场景
        self.play(Create(setosa_dot))
        self.play(Create(versicolor_dot_1))
        self.play(Create(versicolor_dot_2))

        # 隐去非选中的点
        self.play(FadeOut(all_dots))

        self.play(Write(selected_labels))
        self.wait(1)
        self.play(FadeOut(axes, hyperplane, x_label, y_label))

        # 组点和标签
        points = VGroup(setosa_dot, versicolor_dot_1, versicolor_dot_2)
        labels = VGroup(setosa_label, versicolor_label_1, versicolor_label_2)

        # 定义移动目标位置
        top_right_point_target = (
            setosa_dot.copy().to_edge(UP + RIGHT).shift(LEFT * 4 + DOWN * 2)
        )
        spacing = 1.5

        # 创建目标位置列表
        targets = [top_right_point_target]
        for i in range(1, len(points)):
            target = top_right_point_target.copy().shift(DOWN * spacing * i)
            targets.append(target)

            # 创建动画
        animations = []
        for point, label, target in zip(points, labels, targets):
            animations.append(point.animate.move_to(target.get_center()))
            animations.append(label.animate.next_to(target, LEFT * 18).scale(1.5))

            # 播放移动动画
        self.play(AnimationGroup(*animations, lag_ratio=0))

        # 分离标签中的数字
        # 第一组数字（"5", "6", "5.9"）
        first_numbers = ["5", "6", "5.9"]
        # 第二组数字（"3.6", "2.2", "3.2"）
        second_numbers = ["3.6", "2.2", "3.2"]

        # 创建新的分离标签（第一组数字）
        setosa_label_number = MathTex(first_numbers[0], font_size=36).next_to(
            setosa_dot, LEFT * 21
        )
        versicolor_label_number_1 = MathTex(first_numbers[1], font_size=36).next_to(
            versicolor_dot_1, LEFT * 21
        )
        versicolor_label_number_2 = MathTex(first_numbers[2], font_size=36).next_to(
            versicolor_dot_2, LEFT * 21
        )
        number_labels_replace = VGroup(
            setosa_label_number, versicolor_label_number_1, versicolor_label_number_2
        )

        # 创建新的分离标签（第二组数字）
        setosa_label_decimal = MathTex(second_numbers[0], font_size=36).next_to(
            setosa_label_number, RIGHT
        )
        versicolor_label_decimal_1 = MathTex(second_numbers[1], font_size=36).next_to(
            versicolor_label_number_1, RIGHT
        )
        versicolor_label_decimal_2 = MathTex(second_numbers[2], font_size=36).next_to(
            versicolor_label_number_2, RIGHT
        )
        decimal_labels_replace = VGroup(
            setosa_label_decimal, versicolor_label_decimal_1, versicolor_label_decimal_2
        )

        # 将分离后的标签组合
        labels_replace = VGroup(number_labels_replace, decimal_labels_replace)

        # 播放标签替换动画
        replace_animations = []
        for original_label, (new_number, new_decimal) in zip(
            labels, zip(number_labels_replace, decimal_labels_replace)
        ):
            # 替换第一组数字
            replace_animations.append(ReplacementTransform(original_label, new_number))
            # 添加第二组数字
            replace_animations.append(
                ReplacementTransform(original_label.copy(), new_decimal)
            )

        # 播放动画
        self.play(*replace_animations)
        # self.wait(0.5)

        multiplier1 = ["-0.3", "-0.3", "-0.3"]
        multiplier2 = ["0.9", "0.9", "0.9"]
        bias = ["0.3", "0.3", "0.3"]

        # 首先移动第二组数字（decimal_labels_replace）向右
        # move_decimal = decimal_labels_replace.animate.shift(RIGHT * 1.0)  # 调整移动距离
        self.play(decimal_labels_replace.animate.shift(RIGHT * 1.1))

        # 创建第一列的乘数和乘号
        multiplier1_group = VGroup(
            *[
                VGroup(
                    MathTex(multiplier1[i], font_size=36),  # 乘数
                    MathTex(r"\times"),  # 乘号
                )
                .arrange(RIGHT, buff=0.1)
                .next_to(number_labels_replace[i], LEFT * 0.45)
                for i in range(len(number_labels_replace))
            ]
        )

        # 创建第二列的乘数和乘号
        multiplier2_group = VGroup(
            *[
                VGroup(
                    MathTex(multiplier2[i], font_size=36),  # 乘数
                    MathTex(r"\times"),  # 乘号
                )
                .arrange(RIGHT, buff=0.1)
                .next_to(decimal_labels_replace[i], LEFT * 0.45)
                for i in range(len(decimal_labels_replace))
            ]
        )

        # 动画同时显示所有乘数和乘号
        self.play(FadeIn(multiplier1_group), FadeIn(multiplier2_group))
        # self.wait(1)

        # 创建加号，并逐行对齐
        plus_signs = VGroup()
        for i in range(len(number_labels_replace)):
            # 创建加号
            plus = MathTex("+").scale(0.9)
            # 将加号放置在两列之间，并与对应的行对齐
            plus.next_to(number_labels_replace[i], RIGHT, buff=0.1).align_to(
                number_labels_replace[i], DOWN
            )
            plus_signs.add(plus)

            # 添加加号动画
        self.play(FadeIn(plus_signs))

        # 创建偏置项，并逐行对齐
        bias_group = VGroup()
        for i in range(len(number_labels_replace)):
            # 创建 "+ 0.3"
            plus_bias = MathTex(f"- {bias[i]}", font_size=36)
            # 将偏置项放置在加号的右侧，并与对应的行对齐
            plus_bias.next_to(decimal_labels_replace[i], RIGHT, buff=0.1).align_to(
                decimal_labels_replace[i], DOWN
            )
            bias_group.add(plus_bias)

            # 添加偏置项的动画
        self.play(FadeIn(bias_group))

        # 计算结果，考虑偏置项
        results = [
            f"{float(first_numbers[i]) * float(multiplier1[i]) + float(second_numbers[i]) * float(multiplier2[i]) - float(bias[i]):.2f}"
            for i in range(len(first_numbers))
        ]
        print(results)

        # 创建等号组
        equals_group = VGroup(
            *[
                MathTex("=", font_size=36)
                .next_to(bias_group[i], RIGHT, buff=0.1)
                .align_to(bias_group[i], DOWN)
                for i in range(len(results))
            ]
        )

        # 创建一个空的组来存放数字（包括负号和数字）
        numbers_group = VGroup()

        # 识别负数并去掉负号
        is_negative = [num.startswith("-") for num in results]
        stripped_numbers = [
            num[1:] if neg else num for num, neg in zip(results, is_negative)
        ]

        for eq, num, neg in zip(equals_group, stripped_numbers, is_negative):
            # 创建数字对象
            number = MathTex(num, font_size=36, color=BLUE if not neg else ORANGE)

            # 如果是负数，创建负号对象
            if neg:
                minus = MathTex("-", font_size=36, color=ORANGE)
                # 将负号放置在数字的左侧，稍微调整位置
                minus.next_to(eq, RIGHT, buff=0.1)
                number.next_to(minus, RIGHT, buff=0.05)
                # 组合负号和数字
                combined = VGroup(minus, number).arrange(RIGHT, buff=0.05)
            else:
                # 如果是正数，使用 \phantom{-} 占位，确保对齐
                phantom = MathTex("-", font_size=36, color=WHITE).set_opacity(0)
                phantom.next_to(eq, RIGHT, buff=0.1)
                number.next_to(phantom, RIGHT, buff=0.05)
                # 组合占位符和数字
                combined = VGroup(phantom, number).arrange(RIGHT, buff=0.05)

            # 将组合后的数字添加到数字组中
            numbers_group.add(combined)

        numbers_group.arrange(DOWN, buff=1.5).next_to(equals_group, RIGHT, buff=0.3)

        # 调整数字组的位置，使其与等号组在垂直方向上对齐
        for eq, num in zip(equals_group, numbers_group):
            num.align_to(eq, DOWN)

        # 动画同时显示等号和结果

        self.play(FadeIn(equals_group), FadeIn(numbers_group))
        # self.wait(1)

        multiply_signs = VGroup(
            *[
                MathTex(r"\times", font_size=36).next_to(
                    numbers_group[i], RIGHT, buff=0.4
                )
                for i in range(len(numbers_group))
            ]
        )

        numbers = ["1", "-1", "-1"]
        # 创建数字组，初始时位置与等号相同，但不可见
        number_replacements = VGroup()

        for result in numbers:
            # 判断是否为负数
            is_negative = result.startswith("-")
            stripped_number = result[1:] if is_negative else result

            # 创建数字对象，颜色根据正负决定
            number = MathTex(
                stripped_number, font_size=36, color=BLUE if not is_negative else ORANGE
            )

            if is_negative:
                # 如果是负数，创建负号对象
                minus = MathTex("-", font_size=36, color=ORANGE)
                combined_number = VGroup(minus, number).arrange(RIGHT, buff=0.05)
            else:
                # 如果是正数，使用 \phantom{-} 占位，确保对齐
                phantom = MathTex("-", font_size=36, color=WHITE).set_opacity(0)
                combined_number = VGroup(phantom, number).arrange(RIGHT, buff=0.05)

                # 将组合后的数字添加到数字组中
            number_replacements.add(combined_number)

            # 将数字组垂直排列，与等号组对齐
        number_replacements.arrange(DOWN, buff=1.5).next_to(
            multiply_signs, RIGHT, buff=0.3
        )

        # 调整数字组的位置，使其与等号组在垂直方向上对齐
        for eq, num in zip(multiply_signs, number_replacements):
            num.align_to(eq, DOWN)

        # 使用ReplacementTransform替换点为数字
        self.play(
            ReplacementTransform(setosa_dot, number_replacements[0]),
            ReplacementTransform(versicolor_dot_1, number_replacements[1]),
            ReplacementTransform(versicolor_dot_2, number_replacements[2]),
        )
        self.wait(2)

        # 继续后续动画处理
        self.play(
            FadeOut(
                number_labels_replace,
                decimal_labels_replace,
                multiplier1_group,
                multiplier2_group,
                plus_signs,
                bias_group,
                equals_group,
            )
        )
        self.play(number_replacements.animate)
        self.play(FadeIn(multiply_signs))

        # **添加“>0”或“<0”符号**
        # 定义每个数字对应的符号
        signs = [">0", ">0", "<0"]

        # 创建符号标签
        sign_labels = VGroup()
        for i in range(len(number_replacements)):
            sign_label = MathTex(signs[i], font_size=36).next_to(
                number_replacements[i], RIGHT, buff=0.4
            )
            sign_labels.add(sign_label)

            # 动画显示符号标签
        self.play(FadeIn(sign_labels))
        self.wait(1)

        text1 = Text("分类正确", font_size=36, color=GREEN).next_to(
            numbers_group[0], LEFT
        )
        text2 = Text("分类错误", font_size=36, color=RED).next_to(
            numbers_group[2], LEFT
        )
        text3 = Text("分类正确", font_size=36, color=GREEN).next_to(
            numbers_group[1], LEFT
        )
        TextGroup = VGroup(text1, text2, text3)
        self.play(FadeIn(TextGroup))
        self.wait(1)

        # 找到需要操作的元素
        greater_than_zero = [
            text for text in sign_labels if text.get_tex_string() == ">0"
        ]
        less_than_zero = [text for text in sign_labels if text.get_tex_string() == "<0"]

        # 定义一个函数来应用高亮、放大和旋转晃动效果
        def animate_texts(texts):
            animations = []
            for text in texts:
                # 高亮和放大
                highlight_enlarge = text.animate.set_color(YELLOW).scale(1.5)
                animations.append(highlight_enlarge)
                # 同时应用高亮和放大
            self.play(*animations, run_time=0.5)
            self.wait(0.1)

            # 旋转晃动
            shake_angles = [
                15 * DEGREES,
                -15 * DEGREES,
                15 * DEGREES,
                -15 * DEGREES,
                15 * DEGREES,
                -15 * DEGREES,
                0,
            ]
            for angle in shake_angles:
                shake_animations = [text.animate.rotate(angle) for text in texts]
                self.play(*shake_animations, run_time=0.05)
            self.wait(0.3)

            # 恢复到默认外观
            restore_animations = [
                text.animate.scale(1 / 1.5).set_color(WHITE) for text in texts
            ]
            self.play(*restore_animations, run_time=0.5)
            self.wait(0.5)

            # 先同时对两个 ">0" 进行操作

        animate_texts(greater_than_zero)

        # 然后对 "<0" 进行相同的操作
        animate_texts(less_than_zero)

        replacement_numbers = [
            MathTex("0"),
            MathTex("0"),
            MathTex("-0.81"),  # 如果需要数学公式格式，可以使用 MathTex
        ]
        Group1 = VGroup(
            numbers_group[0], number_replacements[0], multiply_signs[0], sign_labels[0]
        )
        Group2 = VGroup(
            numbers_group[1], number_replacements[1], multiply_signs[1], sign_labels[1]
        )
        Group3 = VGroup(
            numbers_group[2], number_replacements[2], multiply_signs[2], sign_labels[2]
        )

        # 将 replacement_numbers 移动到 Group1 和 Group2 的中心位置
        replacement_numbers[0].move_to(Group1.get_center() - RIGHT)
        replacement_numbers[1].move_to(Group2.get_center() - RIGHT)
        replacement_numbers[2].move_to(Group3.get_center() - RIGHT)

        self.play(
            ReplacementTransform(Group1, replacement_numbers[0]),
            ReplacementTransform(Group2, replacement_numbers[1]),
            run_time=1,
        )
        self.wait(0.5)
        self.play(ReplacementTransform(Group3, replacement_numbers[2]))

        text1 = Text("损失/代价").next_to(replacement_numbers[2], RIGHT)
        self.play(Write(text1))
