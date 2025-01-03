from manim import *


class Comparison(Scene):
    def construct(self):
        # 创建第一个公式
        equation1 = MathTex(
            r"A \times x + B \times y + C = 0",
            font_size=46,
            substrings_to_isolate=['A', 'x', 'B', 'y', 'C', '0']
        ).move_to(UP)

        # 创建第二个公式
        equation2 = MathTex(
            r"w_1 \times x_1 + w_2 \times x_2 + b = Z",
            font_size=46,
            substrings_to_isolate=['w_1', 'x_1', 'w_2', 'x_2', 'b', 'Z']
        ).next_to(equation1, DOWN, buff=1.5)

        # 显示公式
        self.play(Write(equation1), Write(equation2))
        self.wait(2)

        # 创建修改后的第二个公式（独立的Mobject）
        equation2_modified = MathTex(
            r"w_1 \times x_1 + w_2 \times x_2 + b = 0",
            font_size=46,
            substrings_to_isolate=['w_1', 'x_1', 'w_2', 'x_2', 'b', '0']
        ).next_to(equation1, DOWN, buff=1.5)

        # 添加描述文本
        description = Text("寻找合适的w1、w2、b", font_size=26).next_to(equation2_modified, DOWN, buff=0.5)

        # 使用 TransformMatchingTex 进行公式变换
        self.play(TransformMatchingTex(equation2, equation2_modified))
        self.wait(2)

        # 获取公式部分
        eq_parts1 = {name: equation1.get_part_by_tex(name) for name in ['A', 'x', 'B', 'y', 'C']}
        eq_parts2 = {name: equation2_modified.get_part_by_tex(name) for name in ['w_1', 'x_1', 'w_2', 'x_2', 'b']}

        # 定义高亮映射
        # 分为两组：第一组同时高亮 x 和 x1, y 和 x2 为 ORANGE
        # 第二组同时高亮 A 和 w1, B 和 w2 为 GREEN, C 和 b 为 RED
        highlight_steps = [
            {
                'pairs': [('x', 'x_1'), ('y', 'x_2')],
                'color': ORANGE
            },
            {
                'pairs': [('A', 'w_1'), ('B', 'w_2')],
                'color': GREEN
            },
            {
                'pairs': [('C', 'b')],
                'color': RED
            }
        ]

        # 执行高亮步骤
        for step in highlight_steps:
            animations = []
            for part1, part2 in step['pairs']:
                animations.append(eq_parts1[part1].animate.set_color(step['color']))
                animations.append(eq_parts2[part2].animate.set_color(step['color']))
            self.play(*animations, run_time=1.5)
            self.wait(1)
            # 恢复颜色
            reset_animations = []
            for part1, part2 in step['pairs']:
                reset_animations.append(eq_parts1[part1].animate.set_color(WHITE))
                reset_animations.append(eq_parts2[part2].animate.set_color(WHITE))
            self.play(*reset_animations, run_time=1.0)

            # 显示描述文本
        self.play(Write(description))
        self.wait(1)

        # 淡出公式和描述文本
        self.play(FadeOut(equation1, description))

        # 将修改后的公式移动到屏幕上方
        self.play(equation2_modified.animate.to_edge(UP))
        self.wait(1)


