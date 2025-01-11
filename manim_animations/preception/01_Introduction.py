from manim import *


class Prologue(Scene):
    def construct(self):
        text1 = Text("数据分类", color=RED)
        text2 = Text("字迹识别", color=YELLOW)
        text3 = Text("肿瘤识别", color=BLUE)
        text4 = Text("......", color=ORANGE)

        self.play(Write(text1), runtime=1.5)
        self.play(text1.animate.shift(UP * 2))
        self.wait(1)

        text2.next_to(text3, LEFT, buff=1)
        self.play(Write(text2), runtime=1)
        self.wait(1)

        self.play(Write(text3), runtime=1)
        self.wait(1)

        text4.next_to(text3, RIGHT, buff=1)
        self.play(Write(text4))
        self.wait(1)
        self.play(FadeOut(Group(text1, text2, text3, text4)))
        self.wait(1)


# if __name__ == "__main__":
#     config.pixel_height = 720  # 设置垂直分辨率
#     config.pixel_width = 1280  # 设置水平分辨率
#     config.frame_rate = 30  # 设置帧率

#     sence = LirsDataTitle()
#     sence.render(preview=True)
