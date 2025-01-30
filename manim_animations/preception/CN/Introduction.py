"""
摘要：  
该代码使用 Manim 库创建一个简单的动画场景，展示了四个文本元素，分别代表不同的主题：  
1. 数据分类  
2. 字迹识别  
3. 肿瘤识别  
4. 其他主题（以“......”表示）  

动画过程包括：  
- 依次写出每个文本，设置不同的颜色。  
- 将文本移动到不同的位置。  
- 最后将所有文本淡出，结束场景。  
"""  

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