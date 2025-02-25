"""  
摘要：  
该代码使用 Manim 库可视化异或（XOR）逻辑门的工作原理。主要功能包括：  

1. 场景构建：  
   - 创建一个异或门（XorGate）并展示其输入和输出。  
   - 显示不同输入组合（0 和 1）及其对应的输出结果。  

2. 表格展示：  
   - 创建一个表格，展示异或门的输入和输出关系。  
   - 表格包含标题行和数据行，逐行显示数据。  
  
"""  

from manim import *
from CustomClasses import XorGate


class XORProblem(Scene):  
    def construct(self):  
        # 标题  
        title = Text("异或问题").set_color(RED)  
        self.play(Write(title))  
        self.wait(1)  

        self.play(title.animate.to_edge(UP, buff=0.5))  

        # 创建一个 XorGate，放在原点  
        input_color = GREEN  
        output_color = BLUE  
        xor_gate = XorGate(  
            center=ORIGIN - RIGHT,  
            arcs_color=RED,                     # 弧线颜色  
            
            fill_color=RED,                     # 如果想要填充，可以改成 YELLOW 等  
            fill_opacity=0.3                    # 填充透明度  
        ).scale(0.8)  
        self.play(Create(xor_gate))  
        self.wait(2)  

        # 创建输入标签  
        input_label_x1 = MathTex("x_1").move_to(xor_gate.input_point_up.get_center() - RIGHT * 0.5).set_color(input_color)  
        input_label_x2 = MathTex("x_2").move_to(xor_gate.input_point_down.get_center() - RIGHT * 0.5).set_color(input_color)  
        input_label_y = MathTex("y").move_to(xor_gate.output_point.get_center() + RIGHT * 0.5).set_color(output_color)  
        self.play(Create(input_label_x1), Create(input_label_x2), Create(input_label_y))  

        input_combinations = [  
            (0, 0),  
            (1, 1),  
            (1, 0),  
            (0, 1)  
        ]  

        for x1, x2 in input_combinations:
            self.play(
                Transform(input_label_x1, MathTex(x1).move_to(xor_gate.input_point_up.get_center() - RIGHT * 0.5).set_color(input_color)),
                Transform(input_label_x2, MathTex(x2).move_to(xor_gate.input_point_down.get_center() - RIGHT * 0.5).set_color(input_color)),
                run_time=0.5
            )
            self.wait(0.5)

            y = x1 ^ x2
            self.play(
                Transform(input_label_y, MathTex(y).move_to(xor_gate.output_point.get_center() + RIGHT * 0.5).set_color(output_color)),
                run_time=0.5
            )
            self.wait(0.5)
        self.wait(2)

        Group_1 = VGroup(xor_gate, input_label_x1, input_label_x2, input_label_y)
        self.play(Group_1.animate.scale(0.9).to_edge(LEFT, buff=1))
        self.wait(2)
# ── 1. 定义表格数据 ──────────────────────────────────  
        # 包含标题行和数据行  
        table_data = [  
            ["X_{1}", "X_{2}", "X_{1} \\oplus X_{2}"],  # 标题行  
            [1, 1, 0],  
            [0, 0, 0],  
            [0, 1, 1],  
            [1, 0, 1],  
        ]  

        # ── 2. 初始化 Table ────────────────────────────────────  
        # include_outer_lines=True 显示外边框  
        # include_inner_lines=True 可以显示内部的线条（根据需要启用或禁用）  
        # element_to_mobject=lambda x: MathTex(str(x)) 将单元格内容渲染为数学公式形式  
        table = Table(  
            table_data,  
            include_outer_lines=True,   
            element_to_mobject=lambda x: MathTex(x) if isinstance(x, str) else MathTex(str(x)),  
        )  

        # ── 3. 缩放和定位 ──────────────────────────────────────  
        table.scale(0.7)    # 将表格缩放至适当大小  
        table.to_edge(RIGHT, buff=1.0)   # 将表格移动到屏幕右侧 

        table.get_entries((1, 1)).set_color(input_color)  # 将标题行中的 "A" 设置为  
        table.get_entries((1, 2)).set_color(input_color)  # 将标题行中的 "B" 设置为
        table.get_entries((1, 3)).set_color(output_color)  # 将标题行中的 "A XOR B" 设置为
        
        # ── 4. 隐藏数据行的内容 ──────────────────────────────────  
        # 获取所有单元格  
        entries = table.get_entries()  
        num_rows = len(table_data)  
        num_cols = len(table_data[0])  

        # 表头占据第一行（索引从0开始）  
        for i, entry in enumerate(entries):  
            # 计算当前单元格的行和列（行从1开始）  
            current_row = (i // num_cols) + 1  # 行号  
            current_col = (i % num_cols) + 1   # 列号  

            if current_row > 1:  # 如果不是表头行  
                entry.set_opacity(0)  # 隐藏单元格内容  
                
        # ── 5. 播放创建表格的动画 ──────────────────────────────────  
        self.play(Create(table))  # 创建表格边框和表头  
        self.wait(1)               # 等待1秒  

        # ── 6. 逐行显示表格数据 ────────────────────────────────────  
        for row in range(2, len(table_data) + 1):  # 数据行从2开始  
            # 获取当前行的输入A和输入B单元格  
            input_a = table.get_entries((row, 1))  
            input_b = table.get_entries((row, 2))  
            cell = table.get_entries((row, 3))
            # 同时显示输入A和输入B  
            self.play(  
                input_a.animate.set_opacity(1),  
                input_b.animate.set_opacity(1),  
                cell.animate.set_opacity(1),
                run_time=0.5  
            )  
            self.wait(0.2)  
        self.wait(2)  # 最后暂停2秒以展示完整表格 