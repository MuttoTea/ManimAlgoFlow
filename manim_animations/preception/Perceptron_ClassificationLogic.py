"""  
Summary:  
This code utilizes the Manim library to create two animated scenes that illustrate the decision-making process in a perceptron model and the graph of the sign function.  

1. JudgeZ: This scene demonstrates the decision rules of the perceptron, comparing the input \( Z \) with the threshold \( A \) and producing classification results (Iris Setosa or Iris Versicolor). The classification logic is clearly expressed through a combination of mathematical formulas and text labels.  

2. FunSign: This scene presents the definition of the sign function and its graph. By plotting the function and marking discontinuities, it helps the audience understand the behavior and characteristics of the sign function.  

Each scene gradually reveals its content through animations, enhancing the audience's understanding and engagement. 
"""  

from manim import *  


class JudgeZ(Scene):  
    def construct(self):  
        # Define font to support Chinese characters  
        chinese_font = "SimHei"  # Ensure this font is installed on the system  

        # Create the first condition line: Z ≥ A -> 1 -> Iris Setosa  
        condition_greater_equal = MathTex("Z", "\geq", "A").set_color(YELLOW)  
        arrow_to_output1 = MathTex("\\rightarrow").set_color(WHITE)  
        output_positive = MathTex("1").set_color(GREEN)  
        arrow_to_label1 = MathTex("\\rightarrow").set_color(WHITE)  
        label_setosa = Text("山鸢尾", font=chinese_font).set_color(BLUE)  

        # Create the second condition line: Z < A -> -1 -> Iris Versicolor  
        condition_less = MathTex("Z", "<", "A").set_color(YELLOW)  
        arrow_to_output2 = MathTex("\\rightarrow").set_color(WHITE)  
        output_negative = MathTex("-1").set_color(RED)  
        arrow_to_label2 = MathTex("\\rightarrow").set_color(WHITE)  
        label_versicolor = Text("变色鸢尾", font=chinese_font).set_color(BLUE)  

        # Combine the first condition line into a single group  
        line_condition1 = VGroup(condition_greater_equal, arrow_to_output1, output_positive, arrow_to_label1, label_setosa).arrange(RIGHT, buff=0.5)  

        # Combine the second condition line into a single group  
        line_condition2 = VGroup(condition_less, arrow_to_output2, output_negative, arrow_to_label2, label_versicolor).arrange(RIGHT, buff=0.5)  

        # Arrange the two lines vertically, aligning them to the left edge  
        condition_lines = VGroup(line_condition1, line_condition2).arrange(DOWN, buff=1, aligned_edge=LEFT)  

        # Move the combined content to the center of the scene  
        condition_lines.move_to(ORIGIN)  

        # Display each element of the condition lines sequentially  
        self.play(Write(condition_greater_equal), Write(condition_less))  
        self.play(Write(arrow_to_output1), Write(output_positive))  
        self.play(Write(arrow_to_output2), Write(output_negative))  
        self.play(Write(arrow_to_label1), Write(label_setosa))  
        self.play(Write(arrow_to_label2), Write(label_versicolor))  
        self.wait(2)  


class FunSign(Scene):  
    def construct(self):  
        # Define the piecewise function expression  
        piecewise_expression = MathTex(  
            r"f(x) = \begin{cases} 1 & x > 0 \\ 0 & x = 0 \\ -1 & x < 0 \end{cases}"  
        )  
        # Center the expression on the screen  
        self.play(Write(piecewise_expression))  
        self.wait(1)  

        # Move the expression up and scale it down  
        self.play(piecewise_expression.animate.scale(0.8).to_edge(UP), run_time=2)  
        self.wait(1)  

        # Create the axes for the graph  
        axes = Axes(  
            x_range=[-3, 3, 1],  # X-axis range  
            y_range=[-2, 2, 1],  # Y-axis range  
            axis_config={"color": BLUE},  # Axis color  
        )  

        # Add labels to the axes  
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")  

        # Define the sign function  
        def sign_function(x):  
            if x > 0:  
                return 1  
            elif x < 0:  
                return -1  
            else:  
                return 0  

        # Generate the graph of the sign function, specifying discontinuities  
        graph = axes.plot(sign_function, color=WHITE, discontinuities=[0])  

        # Create markers for the discontinuities  
        # (0, 0) solid circle  
        closed_circle = Dot(axes.c2p(0, 0), color=WHITE)  
        closed_circle.set_fill(WHITE, opacity=1)  

        # (0, 1) and (0, -1) hollow circles  
        open_circle_pos = Dot(axes.c2p(0, 1), color=WHITE)  
        open_circle_pos.set_fill(BLACK, opacity=1)  
        open_circle_pos.set_stroke(WHITE, width=2)  

        open_circle_neg = Dot(axes.c2p(0, -1), color=WHITE)  
        open_circle_neg.set_fill(BLACK, opacity=1)  
        open_circle_neg.set_stroke(WHITE, width=2)  

        # Group all coordinate-related objects together  
        coord_group = VGroup(  
            axes, axes_labels, graph, closed_circle, open_circle_pos, open_circle_neg  
        )  
        coord_group.scale(0.6).move_to(DOWN)  

        # Display the axes and labels  
        self.play(Create(axes), Write(axes_labels))  
        self.wait(1)  

        # Display the function graph  
        self.play(Create(graph))  

        # Add markers for the discontinuities  
        self.play(  
            FadeIn(closed_circle),  
            FadeIn(open_circle_pos),  
            FadeIn(open_circle_neg),  
            run_time=1,  
        )  
        self.wait(3)