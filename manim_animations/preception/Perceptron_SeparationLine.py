"""  
Summary:  
This code uses the Manim library to create an animated scene that demonstrates the comparison between two mathematical formulas. Specifically, it includes:  
1. Displaying the standard form of a linear equation and the output formula of a perceptron model.  
2. Transforming the perceptron model's output formula into standard form.  
3. Highlighting corresponding parts of the formulas to help the audience understand the relationships between the variables.  
4. Finally, moving the modified formula to the top of the screen to conclude the scene.  
"""  

from manim import *  


class Comparison(Scene):  
    def construct(self):  
        # Create the first equation in standard form  
        linear_equation = MathTex(  
            r"A \times x + B \times y + C = 0",  
            font_size=46,  
            substrings_to_isolate=["A", "x", "B", "y", "C", "0"],  
        ).move_to(UP)  

        # Create the second equation representing the perceptron model output  
        perceptron_equation = MathTex(  
            r"w_1 \times x_1 + w_2 \times x_2 + b = Z",  
            font_size=46,  
            substrings_to_isolate=["w_1", "x_1", "w_2", "x_2", "b", "Z"],  
        ).next_to(linear_equation, DOWN, buff=1.5)  

        # Display both equations on the screen  
        self.play(Write(linear_equation), Write(perceptron_equation))  
        self.wait(2)  

        # Create a modified version of the perceptron equation (independent Mobject)  
        modified_perceptron_equation = MathTex(  
            r"w_1 \times x_1 + w_2 \times x_2 + b = 0",  
            font_size=46,  
            substrings_to_isolate=["w_1", "x_1", "w_2", "x_2", "b", "0"],  
        ).next_to(linear_equation, DOWN, buff=1.5)  

        # Add descriptive text below the modified perceptron equation  
        description_text = Text("Finding suitable w1, w2, b", font_size=26).next_to(  
            modified_perceptron_equation, DOWN, buff=0.5  
        )  

        # Use TransformMatchingTex to animate the transformation of the perceptron equation  
        self.play(TransformMatchingTex(perceptron_equation, modified_perceptron_equation))  
        self.wait(2)  

        # Extract parts of the equations for highlighting  
        linear_parts = {  
            name: linear_equation.get_part_by_tex(name) for name in ["A", "x", "B", "y", "C"]  
        }  
        perceptron_parts = {  
            name: modified_perceptron_equation.get_part_by_tex(name)  
            for name in ["w_1", "x_1", "w_2", "x_2", "b"]  
        }  

        # Define highlighting steps with corresponding colors  
        highlight_steps = [  
            {"pairs": [("x", "x_1"), ("y", "x_2")], "color": ORANGE},  
            {"pairs": [("A", "w_1"), ("B", "w_2")], "color": GREEN},  
            {"pairs": [("C", "b")], "color": RED},  
        ]  

        # Execute highlighting steps  
        for step in highlight_steps:  
            animations = []  
            for part1, part2 in step["pairs"]:  
                animations.append(linear_parts[part1].animate.set_color(step["color"]))  
                animations.append(perceptron_parts[part2].animate.set_color(step["color"]))  
            self.play(*animations, run_time=1.5)  
            self.wait(1)  
            # Reset colors after highlighting  
            reset_animations = []  
            for part1, part2 in step["pairs"]:  
                reset_animations.append(linear_parts[part1].animate.set_color(WHITE))  
                reset_animations.append(perceptron_parts[part2].animate.set_color(WHITE))  
            self.play(*reset_animations, run_time=1.0)  

        # Display the description text  
        self.play(Write(description_text))  
        self.wait(1)  

        # Fade out the linear equation and description text  
        self.play(FadeOut(linear_equation, description_text))  

        # Move the modified perceptron equation to the top of the screen  
        self.play(modified_perceptron_equation.animate.to_edge(UP))  
        self.wait(1)