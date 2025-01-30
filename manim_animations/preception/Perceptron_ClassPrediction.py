"""  
Summary:  

This code utilizes the Manim library to create an animated scene that demonstrates the input computation process in a perceptron model. The specific steps include:  
1. Displaying text labels for sepal length and width along with their corresponding variables (\(x_1\) and \(x_2\)).  
2. Sowing weight variables (\(w_1\) and \(w_2\)) and multiplying them with the input variables.  
3. Presenting the input computation formula of the perceptron through addition and equality, ultimately forming the expression for output \(Z\) and bias \(b\).    

"""  


from manim import *  


class NetInput(Scene):  
    def construct(self):  
        # Create text labels for sepal length and width  
        sepal_length_label = Text("花萼长度", color=RED)  
        sepal_width_label = Text("花萼宽度", color=RED)  
        
        # Create variable representations for sepal length and width  
        sepal_length_var = MathTex("x_{1}", color=YELLOW)  
        sepal_width_var = MathTex("x_{2}", color=YELLOW)  
        
        # Create variable representations for weights  
        weight1_var = MathTex("w_{1}", color=GREEN)  
        weight2_var = MathTex("w_{2}", color=GREEN)  

        # Create vertical groups for labels, variables, and weights  
        label_column = VGroup(sepal_length_label, sepal_width_label).arrange(DOWN, buff=0.5)  
        variable_column = VGroup(sepal_length_var, sepal_width_var).arrange(DOWN, buff=0.5)  
        weight_column = VGroup(weight1_var, weight2_var).arrange(DOWN, buff=0.5)  

        # Arrange the three columns horizontally  
        columns = VGroup(label_column, variable_column, weight_column).arrange(RIGHT, buff=1.5)  

        # Move the entire group to the center of the scene (optional)  
        columns.move_to(ORIGIN)  

        # Create animation groups for each column with a delay  
        animations = [  
            AnimationGroup(Write(label_column), lag_ratio=0),  
            AnimationGroup(Write(variable_column), lag_ratio=0),  
            AnimationGroup(Write(weight_column), lag_ratio=0),  
        ]  

        # Use LaggedStart to control the timing between each column's appearance  
        self.play(LaggedStart(*animations, lag_ratio=1))  
        self.wait(2)  

        # 1. Fade out the label column  
        self.play(FadeOut(label_column))  
        self.wait(1)  

        # 2. Create multiplication symbols and position them next to the corresponding variable  
        multiply_symbol1 = MathTex("\\times").set_color(WHITE)  
        multiply_symbol2 = MathTex("\\times").set_color(WHITE)  

        # 3. Position the multiplication symbols next to the variable representations  
        multiply_symbol1.next_to(sepal_length_var, RIGHT, buff=0.2)  
        multiply_symbol2.next_to(sepal_width_var, RIGHT, buff=0.2)  
        multiplication_group = VGroup(multiply_symbol1, multiply_symbol2)  

        # 4. Move each weight variable next to its corresponding multiplication symbol  
        self.play(  
            weight1_var.animate.next_to(multiply_symbol1, RIGHT, buff=0.2),  
            weight2_var.animate.next_to(multiply_symbol2, RIGHT, buff=0.2),  
        )  

        # 5. Display the multiplication symbols  
        self.play(Write(multiplication_group))  
        self.wait(2)  

        # 6. Create addition and equality symbols  
        addition_symbol = MathTex("+").set_color(WHITE)  
        equals_symbol = MathTex("=").set_color(WHITE)  
        output_z = MathTex("Z").set_color(RED).scale(1.2)  
        bias_b = MathTex("b").set_color(BLUE)  

        # 7. Create groups for the terms being added  
        term1 = VGroup(sepal_length_var, multiply_symbol1, weight1_var)  
        term2 = VGroup(sepal_width_var, multiply_symbol2, weight2_var)  

        # 8. Position term1 to the left of term2  
        self.play(term1.animate.next_to(term2, LEFT, buff=0.7))  
        self.wait(0.5)  

        # 9. Position the addition symbol next to term1 without displaying it yet  
        addition_symbol.next_to(term1, RIGHT, buff=0.2)  
        term2.next_to(addition_symbol, RIGHT, buff=0.2)  

        # 10. Display the addition symbol  
        self.play(Write(addition_symbol))  
        self.wait(1)  

        # 11. Position the equality symbol next to term1  
        equals_symbol.next_to(term1, LEFT, buff=0.5)  
        output_z.next_to(equals_symbol, LEFT, buff=0.5)  

        # 12. Move the equality symbol and Z to their designated positions  
        self.play(  
            term1.animate.next_to(term2, LEFT, buff=0.7),  
            equals_symbol.animate.move_to(equals_symbol.get_center()),  # Keep equals symbol in place  
            output_z.animate.next_to(equals_symbol, LEFT, buff=0.5),  
        )  
        self.wait(0.5)  

        # 13. Display the equality symbol and Z  
        self.play(Write(equals_symbol), Write(output_z))  
        self.wait(1)  

        # 14. Create and position the second addition symbol  
        second_addition_symbol = MathTex("+").set_color(WHITE)  
        second_addition_symbol.next_to(term2, RIGHT, buff=0.3)  
        bias_b.next_to(second_addition_symbol, RIGHT, buff=0.4)  

        # 15. Display the second addition symbol and b  
        self.play(Write(second_addition_symbol), Write(bias_b))  
        self.wait(1)  

        # Fade out all elements related to the input calculation  
        self.play(FadeOut(Group(term1, term2, addition_symbol, equals_symbol, output_z, second_addition_symbol, bias_b)))  
        self.wait(1)
