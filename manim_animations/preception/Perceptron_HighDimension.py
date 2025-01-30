"""  
Summary:  

This code utilizes the Manim library to create two animation scenes that illustrate linear models and related concepts in machine learning, including:  

1. OutputFunctionTransform: Demonstrates the construction of the output function z = w^T x + b for a linear model. It gradually replaces features and weights to show how the feature vector and weight vector are combined into the output formula, ultimately presenting the expression in matrix form.  

2. HyperplaneTransform: Illustrates different representations of hyperplanes and their loss functions, including 2D and 4D hyperplanes, as well as the vectorized form of the loss function. It concludes by introducing a commonly used loss function in perceptrons and demonstrating the gradient descent update formula.  
"""  

from manim import *  


class OutputFunctionTransform(Scene):  
    def setup_deepness(self):  
        pass  

    def construct(self):  
        # 1) Configure and display the formula (and check indices)  
        # substrings_to_isolate specifies the parts to be isolated  
        output_formula = MathTex(  
            r"z = w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + w_{4}x_{4} + b",  
            substrings_to_isolate=[  
                "z",  
                "w_{1}x_{1}",  
                "w_{2}x_{2}",  
                "w_{3}x_{3}",  
                "w_{4}x_{4}",  
                "b",  
            ],  
        ).to_edge(UP)  

        # Print the index of each part in the formula (for debugging)  
        for index, part in enumerate(output_formula):  
            print(f"index = {index}, content = {part}")  

        # 2) Create and display text related to sepal and petal  
        sepal_length_text = Text("Sepal Length").set_color(BLUE)  
        sepal_width_text = Text("Sepal Width").set_color(BLUE)  
        petal_length_text = Text("Petal Length").set_color(BLUE)  
        petal_width_text = Text("Petal Width").set_color(BLUE)  

        left_group = VGroup(sepal_length_text, sepal_width_text).arrange(DOWN, buff=1)  
        right_group = VGroup(petal_length_text, petal_width_text).arrange(DOWN, buff=1)  
        all_groups = VGroup(left_group, right_group).arrange(RIGHT, buff=3).scale(0.8)  

        self.play(Write(left_group))  
        self.play(Write(right_group))  
        self.wait(1)  

        # 3) Create features corresponding to x_i  
        feature_1 = MathTex("x_{1}").set_color(RED).next_to(sepal_length_text, RIGHT, buff=0.3)  
        feature_2 = MathTex("x_{2}").set_color(RED).next_to(sepal_width_text, RIGHT, buff=0.3)  
        feature_3 = MathTex("x_{3}").set_color(RED).next_to(petal_length_text, RIGHT, buff=0.3)  
        feature_4 = MathTex("x_{4}").set_color(RED).next_to(petal_width_text, RIGHT, buff=0.3)  
        features_group = VGroup(feature_1, feature_2, feature_3, feature_4)  

        self.play(Write(features_group))  
        self.wait(1)  

        # 4) Create weights w_i and replace the original sepal/petal texts  
        weight_1 = MathTex("w_{1}").set_color(BLUE).move_to(sepal_length_text)  
        weight_2 = MathTex("w_{2}").set_color(BLUE).move_to(sepal_width_text)  
        weight_3 = MathTex("w_{3}").set_color(BLUE).move_to(petal_length_text)  
        weight_4 = MathTex("w_{4}").set_color(BLUE).move_to(petal_width_text)  

        self.play(  
            ReplacementTransform(sepal_length_text, weight_1),  
            ReplacementTransform(sepal_width_text, weight_2),  
            ReplacementTransform(petal_length_text, weight_3),  
            ReplacementTransform(petal_width_text, weight_4),  
            feature_1.animate.next_to(weight_1, RIGHT, buff=0.2),  
            feature_2.animate.next_to(weight_2, RIGHT, buff=0.2),  
            feature_3.animate.next_to(weight_3, RIGHT, buff=0.2),  
            feature_4.animate.next_to(weight_4, RIGHT, buff=0.2),  
        )  
        self.wait(1)  

        # 5) Combine (w_i x_i) and move them to the corresponding positions in the formula  
        wx_positions = [  
            (VGroup(weight_1, feature_1), output_formula[2].get_center()),  # w1x1 at index 2  
            (VGroup(weight_2, feature_2), output_formula[4].get_center()),  # w2x2 at index 4  
            (VGroup(weight_3, feature_3), output_formula[6].get_center()),  
            (VGroup(weight_4, feature_4), output_formula[8].get_center()),  
        ]  

        # Move to target positions simultaneously  
        animations = []  
        for wx_group, target_position in wx_positions:  
            animations.append(wx_group.animate.move_to(target_position))  

        self.play(*animations)  
        self.wait(1)  

        # 6) Fade in the remaining parts of the formula  
        # output_formula contains: 0:z, 1:=, 2:w1x1, 3:+, 4:w2x2, 5:+, 6:w3x3, 7:+, 8:w4x4, 9:+, 10:b  
        parts_to_fade_in = [  
            output_formula[0],  # z  
            output_formula[1],  # =  
            output_formula[3],  # +  
            output_formula[5],  # +  
            output_formula[7],  # +  
            output_formula[9],  # +  
            output_formula[10],  # b  
        ]  

        self.play(*[FadeIn(part) for part in parts_to_fade_in])  
        self.wait(1)  

        # 7) Reset the colors of w_i and x_i to white  
        self.play(  
            weight_1.animate.set_color(WHITE),  
            weight_2.animate.set_color(WHITE),  
            weight_3.animate.set_color(WHITE),  
            weight_4.animate.set_color(WHITE),  
            feature_1.animate.set_color(WHITE),  
            feature_2.animate.set_color(WHITE),  
            feature_3.animate.set_color(WHITE),  
            feature_4.animate.set_color(WHITE),  
        )  
        self.wait(1)  

        # 8) Display column vectors W and X, then proceed with subsequent animations  
        vector_W = MathTex(  
            r"\mathbf{W} = \begin{bmatrix} w_{1} \\ w_{2} \\ w_{3} \\ w_{4} \end{bmatrix}"  
        ).to_edge(LEFT).scale(0.8)  
        
        vector_X = MathTex(  
            r"\mathbf{X} = \begin{bmatrix} x_{1} \\ x_{2} \\ x_{3} \\ x_{4} \end{bmatrix}"  
        ).next_to(vector_W, RIGHT, buff=0.3).scale(0.8)  

        self.play(Write(vector_W), Write(vector_X))  
        self.wait(1)  

        # 9) Create the equals sign and vectors  
        transposed_W = MathTex(r"\mathbf{W}^{T}", color=BLUE).next_to(vector_X, RIGHT, buff=1).scale(0.8)  
        vector_X_color = MathTex(r"\mathbf{X}", color=GREEN).next_to(transposed_W, RIGHT, buff=0.075).scale(0.8)  
        equals_sign_1 = MathTex("=", color=WHITE).next_to(vector_X_color, RIGHT, buff=0.2).scale(0.8)  
        equals_sign_2 = MathTex("=", color=WHITE).next_to(equals_sign_1, DOWN, buff=1.5).scale(0.8)  

        self.play(Write(transposed_W), Write(vector_X_color), Write(equals_sign_1))  
        self.wait(1)  

        # 10) Create matrix objects  
        W_matrix = Matrix(  
            [["w_{1}", "w_{2}", "w_{3}", "w_{4}"]],  
            h_buff=0.8,  
            bracket_h_buff=SMALL_BUFF,  
            bracket_v_buff=SMALL_BUFF,  
        ).next_to(equals_sign_1, RIGHT, buff=0.2).scale(0.8)  

        X_matrix = Matrix(  
            [["x_{1}"], ["x_{2}"], ["x_{3}"], ["x_{4}"]],  
            v_buff=0.6,  
            bracket_h_buff=SMALL_BUFF,  
            bracket_v_buff=SMALL_BUFF,  
        ).next_to(W_matrix, RIGHT, buff=0.2).scale(0.8)  

        self.play(Write(W_matrix), Write(X_matrix))  
        self.wait(1)  

        # 11) Create the expanded expression and set the parts to isolate  
        expanded_expression = MathTex(  
            r"w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + w_{4}x_{4}",  
            substrings_to_isolate=[  
                "w_{1}",  
                "x_{1}",  
                "w_{2}",  
                "x_{2}",  
                "w_{3}",  
                "x_{3}",  
                "w_{4}",  
                "x_{4}",  
            ],  
        ).scale(0.8).next_to(equals_sign_2, RIGHT, buff=0.2)  

        self.play(Write(equals_sign_2))  
        self.wait(1)  

        # 12) Get matrix entries for TransformFromCopy  
        w_entries = W_matrix.get_entries()  # [w1, w2, w3, w4]  
        x_entries = X_matrix.get_entries()  # [x1, x2, x3, x4]  

        # Configure the correspondence (w_i, x_i) with the corresponding indices in expanded_expression  
        # Possible split order in expanded_expression: 0:w1, 1:x1, 2:+, 3:w2, 4:x2, 5:+, 6:w3, 7:x3, 8:+, 9:w4, 10:x4  
        pair_data = [  
            (w_entries[0], x_entries[0], 0, 1, 2),  
            (w_entries[1], x_entries[1], 3, 4, 5),  
            (w_entries[2], x_entries[2], 6, 7, 8),  
            (w_entries[3], x_entries[3], 9, 10, None),  
        ]  

        # 13) Loop through and play animations  
        for w_value, x_value, idx_w, idx_x, idx_plus in pair_data:  
            # a) Set the current w and x to color  
            self.play(  
                w_value.animate.set_color(BLUE),  
                x_value.animate.set_color(RED),  
            )  
            self.wait(0.15)  

            # b) TransformFromCopy to the expanded expression  
            self.play(  
                TransformFromCopy(w_value, expanded_expression[idx_w]),  
                TransformFromCopy(x_value, expanded_expression[idx_x]),  
                w_value.animate.set_color(WHITE),  
                x_value.animate.set_color(WHITE),  
            )  
            self.wait(0.15)  

            # c) Add the plus sign (if there is one)  
            if idx_plus is not None:  
                self.play(Write(expanded_expression[idx_plus], run_time=0.25))  

        self.wait(1)  

        # 14) Replace with matrix form (including already displayed w1, x1, w2, x2...)  
        # Avoid residual w1, x1...  
        matrix_formula = MathTex(r"z = \mathbf{w}^T \mathbf{x} + b").move_to(output_formula.get_center()).scale(0.8)  
        self.play(  
            ReplacementTransform(  
                VGroup(  
                    output_formula,  
                    weight_1,  
                    feature_1,  
                    weight_2,  
                    feature_2,  
                    weight_3,  
                    feature_3,  
                    weight_4,  
                    feature_4,  
                ),  
                matrix_formula,  
            )  
        )  
        self.wait(1)  


class HyperplaneTransform(Scene):  
    def construct(self):  
        # 1. Top formula: z = w^T x + b  
        z_equation = MathTex(r"z = \mathbf{w}^T \mathbf{x} + b")  
        z_equation.to_edge(UP).scale(0.8)  
        self.play(Write(z_equation))  

        # 2. 2D hyperplane and loss function  
        two_var_plane_equation = MathTex(r"w_{1} x_{1} + w_{2} x_{2} + b = 0")  
        two_var_loss_equation = MathTex(r"L = (w_{1} x_{1} + w_{2} x_{2} + b)\, y")  

        two_var_plane_equation.next_to(z_equation, DOWN, buff=1.5).scale(0.8)  
        two_var_loss_equation.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(Write(two_var_plane_equation), Write(two_var_loss_equation))  
        self.wait(1)  

        # 3. 4D hyperplane and loss function  
        four_var_plane_equation = MathTex(  
            r"w_{1} x_{1} + w_{2} x_{2} + w_{3} x_{3} + w_{4} x_{4} + b = 0"  
        )  
        four_var_loss_equation = MathTex(  
            r"L = (w_{1} x_{1} + w_{2} x_{2} + w_{3} x_{3} + w_{4} x_{4} + b)\, y"  
        )  

        four_var_plane_equation.next_to(z_equation, DOWN, buff=1.5).scale(0.8)  
        four_var_loss_equation.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(  
            ReplacementTransform(two_var_plane_equation, four_var_plane_equation),  
            ReplacementTransform(two_var_loss_equation, four_var_loss_equation),  
        )  
        self.wait(1)  

        # 4. Vectorized form and corresponding loss function  
        matrix_plane_equation = MathTex(r"\mathbf{w}^T \mathbf{x} + b = 0")  
        vector_loss_equation = MathTex(r"L = \mathbf{w}^T \mathbf{x} + b")  

        matrix_plane_equation.next_to(z_equation, DOWN, buff=1.5).scale(0.8)  
        vector_loss_equation.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(  
            ReplacementTransform(four_var_plane_equation, matrix_plane_equation),  
            ReplacementTransform(four_var_loss_equation, vector_loss_equation),  
        )  
        self.wait(1)  

        # 5. Introduce the commonly used loss function in perceptrons (hinge-like)  
        perceptron_loss_function = MathTex(  
            r"L = \max\left(0,\,-y\bigl(\mathbf{w}^T \mathbf{x} + b\bigr)\right)"  
        )  
        perceptron_loss_function.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(ReplacementTransform(vector_loss_equation, perceptron_loss_function))  
        self.wait(1)  

        # 7. Gradient descent update formulas  
        gradient_update_weights = MathTex(  
            r"\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}"  
        )  
        gradient_update_bias = MathTex(  
            r"b \leftarrow b - \eta \frac{\partial L}{\partial b}"  
        )  

        # Arrange the two formulas horizontally  
        gradient_updates = VGroup(gradient_update_weights, gradient_update_bias)  
        gradient_updates.arrange(RIGHT, buff=0.5).scale(0.8)  
        gradient_updates.next_to(z_equation, DOWN, buff=4.5)  

        self.play(Write(gradient_updates))  
        self.wait(1)