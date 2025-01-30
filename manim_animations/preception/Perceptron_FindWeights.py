"""  
Summary:

This code uses the Manim library to create an animated scene that demonstrates the weight calculation and classification process of the Perceptron model.
Specific steps include:

1. data processing: use IrisDataProcessor to obtain features and labels for the Iris dataset.
2. Setting weights and biases: Define the weights and biases of the perceptron, and calculate the slope and intercept of the line.
3. Create Axes: Create a coordinate system using the Axes class and set the range and labels for the x- and y-axes.
4. Mathematical formula presentation: create mathematical formulas to represent the linear equations of the perceptron and replace the values of weights and biases step by step.
5. Data point visualisation: Filter different categories of Iris data points and plot scatter plots in the coordinate system.
6. Plotting hyperplanes: Plot hyperplanes (decision boundaries) based on calculated slopes and intercepts.
7. select specific data points: select specific Setosa and Versicolor data points and create labels for them.
8. Animation effects: animation to show the process of selecting data points, replacing variables in mathematical formulas, and drawing hyperplanes.
9. Classification Result Display: Calculate the classification result for each data point and display the corresponding text (e.g. "Classification Correct" or "Classification Error") according to the result.
10. Final result: display the result of loss/cost calculation, highlighting the output of the perceptron model.

"""

from manim import *  
from DataProcessor import IrisDataProcessor  

class PerceptronFindWeights(Scene):  
    def construct(self):  
        # Data processing  
        data_processor = IrisDataProcessor()  
        features, labels = data_processor.get_data()  

        # Define weights and bias for the perceptron  
        weight1 = -0.4  
        weight2 = 0.9  
        bias = -0.3  
        slope = -weight1 / weight2  # Calculate the slope of the decision boundary  
        intercept = -bias / weight2  # Calculate the y-intercept of the decision boundary  

        # Create axes for the plot  
        axes = Axes(  
            x_range=[4, 7.5, 0.5],  # x-axis range  
            y_range=[1, 5, 1],  # y-axis range  
            axis_config={"color": BLUE, "include_numbers": True},  
        )  

        # Define variable replacements for the equation  
        replacements = {"w_1": "-0.4", "w_2": "0.9", "b": "-0.3"}  

        # Create the initial mathematical equation and isolate variables for replacement  
        equation = MathTex(  
            r"w_1 \times x_1 + w_2 \times x_2 + b = 0",  
            font_size=46,  
            substrings_to_isolate=["w_1", "x_1", "w_2", "x_2", "+ b", "0"],  
        ).to_edge(UP)  

        # Create the equation with actual values  
        equation_with_values = MathTex(  
            r"-0.4 \times x_1 + 0.9 \times x_2 - 0.3 = 0", font_size=42  
        ).to_edge(UP)  

        self.add(equation)  # Add the equation to the scene  
        self.wait(1)  

        # Add labels for the axes  
        x_label = axes.get_x_axis_label(Text("Sepal Length (cm)").scale(0.6))  
        y_label = axes.get_y_axis_label(Text("Sepal Width (cm)").scale(0.6))  

        # Use numpy to filter data for different species  
        setosa_indices = np.where(labels == 0)[0]  
        versicolor_indices = np.where(labels == 1)[0]  

        setosa_points = [(features[i, 0], features[i, 1]) for i in setosa_indices]  
        versicolor_points = [(features[i, 0], features[i, 1]) for i in versicolor_indices]  

        # Create scatter points for each species  
        setosa_dots = [Dot(axes.c2p(x, y), color=BLUE) for x, y in setosa_points]  
        versicolor_dots = [  
            Dot(axes.c2p(x, y), color=ORANGE) for x, y in versicolor_points  
        ]  

        # Plot the decision boundary (hyperplane)  
        hyperplane = axes.plot(  
            lambda x: slope * x + intercept, color=WHITE, x_range=[4, 7]  
        )  

        # Create a VGroup to hold the axes, labels, scatter points, and hyperplane  
        plot_group = VGroup(  
            axes, x_label, y_label, *setosa_dots, *versicolor_dots, hyperplane  
        )  

        # Scale the entire plot group  
        plot_group.scale(0.8)  

        # Add the plot elements to the scene  
        self.play(Create(axes), Write(x_label), Write(y_label))  
        self.play(*[Create(dot) for dot in setosa_dots + versicolor_dots])  

        # Iterate through each variable to create replacement text and play animations  
        for var, value in replacements.items():  
            # Get the part of the equation to replace  
            var_object = equation.get_part_by_tex(var)  

            if var_object is None:  
                # If the part is not found, skip to the next iteration  
                print(f"Part to replace not found: {var}")  
                continue  

            # Create a text object for the value, maintaining the same position and size  
            value_tex = MathTex(value, font_size=38).move_to(var_object)  

            # Adjust the position of the replacement text based on the variable  
            if var == "w_1":  
                value_tex.shift(LEFT * 0.06, UP * 0.06)  # Adjust offset as needed  
            elif var == "w_2":  
                value_tex.shift(UP * 0.06, RIGHT * 0.05)  

            # Use ReplacementTransform for the replacement animation  
            self.play(ReplacementTransform(var_object, value_tex))  

        self.wait(2)  

        # Create and display the hyperplane  
        self.play(Create(hyperplane))  
        self.wait(1)  

        # Define specific data points to highlight  
        setosa_selected = (5, 3.6)  
        versicolor_selected_1 = (6, 2.2)  
        versicolor_selected_2 = (5.9, 3.2)  

        # Create Dot and Label for Setosa  
        setosa_dot = Dot(axes.c2p(*setosa_selected), color=BLUE)  
        setosa_label = MathTex(r"(5, 3.6)", font_size=24).next_to(  
            setosa_dot, UP + RIGHT  
        )  

        # Create Dot and Label for Versicolor 1  
        versicolor_dot_1 = Dot(axes.c2p(*versicolor_selected_1), color=ORANGE)  
        versicolor_label_1 = MathTex(r"(6, 2.2)", font_size=24).next_to(  
            versicolor_dot_1, UP + RIGHT  
        )  

        # Create Dot and Label for Versicolor 2  
        versicolor_dot_2 = Dot(axes.c2p(*versicolor_selected_2), color=ORANGE)  
        versicolor_label_2 = MathTex(r"(5.9, 3.2)", font_size=24).next_to(  
            versicolor_dot_2, UP + RIGHT  
        )  

        # Create VGroups for all dots and selected dots  
        all_dots = VGroup(*setosa_dots, *versicolor_dots)  
        selected_dots = VGroup(setosa_dot, versicolor_dot_1, versicolor_dot_2)  
        selected_labels = VGroup(setosa_label, versicolor_label_1, versicolor_label_2)  

        # Add selected points and labels to the scene  
        self.play(Create(setosa_dot))  
        self.play(Create(versicolor_dot_1))  
        self.play(Create(versicolor_dot_2))  

        # Fade out non-selected points  
        self.play(FadeOut(all_dots))  

        # Write the selected labels  
        self.play(Write(selected_labels))  
        self.wait(1)  
        self.play(FadeOut(axes, hyperplane, x_label, y_label))  

        # Group points and labels  
        points = VGroup(setosa_dot, versicolor_dot_1, versicolor_dot_2)  
        labels = VGroup(setosa_label, versicolor_label_1, versicolor_label_2)  

        # Define target positions for the selected points  
        top_right_point_target = (  
            setosa_dot.copy().to_edge(UP + RIGHT).shift(LEFT * 4 + DOWN * 2)  
        )  
        spacing = 1.5  

        # Create a list of target positions  
        targets = [top_right_point_target]  
        for i in range(1, len(points)):  
            target = top_right_point_target.copy().shift(DOWN * spacing * i)  
            targets.append(target)  

        # Create animations for moving points and labels  
        animations = []  
        for point, label, target in zip(points, labels, targets):  
            animations.append(point.animate.move_to(target.get_center()))  
            animations.append(label.animate.next_to(target, LEFT * 18).scale(1.5))  

        # Play the movement animations  
        self.play(AnimationGroup(*animations, lag_ratio=0))  

        # Separate the numbers in the labels  
        # First group of numbers (e.g., "5", "6", "5.9")  
        first_numbers = ["5", "6", "5.9"]  
        # Second group of numbers (e.g., "3.6", "2.2", "3.2")  
        second_numbers = ["3.6", "2.2", "3.2"]  

        # Create new labels for the first group of numbers  
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

        # Create new labels for the second group of numbers  
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

        # Combine the separated labels  
        labels_replace = VGroup(number_labels_replace, decimal_labels_replace)  

        # Play the label replacement animations  
        replace_animations = []  
        for original_label, (new_number, new_decimal) in zip(  
            labels, zip(number_labels_replace, decimal_labels_replace)  
        ):  
            # Replace the first group of numbers  
            replace_animations.append(ReplacementTransform(original_label, new_number))  
            # Add the second group of numbers  
            replace_animations.append(  
                ReplacementTransform(original_label.copy(), new_decimal)  
            )  

        # Play the animations  
        self.play(*replace_animations)  

        # Define multipliers and bias for calculations  
        multiplier1 = ["-0.3", "-0.3", "-0.3"]  
        multiplier2 = ["0.9", "0.9", "0.9"]  
        bias = ["0.3", "0.3", "0.3"]  

        # Move the second group of numbers (decimal_labels_replace) to the right  
        self.play(decimal_labels_replace.animate.shift(RIGHT * 1.1))  

        # Create the first column of multipliers and multiplication signs  
        multiplier1_group = VGroup(  
            *[  
                VGroup(  
                    MathTex(multiplier1[i], font_size=36),  # Multiplier  
                    MathTex(r"\times"),  # Multiplication sign  
                )  
                .arrange(RIGHT, buff=0.1)  
                .next_to(number_labels_replace[i], LEFT * 0.45)  
                for i in range(len(number_labels_replace))  
            ]  
        )  

        # Create the second column of multipliers and multiplication signs  
        multiplier2_group = VGroup(  
            *[  
                VGroup(  
                    MathTex(multiplier2[i], font_size=36),  # Multiplier  
                    MathTex(r"\times"),  # Multiplication sign  
                )  
                .arrange(RIGHT, buff=0.1)  
                .next_to(decimal_labels_replace[i], LEFT * 0.45)  
                for i in range(len(decimal_labels_replace))  
            ]  
        )  

        # Animate the simultaneous display of all multipliers and multiplication signs  
        self.play(FadeIn(multiplier1_group), FadeIn(multiplier2_group))  

        # Create plus signs and align them row-wise  
        plus_signs = VGroup()  
        for i in range(len(number_labels_replace)):  
            # Create a plus sign  
            plus = MathTex("+").scale(0.9)  
            # Position the plus sign between the two columns and align it with the corresponding row  
            plus.next_to(number_labels_replace[i], RIGHT, buff=0.1).align_to(  
                number_labels_replace[i], DOWN  
            )  
            plus_signs.add(plus)  

        # Add the plus sign animations  
        self.play(FadeIn(plus_signs))  

        # Create bias terms and align them row-wise  
        bias_group = VGroup()  
        for i in range(len(number_labels_replace)):  
            # Create the bias term (e.g., "+ 0.3")  
            plus_bias = MathTex(f"- {bias[i]}", font_size=36)  
            # Position the bias term to the right of the plus sign and align it with the corresponding row  
            plus_bias.next_to(decimal_labels_replace[i], RIGHT, buff=0.1).align_to(  
                decimal_labels_replace[i], DOWN  
            )  
            bias_group.add(plus_bias)  

        # Add the bias term animations  
        self.play(FadeIn(bias_group))  

        # Calculate results considering the bias term  
        results = [  
            f"{float(first_numbers[i]) * float(multiplier1[i]) + float(second_numbers[i]) * float(multiplier2[i]) - float(bias[i]):.2f}"  
            for i in range(len(first_numbers))  
        ]  
        print(results)  

        # Create equals signs  
        equals_group = VGroup(  
            *[  
                MathTex("=", font_size=36)  
                .next_to(bias_group[i], RIGHT, buff=0.1)  
                .align_to(bias_group[i], DOWN)  
                for i in range(len(results))  
            ]  
        )  

        # Create an empty group to hold the numbers (including negative signs and numbers)  
        numbers_group = VGroup()  

        # Identify negative numbers and remove the negative sign  
        is_negative = [num.startswith("-") for num in results]  
        stripped_numbers = [  
            num[1:] if neg else num for num, neg in zip(results, is_negative)  
        ]  

        for eq, num, neg in zip(equals_group, stripped_numbers, is_negative):  
            # Create number objects  
            number = MathTex(num, font_size=36, color=BLUE if not neg else ORANGE)  

            # If it's a negative number, create a negative sign object  
            if neg:  
                minus = MathTex("-", font_size=36, color=ORANGE)  
                # Position the negative sign to the left of the number  
                minus.next_to(eq, RIGHT, buff=0.1)  
                number.next_to(minus, RIGHT, buff=0.05)  
                # Combine the negative sign and number  
                combined = VGroup(minus, number).arrange(RIGHT, buff=0.05)  
            else:  
                # If it's a positive number, use \phantom{-} as a placeholder for alignment  
                phantom = MathTex("-", font_size=36, color=WHITE).set_opacity(0)  
                phantom.next_to(eq, RIGHT, buff=0.1)  
                number.next_to(phantom, RIGHT, buff=0.05)  
                # Combine the placeholder and number  
                combined = VGroup(phantom, number).arrange(RIGHT, buff=0.05)  

            # Add the combined number to the numbers group  
            numbers_group.add(combined)  

        numbers_group.arrange(DOWN, buff=1.5).next_to(equals_group, RIGHT, buff=0.3)  

        # Adjust the position of the numbers group to align vertically with the equals group  
        for eq, num in zip(equals_group, numbers_group):  
            num.align_to(eq, DOWN)  

        # Animate the simultaneous display of equals signs and results  
        self.play(FadeIn(equals_group), FadeIn(numbers_group))  

        multiply_signs = VGroup(  
            *[  
                MathTex(r"\times", font_size=36).next_to(  
                    numbers_group[i], RIGHT, buff=0.4  
                )  
                for i in range(len(numbers_group))  
            ]  
        )  

        # Define the final numbers for replacement  
        final_numbers = ["1", "-1", "-1"]  
        # Create a group for the initial numbers, positioned the same as the equals group but invisible  
        number_replacements = VGroup()  

        for result in final_numbers:  
            # Check if the number is negative  
            is_negative = result.startswith("-")  
            stripped_number = result[1:] if is_negative else result  

            # Create number objects, color based on positive or negative  
            number = MathTex(  
                stripped_number, font_size=36, color=BLUE if not is_negative else ORANGE  
            )  

            if is_negative:  
                # If it's a negative number, create a negative sign object  
                minus = MathTex("-", font_size=36, color=ORANGE)  
                combined_number = VGroup(minus, number).arrange(RIGHT, buff=0.05)  
            else:  
                # If it's a positive number, use \phantom{-} as a placeholder for alignment  
                phantom = MathTex("-", font_size=36, color=WHITE).set_opacity(0)  
                combined_number = VGroup(phantom, number).arrange(RIGHT, buff=0.05)  

            # Add the combined number to the number replacements group  
            number_replacements.add(combined_number)  

        # Arrange the number replacements vertically, aligning with the equals group  
        number_replacements.arrange(DOWN, buff=1.5).next_to(  
            multiply_signs, RIGHT, buff=0.3  
        )  

        # Adjust the position of the number replacements to align vertically with the equals group  
        for eq, num in zip(multiply_signs, number_replacements):  
            num.align_to(eq, DOWN)  

        # Use ReplacementTransform to replace dots with numbers  
        self.play(  
            ReplacementTransform(setosa_dot, number_replacements[0]),  
            ReplacementTransform(versicolor_dot_1, number_replacements[1]),  
            ReplacementTransform(versicolor_dot_2, number_replacements[2]),  
        )  
        self.wait(2)  

        # Continue with subsequent animations  
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

        # **Add ">0" or "<0" symbols**  
        # Define the signs corresponding to each number  
        signs = [">0", ">0", "<0"]  

        # Create sign labels  
        sign_labels = VGroup()  
        for i in range(len(number_replacements)):  
            sign_label = MathTex(signs[i], font_size=36).next_to(  
                number_replacements[i], RIGHT, buff=0.4  
            )  
            sign_labels.add(sign_label)  

        # Animate the display of sign labels  
        self.play(FadeIn(sign_labels))  
        self.wait(1)

                # Create text labels indicating classification results  
        correct_classification_text = Text("Correct Classification", font_size=36, color=GREEN).next_to(  
            numbers_group[0], LEFT  
        )  
        incorrect_classification_text = Text("Incorrect Classification", font_size=36, color=RED).next_to(  
            numbers_group[2], LEFT  
        )  
        correct_classification_text_2 = Text("Correct Classification", font_size=36, color=GREEN).next_to(  
            numbers_group[1], LEFT  
        )  
        classification_text_group = VGroup(correct_classification_text, incorrect_classification_text, correct_classification_text_2)  

        # Animate the display of classification result texts  
        self.play(FadeIn(classification_text_group))  
        self.wait(1)  

        # Identify the elements that need to be animated  
        greater_than_zero_labels = [  
            text for text in sign_labels if text.get_tex_string() == ">0"  
        ]  
        less_than_zero_labels = [text for text in sign_labels if text.get_tex_string() == "<0"]  

        # Define a function to apply highlighting, scaling, and shaking effects to the text  
        def animate_texts(texts):  
            animations = []  
            for text in texts:  
                # Highlight and enlarge the text  
                highlight_enlarge = text.animate.set_color(YELLOW).scale(1.5)  
                animations.append(highlight_enlarge)  

            # Play highlight and enlarge animations  
            self.play(*animations, run_time=0.5)  
            self.wait(0.1)  

            # Apply shaking effect  
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

            # Restore the text to its default appearance  
            restore_animations = [  
                text.animate.scale(1 / 1.5).set_color(WHITE) for text in texts  
            ]  
            self.play(*restore_animations, run_time=0.5)  
            self.wait(0.5)  

        # Animate the ">0" labels  
        animate_texts(greater_than_zero_labels)  

        # Animate the "<0" labels  
        animate_texts(less_than_zero_labels)  

        # Prepare replacement numbers for the final display  
        replacement_numbers = [  
            MathTex("0"),  
            MathTex("0"),  
            MathTex("-0.81"),  # If mathematical format is needed, use MathTex  
        ]  
        
        # Create groups for the original numbers and their corresponding signs  
        Group1 = VGroup(  
            numbers_group[0], number_replacements[0], multiply_signs[0], sign_labels[0]  
        )  
        Group2 = VGroup(  
            numbers_group[1], number_replacements[1], multiply_signs[1], sign_labels[1]  
        )  
        Group3 = VGroup(  
            numbers_group[2], number_replacements[2], multiply_signs[2], sign_labels[2]  
        )  

        # Move the replacement numbers to the center of the corresponding groups  
        replacement_numbers[0].move_to(Group1.get_center() - RIGHT)  
        replacement_numbers[1].move_to(Group2.get_center() - RIGHT)  
        replacement_numbers[2].move_to(Group3.get_center() - RIGHT)  

        # Play the replacement animations for each group  
        self.play(  
            ReplacementTransform(Group1, replacement_numbers[0]),  
            ReplacementTransform(Group2, replacement_numbers[1]),  
            run_time=1,  
        )  
        self.wait(0.5)  
        self.play(ReplacementTransform(Group3, replacement_numbers[2]))  

        # Create a label for the loss/cost  
        loss_text = Text("Loss/Cost").next_to(replacement_numbers[2], RIGHT)  
        self.play(Write(loss_text))  
        self.wait(2)  