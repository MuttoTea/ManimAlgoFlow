"""  
Summary:  
This code utilizes the Manim library to create an animated scene that demonstrates the functionality of the XOR (exclusive OR) logic gate along with its corresponding truth table. The main components include:  

1. XOR Gate Visualization: A graphical representation of the XOR gate is created, with labels for the inputs and output.  
2. Truth Table Display: The truth table for the XOR gate is generated, showing the inputs \(X_1\) and \(X_2\) along with the corresponding output \(X_1 \oplus X_2\).  
3. Point Visualization: Based on the truth table outputs, different colored dots are used to represent the cases where the output is 1 and 0.  
4. Axes and Function Graphs: Axes are drawn, and graphs related to the XOR logic, including a step function and a Sigmoid function, are displayed.
"""  


from manim import *  
from CustomClasses import XorGate  

class XorCoordinateSystem(Scene):  
    def construct(self):  
        # Create a title for the scene  
        title = Text("XOR Problem").set_color(RED).to_edge(UP, buff=0.5)  

        # Create an XOR gate at the origin  
        input_color = GREEN  
        output_color = BLUE  
        xor_gate = XorGate(  
            center=ORIGIN - RIGHT,  
            arcs_color=RED,  # Color of the arcs in the XOR gate  
            input_line_color=input_color,  # Color of the input lines  
            output_line_color=output_color,  # Color of the output line  
            fill_color=RED,  # Fill color of the XOR gate  
            fill_opacity=0.3,  # Opacity of the fill color  
        ).scale(0.8)  

        # Create labels for the inputs and output  
        input_label_x1 = (  
            MathTex("x_1")  
            .move_to(xor_gate.input_point_up.get_center() - RIGHT * 0.5)  
            .set_color(input_color)  
        )  
        input_label_x2 = (  
            MathTex("x_2")  
            .move_to(xor_gate.input_point_down.get_center() - RIGHT * 0.5)  
            .set_color(input_color)  
        )  
        input_label_y = (  
            MathTex("y")  
            .move_to(xor_gate.output_point.get_center() + RIGHT * 0.5)  
            .set_color(output_color)  
        )  

        # Group the XOR gate and its labels  
        Group_1 = VGroup(xor_gate, input_label_x1, input_label_x2, input_label_y)  
        Group_1.scale(0.9).to_edge(LEFT, buff=1)  

        # Define the truth table data for the XOR gate  
        table_data = [  
            ["X_{1}", "X_{2}", "X_{1} \\oplus X_{2}"],  # Header row  
            [1, 1, 0],  
            [0, 0, 0],  
            [0, 1, 1],  
            [1, 0, 1],  
        ]  

        # Create a table to display the truth table  
        table = Table(  
            table_data,  
            include_outer_lines=True,  
            element_to_mobject=lambda x: (  
                MathTex(x) if isinstance(x, str) else MathTex(str(x))  
            ),  
        )  

        table.scale(0.7)  # Scale the table to an appropriate size  
        table.to_edge(RIGHT, buff=1.0)  # Position the table on the right side of the screen  

        # Lists to hold cells corresponding to output values of 1 and 0  
        y_1_cells = []  
        y_0_cells = []  

        # Iterate through the table rows to categorize output cells  
        for row in range(2, len(table_data) + 1):  
            cell_y = table.get_entries((row, 3))  
            y_value = table_data[row - 1][2]  
            if y_value == 1:  
                y_1_cells.append(cell_y)  
            else:  
                y_0_cells.append(cell_y)  

        # Create colored dots for output values  
        dots_y1 = [  
            Dot(color=RED, radius=0.15).move_to(cell.get_center()).scale(0.7)  
            for cell in y_1_cells  
        ]  
        dots_y0 = [  
            Dot(color=YELLOW, radius=0.15).move_to(cell.get_center()).scale(0.7)  
            for cell in y_0_cells  
        ]  

        # Replace the cells for y=1 with red dots  
        for cell, dot in zip(y_1_cells, dots_y1):  
            cell.remove(*cell.submobjects)  # Remove existing content  
            cell.add(dot)  # Add red dot  

        # Replace the cells for y=0 with yellow dots  
        for cell, dot in zip(y_0_cells, dots_y0):  
            cell.remove(*cell.submobjects)  # Remove existing content  
            cell.add(dot)  

        # Create independent variables for the colored dots  
        dot_y0_row2 = (  
            Dot(color=YELLOW, radius=0.15).move_to(y_0_cells[0].get_center()).scale(0.7)  
        )  
        dot_y0_row3 = (  
            Dot(color=YELLOW, radius=0.15).move_to(y_0_cells[1].get_center()).scale(0.7)  
        )  
        dot_y1_row4 = (  
            Dot(color=RED, radius=0.15).move_to(y_1_cells[0].get_center()).scale(0.7)  
        )  
        dot_y1_row5 = (  
            Dot(color=RED, radius=0.15).move_to(y_1_cells[1].get_center()).scale(0.7)  
        )  

        # Animate the appearance of the title, XOR gate, table, and dots  
        self.play(  
            FadeIn(title),  
            FadeIn(Group_1),  
            FadeIn(table),  
            FadeIn(dot_y0_row2),  
            FadeIn(dot_y0_row3),  
            FadeIn(dot_y1_row4),  
            FadeIn(dot_y1_row5),  
        )  
        self.wait(1)  

        # Fade out the title, XOR gate, and table  
        self.play(FadeOut(title), FadeOut(Group_1), FadeOut(table))  

        # Create axes for the coordinate system  
        axes = (  
            Axes(  
                x_range=[-0.5, 1.5, 1],  
                y_range=[-0.5, 1.5, 1],  
                axis_config={"include_numbers": True},  
                tips=True,  
            )  
            .to_edge(DOWN, buff=1)  
            .scale(0.6)  
        )  

        # Add labels to the axes  
        axes_labels = axes.get_axis_labels(x_label="x_1", y_label="x_2")  
        self.play(FadeIn(axes), FadeIn(axes_labels))  
        self.wait(1)  

        # Define the positions for the dots based on the truth table  
        points_positions = {  
            dot_y0_row2: (1, 1),  
            dot_y0_row3: (0, 0),  
            dot_y1_row4: (0, 1),  
            dot_y1_row5: (1, 0),  
        }  

        # Move the dots to their corresponding positions on the axes  
        self.play(  
            *[  
                dot.animate.move_to(axes.c2p(x1, x2))  
                for dot, (x1, x2) in points_positions.items()  
            ],  
            run_time=2,  
        )  
        self.wait(1)  

        # Plot the function graphs related to the XOR logic  
        graph1 = axes.plot(  
            lambda x: self.func1(x), x_range=[-0.5, 1.5, 0.01], color=BLUE  
        )  
        graph2 = axes.plot(lambda x: self.func2(x), color=BLUE)  
        graph3 = axes.plot(lambda x: self.func3(x), color=BLUE)  
        graph4 = axes.plot(  
            lambda x: self.func4(x), x_range=[-0.5, 1.5, 0.01], color=BLUE  
        )  

        # Animate the creation of the graphs  
        self.play(Create(graph1))  
        self.wait(0.5)  
        self.play(FadeOut(graph1))  

        self.play(Create(graph2))  
        self.wait(0.5)  
        self.play(FadeOut(graph2))  

        self.play(Create(graph3))  
        self.wait(0.5)  
        self.play(FadeOut(graph3))  

        self.play(Create(graph1))  
        self.wait(1)  
        self.play(Create(graph4))  
        self.wait(1)  

        # Group the axes and graphs for final positioning  
        Group_2 = VGroup(  
            axes,  
            graph1,  
            graph4,  
            dot_y0_row2,  
            dot_y0_row3,  
            dot_y1_row4,  
            dot_y1_row5,  
            axes_labels,  
        )  
        self.play(Group_2.animate.shift(UP * 2.5).scale(0.7))  
        self.wait(1)  

        # Set colors for the neuron layers  
        input_color = "#6ab7ff"  
        receive_color = "#ff9e9e"  

        ## Create the first neuron layer  
        # Input layer  
        input_layer_1 = VGroup(  
            *[Circle(radius=0.2, fill_opacity=1, color=input_color) for _ in range(2)]  
        )  
        input_layer_1.arrange(DOWN, buff=1.5)  

        # Receiving layer  
        receive_layer_1 = Circle(  
            radius=0.5, fill_opacity=1, color=receive_color  
        ).move_to(ORIGIN + UP)  

        # Output layer  
        output_layer_1 = Circle(radius=0.2)  

        # Arrange the layers horizontally  
        layers_1 = VGroup(input_layer_1, receive_layer_1, output_layer_1)  
        layers_1.arrange(RIGHT, buff=2.5)  

        # Add connection lines between input and receiving layers  
        connections_1 = VGroup()  
        for i, input_neuron in enumerate(input_layer_1):  
            connection = Arrow(  
                start=input_neuron.get_center(),  
                end=receive_layer_1.get_center(),  
                stroke_opacity=0.3,  
                tip_length=0.2,  
                buff=0.6,  
            )  
            connections_1.add(connection)  

        # Add input and output labels  
        input_labels_1 = VGroup(  
            *[  
                MathTex(f"x_{i + 1}").move_to(input_neuron.get_center())  
                for i, input_neuron in enumerate(input_layer_1)  
            ]  
        )  

        # Create an arrow from the receiving layer to the output layer  
        theta = np.deg2rad(0)  
        direction = np.array([np.cos(theta), np.sin(theta), 0])  
        start_point = receive_layer_1.get_center()  
        end_point = output_layer_1.get_center()  
        arrow_1 = Arrow(  
            start=start_point,  
            end=end_point,  
            color=GREEN,  
            stroke_opacity=0.6,  
            tip_length=0.3,  
            buff=0.6,  
        )  
        output_label_y_1 = MathTex("y").move_to(output_layer_1.get_center())  
        p1 = VGroup(  
            input_labels_1,  
            connections_1,  
            receive_layer_1,  
            arrow_1,  
            receive_layer_1,  
            output_label_y_1,  
        )  

        ## Create the second neuron layer  
        # Input layer  
        input_layer_2 = VGroup(  
            *[Circle(radius=0.2, fill_opacity=1, color=input_color) for _ in range(2)]  
        )  
        input_layer_2.arrange(DOWN, buff=1.5)  

        # Receiving layer  
        receive_layer_2 = Circle(  
            radius=0.5, fill_opacity=1, color=receive_color  
        ).move_to(ORIGIN + UP)  

        # Output layer  
        output_layer_2 = Circle(radius=0.2)  

        # Arrange the layers horizontally  
        layers_2 = VGroup(input_layer_2, receive_layer_2, output_layer_2)  
        layers_2.arrange(RIGHT, buff=2.5)  

        # Add connection lines between input and receiving layers  
        connections_2 = VGroup()  
        for i, input_neuron in enumerate(input_layer_2):  
            connection = Arrow(  
                start=input_neuron.get_center(),  
                end=receive_layer_2.get_center(),  
                stroke_opacity=0.3,  
                tip_length=0.2,  
                buff=0.6,  
            )  
            connections_2.add(connection)  

        # Add input and output labels  
        input_labels_2 = VGroup(  
            *[  
                MathTex(f"x_{i + 1}").move_to(input_neuron.get_center())  
                for i, input_neuron in enumerate(input_layer_2)  
            ]  
        )  
        theta = np.deg2rad(0)  
        direction = np.array([np.cos(theta), np.sin(theta), 0])  
        start_point = receive_layer_2.get_center()  
        end_point = output_layer_2.get_center()  
        arrow_2 = Arrow(  
            start=start_point,  
            end=end_point,  
            color=GREEN,  
            stroke_opacity=0.6,  
            tip_length=0.3,  
            buff=0.6,  
        )  
        output_label_y_2 = MathTex("y").move_to(output_layer_2.get_center())  
        p2 = VGroup(  
            input_labels_2,  
            connections_2,  
            receive_layer_2,  
            arrow_2,  
            receive_layer_2,  
            output_label_y_2,  
        )  

        # Create the third neuron layer  
        # Note: This is a separate creation despite having a perceptron class to avoid further modifications  
        # Input layer  
        input_layer_3 = VGroup(  
            *[Circle(radius=0.2, fill_opacity=1, color=input_color) for _ in range(2)]  
        )  
        input_layer_3.arrange(DOWN, buff=1.5)  

        # Receiving layer  
        receive_layer_3 = Circle(  
            radius=0.5, fill_opacity=1, color=receive_color  
        ).move_to(ORIGIN + UP)  

        # Output layer  
        output_layer_3 = Circle(radius=0.2)  

        # Arrange the layers horizontally  
        layers_3 = VGroup(input_layer_3, receive_layer_3, output_layer_3)  
        layers_3.arrange(RIGHT, buff=2.5)  

        # Add connection lines between input and receiving layers  
        connections_3 = VGroup()  
        for i, input_neuron in enumerate(input_layer_3):  
            connection = Arrow(  
                start=input_neuron.get_center(),  
                end=receive_layer_3.get_center(),  
                stroke_opacity=0.3,  
                tip_length=0.2,  
                buff=0.6,  
            )  
            connections_3.add(connection)  

        # Add input and output labels  
        input_labels_3 = VGroup(  
            *[  
                MathTex(f"x_{i + 1}").move_to(input_neuron.get_center())  
                for i, input_neuron in enumerate(input_layer_3)  
            ]  
        )  
        theta = np.deg2rad(0)  
        direction = np.array([np.cos(theta), np.sin(theta), 0])  
        start_point = receive_layer_3.get_center()  
        end_point = output_layer_3.get_center()  
        arrow_3 = Arrow(  
            start=start_point,  
            end=end_point,  
            color=GREEN,  
            stroke_opacity=0.6,  
            tip_length=0.3,  
            buff=0.6,  
        )  
        output_label_y_3 = MathTex("y").move_to(output_layer_3.get_center())  
        p3 = VGroup(receive_layer_3, arrow_3, receive_layer_3, output_label_y_3)  

        # Position the neuron layers on the screen  
        p1.scale(0.6).to_edge(DOWN, buff=2.5).to_edge(LEFT, buff=3.5)  
        p2.scale(0.6).next_to(p1, DOWN, buff=0.5)  
        p3.scale(0.6).to_edge(DOWN, buff=2)  
        self.play(Create(p1), Create(p2))  
        self.wait(1)  

        # Create arrows connecting the receiving layers to the output layer  
        start_point_new_1 = receive_layer_1.get_center()  
        end_point_new_1 = receive_layer_3.get_center()  
        arrow_new_1 = Arrow(  
            start=start_point_new_1,  
            end=end_point_new_1,  
            color=YELLOW,  
            stroke_opacity=0.6,  
            tip_length=0.2,  
            buff=0.3,  
        )  

        start_point_new_2 = receive_layer_2.get_center()  
        end_point_new_2 = receive_layer_3.get_center()  
        arrow_new_2 = Arrow(  
            start=start_point_new_2,  
            end=end_point_new_2,  
            color=YELLOW,  
            stroke_opacity=0.6,  
            tip_length=0.2,  
            buff=0.3,  
        )  

        # Create labels for the inputs  
        x1 = (  
            MathTex("x_{1}").move_to(receive_layer_1.get_center() + LEFT * 2).scale(0.6)  
        )  
        x2 = (  
            MathTex("x_{2}").move_to(receive_layer_2.get_center() + LEFT * 2).scale(0.6)  
        )  

        # Create arrows from the input labels to the receiving layers  
        input_arrow_1_1 = Arrow(  
            start=x1.get_center(),  
            end=receive_layer_1.get_center(),  
            color=input_color,  
            stroke_opacity=0.6,  
            tip_length=0.2,  
            buff=0.3,  
        )  
        input_arrow_1_2 = Arrow(  
            start=x1.get_center(),  
            end=receive_layer_2.get_center(),  
            color=input_color,  
            stroke_opacity=0.6,  
            tip_length=0.2,  
            buff=0.3,  
        )  
        input_arrow_2_1 = Arrow(  
            start=x2.get_center(),  
            end=receive_layer_1.get_center(),  
            color=input_color,  
            stroke_opacity=0.6,  
            tip_length=0.2,  
            buff=0.3,  
        )  
        input_arrow_2_2 = Arrow(  
            start=x2.get_center(),  
            end=receive_layer_2.get_center(),  
            color=input_color,  
            stroke_opacity=0.6,  
            tip_length=0.2,  
            buff=0.3,  
        )  

        # Animate the transformation of arrows and labels  
        self.play(  
            Transform(  
                Group(arrow_1, arrow_2, output_label_y_1, output_label_y_2),  
                Group(arrow_new_1, arrow_new_2),  
            ),  
            FadeIn(p3),  
        )  
        self.wait(1)  

        # Transform the connections and input labels  
        self.play(  
            Transform(  
                Group(connections_1, connections_2),  
                Group(  
                    input_arrow_1_1, input_arrow_1_2, input_arrow_2_1, input_arrow_2_2  
                ),  
            ),  
            Transform(Group(input_labels_1, input_labels_2), Group(x1, x2)),  
        )  
        self.wait(1)  

        # Fade out the axes, labels, and other elements  
        self.play(FadeOut(axes), FadeOut(axes_labels), FadeOut(dot_y0_row2), FadeOut(dot_y0_row3), FadeOut(dot_y1_row4), FadeOut(dot_y1_row5), FadeOut(graph1), FadeOut(graph4))
        # Fade out the remaining elements  
        self.play(FadeOut(dot_y1_row5), FadeOut(graph1), FadeOut(graph4))  
        self.wait(1)  

        # Set up the axes for the next set of graphs  
        axes_1 = Axes(  
            x_range=[-5, 5, 1],  # X-axis range  
            y_range=[-0.5, 1.5, 0.5],  # Y-axis range  
            axis_config={"include_numbers": True},  # Include numbers on the axes  
        )  

        axes_2 = Axes(  
            x_range=[-5, 5, 1],  # X-axis range  
            y_range=[-0.5, 1.5, 0.5],  # Y-axis range  
            axis_config={"include_numbers": True},  # Include numbers on the axes  
        )  

        # Create labels for the axes  
        axes_labels_1 = axes_1.get_axis_labels(x_label="x")  
        axes_labels_2 = axes_2.get_axis_labels(x_label="x")  

        # Define the step function for the first graph  
        def step_func(x):  
            return 1 if x >= 0 else 0  # Returns 1 for x >= 0, otherwise 0  

        # Plot the step function graph  
        step_graph = axes_1.plot(step_func, color=BLUE, discontinuities=[0], use_smoothing=False)  

        # Define the Sigmoid function for the second graph  
        def sigmoid(x):  
            return 1 / (1 + np.exp(-x))  # Sigmoid function formula  

        # Plot the Sigmoid function graph  
        sigmoid_graph = axes_1.plot(sigmoid, color=RED)  

        # Add labels for the function graphs  
        step_label = MathTex("f(x)=\\begin{cases} 1 & x \\geq 0 \\\\ 0 & x < 0 \\end{cases}").set_color(BLUE).next_to(step_graph, UP, buff=2.2)  
        sigmoid_label = MathTex("f(x)=\\frac{1}{1 + e^{-x}}").set_color(RED).next_to(sigmoid_graph, UP, buff=2.2)  

        # Group the first graph components  
        Group1 = VGroup(axes_1, axes_labels_1, step_graph, step_label).scale(0.5)  
        # Group the second graph components  
        Group2 = VGroup(axes_2, axes_labels_2, sigmoid_graph, sigmoid_label).scale(0.5).to_edge(RIGHT)  
        # Combine both groups for final positioning  
        Group3 = VGroup(Group1, Group2).to_edge(UP, buff=0.1)  

        # Animate the creation of the first graph  
        self.play(Create(Group1))  
        self.wait(1)  
        # Move the first graph to the left  
        self.play(Group1.animate.to_edge(LEFT))  
        self.wait(1)  
        # Animate the creation of the second graph  
        self.play(Create(Group2))  
        self.wait(1)  

    # Define additional functions for graphing  
    def func1(self, x):  
        return x + 0.5  # Linear function for graphing  

    def func2(self, x):  
        return x - 0.5  # Linear function for graphing  

    def func3(self, x):  
        return -x + 1.5  # Linear function for graphing  

    def func4(self, x):  
        return x - 0.5  # Linear function for graphing  