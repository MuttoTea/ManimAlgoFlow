"""  
Summary:  
This code uses the Manim library to create an animation scene that illustrates a simple neural network structure along with its associated mathematical formulas. The main components include:  

1. Linear Model Formula: Displays the output formula of the linear model z = w^T x + b, along with the corresponding loss function and gradient update formulas.  
2. Neuron Structure: Visualizes the neurons in the input layer, hidden layer, and output layer, using arrows to represent the connections between them.  
3. Dynamic Input Layer: Demonstrates a dynamic input layer that includes ellipses to indicate that the number of input features is variable.  
4. Weight and Input Matrices: Shows the mathematical representation of the weight matrix and input matrix.  
"""

from manim import *


class NeuralNetworkScene(Scene):
    def construct(self):
        # Define the output equation of the linear model
        output_equation = (
            MathTex(r"z = \mathbf{w}^T \mathbf{x} + b").to_edge(UP).scale(0.8)
        )

        # Define the matrix form of the equation
        matrix_equation = MathTex(r"\mathbf{w}^T \mathbf{x} + b = 0")
        matrix_equation.next_to(output_equation, DOWN, buff=1.5).scale(0.8)

        # Define the loss function
        loss_function = MathTex(
            r"L = \max\left(0,\,-y\bigl(\mathbf{w}^T \mathbf{x} + b\bigr)\right)"
        )
        loss_function.next_to(output_equation, DOWN, buff=3).scale(0.8)

        # Define the gradient update formulas for weights and bias
        weight_update = MathTex(
            r"\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}"
        )
        bias_update = MathTex(r"b \leftarrow b - \eta \frac{\partial L}{\partial b}")

        # Group the gradient updates together
        gradient_updates = VGroup(weight_update, bias_update)
        gradient_updates.arrange(RIGHT, buff=0.5).scale(0.8)
        gradient_updates.next_to(output_equation, DOWN, buff=4.5)

        # Animate the writing of the equations
        self.play(
            Write(output_equation),
            Write(matrix_equation),
            Write(loss_function),
            Write(weight_update),
            Write(bias_update),
        )

        # Group all equations for later use
        equations_group = VGroup(
            output_equation,
            matrix_equation,
            loss_function,
            weight_update,
            bias_update,
        )

        # Set colors for the neurons
        input_neuron_color = "#6ab7ff"
        hidden_neuron_color = "#ff9e9e"

        # Create the input layer neurons
        input_layer_neurons = VGroup(
            *[
                Circle(radius=0.2, fill_opacity=1, color=input_neuron_color)
                for _ in range(4)
            ]
        )
        input_layer_neurons.arrange(DOWN, buff=0.7)

        # Create the hidden layer neuron
        hidden_layer_neuron = Circle(
            radius=0.5, fill_opacity=1, color=hidden_neuron_color
        ).move_to(ORIGIN + UP)

        # Create the output layer neuron
        output_layer_neuron = Circle(radius=0.2)

        # Arrange the layers horizontally
        layers_group = VGroup(
            input_layer_neurons, hidden_layer_neuron, output_layer_neuron
        )
        layers_group.arrange(RIGHT, buff=2.5)

        # Create connections from input neurons to the hidden layer neuron
        connections = VGroup()
        for input_neuron in input_layer_neurons:
            connection_arrow = Arrow(
                start=input_neuron.get_center(),
                end=hidden_layer_neuron.get_center(),
                stroke_opacity=0.3,
                tip_length=0.2,
                buff=0.6,
            )
            connections.add(connection_arrow)

        # Create labels for the input neurons
        input_labels = VGroup(
            *[
                MathTex(f"x_{i + 1}").scale(0.8).move_to(input_neuron.get_center())
                for i, input_neuron in enumerate(input_layer_neurons)
            ]
        )

        # Create an arrow from the hidden layer to the output layer
        arrow_direction = np.array([np.cos(0), np.sin(0), 0])
        start_point = hidden_layer_neuron.get_center()
        end_point = output_layer_neuron.get_center()
        connection_arrow = Arrow(
            start=start_point,
            end=end_point,
            color=GREEN,
            stroke_opacity=0.6,
            tip_length=0.3,
            buff=0.6,
        )
        output_label = MathTex("y").scale(0.8).move_to(output_layer_neuron.get_center())

        # Animate the disappearance of the equations and the appearance of the neural network
        self.play(FadeOut(equations_group))
        self.play(Create(hidden_layer_neuron))
        self.play(Create(connections))
        self.play(Write(input_labels))
        self.play(Create(connection_arrow))
        self.play(Write(output_label))
        self.wait(2)

        # Create a new input layer with ellipses to indicate variable input size
        dynamic_input_layer = VGroup(
            Circle(radius=0.2, fill_opacity=1, color=input_neuron_color),
            Circle(radius=0.2, fill_opacity=1, color=input_neuron_color),
            Tex("...").scale(0.8),  # Use ellipses to replace the third neuron
            Circle(radius=0.2, fill_opacity=1, color=input_neuron_color),
        )
        dynamic_input_layer.arrange(DOWN, buff=0.7)
        dynamic_input_layer.next_to(hidden_layer_neuron, LEFT, buff=2.5)

        # Create new labels for the dynamic input layer
        dynamic_labels = VGroup()
        dynamic_labels.add(
            MathTex("x_{1}").move_to(dynamic_input_layer[0].get_center()).scale(0.8)
        )
        dynamic_labels.add(
            MathTex("x_{2}").move_to(dynamic_input_layer[1].get_center()).scale(0.8)
        )
        dynamic_labels.add(
            MathTex("...").move_to(dynamic_input_layer[2].get_center()).scale(0.8)
        )
        dynamic_labels.add(
            MathTex("x_{n}").move_to(dynamic_input_layer[3].get_center()).scale(0.8)
        )

        # Create new connection arrows for the dynamic input layer
        new_connections = VGroup()
        for input_neuron in dynamic_input_layer:
            if isinstance(input_neuron, Circle):
                new_connection_arrow = Arrow(
                    start=input_neuron.get_right(),  # Start the arrow from the right side of the neuron
                    end=hidden_layer_neuron.get_left(),  # Connect to the left side of the hidden layer
                    stroke_opacity=0.3,
                    tip_length=0.2,
                    buff=0.6,
                    color=WHITE,
                )
                new_connections.add(new_connection_arrow)

        # Animate the replacement of the old connections and labels with the new ones
        self.play(
            ReplacementTransform(
                Group(connections, input_labels), Group(new_connections, dynamic_labels)
            )
        )
        self.wait(2)

        # Create the weight matrix representation
        weight_matrix = (
            MathTex(
                r"\begin{bmatrix} w_{1} & w_{2} & \dots  & w_{n} \end{bmatrix}^{T} "
            )
            .scale(0.8)
            .next_to(hidden_layer_neuron, DOWN, buff=1.5)
        )
        # Create the input matrix representation
        input_matrix = (
            MathTex(r"\begin{bmatrix} x_{1} & x_{2} & \dots & x_{n} \end{bmatrix}^{T} ")
            .scale(0.8)
            .next_to(weight_matrix, DOWN, buff=1)
        )
        self.play(Write(weight_matrix))
        self.play(Write(input_matrix))
        self.wait(2)