"""  
Summary:  
This code uses the Manim and TensorFlow libraries to create an animation scene that demonstrates how a Multi-Layer Perceptron (MLP) works for handwritten digit recognition. The main components include:  

1. MNIST Data Loading: Loads handwritten digit images from the MNIST dataset and selects one image for each digit.  
2. Pixel Group Visualization: Converts the selected handwritten digit images into pixel groups and arranges them on the screen.  
3. MLP Construction: Creates a visual representation of a Multi-Layer Perceptron, including input, hidden, and output layers.  
4. Activation Value Animation: Animates the propagation of input data through the MLP, showing the activation value changes at each layer.  
5. Connection Animation: Displays the connections between layers and the changes in activation states during data propagation.  
"""  

from manim import *  
import tensorflow as tf  
from CustomFunction import *  
import matplotlib.pyplot as plt  
import numpy as np  


class MNISTExample(Scene):  
    def construct(self):  
        selected_images = []  
        selected_labels = []  
        
        # Select one image for each handwritten digit (0-9)  
        for digit in range(10):  
            try:  
                image, label = load_mnist_image(label_target=digit, index=0)  
                selected_images.append(image)  
                selected_labels.append(label)  
            except IndexError as e:  
                print(e)  
        
        # Create pixel groups for each selected image  
        pixel_groups = [create_pixel_group(img) for img in selected_images]   
        
        # Arrange pixel groups on the screen  
        arranged_pixel_groups = VGroup(*pixel_groups).arrange_in_grid(rows=2, cols=5, buff=1).scale(0.3)   
        
        self.play(Create(arranged_pixel_groups))   
        self.wait(0.5)   
        self.play(FadeOut(arranged_pixel_groups))  
        self.wait(2)  

        # Initialize the Multi-Layer Perceptron (MLP)  
        mlp = self.initialize_network()  
        
        # Extract elements from the MLP dictionary  
        input_circles = mlp["input_circles"]  
        hidden_circles_1 = mlp["hidden_circles_1"]  
        hidden_circles_2 = mlp["hidden_circles_2"]  
        hidden_circles_3 = mlp["hidden_circles_3"]  
        output_circles = mlp["output_circles"]  
        connection_in_h1 = mlp["connection_in_h1"]  
        connection_h1_h2 = mlp["connection_h1_h2"]  
        connection_h2_h3 = mlp["connection_h2_h3"]  
        connection_h3_o = mlp["connection_h3_o"]  
        
        # Combine all MLP elements into a single VGroup  
        mlp_group = VGroup(input_circles, hidden_circles_1, hidden_circles_2, hidden_circles_3, output_circles,   
                           connection_in_h1, connection_h1_h2, connection_h2_h3, connection_h3_o)  
        
        # Create the MLP visualization  
        self.play(Create(mlp_group))  
        self.wait(0.5)  
        self.play(mlp_group.animate.to_edge(RIGHT, buff=1))  
        self.wait(0.5)  

        # Animate activation values for each selected image  
        for index, (img, label) in enumerate(zip(selected_images, selected_labels)):  
            label = int(label)  # Convert label to integer  
            activation_values = self.activation(label, mlp, seed=label)  # Use label as seed for randomness  

            # Set activation animations for each layer  
            input_circles_animation = set_activation(input_circles, activation_values["input"])  
            h1_circles_animation = set_activation(hidden_circles_1, activation_values["hidden_1"])  
            h2_circles_animation = set_activation(hidden_circles_2, activation_values["hidden_2"])  
            h3_circles_animation = set_activation(hidden_circles_3, activation_values["hidden_3"])  
            output_circles_animation = set_activation(output_circles, activation_values["output"])  
            
            img_group = create_pixel_group(img)  
            img_group.scale(0.6).to_edge(LEFT, buff=1)  
            
            # Play the activation animations  
            self.play(FadeIn(img_group), *input_circles_animation, *h1_circles_animation, *h2_circles_animation, *h3_circles_animation, *output_circles_animation)  
            self.play(FadeOut(img_group))  
            self.wait(0.5)  
        self.wait(2)  

    # Set MLP activation function  
    def activation(self, number, mlp, seed=None):  
        """Generate activation values for the MLP layers based on the input number."""  
        if seed is not None:  
            np.random.seed(seed)  # Set random seed for reproducibility  

        input_circles = mlp["input_circles"]  
        hidden_circles_1 = mlp["hidden_circles_1"]  
        hidden_circles_2 = mlp["hidden_circles_2"]  
        hidden_circles_3 = mlp["hidden_circles_3"]  
        output_circles = mlp["output_circles"]  

        # Find the index of the ellipsis in the input layer  
        non_circle_index = [  
            index for index, obj in enumerate(input_circles) if not isinstance(obj, Circle)  
        ]  
        ellipsis_index = non_circle_index[0] if non_circle_index else None  # Ensure ellipsis exists  

        # Generate random activation values for the input layer  
        activation_input_values = np.random.random(len(input_circles))  
        if ellipsis_index is not None:  
            activation_input_values[ellipsis_index] = -1  # Set ellipsis activation value to -1  

        # Generate random activation values for hidden layers  
        activation_hidden_1_values = np.random.rand(len(hidden_circles_1))  
        activation_hidden_2_values = np.random.rand(len(hidden_circles_2))  
        activation_hidden_3_values = np.random.rand(len(hidden_circles_3))  

        # Set the corresponding output layer activation value to 1, others to random values between 0 and 0.8  
        activation_output_values = np.random.rand(len(output_circles)) * 0.8  
        activation_output_values[number] = 1  # Set the corresponding output layer activation value to 1  

        return {  
            "input": activation_input_values,  
            "hidden_1": activation_hidden_1_values,  
            "hidden_2": activation_hidden_2_values,  
            "hidden_3": activation_hidden_3_values,  
            "output": activation_output_values  
        }  

    def initialize_network(self):  
        """Initialize the MLP structure with circles and connections."""  
        # Generate circles for the input layer with an ellipsis  
        input_circles, Icircles = generate_circles_with_vertical_ellipsis(  
            n=20,  # Example: 20 circles  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        input_circles.to_edge(LEFT, buff=3)  

        # Generate circles for the first hidden layer  
        hidden_circles_1, Hcircles_1 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_1.next_to(input_circles, RIGHT * 6)  

        # Generate circles for the second hidden layer  
        hidden_circles_2, Hcircles_2 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_2.next_to(hidden_circles_1, RIGHT * 6)  

        # Generate circles for the third hidden layer  
        hidden_circles_3, Hcircles_3 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_3.next_to(hidden_circles_2, RIGHT * 6)  

        # Generate circles for the output layer  
        output_circles, Ocircles = generate_circles_with_vertical_ellipsis(  
            n=10,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        output_circles.next_to(hidden_circles_3, RIGHT * 6)  

        # Add connections between layers  
        connection_in_h1 = add_full_connections_between_groups(Icircles, Hcircles_1, connection_color=BLUE)  
        connection_h1_h2 = add_full_connections_between_groups(Hcircles_1, Hcircles_2, connection_color=BLUE)  
        connection_h2_h3 = add_full_connections_between_groups(Hcircles_2, Hcircles_3, connection_color=BLUE)  
        connection_h3_o = add_full_connections_between_groups(Hcircles_3, Ocircles, connection_color=BLUE)  

        # Create a VGroup containing all connections and circles  
        mlp_group_vgroup = VGroup(  
            input_circles, hidden_circles_1, hidden_circles_2, hidden_circles_3,  
            output_circles, connection_h1_h2, connection_h2_h3, connection_h3_o,  
            connection_in_h1  
        )  
        mlp_group_vgroup.scale(0.8)  

        # Add corresponding handwritten digit results to the output layer circles  
        numbers = VGroup()  
        for i, circle in enumerate(output_circles):  
            number = MathTex(str(i), font_size=24, color=BLUE).next_to(circle, RIGHT)  
            numbers.add(number)  

        # Combine all elements into a dictionary  
        mlp_group = {  
            "input_circles": input_circles,  
            "Icircles": Icircles,  
            "hidden_circles_1": hidden_circles_1,  
            "Hcircles_1": Hcircles_1,  
            "hidden_circles_2": hidden_circles_2,  
            "Hcircles_2": Hcircles_2,  
            "hidden_circles_3": hidden_circles_3,  
            "Hcircles_3": Hcircles_3,  
            "output_circles": output_circles,  
            "Ocircles": Ocircles,  
            "connection_in_h1": connection_in_h1,  
            "connection_h1_h2": connection_h1_h2,  
            "connection_h2_h3": connection_h2_h3,  
            "connection_h3_o": connection_h3_o,  
            "numbers": numbers  
        }  

        return mlp_group  
    

class NeuralNetworkAnimation(Scene):  
    def construct(self):  
        # Initialize the network and get the MLP group  
        mlp_group = self.initialize_network()  
        
        # Execute data propagation animation  
        self.data_propagation_animation(mlp_group)  
        
        self.wait(2)  

    def initialize_network(self):  
        """Initialize the MLP structure with circles and connections."""  
        # Generate circles for the input layer with an ellipsis  
        input_circles, Icircles = generate_circles_with_vertical_ellipsis(  
            n=20,  # Example: 20 circles  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        input_circles.to_edge(LEFT, buff=3)  

        # Generate circles for the first hidden layer  
        hidden_circles_1, Hcircles_1 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_1.next_to(input_circles, RIGHT * 6)  

        # Generate circles for the second hidden layer  
        hidden_circles_2, Hcircles_2 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_2.next_to(hidden_circles_1, RIGHT * 6)  

        # Generate circles for the third hidden layer  
        hidden_circles_3, Hcircles_3 = generate_circles_with_vertical_ellipsis(  
            n=14,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        hidden_circles_3.next_to(hidden_circles_2, RIGHT * 6)  

        # Generate circles for the output layer  
        output_circles, Ocircles = generate_circles_with_vertical_ellipsis(  
            n=10,  
            radius=0.15,  
            spacing=0.4,  
            color=BLUE  
        )  
        output_circles.next_to(hidden_circles_3, RIGHT * 6)  

        # Add connections between layers  
        connection_in_h1 = add_full_connections_between_groups(Icircles, Hcircles_1, connection_color=BLUE)  
        connection_h1_h2 = add_full_connections_between_groups(Hcircles_1, Hcircles_2, connection_color=BLUE)  
        connection_h2_h3 = add_full_connections_between_groups(Hcircles_2, Hcircles_3, connection_color=BLUE)  
        connection_h3_o = add_full_connections_between_groups(Hcircles_3, Ocircles, connection_color=BLUE)  

        # Create a VGroup containing all connections and circles  
        mlp_group = {  
            "input_circles": input_circles,  
            "Icircles": Icircles,  
            "hidden_circles_1": hidden_circles_1,  
            "Hcircles_1": Hcircles_1,  
            "hidden_circles_2": hidden_circles_2,  
            "Hcircles_2": Hcircles_2,  
            "hidden_circles_3": hidden_circles_3,  
            "Hcircles_3": Hcircles_3,  
            "output_circles": output_circles,  
            "Ocircles": Ocircles,  
            "connection_in_h1": connection_in_h1,  
            "connection_h1_h2": connection_h1_h2,  
            "connection_h2_h3": connection_h2_h3,  
            "connection_h3_o": connection_h3_o  
        }  

        # Create VGroup containing all connections and circles  
        mlp_group_vgroup = VGroup(  
            input_circles, hidden_circles_1, hidden_circles_2, hidden_circles_3,  
            output_circles, connection_h1_h2, connection_h2_h3, connection_h3_o,  
            connection_in_h1  
        )  
        mlp_group_vgroup.scale(0.8)  

        # Animate the creation of the MLP structure  
        self.play(Create(input_circles))  
        self.wait(0.5)  
        self.play(Create(hidden_circles_1), Create(connection_in_h1))  
        self.wait(0.5)  
        self.play(Create(hidden_circles_2), Create(connection_h1_h2))  
        self.wait(0.5)  
        self.play(Create(hidden_circles_3), Create(connection_h2_h3))  
        self.wait(0.5)  
        self.play(Create(output_circles), Create(connection_h3_o))  
        self.wait(0.5)  

        return mlp_group  

    def data_propagation_animation(self, mlp_group):  
        """Animate the data propagation through the MLP."""  
        # Extract elements from the MLP group  
        input_circles = mlp_group["input_circles"]  
        hidden_circles_1 = mlp_group["hidden_circles_1"]  
        hidden_circles_2 = mlp_group["hidden_circles_2"]  
        hidden_circles_3 = mlp_group["hidden_circles_3"]  
        output_circles = mlp_group["output_circles"]  
        connection_in_h1 = mlp_group["connection_in_h1"]  
        connection_h1_h2 = mlp_group["connection_h1_h2"]  
        connection_h2_h3 = mlp_group["connection_h2_h3"]  
        connection_h3_o = mlp_group["connection_h3_o"]  

        # Find the index of the ellipsis in the input layer  
        non_circle_index = [  
            index for index, obj in enumerate(input_circles) if not isinstance(obj, Circle)  
        ]  
        ellipsis_index = non_circle_index[0]  

        # Generate random activation values for the input layer  
        activation_input_values = [np.random.random() for _ in range(len(input_circles))]  
        activation_input_values[ellipsis_index] = -1  # Set ellipsis activation value to -1  

        # Generate random activation values for hidden layers  
        activation_hidden_1_values = np.random.rand(len(hidden_circles_1))  
        activation_hidden_2_values = np.random.rand(len(hidden_circles_2))  
        activation_hidden_3_values = np.random.rand(len(hidden_circles_3))  
        activation_output_values = np.random.rand(len(output_circles))  

        # Define connection pulse animations  
        def play_pulse(connection):  
            """Play pulse animation for the connections."""  
            wave_animations = []  
            for line in connection:  
                sublines, wave_animation = animate_line_wave(line, wave_type="pulse", stroke_width=0.3)  
                wave_animations.append(wave_animation)  
            return wave_animations  

        # Step-by-step animation playback  
        # 1. Input layer activation  
        self.play(*set_activation(input_circles, activation_input_values), run_time=1)  

                # 2. Connection pulse from input to hidden layer 1  
        I_h1_pulse = play_pulse(connection_in_h1)  
        self.play(*I_h1_pulse, run_time=1)  

        # 3. Activate hidden layer 1  
        self.play(*set_activation(hidden_circles_1, activation_hidden_1_values), run_time=1)  

        # 4. Connection pulse from hidden layer 1 to hidden layer 2  
        H1_H2_pulse = play_pulse(connection_h1_h2)  
        self.play(*H1_H2_pulse, run_time=1)  

        # 5. Activate hidden layer 2  
        self.play(*set_activation(hidden_circles_2, activation_hidden_2_values), run_time=1)  

        # 6. Connection pulse from hidden layer 2 to hidden layer 3  
        H2_H3_pulse = play_pulse(connection_h2_h3)  
        self.play(*H2_H3_pulse, run_time=1)  

        # 7. Activate hidden layer 3  
        self.play(*set_activation(hidden_circles_3, activation_hidden_3_values), run_time=1)  

        # 8. Connection pulse from hidden layer 3 to output layer  
        H3_O_pulse = play_pulse(connection_h3_o)  
        self.play(*H3_O_pulse, run_time=1)  

        # 9. Activate output layer  
        self.play(*set_activation(output_circles, activation_output_values), run_time=1)  

        # Add arrows to indicate forward and backward propagation  
        # Get the center positions of the circles for arrow placement  
        forward_arrow_start = hidden_circles_1[0].get_center() + 3 * LEFT  
        forward_arrow_end = hidden_circles_3[0].get_center() + 3 * RIGHT  

        # Create forward propagation arrow  
        forward_arrow = Arrow(forward_arrow_start, forward_arrow_end, color=RED, buff=1).to_edge(UP, buff=0.75)  

        # Define start and end points for the backward arrow  
        backward_arrow_start = hidden_circles_3[0].get_center() + 3 * RIGHT  
        backward_arrow_end = hidden_circles_1[0].get_center() + 3 * LEFT  

        # Create backward propagation arrow  
        backward_arrow = Arrow(backward_arrow_start, backward_arrow_end, color=RED, buff=1).to_edge(DOWN, buff=0.75)  

        # Play the arrow animations  
        self.play(Create(backward_arrow), Create(forward_arrow), run_time=1)  
        self.wait(1)  
        
        # Remove arrows and reset circles to their initial state  
        self.play(  
            FadeOut(backward_arrow),   
            FadeOut(forward_arrow),   
            *set_activation(input_circles, [0] * len(input_circles)),  
            *set_activation(hidden_circles_1, [0] * len(hidden_circles_1)),  
            *set_activation(hidden_circles_2, [0] * len(hidden_circles_2)),  
            *set_activation(hidden_circles_3, [0] * len(hidden_circles_3)),  
            run_time=1  
        )  
        self.wait(1)