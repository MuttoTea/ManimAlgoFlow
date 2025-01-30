"""  
Summary:  
This code uses the Manim library to create multiple animation scenes that illustrate various concepts in mathematics and machine learning, including:  

1. UnivariateQuadratic:   
   - Functionality: Displays the graph of a univariate quadratic function.  
   - Content: Demonstrates how to find the minimum value of the function through differentiation and labels the minimum point on the graph.  

2. QuadraticSurfaceVisualization:   
   - Functionality: Visualizes a complex quadratic surface in three-dimensional space.  
   - Content: Showcases the shape and characteristics of the surface, helping viewers understand the geometric representation of multivariable functions.  

3. GradientDescentOneVariable:   
   - Functionality: Demonstrates the gradient descent method in one dimension.  
   - Content: Includes implementations of both variable and fixed step sizes, aiding viewers in understanding the gradient descent process.  

4. ComplexTVariable:   
   - Functionality: Displays the graph of a complex function and its derivative.  
   - Content: Illustrates the gradient descent process for this function.  

5. GradientDescentTwoVariable:   
   - Functionality: Visualizes the surface of a complex function in three-dimensional space.  
   - Content: Demonstrates the gradient descent process occurring on this surface.  

Note: The function selected for the GradientDescentOneVariable scene is chosen from the demonstration functions in `algorithm_implementations/perception/gradient_show.py`. You can select any function you like from there for demonstration or define your own function for the presentation.  
"""


from manim import *  
import torch  
import math  
import sympy as sp  

class UnivariateQuadratic(Scene):  
    def construct(self):  
        # Create the axes for the graph  
        axes = Axes(  
            x_range=[-3, 7, 1],  
            y_range=[-2, 8, 1],  
            axis_config={"color": BLUE},  
        )  

        # Plot the quadratic function graph  
        graph = axes.plot(lambda x: (x - 2) ** 2 + 1, color=YELLOW)  
        graph_label = axes.get_graph_label(graph, label="y=(x-2)^2+1")  

        self.play(Create(axes))  
        self.play(Create(graph))  

        # Pose the question about the minimum value of the function  
        question = Text("What value of x gives the minimum of the function?").to_edge(UP).scale(0.6)  
        self.play(Write(question))  
        self.wait(1)  
        self.play(FadeOut(question))  

        # Method 1: Find the extremum by calculating the derivative  
        derivative_label = (  
            MathTex(r"\frac{\mathrm{d} y}{\mathrm{d} x} ")  
            .to_edge(RIGHT, buff=2)  
            .shift(UP * 0.5)  
        )  
        equal_zero_label = MathTex(r"= 0").next_to(derivative_label, RIGHT, buff=0.15)  

        # Display the derivative  
        self.play(FadeIn(derivative_label))  
        self.wait(1)  

        # Set the derivative equal to zero  
        self.play(FadeIn(equal_zero_label))  
        self.wait(1)  

        # Mark the minimum value point  
        min_x = 2  
        min_y = (min_x - 2) ** 2 + 1  # min_y = 1  

        # Create a point at the minimum  
        point_position = axes.coords_to_point(min_x, 0)  
        min_point = Dot(point_position, color=RED)  
        self.play(FadeIn(min_point))  

        # Add label for the minimum point  
        min_point_label = MathTex(r"x_{min}").next_to(min_point, DR, buff=0.3)  
        self.play(Write(min_point_label))  
        self.wait(1)  

        # Create dashed lines to indicate the minimum point  
        vertical_line = DashedLine(  
            start=axes.coords_to_point(min_x, 0),  
            end=axes.coords_to_point(min_x, min_y),  
            color=GREY,  
        )  
        horizontal_line = DashedLine(  
            start=axes.coords_to_point(0, min_y),  
            end=axes.coords_to_point(min_x, min_y),  
            color=GREY,  
        )  
        self.play(Create(vertical_line), Create(horizontal_line))  
        self.wait(2)  

        # Final display  
        self.wait(2)  

        # Display the formula for the loss function  
        loss_formula = (  
            MathTex(  
                r"L = \max(0, -y(w \cdot x + b))",  
                substrings_to_isolate=["L", "=", r"\max", "w", "x", "b"],  
            )  
            .scale(0.8)  
            .to_edge(UP)  
        )  

        # Highlight parts of the formula  
        part_w = loss_formula[6]  # w  
        part_x = loss_formula[8]  # x  
        part_b = loss_formula[10]  # b  

        self.play(Write(loss_formula))  
        self.wait(0.5)  
        self.play(  
            part_w.animate.set_color(RED),  
            part_x.animate.set_color(RED),  
            part_b.animate.set_color(RED),  
        )  
        self.play(  
            FadeOut(loss_formula),  
            FadeOut(vertical_line),  
            FadeOut(horizontal_line),  
            FadeOut(min_point),  
            FadeOut(axes),  
            FadeOut(graph),  
            FadeOut(min_point_label),  
            FadeOut(derivative_label),  
            FadeOut(equal_zero_label),  
        )  


class QuadraticSurfaceVisualization(ThreeDScene):  
    def construct(self):  
        # Define the function for the surface  
        def surface_function(x, y):  
            x_tensor = torch.tensor(x, dtype=torch.float32)  
            y_tensor = torch.tensor(y, dtype=torch.float32)  
            term1 = 1.5 * (1 - x_tensor) ** 2 * torch.exp(-0.5 * x_tensor**2 - (y_tensor + 2) ** 2)  
            term2 = -2 * (x_tensor / 10 - x_tensor**3 / 2 - y_tensor**3) * torch.exp(-0.5 * (x_tensor**2 + y_tensor**2))  
            term3 = -0.1 * torch.exp(-0.5 * (x_tensor + 2) ** 2 - 0.5 * y_tensor**2)  
            return (term1 + term2 + term3) * 3  

        # Create 3D axes  
        axes = ThreeDAxes(  
            x_range=[-5, 5, 1],  
            y_range=[-5, 5, 1],  
            z_range=[-10, 10, 2],  
            axis_config={"color": BLUE},  
        )  

        # Create the surface  
        resolution = 20  # Grid resolution  
        surface = Surface(  
            lambda u, v: axes.c2p(u, v, surface_function(u, v)),  
            u_range=[-5, 5],  
            v_range=[-5, 5],  
            resolution=(resolution, resolution),  
            fill_color=YELLOW,  
            fill_opacity=1,  
            checkerboard_colors=[RED, RED_E],  
        )  

        # Set surface style  
        surface.set_style(fill_opacity=0.8, stroke_width=0.8)  
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)  

        # Add axes and surface to the scene  
        self.play(Create(axes))  
        self.play(Create(surface))  

        # Start ambient camera rotation  
        self.begin_ambient_camera_rotation(rate=PI / 24)  # Set rotation speed  

        # Rotate for a while  
        self.wait(8)  

        # Stop camera rotation  
        self.stop_ambient_camera_rotation()  

        # Wait for the animation to finish  
        self.wait(2)  


class GradientDescentOneVariable(Scene):  
    def construct(self):  
        # Set option: "T" for variable learning rate, "F" for fixed learning rate, "A" for both  
        option = "F"  # Change as needed to "T", "F", or "A"  

        # Create the axes for the graph  
        axes = Axes(  
            x_range=[-3, 7, 1],  
            y_range=[-2, 8, 1],  
            axis_config={"color": BLUE},  
        )  

        # Plot the function graph  
        graph = axes.plot(lambda x: self.quadratic_function(x), color=YELLOW)  
        graph_label = axes.get_graph_label(graph, label="y=(x-2)^2+1")  

        self.play(Create(axes), Create(graph), Write(graph_label))  

        # Demonstrate different gradient descent methods based on the option  
        if option == "A":  
            # Demonstrate variable learning rate gradient descent  
            self.perform_gradient_descent(  
                axes=axes,  
                initial_x=4,  
                learning_rate=None,  # None indicates variable learning rate  
                iterations=7,  
                color=RED,  
            )  
            self.wait(1)  

            # Demonstrate fixed learning rate gradient descent  
            self.perform_gradient_descent(  
                axes=axes,  
                initial_x=4,  
                learning_rate=0.5,  # Fixed learning rate  
                iterations=4,  
                color=BLUE,  
            )  
            self.wait(1)  

        elif option == "T":  
            # Demonstrate only variable learning rate gradient descent  
            self.perform_gradient_descent(  
                axes=axes,  
                initial_x=4,  
                learning_rate=None,  # None indicates variable learning rate  
                iterations=7,  
                color=RED,  
            )  
            self.wait(1)  

        elif option == "F":  
            # Demonstrate only fixed learning rate gradient descent  
            self.perform_gradient_descent(  
                axes=axes,  
                initial_x=4,  
                learning_rate=0.5,  # Fixed learning rate  
                iterations=4,  
                color=BLUE,  
            )  
            self.wait(1)  

        else:  
            self.play(Write(Text("Invalid option selected!", font_size=24)))  

        self.wait(2)  

    def quadratic_function(self, x):  
        """Define the quadratic function y = (x - 2)^2 + 1"""  
        return (x - 2) ** 2 + 1  

    def compute_gradient(self, x):  
        """Calculate the gradient dy/dx = 2*(x - 2)"""  
        return 2 * (x - 2)  

    def create_tangent_line(self, axes, x):  
        """Create a tangent line object based on the given x value"""  
        y = self.quadratic_function(x)  
        slope = self.compute_gradient(x)  
        intercept = y - slope * x  
        return axes.plot(lambda x_val: slope * x_val + intercept, x_range=[-3, 7], color=GREEN)  

    def perform_gradient_descent(self, axes, initial_x, learning_rate, iterations, color):  
        """  
        Execute the gradient descent animation  
        :param axes: Axes object for the graph  
        :param initial_x: Initial x value for the descent  
        :param learning_rate: Learning rate; if None, use variable learning rate  
        :param iterations: Number of iterations for the descent  
        :param color: Color of the point representing the current position  
        """  
        current_point = Dot(color=color).move_to(axes.c2p(initial_x, self.quadratic_function(initial_x)))  
        self.play(FadeIn(current_point))  

        # Calculate and draw the initial tangent line  
        current_x = initial_x  
        tangent_line = self.create_tangent_line(axes, current_x)  
        self.play(Create(tangent_line))  

        for i in range(iterations):  
            self.wait(0.5)  

            # Calculate the gradient  
            gradient_value = self.compute_gradient(current_x)  

            # Determine the step size  
            if learning_rate is not None:  
                step_size = learning_rate  # Fixed step size  
                new_x = current_x - step_size  
            else:  
                step_size = self.variable_learning_rate(i)  # Variable step size  
                new_x = current_x - step_size * gradient_value  

            # Update the y value based on the new x  
            new_y = self.quadratic_function(new_x)  

            # Calculate and draw the new tangent line  
            new_tangent_line = self.create_tangent_line(axes, new_x)  

            # Update the point and tangent line  
            new_point = Dot(color=color).move_to(axes.c2p(new_x, new_y))  
            self.play(  
                Transform(current_point, new_point), Transform(tangent_line, new_tangent_line)  
            )  
            self.wait(0.5)  

            # Update the current x value  
            current_x = new_x  

    def variable_learning_rate(self, iteration):  
        """  
        Variable learning rate strategy: gradually decreasing learning rate  
        For example, lr = 0.5 / (iteration + 1)  
        """  
        return 0.5 / (iteration + 1)


class ComplexTVariable(ThreeDScene):  
    def ComplexFunction(self, x):  
        """Define the function: y = (x^2)/20 - cos(x) + sin(2x)/2"""  
        return (x**2) / 20 - math.cos(x) + math.sin(2 * x) / 2  

    def get_derivative_function(self):  
        """Calculate the derivative of ComplexFunction using SymPy and return a callable numerical function."""  
        x = sp.symbols("x")  
        # Define the symbolic expression  
        f = (x**2) / 20 - sp.cos(x) + sp.sin(2 * x) / 2  
        # Compute the derivative  
        f_prime = sp.diff(f, x)  
        # Convert the symbolic expression to a numerical function  
        f_prime_lambdified = sp.lambdify(x, f_prime, modules=["numpy"])  
        return f_prime_lambdified  

    def get_function_range(self, x_min, x_max, num_samples=200):  
        """Dynamically calculate the range of y values for the function."""  
        x_values = np.linspace(x_min, x_max, num_samples)  
        y_values = [self.ComplexFunction(x) for x in x_values]  
        y_min, y_max = min(y_values), max(y_values)  
        return y_min, y_max  

    def create_tangent_line(self, axes, x, derivative_func, length=4):  
        """  
        Create a short tangent line object based on the given x value.  

        Parameters:  
        - axes: The axes object.  
        - x: The x-coordinate of the current point.  
        - derivative_func: The derivative function.  
        - length: The total length of the tangent line (default is 4).  
        """  
        y = self.ComplexFunction(x)  
        slope = derivative_func(x)  # Calculate the slope  
        intercept = y - slope * x  # Calculate the y-intercept  

        # Define the range for the tangent line, ensuring it does not exceed the x range of the axes  
        half_length = length / 2  
        x_min, x_max = axes.x_range[:2]  
        x_start = max(x - half_length, x_min)  
        x_end = min(x + half_length, x_max)  

        return axes.plot(  
            lambda x_val: slope * x_val + intercept, x_range=[x_start, x_end], color=GREEN  
        )  

    def construct(self):  
        # Define the range for x  
        x_min, x_max = -10, 10  

        # Automatically calculate the range for y  
        y_min, y_max = self.get_function_range(x_min, x_max)  

        # Define dynamic axes  
        axes = Axes(  
            x_range=[x_min, x_max, 5],  # x tick interval  
            y_range=[y_min - 5, y_max + 5, 5],  # y tick interval with buffer  
            axis_config={"color": BLUE},  
        ).add_coordinates()  

        # Plot the function  
        graph = axes.plot(lambda x: self.ComplexFunction(x), color=YELLOW)  
        graph_label = axes.get_graph_label(  
            graph, label="f(x) = \\frac{x^2}{20} - \\cos(x) + \\frac{\\sin(2x)}{2}"  
        )  

        # Animate the display of axes and function graph  
        self.play(Create(axes), Create(graph), Write(graph_label))  
        self.wait(1)  

        # Get the derivative function  
        derivative_func = self.get_derivative_function()  

        # Gradient descent parameters  
        initial_x = 4.2  # Initial point (ensure it's within the x range)  
        learning_rate = 0.15  # Learning rate  
        iterations = 10  # Number of iterations  

        # Create the initial point  
        initial_y = self.ComplexFunction(initial_x)  
        point = Dot(color=GREEN).move_to(axes.c2p(initial_x, initial_y))  
        self.play(FadeIn(point))  
        self.wait(0.5)  

        # Create the initial tangent line  
        tangent_line = self.create_tangent_line(axes, initial_x, derivative_func)  
        self.play(Create(tangent_line))  
        self.wait(0.5)  

        current_x = initial_x  

        for i in range(iterations):  
            # Calculate the gradient (derivative)  
            grad = derivative_func(current_x)  

            # Calculate the new x value  
            new_x = current_x + learning_rate  
            new_y = self.ComplexFunction(new_x)  

            # Create the new tangent line  
            new_tangent_line = self.create_tangent_line(axes, new_x, derivative_func)  

            # Create the new point  
            new_point = Dot(color=RED).move_to(axes.c2p(new_x, new_y))  

            # Animate: move the point and update the tangent line  
            self.play(  
                Transform(point, new_point), Transform(tangent_line, new_tangent_line)  
            )  

            # Update the current x value  
            current_x = new_x  

        self.wait(2)  


class GradientDescentTwoVariable(ThreeDScene):  
    def construct(self):  
        # Define the function for the surface  
        def surface_function(x, y):  
            x_tensor = torch.tensor(x, dtype=torch.float32)  
            y_tensor = torch.tensor(y, dtype=torch.float32)  
            term1 = 1.5 * (1 - x_tensor) ** 2 * torch.exp(-0.5 * x_tensor**2 - (y_tensor + 2) ** 2)  
            term2 = -2 * (x_tensor / 10 - x_tensor**3 / 2 - y_tensor**3) * torch.exp(-0.5 * (x_tensor**2 + y_tensor**2))  
            term3 = -0.1 * torch.exp(-0.5 * (x_tensor + 2) ** 2 - 0.5 * y_tensor**2)  
            return (term1 + term2 + term3) * 3  

        # Create 3D axes  
        axes = ThreeDAxes(  
            x_range=[-3, 3, 1],  
            y_range=[-3, 3, 1],  
            z_range=[-10, 10, 2],  
            axis_config={"color": BLUE},  
        )  
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y", z_label="z")  

        # Create the surface  
        resolution = 20  # Grid resolution  
        surface = Surface(  
            lambda u, v: axes.c2p(u, v, surface_function(u, v)),  
            u_range=[-5, 5],  
            v_range=[-5, 5],  
            resolution=(resolution, resolution),  
            fill_color=YELLOW,  
            fill_opacity=1,  
            checkerboard_colors=[RED, RED_E],  
        )  

        # Set surface style  
        surface.set_style(fill_opacity=0.8, stroke_width=0.8)  
        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)  

        # Scale down the entire axes and labels  
        axes.scale(0.5)  
        surface.scale(0.5)  

        # Add axes and surface to the scene  
        self.play(Create(axes), Write(axes_labels))  
        self.play(Create(surface))  
        self.wait(1)  

        # Store the coordinates of selected points  
        selected_points = {  
            1: np.array([-0.8, -2.3, 5.847186]),  
            2: np.array([-0.445482, -2.384923, 3.047376]),  
            3: np.array([-0.178311, -2.390921, 0.649276]),  
            4: np.array([0.272616, -2.238446, -3.136957]),  
            5: np.array([0.554393, -1.826272, -5.143542]),  
        }  

        # List of points (ensured to be in order)  
        point_list = list(selected_points.values())  

        # Set the 3D view  
        self.set_camera_orientation(phi=60 * DEGREES, theta=-30 * DEGREES)  

        for i, point in enumerate(point_list):  
            # (a) Convert the original point (data coordinates) to Manim coordinates  
            manim_point = axes.coords_to_point(*point)  
            # (b) Add a 3D point  
            dot = Dot3D(manim_point, color=BLUE, radius=0.05)  # Reduce the radius of the point  

            # If there is a previous point, connect with a line segment  
            if i > 0:  
                prev_manim_point = axes.coords_to_point(*point_list[i - 1])  
                line = Line3D(prev_manim_point, manim_point, color=GREEN)  
                self.play(Create(line), run_time=0.5)  
                self.wait(0.5)  
            # Draw the current point  
            self.play(FadeIn(dot), run_time=0.2)  

        self.wait(2)  