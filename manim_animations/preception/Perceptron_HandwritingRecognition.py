"""  
Summary:  
This code uses TensorFlow and Manim libraries to create multiple animation scenes that demonstrate the process of handwritten digit recognition, particularly through a perceptron for binary classification. The main content includes:  

1. Handwritten Digit Visualization: Loads images of handwritten digits from the MNIST dataset and displays them in a pixel block format. Animations show changes in grayscale values and image size annotations.  

2. Simplified Pixel Row: Simplifies the pixel rows of handwritten digits into a long row and displays its matrix form.  

3. Perceptron Structure: Demonstrates the input and output layers of the perceptron, illustrating how input data (pixel values of handwritten digits) is passed into the perceptron for classification.  

4. Binary Classification Example: Selects images of digits 0 and 1 from the MNIST dataset and shows how the perceptron classifies these images.  
"""  

import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
from manim import *  
from CustomClasses import Perceptron  
from CustomFunction import load_mnist_image, create_pixel_group  

class HandwritingVisualization(Scene):  
    def construct(self):  
        # Load MNIST dataset and select images for digits 0 and 1  
        digit_one_image, digit_one_label = load_mnist_image(label_target=1, index=0)  
        digit_zero_image, digit_zero_label = load_mnist_image(label_target=0, index=0)  

        print(f'Grayscale values: {digit_one_image}')  

        # Create and scale pixel groups for both images  
        pixel_group_one = create_pixel_group(digit_one_image).scale(0.8).to_edge(LEFT, buff=1)  
        pixel_group_zero = create_pixel_group(digit_zero_image).scale(0.8).to_edge(RIGHT, buff=1)  

        # Add pixel groups to the scene  
        self.play(Create(pixel_group_one), Create(pixel_group_zero))  
        self.wait(2)  

        # Display the title  
        title = self.create_title("How to Recognize Handwritten Digits with a Perceptron")  
        self.play(Write(title))  
        self.wait(1)  
        self.play(FadeOut(title))  
        self.wait(1)  

        # Move pixel groups to the center  
        self.play(FadeOut(pixel_group_zero), pixel_group_one.animate.move_to(ORIGIN))  
        self.wait(1)  

        # Display image dimensions  
        image_dimensions = {"width": 28, "height": 28}  
        self.display_image_dimensions(pixel_group_one, image_dimensions)  
        self.wait(1)  

        # Replace pixels with grayscale value text  
        text_group = self.create_text_group(pixel_group_one, digit_one_image)  
        self.play(FadeIn(text_group))  
        self.wait(1)  

        # Animate the change of a specific pixel's grayscale value  
        self.animate_selected_pixel(pixel_group_one, digit_one_image, text_group, row=8, col=16)  
        self.wait(1)  

    def create_title(self, text):  
        """Create a title text object."""  
        return Text(text).to_edge(UP, buff=0.5).scale(0.6)  

    def display_image_dimensions(self, pixel_group, dimensions):  
        """Display the width and height annotations of the image."""  
        braces = {  
            "width": Brace(pixel_group, DOWN),  
            "height": Brace(pixel_group, LEFT)  
        }  
        labels = {  
            "width": braces["width"].get_tex(f"W={dimensions['width']}"),  
            "height": braces["height"].get_tex(f"H={dimensions['height']}")  
        }  
        label_groups = {  
            key: VGroup(braces[key], labels[key]) for key in braces  
        }  

        self.play(FadeIn(braces["width"]), FadeIn(labels["width"]))  
        self.wait(1)  
        self.play(FadeIn(braces["height"]), Write(labels["height"]))  
        self.wait(1)  
        self.play(FadeOut(label_groups["width"]), FadeOut(label_groups["height"]))  

    def create_text_group(self, pixel_group, image):  
        """Create a group of text objects representing grayscale values."""  
        text_group = VGroup()  
        for i, square in enumerate(pixel_group):  
            row, col = divmod(i, 28)  
            gray_value = image[row, col] / 255  
            formatted_gray = f"{gray_value:.1f}"  
            text = Text(formatted_gray, font_size=8).move_to(square.get_center())  
            text.set_color(WHITE if gray_value < 0.5 else BLACK)  
            text_group.add(text)  
        return text_group  

    def animate_selected_pixel(self, pixel_group, image, text_group, row, col):  
        """Animate the grayscale change of a selected pixel, then return the pixel to its original position and remove all grayscale value texts."""  
        index = row * 28 + col  
        selected_pixel = pixel_group[index]  
        initial_gray = image[row, col] / 255  
        gray_tracker = ValueTracker(initial_gray)  

        # Create a text object to display the grayscale value  
        gray_text = DecimalNumber(  
            gray_tracker.get_value(),  
            num_decimal_places=1,  
            include_sign=False,  
            font_size=8  
        ).add_updater(lambda m: m.set_value(gray_tracker.get_value()))  
        gray_text.move_to(selected_pixel.get_center() + RIGHT * 4 + DOWN * 1).scale(2.5)  

        # Move and scale the selected pixel  
        self.play(  
            selected_pixel.animate.shift(RIGHT * 4 + DOWN * 1).scale(2.5)  
        )  
        self.add(gray_text)  
        self.remove(text_group[index])  

        # Update text color based on grayscale value  
        gray_text.add_updater(  
            lambda d: d.set_color(WHITE if gray_tracker.get_value() < 0.5 else BLACK)  
        )  

        # Animate grayscale change from initial value to 1.0 (white)  
        self.play(  
            gray_tracker.animate.set_value(1.0),  
            selected_pixel.animate.set_fill(WHITE, opacity=1),  
            run_time=2,  
        )  

        # Animate grayscale change from 1.0 to 0.0 (black)  
        self.play(  
            gray_tracker.animate.set_value(0.0),  
            selected_pixel.animate.set_fill(BLACK, opacity=1),  
            run_time=2,  
        )  

        # Restore the initial state and return the pixel to its original position  
        self.play(  
            selected_pixel.animate.set_fill(WHITE, opacity=initial_gray)  
                                .shift(LEFT * 4 + UP * 1)  
                                .scale(1 / 2.5),  
        )  
        # Fade out all grayscale value texts  
        self.play(FadeOut(text_group))  

class SimplifyLongRow(Scene):  
    def construct(self):  
        # Load MNIST dataset and select the first image labeled as 1  
        image, label = load_mnist_image(label_target=1, index=0)  

        # Create pixel group and play creation animation  
        pixel_group = create_pixel_group(image)  
        self.play(Create(pixel_group))  
        self.wait(1)  

        # Arrange pixel group and play transformation animation  
        arranged_group = self.arrange_pixel_group(pixel_group)  
        self.play(ReplacementTransform(pixel_group, arranged_group))  
        self.wait(1)  

        # Flatten multiple rows of pixels into a long row  
        long_row = VGroup(*[pixel for row in arranged_group for pixel in row]).arrange(RIGHT, buff=0.05)  

        # Simplify the long row and play transformation animation  
        simplified_group = self.simplify_long_row(long_row)  
        self.play(ReplacementTransform(arranged_group, simplified_group))  
        self.wait(1)  

        # Add a brace to indicate the number of pixels  
        brace = Brace(simplified_group, DOWN)  
        brace_text = brace.get_text("728")  
        self.play(FadeIn(brace), FadeIn(brace_text))  
        self.wait(1)  

        # Transform the simplified pixel group into a matrix and play transformation animation  
        matrix = MathTex("\\begin{bmatrix} x_{1} & x_{2} & \dots & x_{728} \\end{bmatrix}").scale(0.6)  
        matrix.move_to(simplified_group.get_center())  
        self.play(  
            ReplacementTransform(simplified_group, matrix),  
            FadeOut(brace),  
            FadeOut(brace_text)  
        )  
        self.wait(1)  
        self.play(matrix.animate.shift(UP))  

    def get_row_group(self, pixel_group, row_index, num_columns=28):  
        """Retrieve the pixel group for a specified row.  

        Parameters:  
            pixel_group (VGroup): Group containing all pixel squares.  
            row_index (int): Index of the row to retrieve (0-based).  
            num_columns (int): Number of pixels per row.  

        Returns:  
            VGroup: Group of pixel squares for the specified row.  
        """  
        total_rows = len(pixel_group) // num_columns  
        if not 0 <= row_index < total_rows:  
            raise ValueError(f"Row index must be between 0 and {total_rows - 1}.")  
        
        start = row_index * num_columns  
        end = start + num_columns  
        return VGroup(*pixel_group[start:end])  

    def arrange_pixel_group(self, pixel_group, center_row=None, num_columns=28):  
        """  
        Arrange pixels in specified rows, centering the specified row with others arranged above and below.  

        Parameters:  
            pixel_group (VGroup): Group containing all pixel squares.  
            center_row (int, optional): Index of the row to center. Defaults to the middle row.  
            num_columns (int): Number of pixels per row.  

        Returns:  
            VGroup: Arranged pixel group.  
        """  
        total_rows = len(pixel_group) // num_columns  
        center_row = center_row if center_row is not None else total_rows // 2  

        if not 0 <= center_row < total_rows:  
            raise ValueError(f"Center row index must be between 0 and {total_rows - 1}.")  
        
        rows = [self.get_row_group(pixel_group, i, num_columns) for i in range(total_rows)]  
        arranged_rows = VGroup(rows[center_row].copy())  

        for offset in range(1, max(center_row + 1, total_rows - center_row)):  
            if center_row - offset >= 0:  
                arranged_rows.add(rows[center_row - offset].copy().next_to(arranged_rows[-1], DOWN, buff=0))  
            if center_row + offset < total_rows:  
                arranged_rows.add(rows[center_row + offset].copy().next_to(arranged_rows[-1], DOWN, buff=0))  
                    
        return arranged_rows.arrange(RIGHT)  

    def simplify_long_row(self, long_row, left_keep=4, right_keep=4, pixel_size=0.5, spacing=0.05):  
        """  
        Simplify a long row by keeping a specified number of pixels on the left and right, with ellipsis in the middle.  

        Parameters:  
            long_row (VGroup): Group of long row pixels.  
            left_keep (int): Number of pixels to keep on the left.  
            right_keep (int): Number of pixels to keep on the right.  
            pixel_size (float): Size of the pixel squares.  
            spacing (float): Spacing between squares.  

        Returns:  
            VGroup: Simplified pixel group.  
        """  
        total_pixels = len(long_row)  
        if total_pixels <= left_keep + right_keep:  
            return long_row.copy()  
        
        left_group = VGroup(*long_row[:left_keep]).copy()  
        right_group = VGroup(*long_row[-right_keep:]).copy()  
        ellipsis = Text("...", font_size=24).scale(pixel_size)  
        
        simplified = VGroup(left_group, ellipsis, right_group).arrange(RIGHT, buff=spacing)  
        return simplified  

    
class MatrixToPerceptronScene(Scene):  
    def construct(self):  
        """  
        Display the matrix X_matrix in the scene and copy/move its entries to the input layer positions of the perceptron.  
        Finally, show the perceptron.  
        """  
        # 1. Create the matrix and display it  
        def custom_element_to_mobject(element):  
            return MathTex(element, font_size=36)  # Set font size to 36  

        # Create the matrix and apply custom font size  
        X_matrix = Matrix(  
            [  
                ["x_{1}", "x_{2}", "\\dots", "x_{784}"]  
            ],  
            h_buff=0.8,  
            bracket_h_buff=SMALL_BUFF,  
            bracket_v_buff=SMALL_BUFF,  
            element_to_mobject=custom_element_to_mobject  # Apply custom function  
        ).to_edge(UP).scale(0.8)  

        self.play(Create(X_matrix))  
        self.wait(1)  

        # 2. Retrieve matrix entries  
        # Assume the matrix has 4 entries (x_{1}, x_{2}, \dots, x_{784})  
        X_entries = X_matrix.get_entries()  

        # 3. Create the perceptron, but do not display it on the screen  
        # show_input_labels and show_input_circles are both False by default, so no input layer visual objects are shown  
        perceptron = Perceptron(  
            show_input_labels=False,  
            show_input_circles=False,  
            n_inputs=784  
        )  

        # 4. Get the input layer positions of the perceptron. Since labels and circles are not shown, handle accordingly in Perceptron (or return input_layer_centers directly)  
        positions = perceptron.get_positions()  
        # The first four positions correspond to x_{1}, x_{2}, \dots, x_{784}  
        input_label_positions = positions["input_layer"][:4]  

        # Print input layer positions  
        for idx, pos in enumerate(input_label_positions, start=1):  
            print(f'Input Layer_{idx}: {pos}')  
        
        # 5. Create animations: copy matrix entries and move them to the perceptron input layer positions  
        copied_entries = VGroup()  # Group to hold copied entries  
        animations = []  
        for entry, target_pos in zip(X_entries, input_label_positions):  
            # Copy and move to the corresponding position  
            copied_entry = entry.copy()  
            copied_entries.add(copied_entry)  
            animations.append(copied_entry.animate.move_to(target_pos))  
        
        # Play all move animations at once  
        self.play(*animations, run_time=1)  
        self.wait(1)  

        # 6. Create and display the perceptron on the screen  
        self.play(Create(perceptron))  
        self.wait(1)  

        # 7. Remove the copied matrix entries  
        self.play(FadeOut(copied_entries))  
        perceptron.enable_input_circles()  
        self.wait(1)  

        # 8. Move the perceptron to a specified position  
        self.play(perceptron.animate.to_edge(RIGHT, buff=1).scale(0.6))  
        self.wait(1)  


class PerceptronBinaryClassification(Scene):  
    def construct(self):  
        # Load the MNIST dataset  
        mnist = tf.keras.datasets.mnist  
        (image_train, label_train), (image_test, label_test) = mnist.load_data()  
        
        # Select images for digits 0 and 1  
        zero_indices = np.where(label_train == 0)[0]  # Get all indices for digit 0  
        one_indices  = np.where(label_train == 1)[0]  # Get all indices for digit 1  

        # Select 2 images for digits 0 and 1  
        zero_images = image_train[zero_indices[:2]]  
        one_images  = image_train[one_indices[:2]]  

        # Convert images to grayscale pixel groups  
        zero_images_gray = [create_pixel_group(img) for img in zero_images]  
        one_images_gray  = [create_pixel_group(img) for img in one_images]  

        # Combine and label images  
        labeled_images = [(img, 0) for img in zero_images_gray] + [(img, 1) for img in one_images_gray]  

        # Shuffle the order randomly  
        np.random.shuffle(labeled_images)  

        # Separate images and labels  
        images, labels = zip(*labeled_images)  

        # Create the perceptron  
        perceptron = Perceptron(  
            show_input_labels=False,  
        )  
        perceptron.to_edge(RIGHT, buff=1)  # Place the perceptron on the right side of the screen  
        self.play(Create(perceptron))  
        self.wait(1)  

        for img, label in zip(images, labels):  
            # Add images to the left of the perceptron  
            pixel_group = img  
            pixel_group.next_to(perceptron.input_circles, LEFT, buff=0.5).scale(0.6)  

            # Display the classification result from the perceptron output layer  
            output_label = MathTex(str(label), font_size=36, color=WHITE)  
            output_label.move_to(perceptron.output_layer.get_center())  
            perceptron.disable_output_label()  
            self.play(FadeIn(pixel_group), FadeIn(output_label))  
            self.wait(1)  
            self.play(FadeOut(pixel_group), FadeOut(output_label))