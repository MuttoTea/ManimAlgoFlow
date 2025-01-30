"""  
Summary:  
This code uses TensorFlow and the Manim library to load handwritten digit images from the MNIST dataset and visualize them as pixel squares. The main functionalities include:  

1. Loading MNIST Images: The `load_mnist_image` function loads a specified label's handwritten digit image and returns the image array and label.  
2. Creating Pixel Group: The `create_pixel_group` function converts the loaded 28x28 grayscale image into a group of visual pixel squares, with each square's opacity set according to the corresponding pixel value.  
"""  

from manim import *  
import tensorflow as tf  
import numpy as np  

def load_mnist_image(label_target=1, index=0):  
    """  
    Load the MNIST dataset and select the specified label's image.  

    Parameters:  
        label_target (int): Target label (0-9).  
        index (int): The index of the selected image within the target label.  

    Returns:  
        image (np.ndarray): The selected image array.  
        label (int): The label of the image.  
    """  
    # Load the MNIST dataset  
    mnist = tf.keras.datasets.mnist  
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  
    images = np.concatenate((x_train, x_test), axis=0)  # Combine training and testing images  
    labels = np.concatenate((y_train, y_test), axis=0)  # Combine training and testing labels  
    
    # Find all indices of the target label  
    target_indices = np.where(labels == label_target)[0]  
    
    if index >= len(target_indices):  
        raise IndexError(f"There are not enough images for label {label_target} at index {index + 1}.")  
    
    selected_index = target_indices[index]  # Select the index of the desired image  
    image = images[selected_index]  # Get the image array  
    label = labels[selected_index]  # Get the corresponding label  
    
    print(f'Selected image index: {selected_index}')  
    print(f'Label: {label}')  
    
    return image, label  

def create_pixel_group(image, pixel_size=0.2, spacing=0.05):  
    """  
    Create a pixel group based on the input image array.  

    Parameters:  
        image (np.ndarray): 28x28 grayscale image array.  
        pixel_size (float): The side length of each pixel square.  
        spacing (float): The spacing between the squares.  

    Returns:  
        pixel_group (VGroup): A group containing all pixel squares.  
    """  
    pixel_group = VGroup()  # Create a VGroup to hold pixel squares  
    
    for i in range(28):  # Iterate over rows  
        for j in range(28):  # Iterate over columns  
            # Get pixel value (0-255), convert to grayscale color and invert  
            pixel_value = image[i, j]  
            gray = pixel_value / 255  # Normalize pixel value to range [0, 1]  
            
            # Create a square to represent the pixel  
            square = Square(  
                side_length=pixel_size,  
                fill_color=WHITE,  
                fill_opacity=gray,  # Set opacity based on grayscale value  
                stroke_width=0.2,  
                stroke_color=WHITE  
            )  
            # Position the square in the grid  
            square.shift(  
                (j - 14) * (pixel_size + spacing) * RIGHT +  
                (14 - i) * (pixel_size + spacing) * UP  
            )  
            pixel_group.add(square)  # Add the square to the pixel group  
    
    return pixel_group  

# Set colors and opacities based on activation values  
def set_activation(circles, activation_values):  
    animations = []  
    for circle, value in zip(circles, activation_values):  
        if value == -1:  
            continue  # Skip if value is -1 (indicating no activation)  
        color = interpolate_color(BLACK, WHITE, value)  # Interpolate color based on activation value  
        opacity = value  # Set opacity based on activation value  
        animations.append(circle.animate.set_fill(color, opacity=opacity))  # Animate the color change  
    return animations  

def generate_circles_with_vertical_ellipsis(  
    n,  
    radius=0.3,  
    spacing=0.8,  
    color=WHITE,  
    ellipsis_dots=3,  
    ellipsis_dot_radius=None,  
    ellipsis_buff=0.1,  
    ellipsis_color=None,  
    stroke_width=1.5  # Set the thickness of the circle stroke  
):  
    """  
    Generate a column of circles. If there are more than 16, display eight on top and bottom,  
    with vertically arranged ellipses in the middle.  

    Parameters:  
    - n (int): Total number of circles  
    - radius (float): Circle radius  
    - spacing (float): Vertical spacing between circles  
    - color (str/Color): Color of circles and ellipses  
    - ellipsis_dots (int): Number of dots in the ellipsis, default is 3  
    - ellipsis_dot_radius (float): Radius of each dot in the ellipsis, default is radius/6  
    - ellipsis_buff (float): Vertical spacing between dots in the ellipsis, default is 0.1  
    - ellipsis_color (str/Color): Color of dots in the ellipsis, default is the same as circle color  
    - stroke_width (float): Thickness of the circle border, default is 1.5  

    Returns:  
    - elements (VGroup): A group containing circles and ellipses  
    - circles (List[Circle]): A list of all circles  
    """  
    elements = VGroup()  # Create a VGroup to hold circles and ellipses  
    circles = []  # List to store all circles  

    # Set the radius and color for ellipsis dots  
    if ellipsis_dot_radius is None:  
        ellipsis_dot_radius = radius / 6  
    if ellipsis_color is None:  
        ellipsis_color = color  

    if n <= 16:  
        # Display all circles  
        for i in range(n):  
            circle = Circle(  
                radius=radius,  
                color=color,  
                stroke_width=stroke_width  # Set the thickness of the circle stroke  
            )  
            circle.number = i + 1  # Add a numbering attribute  
            circles.append(circle)  
            elements.add(circle)  
    else:  
        # Top eight circles  
        for i in range(8):  
            circle = Circle(  
                radius=radius,  
                color=color,  
                stroke_width=stroke_width  # Set the thickness of the circle stroke  
            )  
            circle.number = i + 1  # Add a numbering attribute  
            circles.append(circle)  
            elements.add(circle)  
        
        # Add vertically arranged ellipses  
        ellipsis = VGroup()  
        for _ in range(ellipsis_dots):  
            dot = Dot(  
                radius=ellipsis_dot_radius,  
                color=ellipsis_color  
            )  
            ellipsis.add(dot)  
        ellipsis.arrange(DOWN, buff=ellipsis_buff)  # Arrange dots vertically  
        elements.add(ellipsis)  
        
        # Bottom eight circles  
        for i in range(n - 8, n):  
            circle = Circle(  
                radius=radius,  
                color=color,  
                stroke_width=stroke_width  # Set the thickness of the circle stroke  
            )  
            circle.number = i + 1  # Add a numbering attribute  
            circles.append(circle)  
            elements.add(circle)  

    # Set spacing between circles  
    elements.arrange(DOWN, buff=spacing / 4)  
    return elements, circles  

def add_full_connections_between_groups(  
    group1_circles,  
    group2_circles,  
    connection_color=GRAY,  
    stroke_width=0.5,  
    buff=0.1  
):  
    """  
    Add full connections between two groups of circles.  

    Parameters:  
        group1_circles (List[Circle]): The first group of circles.  
        group2_circles (List[Circle]): The second group of circles.  
        connection_color (Color): Color of the connecting lines.  
        stroke_width (float): Thickness of the connecting lines.  
        buff (float): Buffer space between circles and lines.  

    Returns:  
        lines (VGroup): A group containing all connecting lines.  
    """  
    lines = VGroup()  # Create a VGroup to hold connecting lines  
    for circle1 in group1_circles:  
        for circle2 in group2_circles:  
            line = Line(  
                circle1.get_right() + RIGHT * buff,  
                circle2.get_left() + LEFT * buff,  
                color=connection_color,  
                stroke_width=stroke_width  
            )  
            lines.add(line)  # Add the line to the group  
    return lines  

def split_line_into_sublines(line, num_subsegments=100, overlap=0.01, stroke_width=2):  
    """  
    Split a line segment into multiple sub-segments and add overlap to reduce the visibility of boundaries.  

    Parameters:  
        line (Line): The line to be split.  
        num_subsegments (int): Number of sub-segments to create.  
        overlap (float): Amount of overlap between sub-segments.  
        stroke_width (float): Thickness of the sub-segments.  

    Returns:  
        sublines (VGroup): A group containing all sub-segments.  
    """  
    sublines = VGroup(*[  
        Line(  
            line.point_from_proportion(i / num_subsegments),  
            line.point_from_proportion((i + 1) / num_subsegments) + overlap * RIGHT,  
            color=BLUE,  
            stroke_width=stroke_width  
        ) for i in range(num_subsegments)  
    ])  
    return sublines  

def continuous_color_wave(mob, alpha):  
    """  
    Define a function for continuous color wave fluctuation.  
    The color transitions from blue to yellow and back to blue in a loop.  

    Parameters:  
        mob (Mobject): The Manim object to apply the color wave to.  
        alpha (float): A value between 0 and 1 representing the progress of the animation.  
    """  
    phase = alpha * TAU  # Calculate the phase based on alpha  
    gradient_colors = color_gradient([BLUE, YELLOW, BLUE], 100)  # Create a gradient color array  
    num_subsegments = len(mob)  # Get the number of segments in the object  
    
    for i in range(num_subsegments):  
        t = i / num_subsegments  
        offset = (t * TAU + phase) % TAU  # Calculate the offset for color cycling  
        color_value = (np.sin(offset) + 1) / 2  # Normalize to range [0, 1]  
        color_index = int(color_value * (len(gradient_colors) - 1))  # Get the index for the gradient color  
        current_color = gradient_colors[color_index]  # Get the current color from the gradient  
        mob[i].set_color(current_color)  # Set the color of the segment  

def single_pulse_wave(mob, alpha):  
    """  
    Create a pulse effect where the pulse center moves from -pulse_width to 1 + pulse_width,  
    ensuring that the pulse tail reaches the end of the line segment at the end of the animation.  

    Parameters:  
        mob (Mobject): The Manim object to apply the pulse effect to.  
        alpha (float): A value between 0 and 1 representing the progress of the animation.  
    """  
    pulse_width = 0.05  # Width of the pulse  
    num_subsegments = len(mob)  # Get the number of segments in the object  

    # Map alpha from [0, 1] to the pulse center range [-pulse_width, 1 + pulse_width]  
    current_pulse = -pulse_width + alpha * (1 + pulse_width - (-pulse_width))  

    for i in range(num_subsegments):  
        t = i / num_subsegments  
        if current_pulse - pulse_width < t < current_pulse + pulse_width:  
            mob[i].set_color(YELLOW)  # Set color to yellow within the pulse range  
        else:  
            mob[i].set_color(BLUE)  # Set color to blue outside the pulse range  

def animate_line_wave(line, wave_type="continuous",  
                      num_subsegments=100, overlap=0.01, stroke_width=2):  
    """  
    Accept a Line, split it into segments, and play the corresponding animation based on the wave_type parameter.  
    wave_type supports "continuous" (continuous wave) or "pulse" (pulse wave).  

    Parameters:  
        line (Line): The line to animate.  
        wave_type (str): The type of wave animation to apply.  
        num_subsegments (int): Number of sub-segments to create.  
        overlap (float): Amount of overlap between sub-segments.  
        stroke_width (float): Thickness of the sub-segments.  

    Returns:  
        sublines (VGroup): A group containing all sub-segments.  
        line_wave_animation (Animation): The animation for the wave effect.  
    """  
    # 1. Split the line into sub-segments  
    sublines = split_line_into_sublines(  
        line, num_subsegments, overlap, stroke_width  
    )  
    
    # 2. Choose the animation update function based on wave type  
    if wave_type == "continuous":  
        wave_func = continuous_color_wave  
    elif wave_type == "pulse":  
        wave_func = single_pulse_wave  
    else:  
        raise ValueError("wave_type must be 'continuous' or 'pulse'.")  
    
    # 3. Create the animation (UpdateFromAlphaFunc will continuously call wave_func during playback)  
    line_wave_animation = UpdateFromAlphaFunc(  
        sublines,  
        lambda m, alpha: wave_func(m, alpha)  
    )  
    
    return sublines, line_wave_animation