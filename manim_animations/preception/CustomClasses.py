from manim import *


class XorGate(VGroup):  
    def __init__(  
        self,  
        center=ORIGIN,  # Center coordinates of the XOR gate  
        arc_color=RED,  # Color of the arcs  
        input_line_color=RED,  # Color of the input lines  
        output_line_color=RED,  # Color of the output line  
        fill_color=None,  # Fill color (if needed)  
        fill_opacity=0.5,  # Fill opacity  
        **kwargs,  
    ):  
        """  
        Encapsulates an XOR Gate.  

        Parameters:  
            center (np.ndarray): Center coordinates of the XOR gate, default is the origin.  
            arc_color (Manim color): Color of the arcs.  
            input_line_color (Manim color): Color of the input lines.  
            output_line_color (Manim color): Color of the output line.  
            fill_color (Manim color or None): Fill color. If None, no fill is applied.  
            fill_opacity (float): Opacity of the fill.  
            **kwargs: Additional arguments passed to VGroup.  
        """  
        super().__init__(**kwargs)  # Call the parent class constructor  

        # Define the major and minor axis radii for the outer and inner ellipses  
        outer_major_axis, outer_minor_axis = 5, 2  # Major and minor axis radii of the outer ellipse  
        inner_major_axis, inner_minor_axis = 1.5, 2  # Major and minor axis radii of the inner ellipse  
        start_angle, end_angle = -PI / 2, PI / 2  # Start and end angles of the arcs  

        # Create the outer elliptical arc  
        outer_arc = self.create_elliptical_arc(  
            a=outer_major_axis,  
            b=outer_minor_axis,  
            center=center,  
            color=arc_color,  
            scale_factor=0.4,  
            t_range=[start_angle, end_angle],  
        )  

        # Create the inner elliptical arcs  
        inner_arc1 = self.create_elliptical_arc(  
            a=inner_major_axis,  
            b=inner_minor_axis,  
            center=center,  
            color=arc_color,  
            scale_factor=0.4,  
            t_range=[start_angle, end_angle],  
        )  

        inner_arc2 = self.create_elliptical_arc(  
            a=inner_major_axis,  
            b=inner_minor_axis,  
            center=center + np.array([-inner_major_axis / 2, 0, 0]),  # Shift the second inner arc to the left  
            color=arc_color,  
            scale_factor=0.4,  
            t_range=[start_angle, end_angle],  
        )  

        # Create input lines for the XOR gate  
        input_lines = self.create_input_lines(  
            center=center,  
            inner_major_axis=inner_major_axis,  
            inner_minor_axis=inner_minor_axis,  
            input_line_color=input_line_color,  
        )  

        # Create the output line for the XOR gate  
        output_line = self.create_output_line(  
            center=center,  
            outer_major_axis=outer_major_axis,  
            outer_minor_axis=outer_minor_axis,  
            output_line_color=output_line_color,  
        )  

        # Add all components to the XOR gate  
        self.add(  
            outer_arc,  
            inner_arc1,  
            inner_arc2,  
            *input_lines,  
            output_line,  
        )  

        # If a fill color is specified, create a filled area  
        if fill_color and fill_opacity > 0:  
            filled_area = self.create_filled_area(  
                arc_outer=outer_arc,  
                arc_inner=inner_arc1,  
                fill_color=fill_color,  
                fill_opacity=fill_opacity,  
            )  
            self.add(filled_area)  

    def create_elliptical_arc(self, major_axis, minor_axis, center, color, scale_factor, t_range):  
        """  
        Create an elliptical arc.  

        Parameters:  
            major_axis (float): Major axis radius.  
            minor_axis (float): Minor axis radius.  
            center (np.ndarray): Center coordinates.  
            color (Manim color): Color of the arc.  
            scale_factor (float): Scaling factor for the arc.  
            t_range (list): Parameter range [start parameter, end parameter].  

        Returns:  
            ParametricFunction: Created elliptical arc object.  
        """  
        return ParametricFunction(  
            lambda t: np.array(  
                [major_axis * np.cos(t) + center[0], minor_axis * np.sin(t) + center[1], 0]  
            ),  
            t_range=t_range,  
            color=color,  
        ).scale(scale_factor, about_point=center)  # Scale the arc around its center  

    def create_input_lines(self, center, inner_major_axis, inner_minor_axis, input_line_color):  
        """  
        Create input lines and their hidden point markers.  

        Parameters:  
            center (np.ndarray): Center coordinates of the XOR gate.  
            inner_major_axis (float): Major axis radius of the inner ellipse.  
            inner_minor_axis (float): Minor axis radius of the inner ellipse.  
            input_line_color (Manim color): Color of the input lines.  

        Returns:  
            list: List of input line objects.  
        """  
        offset = np.array([inner_major_axis / 3.2, 0, 0])  # Define offset for reference point  
        reference_coord = center + offset  # Calculate the reference point for input lines  

        x_coordinate = reference_coord[0]  # X-coordinate of the reference point  
        fraction = (x_coordinate**2) / (inner_major_axis**2)  # Calculate the fraction for intersection points  

        if fraction <= 1:  
            # Calculate upper and lower intersection points on the inner ellipse  
            y_distance = np.sqrt(1 - fraction) / inner_minor_axis  
            intersection_up = np.array([x_coordinate, y_distance + center[1], 0])  
            intersection_down = np.array([x_coordinate, -y_distance + center[1], 0])  

            # Define start points of input lines (extending to the left)  
            input_up_start = intersection_up.copy()  
            input_up_start[0] -= 2.0  # Extend the line to the left  
            input_down_start = intersection_down.copy()  
            input_down_start[0] -= 2.0  # Extend the line to the left  

            # Create input lines  
            input_line_up = Line(  
                start=input_up_start, end=intersection_up, color=input_line_color  
            )  
            input_line_down = Line(  
                start=input_down_start, end=intersection_down, color=input_line_color  
            )  

            # Create hidden point markers for the input lines  
            input_point_up = Dot(point=input_up_start, color=RED).set_opacity(0)  # Hidden marker  
            input_point_down = Dot(point=input_down_start, color=RED).set_opacity(0)  # Hidden marker  

            # Add input lines and points to VGroup  
            self.add(input_line_up, input_line_down, input_point_up, input_point_down)  

            # Store references to points for later use  
            self.input_point_up = input_point_up  
            self.input_point_down = input_point_down  

            return [input_line_up, input_line_down]  
        else:  
            # If the reference point is outside the inner ellipse, do not draw input lines  
            self.input_point_up = None  
            self.input_point_down = None  
            return []  

    def create_output_line(self, center, outer_major_axis, outer_minor_axis, output_line_color):  
        """  
        Create the output line and its hidden point marker.  

        Parameters:  
            center (np.ndarray): Center coordinates of the XOR gate.  
            outer_major_axis (float): Major axis radius of the outer ellipse.  
            outer_minor_axis (float): Minor axis radius of the outer ellipse.  
            output_line_color (Manim color): Color of the output line.  

        Returns:  
            Line: Created output line object.  
        """  
        output_offset = np.array([outer_minor_axis, 0, 0])  # Define offset for the output point  
        output_coord = center + output_offset  # Calculate the output point coordinates  

        # Define the end point of the output line (extending to the right)  
        output_line_end = output_coord.copy()  
        output_line_end[0] += 2.0  # Extend the line to the right  

        # Create the output line  
        output_line = Line(  
            start=output_coord, end=output_line_end, color=output_line_color  
        )  

        # Create the hidden output point marker  
        self.output_point = Dot(point=output_line_end, color=RED).set_opacity(0)  # Hidden marker  

        # Add the output line and point to VGroup  
        self.add(output_line, self.output_point)  

        return output_line  

    def create_filled_area(self, arc_outer, arc_inner, fill_color, fill_opacity):  
        """  
        Create a filled area between the outer and inner arcs.  

        Parameters:  
            arc_outer (ParametricFunction): Outer arc object.  
            arc_inner (ParametricFunction): Inner arc object.  
            fill_color (Manim color): Fill color.  
            fill_opacity (float): Fill opacity.  

        Returns:  
            VMobject: Created filled area object.  
        """  
        # Get points from the arcs  
        outer_arc_points = arc_outer.get_points()  # Points of the outer arc  
        inner_arc_points = arc_inner.get_points()[::-1]  # Reverse the inner arc points  

        # Combine points to form the filled area  
        filled_area = VMobject()  
        filled_area.set_points_as_corners(  
            np.concatenate([outer_arc_points, inner_arc_points])  # Create a closed shape  
        )  
        filled_area.set_fill(fill_color, opacity=fill_opacity)  # Set fill color and opacity  
        filled_area.set_stroke(None, 0)  # Remove the border  

        return filled_area  # Return the filled area object


class Perceptron(VGroup):  
    def __init__(  
        self,  
        center=ORIGIN,  # Center position of the perceptron layer  
        num_inputs=4,  # Default number of input neurons  
        input_line_color="#6ab7ff",  # Color of the input lines  
        input_line_tip_length=0.2,  # Length of the input line arrow tips  
        input_line_opacity=0.6,  # Opacity of the input lines  
        receive_color="#ff9e9e",  # Color of the receiving layer  
        output_line_color=GREEN,  # Color of the output line  
        output_line_tip_length=0.3,  # Length of the output line arrow tip  
        output_line_opacity=0.6,  # Opacity of the output line  
        show_input_circles=True,  # Whether to show input layer circles  
        show_input_labels=True,  # Whether to show input layer labels  
        input_opacity=1.0,  # Opacity of the input layer circles  
        show_output_layer=True,  # Whether to show the output layer circle  
        show_output_label=True,  # Whether to show the output layer label  
        output_opacity=1.0,  # Opacity of the output layer circle  
        show_input_layer=True,  # Whether to show the entire input layer  
        **kwargs,  
    ):  
        super().__init__(**kwargs)  # Initialize the parent class  

        # Validate parameters to ensure they meet the expected criteria  
        self._validate_parameters(  
            num_inputs,  
            input_line_tip_length,  
            input_line_opacity,  
            output_line_tip_length,  
            output_line_opacity,  
            input_opacity,  
            output_opacity,  
            show_input_circles,  
            show_input_labels,  
            show_output_layer,  
            show_output_label,  
            show_input_layer,  
        )  

        # Store parameters for later use  
        self.receive_center = center  # Center position of the receiving layer  
        self.input_line_color = input_line_color  # Color for input lines  
        self.input_line_opacity = input_line_opacity  # Opacity for input lines  
        self.input_line_tip_length = input_line_tip_length  # Arrow tip length for input lines  
        self.output_line_color = output_line_color  # Color for output line  
        self.output_opacity = output_opacity  # Opacity for output layer circle  
        self.output_line_opacity = output_line_opacity  # Opacity for output line  
        self.output_line_tip_length = output_line_tip_length  # Arrow tip length for output line  
        self.receive_color = receive_color  # Color for the receiving layer  
        self.num_inputs = num_inputs  # Number of input neurons  
        self.show_input_circles = show_input_circles  # Flag to show input circles  
        self.show_input_labels = show_input_labels  # Flag to show input labels  
        self.input_opacity = input_opacity  # Opacity for input circles  
        self.show_output_layer = show_output_layer  # Flag to show output layer  
        self.show_output_label = show_output_label  # Flag to show output label  
        self.show_input_layer = show_input_layer  # Flag to show input layer  

        # Calculate the center positions for the input layer neurons  
        self.input_layer_centers = self._calculate_input_layer_centers()  

        # Create the components of the perceptron  
        self.receive_layer = self._create_receive_layer()  # Create the receiving layer  
        self.output_layer, self.output_label = self._create_output_layer()  # Create output layer and label  
        self.input_circles, self.input_labels = self._create_input_layer()  # Create input layer circles and labels  
        self.connections = self._create_connections()  # Create connections (arrows) from inputs to receiving layer  
        self.output_arrow = self._create_output_arrow()  # Create output arrow from receiving layer to output layer  

        # Add all components to the VGroup for rendering  
        self.add(  
            self.receive_layer,  
            self.output_layer,  
            self.output_label,  
            self.input_circles,  
            self.input_labels,  
            self.connections,  
            self.output_arrow,  
        )  

        # Control the visibility of the entire input layer based on the show_input_layer flag  
        if not self.show_input_layer:  
            self.disable_input_layer()  # Hide the input layer if the flag is set to False  

    def _validate_parameters(  
        self,  
        num_inputs,  
        input_line_tip_length,  
        input_line_opacity,  
        output_line_tip_length,  
        output_line_opacity,  
        input_opacity,  
        output_opacity,  
        show_input_circles,  
        show_input_labels,  
        show_output_layer,  
        show_output_label,  
        show_input_layer,  
    ):  
        """Validate the initialization parameters to ensure they are within acceptable ranges."""  
        # Check if num_inputs is a positive integer  
        if not isinstance(num_inputs, int) or num_inputs <= 0:  
            raise ValueError("num_inputs must be a positive integer.")  

        # Check if input_line_tip_length is a positive number  
        if not (  
            isinstance(input_line_tip_length, (int, float))  
            and input_line_tip_length > 0  
        ):  
            raise ValueError("input_line_tip_length must be a positive number.")  

        # Check if input_line_opacity is in the range [0.0, 1.0]  
        if not (  
            isinstance(input_line_opacity, (int, float))  
            and 0.0 <= input_line_opacity <= 1.0  
        ):  
            raise ValueError("input_line_opacity must be in the range [0.0, 1.0].")  

        # Check if output_line_tip_length is a positive number  
        if not (  
            isinstance(output_line_tip_length, (int, float))  
            and output_line_tip_length > 0  
        ):  
            raise ValueError("output_line_tip_length must be a positive number.")  

        # Check if output_line_opacity is in the range [0.0, 1.0]  
        if not (  
            isinstance(output_line_opacity, (int, float))  
            and 0.0 <= output_line_opacity <= 1.0  
        ):  
            raise ValueError("output_line_opacity must be in the range [0.0, 1.0].")  

        # Check if input_opacity is in the range [0.0, 1.0]  
        if not (  
            isinstance(input_opacity, (int, float)) and 0.0 <= input_opacity <= 1.0  
        ):  
            raise ValueError("input_opacity must be in the range [0.0, 1.0].")  

        # Check if output_opacity is in the range [0.0, 1.0]  
        if not (  
            isinstance(output_opacity, (int, float)) and 0.0 <= output_opacity <= 1.0  
        ):  
            raise ValueError("output_opacity must be in the range [0.0, 1.0].")  

        # Check if show_input_circles is a boolean  
        if not isinstance(show_input_circles, bool):  
            raise TypeError("show_input_circles must be a boolean.")  

        # Check if show_input_labels is a boolean  
        if not isinstance(show_input_labels, bool):  
            raise TypeError("show_input_labels must be a boolean.")  

        # Check if show_output_layer is a boolean  
        if not isinstance(show_output_layer, bool):  
            raise TypeError("show_output_layer must be a boolean.")  

        # Check if show_output_label is a boolean  
        if not isinstance(show_output_label, bool):  
            raise TypeError("show_output_label must be a boolean.")  

        # Check if show_input_layer is a boolean  
        if not isinstance(show_input_layer, bool):  
            raise TypeError("show_input_layer must be a boolean.")  

    def _calculate_input_layer_centers(self):  
        """Calculate the center positions of the input layer neurons based on the number of inputs."""  
        horizontal_distance = 3  # Horizontal distance from the receiving layer  

        vertical_spacing = 0.7  # Vertical spacing between input neurons  

        # Determine how many neurons to display  
        if self.num_inputs > 3:  
            display_count = 4  # Show first two, one ellipsis, and the last one  
        else:  
            display_count = self.num_inputs  # Show all neurons if 3 or fewer  

        # Calculate total height for displaying neurons  
        total_height = (display_count - 1) * vertical_spacing  

        # Starting y-coordinate for vertical centering with the receiving layer  
        start_y = self.receive_center[1] + total_height / 2  

        # Generate list of center positions for input neurons  
        centers = [  
            self.receive_center  
            + LEFT * horizontal_distance  
            + np.array([0, start_y - i * vertical_spacing, 0])  
            for i in range(display_count)  
        ]  
        self.input_layer_centers = centers  # Store the calculated centers  

        return centers  # Return the list of center positions  

    def _create_receive_layer(self):  
        """Create the receiving layer of the perceptron."""  
        receive_layer = Circle(  
            radius=0.3, fill_opacity=0.5, color=self.receive_color  
        )  # Create a circle to represent the receiving layer  
        receive_layer.move_to(self.receive_center)  # Position it at the specified center  
        return receive_layer  # Return the created layer  

    def _create_output_layer(self):  
        """Create the output layer of the perceptron."""  
        if not self.show_output_layer:  
            # If the output layer is not to be shown, return empty groups  
            self.output_layer = VGroup()  
            self.output_label = VGroup()  
            return self.output_layer, self.output_label  

        # Create the output layer circle  
        self.output_layer = Circle(  
            radius=0.2, fill_opacity=self.output_opacity, color=self.output_line_color  
        )  
        self.output_layer.next_to(self.receive_layer, RIGHT, buff=2.5)  # Position it to the right of the receiving layer  

        # Create the output label if it is to be shown  
        if self.show_output_label:  
            self.output_label = (  
                MathTex("y").scale(0.8).move_to(self.output_layer.get_center())  
            )  
        else:  
            self.output_label = VGroup()  # Empty object if label is not shown  

        return self.output_layer, self.output_label  # Return the output layer and label  

    def _create_input_layer(self):  
        """Create the input layer circles and labels."""  
        input_circles = VGroup()  # Group to hold input circles  
        input_labels = VGroup()  # Group to hold input labels  
        input_circles_opacity = 1 if self.show_input_circles else 0  # Set opacity based on visibility flag  
        input_labels_opacity = 1 if self.show_input_labels else 0  # Set opacity based on visibility flag  

        # Loop through each center position to create circles and labels  
        for i, center in enumerate(self.input_layer_centers):  
            if self.num_inputs > 3 and i == 2:  
                # If there are more than 3 inputs, show ellipsis at the third position  
                ellipsis = Tex("...").scale(0.8)  
                ellipsis.move_to(center)  # Position ellipsis at the center  
                input_labels.add(ellipsis)  # Add ellipsis to labels  
                continue  # Skip to the next iteration  

            # Create a circle for the input neuron  
            circle = Circle(  
                radius=0.2, fill_opacity=self.input_opacity, color=self.input_line_color  
            )  
            circle.move_to(center)  # Position the circle at the calculated center  
            circle.set_opacity(input_circles_opacity)  # Set the opacity of the circle  
            input_circles.add(circle)  # Add the circle to the group  

            # Create a label for the input neuron  
            label_text = (  
                f"x_{{{i+1}}}"  
                if not (self.num_inputs > 3 and i == (self.num_inputs - 1))  
                else f"x_{{{self.num_inputs}}}"  
            )  
            label = MathTex(label_text).scale(0.8)  # Create a label with the appropriate text  
            label.shift(UP * 0.4).move_to(center)  # Position the label above the circle  
            label.set_opacity(input_labels_opacity)  # Set the opacity of the label  
            input_labels.add(label)  # Add the label to the group  

        return input_circles, input_labels  # Return the groups of circles and labels  

    def _create_connections(self):  
        """Create connections (arrows) from input circles to the receiving layer."""  
        connections = VGroup()  # Group to hold connection arrows  
        for i, center in enumerate(self.input_layer_centers):  
            start_point = center + RIGHT * 0.2  # Start point of the connection arrow  
            end_point = self.receive_layer.get_left() + RIGHT * 0.2  # End point at the receiving layer  

            # Create an arrow to represent the connection  
            connection = Arrow(  
                start=start_point,  
                end=end_point,  
                color=self.input_line_color,  
                stroke_opacity=self.input_line_opacity,  
                tip_length=self.input_line_tip_length,  
                buff=0.2,  
            )  
            connections.add(connection)  # Add the connection to the group  
        return connections  # Return the group of connections  

    def _create_output_arrow(self):  
        """Create the output arrow from the receiving layer to the output layer."""  
        if not self.show_output_layer:  
            return VGroup()  # Return an empty group if the output layer is not shown  

        # Create an arrow to represent the output connection  
        output_arrow = Arrow(  
            start=self.receive_layer.get_right() + LEFT * 0.2,  # Start point at the right of the receiving layer  
            end=self.output_layer.get_left() + RIGHT * 0.2,  # End point at the left of the output layer  
            color=self.output_line_color,  
            stroke_opacity=self.output_line_opacity,  
            tip_length=self.output_line_tip_length,  
            buff=0.4,  
        )  
        return output_arrow  # Return the output arrow  

    def get_positions(self):  
        """Get the positions of the perceptron components for external use."""  
        positions = {  
            "receive_layer": self.receive_layer.get_center(),  # Center position of the receiving layer  
            "output_layer": (  
                self.output_layer.get_center() if self.show_output_layer else None  
            ),  # Center position of the output layer if visible  
            "output_label": (  
                self.output_label.get_center() if self.show_output_label else None  
            ),  # Center position of the output label if visible  
            "output_arrow_start": (  
                self.output_arrow.get_start() if self.show_output_layer else None  
            ),  # Start position of the output arrow if visible  
            "output_arrow_end": (  
                self.output_arrow.get_end() if self.show_output_layer else None  
            ),  # End position of the output arrow if visible  
        }  

        if self.show_input_labels or self.show_input_circles:  
            # Extract positions from the input layer if labels or circles are shown  
            input_positions = []  
            for obj in self.input_circles:  
                input_positions.append(obj.get_center())  # Get the center of each input circle  
            positions["input_layer"] = input_positions  # Store the positions in the dictionary  
        else:  
            # Return pre-calculated center positions if labels and circles are not shown  
            positions["input_layer"] = self.input_layer_centers.copy()  

        return positions  # Return the dictionary of positions  

    def add_position_markers(self):  
        """Add markers to indicate the positions of the layers for visualization."""  
        markers = VGroup()  # Group to hold all markers  

        # Marker for the receiving layer  
        receive_marker = Dot(point=self.receive_layer.get_center(), color=RED)  # Create a dot marker  
        receive_label = Text("Receiving Layer", font_size=24).next_to(receive_marker, UP)  # Create a label for the marker  
        markers.add(VGroup(receive_marker, receive_label))  # Add marker and label to the group  

        # Markers for the input layer  
        for idx, pos in enumerate(self.input_layer_centers, start=1):  
            if self.num_inputs > 3 and idx == 3:  
                # If there are more than 3 inputs, show ellipsis at the third position  
                input_marker = Dot(point=pos, color=RED)  # Create a dot marker  
                input_label = Text("...", font_size=24).next_to(input_marker, UP)  # Create an ellipsis label  
            else:  
                input_marker = Dot(point=pos, color=RED)  # Create a dot marker for input  
                input_label = Text(f"Input Layer {idx}", font_size=24).next_to(  
                    input_marker, UP  
                )  # Create a label for the input layer  
            markers.add(VGroup(input_marker, input_label))  # Add marker and label to the group  

        # Marker for the output layer  
        if self.show_output_layer:  
            output_marker = Dot(point=self.output_layer.get_center(), color=RED)  # Create a dot marker for output  
            output_label = Text("Output Layer", font_size=24).next_to(output_marker, UP)  # Create a label for output  
            markers.add(VGroup(output_marker, output_label))  # Add marker and label to the group  

        self.add(markers)  # Add all markers to the main group  
        return markers  # Return the group of markers  

    def enable_input_labels(self):  
        """Show input layer labels by setting their opacity to 1."""  
        if self.show_input_labels and hasattr(self, "input_labels"):  
            self.input_labels.set_opacity(1)  # Set opacity to 1 to make labels visible  

    def disable_input_labels(self):  
        """Hide input layer labels by setting their opacity to 0."""  
        if hasattr(self, "input_labels"):  
            self.input_labels.set_opacity(0)  # Set opacity to 0 to hide labels  

    def enable_output_label(self):  
        """Show output layer label by setting its opacity to 1."""  
        if self.show_output_label and hasattr(self, "output_label"):  
            self.output_label.set_opacity(1)  # Set opacity to 1 to make the label visible  

    def disable_output_label(self):  
        """Hide output layer label by setting its opacity to 0."""  
        if hasattr(self, "output_label"):  
            self.output_label.set_opacity(0)  # Set opacity to 0 to hide the label  

    def enable_input_connections(self):  
        """Show input layer connections by setting their opacity to 1."""  
        self.connections.set_opacity(1)  # Set opacity to 1

    def disable_input_connections(self):  
        """Hide input layer connections by setting their opacity to 0."""  
        self.connections.set_opacity(0)  # Set opacity to 0 to hide the connections  

    def enable_output_circle(self):  
        """Show output layer circle by setting its opacity to 1."""  
        if self.show_output_layer and hasattr(self, "output_layer"):  
            self.output_layer.set_opacity(1)  # Set opacity to 1 to make the output circle visible  

    def disable_output_circle(self):  
        """Hide output layer circle by setting its opacity to 0."""  
        if self.show_output_layer and hasattr(self, "output_layer"):  
            self.output_layer.set_opacity(0)  # Set opacity to 0 to hide the output circle  

    def enable_input_circles(self):  
        """Show input layer circles by setting their opacity to 1."""  
        if (  
            self.show_input_layer  
            and hasattr(self, "input_circles")  
            and hasattr(self, "input_labels")  
        ):  
            self.input_circles.set_opacity(1)  # Set opacity to 1 to make input circles visible  

    def disable_input_circles(self):  
        """Hide input layer circles by setting their opacity to 0."""  
        if (  
            self.show_input_layer  
            and hasattr(self, "input_circles")  
            and hasattr(self, "input_labels")  
        ):  
            self.input_circles.set_opacity(0)  # Set opacity to 0 to hide input circles  

    def enable_input_layer(self):  
        """Show the entire input layer by setting the opacity of circles and labels to 1."""  
        if (  
            self.show_input_layer  
            and hasattr(self, "input_circles")  
            and hasattr(self, "input_labels")  
        ):  
            self.input_circles.set_opacity(1)  # Set opacity of input circles to 1  
            self.input_labels.set_opacity(1)  # Set opacity of input labels to 1  

    def disable_input_layer(self):  
        """Hide the entire input layer by setting the opacity of circles and labels to 0."""  
        if (  
            self.show_input_layer  
            and hasattr(self, "input_circles")  
            and hasattr(self, "input_labels")  
        ):  
            self.input_circles.set_opacity(0)  # Set opacity of input circles to 0  
            self.input_labels.set_opacity(0)  # Set opacity of input labels to 0