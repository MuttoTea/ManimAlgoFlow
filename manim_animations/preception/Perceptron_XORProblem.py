"""  
Summary:  
This code uses the Manim library to visualize the operation of an XOR (exclusive OR) logic gate. The main features include:  

1. Scene Construction:  
   - Create an XOR gate (XorGate) and display its inputs and output.  
   - Show different input combinations (0 and 1) and their corresponding output results.  

2. Table Display:  
   - Create a table that illustrates the input-output relationship of the XOR gate.  
   - The table includes a header row and data rows, displaying the data row by row.  
"""  

from manim import *  
from CustomClasses import XorGate  


class XORProblem(Scene):  
    def construct(self):  
        # Title  
        title = Text("XOR Problem").set_color(RED)  
        self.play(Write(title))  
        self.wait(1)  

        self.play(title.animate.to_edge(UP, buff=0.5))  

        # Create an XOR gate positioned at the origin  
        input_color = GREEN  
        output_color = BLUE  
        xor_gate = XorGate(  
            center=ORIGIN - RIGHT,  
            arcs_color=RED,                     # Color of the arcs  
            fill_color=RED,                     # Fill color (can be changed to YELLOW, etc.)  
            fill_opacity=0.3                    # Fill opacity  
        ).scale(0.8)  
        self.play(Create(xor_gate))  
        self.wait(2)  

        # Create input labels  
        input_label_x1 = MathTex("x_1").move_to(xor_gate.input_point_up.get_center() - RIGHT * 0.5).set_color(input_color)  
        input_label_x2 = MathTex("x_2").move_to(xor_gate.input_point_down.get_center() - RIGHT * 0.5).set_color(input_color)  
        input_label_y = MathTex("y").move_to(xor_gate.output_point.get_center() + RIGHT * 0.5).set_color(output_color)  
        self.play(Create(input_label_x1), Create(input_label_x2), Create(input_label_y))  

        # Define input combinations for the XOR gate  
        input_combinations = [  
            (0, 0),  
            (1, 1),  
            (1, 0),  
            (0, 1)  
        ]  

        # Iterate through input combinations and display results  
        for x1, x2 in input_combinations:  
            self.play(  
                Transform(input_label_x1, MathTex(x1).move_to(xor_gate.input_point_up.get_center() - RIGHT * 0.5).set_color(input_color)),  
                Transform(input_label_x2, MathTex(x2).move_to(xor_gate.input_point_down.get_center() - RIGHT * 0.5).set_color(input_color)),  
                run_time=0.5  
            )  
            self.wait(0.5)  

            # Calculate the output of the XOR gate  
            y = x1 ^ x2  
            self.play(  
                Transform(input_label_y, MathTex(y).move_to(xor_gate.output_point.get_center() + RIGHT * 0.5).set_color(output_color)),  
                run_time=0.5  
            )  
            self.wait(0.5)  
        self.wait(2)  

        # Group the XOR gate and labels for scaling and positioning  
        group_elements = VGroup(xor_gate, input_label_x1, input_label_x2, input_label_y)  
        self.play(group_elements.animate.scale(0.9).to_edge(LEFT, buff=1))  
        self.wait(2)  

        # ── 1. Define table data ──────────────────────────────────  
        # Contains header row and data rows  
        table_data = [  
            ["X_{1}", "X_{2}", "X_{1} \\oplus X_{2}"],  # Header row  
            [1, 1, 0],  
            [0, 0, 0],  
            [0, 1, 1],  
            [1, 0, 1],  
        ]  

        # ── 2. Initialize Table ────────────────────────────────────  
        # include_outer_lines=True to show outer borders  
        # include_inner_lines=True to show inner lines (enable or disable as needed)  
        # element_to_mobject=lambda x: MathTex(str(x)) to render cell content as math  
        table = Table(  
            table_data,  
            include_outer_lines=True,  
            element_to_mobject=lambda x: MathTex(x) if isinstance(x, str) else MathTex(str(x)),  
        )  

        # ── 3. Scale and position ──────────────────────────────────────  
        table.scale(0.7)    # Scale the table to an appropriate size  
        table.to_edge(RIGHT, buff=1.0)   # Move the table to the right side of the screen  

        # Set colors for header entries  
        table.get_entries((1, 1)).set_color(input_color)  # Set color for "X1" in header  
        table.get_entries((1, 2)).set_color(input_color)  # Set color for "X2" in header  
        table.get_entries((1, 3)).set_color(output_color)  # Set color for "X1 XOR X2" in header  

        # ── 4. Hide data row contents ──────────────────────────────────  
        # Get all table entries  
        entries = table.get_entries()  
        num_rows = len(table_data)  
        num_cols = len(table_data[0])  

        # The header occupies the first row (index starts from 0)  
        for i, entry in enumerate(entries):  
            # Calculate the current cell's row and column (row starts from 1)  
            current_row = (i // num_cols) + 1  # Row number  
            current_col = (i % num_cols) + 1   # Column number  

            if current_row > 1:  # If not the header row  
                entry.set_opacity(0)  # Hide cell content  

        # ── 5. Play table creation animation ──────────────────────────────────  
        self.play(Create(table))  # Create table borders and header  
        self.wait(1)               # Wait for 1 second  

        # ── 6. Display table data row by row ────────────────────────────────────  
        for row in range(2, len(table_data) + 1):  # Data rows start from 2  
            # Get the current row's input A and input B cells  
            input_a = table.get_entries((row, 1))  
            input_b = table.get_entries((row, 2))  
            cell = table.get_entries((row, 3))  
            # Show input A and input B simultaneously  
            self.play(  
                input_a.animate.set_opacity(1),  
                input_b.animate.set_opacity(1),  
                cell.animate.set_opacity(1),  
                run_time=0.5  
            )  
            self.wait(0.2)  
        self.wait(2)  # Finally pause for 2 seconds to display the complete table