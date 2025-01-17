from manim import *


class XorGate(VGroup):
    def __init__(
        self,
        center=ORIGIN,  # 异或门的中心坐标
        arcs_color=RED,  # 弧线颜色
        input_line_color=RED,  # 输入线的颜色
        output_line_color=RED,  # 输出线的颜色
        fill_color=None,  # 填充颜色（若需要填充）
        fill_opacity=0.5,  # 填充透明度
        **kwargs,
    ):
        """
        封装一个「异或门」(XOR Gate)。

        参数：
            center (np.ndarray): 异或门的中心坐标，默认为原点。
            arcs_color (Manim 颜色): 弧线的颜色。
            input_line_color (Manim 颜色): 输入线和输出线的颜色。
            fill_color (Manim 颜色 或 None): 填充颜色。如果为 None，则不进行填充。
            fill_opacity (float): 填充的透明度。
            **kwargs: 传递给 VGroup 的其他参数。
        """
        super().__init__(**kwargs)  # 调用父类构造方法

        # ── 1. 定义几何参数 ──────────────────────────────────────
        a_outer, b_outer = 5, 2  # 大椭圆的长轴和短轴半径
        a_inner, b_inner = 1.5, 2  # 小椭圆的长轴和短轴半径
        start_angle, end_angle = -PI / 2, PI / 2  # 弧线的起始和结束角度

        # ── 2. 创建主体三条弧线 ───────────────────────────────────
        elliptical_arc_outer = self.create_elliptical_arc(
            a=a_outer,
            b=b_outer,
            center=center,
            color=arcs_color,
            scale_factor=0.4,
            t_range=[start_angle, end_angle],
        )

        elliptical_arc_inner1 = self.create_elliptical_arc(
            a=a_inner,
            b=b_inner,
            center=center,
            color=arcs_color,
            scale_factor=0.4,
            t_range=[start_angle, end_angle],
        )

        elliptical_arc_inner2 = self.create_elliptical_arc(
            a=a_inner,
            b=b_inner,
            center=center + np.array([-a_inner / 2, 0, 0]),
            color=arcs_color,
            scale_factor=0.4,
            t_range=[start_angle, end_angle],
        )

        # ── 3. 创建输入线 ─────────────────────────────────────────
        input_lines = self.create_input_lines(
            center=center,
            a_inner=a_inner,
            b_inner=b_inner,
            input_line_color=input_line_color,
        )

        # ── 4. 创建输出线 ─────────────────────────────────────────
        output_line = self.create_output_line(
            center=center,
            a_outer=a_outer,
            b_outer=b_outer,
            output_line_color=output_line_color,
        )

        # ── 5. 添加所有组成部分到 VGroup ─────────────────────────
        self.add(
            elliptical_arc_outer,
            elliptical_arc_inner1,
            elliptical_arc_inner2,
            *input_lines,
            output_line,
        )

        # ── 6. 创建填充区域（如果需要） ───────────────────────────
        if fill_color and fill_opacity > 0:
            filled_area = self.create_filled_area(
                arc_outer=elliptical_arc_outer,
                arc_inner=elliptical_arc_inner1,
                fill_color=fill_color,
                fill_opacity=fill_opacity,
            )
            self.add(filled_area)

    def create_elliptical_arc(self, a, b, center, color, scale_factor, t_range):
        """
        创建椭圆弧线。

        参数：
            a (float): 长轴半径。
            b (float): 短轴半径。
            center (np.ndarray): 中心坐标。
            color (Manim 颜色): 弧线颜色。
            scale_factor (float): 缩放因子。
            t_range (list): 参数范围 [起始参数, 结束参数]。

        返回：
            ParametricFunction: 创建的椭圆弧线对象。
        """
        return ParametricFunction(
            lambda t: np.array(
                [a * np.cos(t) + center[0], b * np.sin(t) + center[1], 0]
            ),
            t_range=t_range,
            color=color,
        ).scale(scale_factor, about_point=center)

    def create_input_lines(self, center, a_inner, b_inner, input_line_color):
        """
        创建输入线及其隐藏的点标记。

        参数：
            center (np.ndarray): 异或门的中心坐标。
            a_inner (float): 小椭圆的长轴半径。
            b_inner (float): 小椭圆的短轴半径。
            input_line_color (Manim 颜色): 输入线的颜色。

        返回：
            list: 输入线对象的列表。
        """
        offset = np.array([a_inner / 3.2, 0, 0])  # 定义参考点的偏移
        reference_coord = center + offset

        x_r = reference_coord[0]
        fraction = (x_r**2) / (a_inner**2)

        if fraction <= 1:
            y_dist = np.sqrt(1 - fraction) / b_inner
            # 计算上、下交点坐标
            intersection_up = np.array([x_r, y_dist + center[1], 0])
            intersection_down = np.array([x_r, -y_dist + center[1], 0])

            # 定义输入线的起点（向左延伸）
            input_up_start = intersection_up.copy()
            input_up_start[0] -= 2.0
            input_down_start = intersection_down.copy()
            input_down_start[0] -= 2.0

            # 创建输入线
            input_line_up = Line(
                start=input_up_start, end=intersection_up, color=input_line_color
            )
            input_line_down = Line(
                start=input_down_start, end=intersection_down, color=input_line_color
            )

            # 创建隐藏的点标记
            input_point_up = Dot(point=input_up_start, color=RED).set_opacity(0)
            input_point_down = Dot(point=input_down_start, color=RED).set_opacity(0)

            # 添加输入线和点到 VGroup
            self.add(input_line_up, input_line_down, input_point_up, input_point_down)

            # 存储点的引用以便后续使用
            self.input_point_up = input_point_up
            self.input_point_down = input_point_down

            return [input_line_up, input_line_down]
        else:
            # 如果参考点超出小椭圆的范围，则不绘制输入线
            self.input_point_up = None
            self.input_point_down = None
            return []

    def create_output_line(self, center, a_outer, b_outer, output_line_color):
        """
        创建输出线及其隐藏的点标记。

        参数：
            center (np.ndarray): 异或门的中心坐标。
            a_outer (float): 大椭圆的长轴半径。
            b_outer (float): 大椭圆的短轴半径。
            output_line_color (Manim 颜色): 输出线的颜色。

        返回：
            Line: 创建的输出线对象。
        """
        output_offset = np.array([b_outer, 0, 0])  # 定义输出点的偏移
        output_coord = center + output_offset

        # 定义输出线的终点（向右延伸）
        output_line_end = output_coord.copy()
        output_line_end[0] += 2.0

        # 创建输出线
        output_line = Line(
            start=output_coord, end=output_line_end, color=output_line_color
        )

        # 创建隐藏的输出点标记
        self.output_point = Dot(point=output_line_end, color=RED).set_opacity(0)

        # 添加输出线和点到 VGroup
        self.add(output_line, self.output_point)

        return output_line

    def create_filled_area(self, arc_outer, arc_inner, fill_color, fill_opacity):
        """
        创建填充区域。

        参数：
            arc_outer (ParametricFunction): 外部弧线对象。
            arc_inner (ParametricFunction): 内部弧线对象。
            fill_color (Manim 颜色): 填充颜色。
            fill_opacity (float): 填充透明度。

        返回：
            VMobject: 创建的填充区域对象。
        """
        # 获取弧线的点
        arc_outer_points = arc_outer.get_points()
        arc_inner_points = arc_inner.get_points()[::-1]  # 反向小弧

        # 合并点以形成填充区域
        filled_area = VMobject()
        filled_area.set_points_as_corners(
            np.concatenate([arc_outer_points, arc_inner_points])
        )
        filled_area.set_fill(fill_color, opacity=fill_opacity)
        filled_area.set_stroke(None, 0)  # 去除边框

        return filled_area


class Perceptron(VGroup):  
    def __init__(  
        self,  
        center=ORIGIN,                  # 接受层中心位置  
        n_inputs=4,                     # 默认生成的输入层神经元数量  
        input_line_color="#6ab7ff",     # 输入线颜色  
        input_line_tip_length=0.2,      # 输入线箭头长度  
        input_line_opacity=0.6,         # 输入线不透明度  
        receive_color="#ff9e9e",        # 接受层颜色  
        output_line_color=GREEN,        # 输出线颜色  
        output_line_tip_length=0.3,     # 输出线箭头长度  
        output_line_opacity=0.6,        # 输出线不透明度  
        show_input_circles=True,        # 是否显示输入层圆圈  
        show_input_labels=True,         # 是否显示输入层标签  
        input_opacity=1.0,              # 输入层圆圈的不透明度  
        show_output_layer=True,         # 是否显示输出层圆圈  
        show_output_label=True,         # 是否显示输出层标签  
        output_opacity=1.0,             # 输出层圆圈的不透明度  
        show_input_layer=True,          # 是否显示整个输入层  
        **kwargs  
    ):  
        super().__init__(**kwargs)  

        # 参数验证  
        self._validate_parameters(  
            n_inputs,  
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
            show_input_layer  
        )  

        # 参数存储  
        self.receive_center = center  
        self.input_line_color = input_line_color  
        self.input_line_opacity = input_line_opacity  
        self.input_line_tip_length = input_line_tip_length  
        self.output_line_color = output_line_color  
        self.output_opacity = output_opacity  
        self.output_line_opacity = output_line_opacity  
        self.output_line_tip_length = output_line_tip_length  
        self.receive_color = receive_color  
        self.n_inputs = n_inputs  
        self.show_input_circles = show_input_circles  
        self.show_input_labels = show_input_labels  
        self.input_opacity = input_opacity  
        self.show_output_layer = show_output_layer  
        self.show_output_label = show_output_label  
        self.show_input_layer = show_input_layer  
    
        # 计算输入层中心位置  
        self.input_layer_centers = self._calculate_input_layer_centers()  

        # 创建感知机组件  
        self.receive_layer = self._create_receive_layer()  
        self.output_layer, self.output_label = self._create_output_layer()  
        self.input_circles, self.input_labels = self._create_input_layer()  
        self.connections = self._create_connections()  
        self.output_arrow = self._create_output_arrow()  
        
        # 将所有组件添加到 VGroup  
        self.add(  
            self.receive_layer,  
            self.output_layer,  
            self.output_label,  
            self.input_circles,  
            self.input_labels,  
            self.connections,  
            self.output_arrow  
        )  

        # 根据 show_input_layer 控制整个输入层的显示状态  
        if not self.show_input_layer:  
            self.disable_input_layer()  

    def _validate_parameters(  
        self,  
        n_inputs,  
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
        show_input_layer  
    ):  
        """验证初始化参数的合法性"""  
        # 检查 n_inputs 是否为正整数  
        if not isinstance(n_inputs, int) or n_inputs <= 0:  
            raise ValueError("n_inputs 必须是一个正整数。")  

        # 检查 input_line_tip_length 是否为正数  
        if not (isinstance(input_line_tip_length, (int, float)) and input_line_tip_length > 0):  
            raise ValueError("input_line_tip_length 必须是一个正数。")  

        # 检查 input_line_opacity 是否在 [0.0, 1.0] 范围内  
        if not (isinstance(input_line_opacity, (int, float)) and 0.0 <= input_line_opacity <= 1.0):  
            raise ValueError("input_line_opacity 必须在 [0.0, 1.0] 范围内。")  

        # 检查 output_line_tip_length 是否为正数  
        if not (isinstance(output_line_tip_length, (int, float)) and output_line_tip_length > 0):  
            raise ValueError("output_line_tip_length 必须是一个正数。")  

        # 检查 output_line_opacity 是否在 [0.0, 1.0] 范围内  
        if not (isinstance(output_line_opacity, (int, float)) and 0.0 <= output_line_opacity <= 1.0):  
            raise ValueError("output_line_opacity 必须在 [0.0, 1.0] 范围内。")  

        # 检查 input_opacity 是否在 [0.0, 1.0] 范围内  
        if not (isinstance(input_opacity, (int, float)) and 0.0 <= input_opacity <= 1.0):  
            raise ValueError("input_opacity 必须在 [0.0, 1.0] 范围内。")  

        # 检查 output_opacity 是否在 [0.0, 1.0] 范围内  
        if not (isinstance(output_opacity, (int, float)) and 0.0 <= output_opacity <= 1.0):  
            raise ValueError("output_opacity 必须在 [0.0, 1.0] 范围内。")  

        # 检查 show_input_circles 是否为布尔值  
        if not isinstance(show_input_circles, bool):  
            raise TypeError("show_input_circles 必须是布尔值。")  

        # 检查 show_input_labels 是否为布尔值  
        if not isinstance(show_input_labels, bool):  
            raise TypeError("show_input_labels 必须是布尔值。")  

        # 检查 show_output_layer 是否为布尔值  
        if not isinstance(show_output_layer, bool):  
            raise TypeError("show_output_layer 必须是布尔值。")  

        # 检查 show_output_label 是否为布尔值  
        if not isinstance(show_output_label, bool):  
            raise TypeError("show_output_label 必须是布尔值。")  

        # 检查 show_input_layer 是否为布尔值  
        if not isinstance(show_input_layer, bool):  
            raise TypeError("show_input_layer 必须是布尔值。")  
            
    def _calculate_input_layer_centers(self):  
        """  
        内部计算输入层神经元的中心位置，确保圆圈、标签和箭头互不干扰。  
        输入层的神经元将在垂直方向上对称且均匀分布，基于接受层的中心位置。  
        """  
        # 设定接受层到输入层的水平距离  
        horizontal_distance = 3  

        # 设定输入层神经元的垂直间距  
        vertical_spacing = 0.7  

        # 计算输入层神经元的总数量需要显示的神经元数  
        # 当 n_inputs > 3 时，仅显示前两个和最后一个，再加上省略号  
        if self.n_inputs > 3:  
            display_n = 4  # 前两个、一个省略号、最后一个  
        else:  
            display_n = self.n_inputs  

        # 计算显示神经元的总高度  
        total_height = (display_n - 1) * vertical_spacing  

        # 输入层起始 y 坐标，使其垂直居中对齐接受层  
        start_y = self.receive_center[1] + total_height / 2  

        # 生成中心位置列表  
        centers = [  
            self.receive_center + LEFT * horizontal_distance + np.array([0, start_y - i * vertical_spacing, 0])  
            for i in range(display_n)  
        ]  
        self.input_layer_centers = centers  

        return centers  

    def _create_receive_layer(self):  
        """创建接受层"""  
        receive_layer = Circle(  
            radius=0.5,  
            fill_opacity=1,  
            color=self.receive_color  
        )  
        receive_layer.move_to(self.receive_center)  
        return receive_layer  

    def _create_output_layer(self):  
        """创建输出层及其标签"""  
        if not self.show_output_layer:  
            self.output_layer = VGroup()  
            self.output_label = VGroup()  
            return self.output_layer, self.output_label  
        
        self.output_layer = Circle(  
            radius=0.2,  
            fill_opacity=self.output_opacity,  
            color=self.output_line_color  
        )  
        self.output_layer.next_to(self.receive_layer, RIGHT, buff=2.5)  

        if self.show_output_label:  
            self.output_label = MathTex("y").scale(0.8).move_to(self.output_layer.get_center())  
        else:  
            self.output_label = VGroup()  # 空对象  

        return self.output_layer, self.output_label  

    def _create_input_layer(self):  
        """  
        创建输入层的神经元和标签，使用内部计算的 input_layer_centers 定位每个圆圈和标签。  
        根据参数 show_input_circles 和 show_input_labels 决定是否显示圆圈和标签。  
        当输入层神经元数量超过3时，仅显示前两个和最后一个，并在中间显示省略号。  
        """  
        input_circles = VGroup()  
        input_labels = VGroup()  
        input_circles_opacity = 1 if self.show_input_circles else 0
        input_labels_opacity = 1 if self.show_input_labels else 0

        for i, center in enumerate(self.input_layer_centers):  
            if self.n_inputs > 3 and i == 2:  
                # 在第三个位置添加省略号  
                ellipsis = Tex("...").scale(0.8)  
                ellipsis.move_to(center)  
                input_labels.add(ellipsis)  # 将省略号作为标签的一部分  
                continue  

            # 创建圆圈  
            
            circle = Circle(  
                radius=0.2,  
                fill_opacity=self.input_opacity,  
                color=self.input_line_color  
            )  
            circle.move_to(center)  
            circle.set_opacity(input_circles_opacity)
            input_circles.add(circle)  

            # 创建标签  
            
            label_text = f"x_{{{i+1}}}" if not (self.n_inputs > 3 and i == (self.n_inputs - 1)) else f"x_{{{self.n_inputs}}}"  
            label = MathTex(label_text).scale(0.8)  
            label.shift(UP * 0.4).move_to(center)  
            label.set_opacity(input_labels_opacity)
            input_labels.add(label)  

        return input_circles, input_labels  

    def _create_connections(self):  
        """  
        创建从输入层到接收层的连接箭头，基于内部计算的 input_layer_centers。  
        """  
        connections = VGroup()  
        for i, center in enumerate(self.input_layer_centers):  
            # 定义箭头开始点：输入层圆圈的右边缘  
            start_point = center + RIGHT * 0.2  

            # 定义箭头结束点：接受层的左边缘  
            end_point = self.receive_layer.get_left() + RIGHT * 0.2  

            connection = Arrow(  
                start=start_point,  
                end=end_point,  
                color=self.input_line_color,  
                stroke_opacity=self.input_line_opacity,  
                tip_length=self.input_line_tip_length,  
                buff=0.2  
            )  
            connections.add(connection)  
        return connections  

    def _create_output_arrow(self):  
        """  
        创建从接收层到输出层的箭头，基于接收层和输出层的位置。  
        """  
        if not self.show_output_layer:  
            return VGroup()  

        output_arrow = Arrow(  
            start=self.receive_layer.get_right() + LEFT * 0.2,  
            end=self.output_layer.get_left() + RIGHT * 0.2,  
            color=self.output_line_color,  
            stroke_opacity=self.output_line_opacity,  
            tip_length=self.output_line_tip_length,  
            buff=0.4  
        )  
        return output_arrow  

    def get_positions(self):  
        """  
        返回各组件的当前位置，确保在整体变换后位置是最新的。  
        """  
        positions = {  
            "receive_layer": self.receive_layer.get_center(),  
            "output_layer": self.output_layer.get_center() if self.show_output_layer else None,  
            "output_label": self.output_label.get_center() if self.show_output_label else None,  
            "output_arrow_start": self.output_arrow.get_start() if self.show_output_layer else None,  
            "output_arrow_end": self.output_arrow.get_end() if self.show_output_layer else None,  
        }  

        if self.show_input_labels or self.show_input_circles:  
            # 从 input_layer 中提取位置  
            input_positions = []  
            for obj in self.input_layer:  
                if isinstance(obj, Tex) or isinstance(obj, MathTex):  
                    input_positions.append(obj.get_center())  
                elif isinstance(obj, VGroup) and len(obj) > 0:  
                    input_positions.append(obj.get_center())  
            positions["input_layer"] = input_positions  
        else:  
            # 直接返回预先计算的中心位置  
            positions["input_layer"] = self.input_layer_centers.copy()  

        return positions

    def add_position_markers(self):  
        """为各组件添加红点位置标记"""  
        markers = VGroup()  
        
        # 接受层位置标记  
        receive_marker = Dot(point=self.receive_layer.get_center(), color=RED)  
        receive_label = Text("接受层", font_size=24).next_to(receive_marker, UP)  
        markers.add(VGroup(receive_marker, receive_label))  
        
        # 输入层位置标记  
        for idx, pos in enumerate(self.input_layer_centers, start=1):  
            if self.n_inputs > 3 and idx == 3:  
                input_marker = Dot(point=pos, color=RED)  
                input_label = Text("...", font_size=24).next_to(input_marker, UP)  
            else:  
                input_marker = Dot(point=pos, color=RED)  
                input_label = Text(f"输入层_{idx}", font_size=24).next_to(input_marker, UP)  
            markers.add(VGroup(input_marker, input_label))  
        
        # 输出层位置标记  
        if self.show_output_layer:  
            output_marker = Dot(point=self.output_layer.get_center(), color=RED)  
            output_label = Text("输出层", font_size=24).next_to(output_marker, UP)  
            markers.add(VGroup(output_marker, output_label))  
        
        self.add(markers)  
        return markers  
    
    def enable_input_labels(self):  
        """显示输入层标签"""  
        if self.show_input_labels and hasattr(self, 'input_labels'):  
            self.input_labels.set_opacity(1)  

    def disable_input_labels(self):  
        """隐藏输入层标签"""  
        if hasattr(self, 'input_labels'):  
            self.input_labels.set_opacity(0)  

    def enable_output_label(self):  
        """显示输出层标签"""  
        if self.show_output_label and hasattr(self, 'output_label'):  
            self.output_label.set_opacity(1)  

    def disable_output_label(self):  
        """隐藏输出层标签"""  
        if hasattr(self, 'output_label'):  
            self.output_label.set_opacity(0)  

    def enable_input_connections(self):  
        """显示输入层连接"""  
        self.connections.set_opacity(1)  

    def disable_input_connections(self):  
        """隐藏输入层连接"""  
        self.connections.set_opacity(0)  

    def enable_output_circle(self):  
        """显示输出层圆圈"""  
        if self.show_output_layer and hasattr(self, 'output_layer'):  
            self.output_layer.set_opacity(1)  

    def disable_output_circle(self):  
        """隐藏输出层圆圈"""  
        if self.show_output_layer and hasattr(self, 'output_layer'):  
            self.output_layer.set_opacity(0)  

    def enable_input_circles(self):  
        """显示输入层圆圈"""  
        if self.show_input_layer and hasattr(self, 'input_circles') and hasattr(self, 'input_labels'):
            self.input_circles.set_opacity(1)  

    def disable_input_circles(self):  
        """隐藏输入层圆圈"""  
        if self.show_input_layer and hasattr(self, 'input_circles') and hasattr(self, 'input_labels'):  
            self.input_circles.set_opacity(0)  

    def enable_input_layer(self):  
        """显示输入层"""  
        if self.show_input_layer and hasattr(self, 'input_circles') and hasattr(self, 'input_labels'):  
            self.input_circles.set_opacity(1)  
            self.input_labels.set_opacity(1)  
    
    def disable_input_layer(self):  
        """隐藏输入层"""  
        if self.show_input_layer and hasattr(self, 'input_circles') and hasattr(self, 'input_labels'):  
            self.input_circles.set_opacity(0)  
            self.input_labels.set_opacity(0)  