from manim import *  

class XorGate(VGroup):  
    def __init__(  
        self,  
        center=ORIGIN,              # 异或门的中心坐标  
        arcs_color=RED,             # 弧线颜色  
        input_line_color=RED,       # 输入线的颜色  
        output_line_color=RED,      # 输出线的颜色  
        fill_color=None,            # 填充颜色（若需要填充）  
        fill_opacity=0.5,           # 填充透明度  
        **kwargs  
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
        a_outer, b_outer = 5, 2          # 大椭圆的长轴和短轴半径  
        a_inner, b_inner = 1.5, 2        # 小椭圆的长轴和短轴半径  
        start_angle, end_angle = -PI / 2, PI / 2  # 弧线的起始和结束角度  

        # ── 2. 创建主体三条弧线 ───────────────────────────────────  
        elliptical_arc_outer = self.create_elliptical_arc(  
            a=a_outer,  
            b=b_outer,  
            center=center,  
            color=arcs_color,  
            scale_factor=0.4,  
            t_range=[start_angle, end_angle]  
        )  

        elliptical_arc_inner1 = self.create_elliptical_arc(  
            a=a_inner,  
            b=b_inner,  
            center=center,  
            color=arcs_color,  
            scale_factor=0.4,  
            t_range=[start_angle, end_angle]  
        )  

        elliptical_arc_inner2 = self.create_elliptical_arc(  
            a=a_inner,  
            b=b_inner,  
            center=center + np.array([-a_inner/2, 0, 0]),  
            color=arcs_color,  
            scale_factor=0.4,  
            t_range=[start_angle, end_angle]  
        )  

        # ── 3. 创建输入线 ─────────────────────────────────────────  
        input_lines = self.create_input_lines(  
            center=center,  
            a_inner=a_inner,  
            b_inner=b_inner,  
            input_line_color=input_line_color  
        )  

        # ── 4. 创建输出线 ─────────────────────────────────────────  
        output_line = self.create_output_line(  
            center=center,  
            a_outer=a_outer,  
            b_outer=b_outer,  
            output_line_color=output_line_color  
        )  

        # ── 5. 添加所有组成部分到 VGroup ─────────────────────────  
        self.add(  
            elliptical_arc_outer,  
            elliptical_arc_inner1,  
            elliptical_arc_inner2,  
            *input_lines,  
            output_line  
        )  

        # ── 6. 创建填充区域（如果需要） ───────────────────────────  
        if fill_color and fill_opacity > 0:  
            filled_area = self.create_filled_area(  
                arc_outer=elliptical_arc_outer,  
                arc_inner=elliptical_arc_inner1,  
                fill_color=fill_color,  
                fill_opacity=fill_opacity  
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
            lambda t: np.array([  
                a * np.cos(t) + center[0],  
                b * np.sin(t) + center[1],  
                0  
            ]),  
            t_range=t_range,  
            color=color  
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
        fraction = (x_r ** 2) / (a_inner ** 2)  

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
                start=input_up_start,  
                end=intersection_up,  
                color=input_line_color  
            )  
            input_line_down = Line(  
                start=input_down_start,  
                end=intersection_down,  
                color=input_line_color  
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
            start=output_coord,  
            end=output_line_end,  
            color=output_line_color  
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
        filled_area.set_points_as_corners(np.concatenate([arc_outer_points, arc_inner_points]))  
        filled_area.set_fill(fill_color, opacity=fill_opacity)  
        filled_area.set_stroke(None, 0)  # 去除边框  

        return filled_area  