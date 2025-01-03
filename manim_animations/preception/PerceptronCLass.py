from manim import *


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
        **kwargs
    ):
        super().__init__(**kwargs)

        self.receive_center = center

        self.input_line_color = input_line_color  
        self.input_line_opacity = input_line_opacity  
        self.input_line_tip_length = input_line_tip_length  
        
        self.output_line_color = output_line_color  
        self.output_line_opacity = output_line_opacity  
        self.output_line_tip_length = output_line_tip_length  
        
        self.receive_color = receive_color  
        self.n_inputs = n_inputs  

       # 创建感知机组件  
        self.receive_layer = self._create_receive_layer()  
        self.output_layer, self.output_label = self._create_output_layer()  
        self.input_layer, self.input_labels = self._create_input_layer()  
        self.connections = self._create_connections()  
        self.output_arrow = self._create_output_arrow()  
        
        # 将所有组件添加到 VGroup  
        self.add(  
            self.receive_layer,  
            # self.output_layer,  
            self.output_label,  
            # self.input_layer,  
            self.input_labels,  
            self.connections,  
            self.output_arrow  
        )  


    def _create_receive_layer(self):
        """创建接受层"""
        receive_layer = Circle(radius=0.5, fill_opacity=1, color=self.receive_color)
        receive_layer.move_to(self.receive_center)
        return receive_layer
    
    def _create_output_layer(self):
        """创建输出层及其标签"""
        output_layer = Circle(radius=0.2)
        output_layer.next_to(self.receive_layer, RIGHT, buff=2.5)
        output_label = MathTex("y").scale(0.8).move_to(output_layer.get_center())
        return output_layer, output_label

    def _create_input_layer(self):  
        """创建输入层及其标签，支持省略号显示"""  
        if self.n_inputs <= 3:
            # 输入数量不超过3，全部显示 
            input_circles = VGroup(*[
                Circle(radius=0.2, fill_opacity=1, color=self.input_line_color)  
                for _ in range(self.n_inputs)  
            ]) 
            input_circles.arrange(DOWN, buff=0.7)
            input_circles.next_to(self.receive_center, LEFT, buff=2.5)

            input_labels = VGroup(*[
                MathTex(f"x_{{{i+1}}}").scale(0.8).move_to(neuron.get_center())  
                for i, neuron in enumerate(input_circles)
            ])

        else:  
            # 输入数量超过3，显示x1, x2, ..., xn  
            input_circles = VGroup(  
                Circle(radius=0.2, fill_opacity=1, color=self.input_line_color),  
                Circle(radius=0.2, fill_opacity=1, color=self.input_line_color),  
                Tex("...").scale(0.8),  
                Circle(radius=0.2, fill_opacity=1, color=self.input_line_color),  
            )  
            input_circles.arrange(DOWN, buff=0.7)  
            input_circles.next_to(self.receive_center, LEFT, buff=2.5)
            
            input_labels = VGroup(  
                MathTex("x_{1}").scale(0.8).move_to(input_circles[0].get_center()),  
                MathTex("x_{2}").scale(0.8).move_to(input_circles[1].get_center()),  
                MathTex("...").scale(0.8).move_to(input_circles[2].get_center()),  
                MathTex(f"x_{{{self.n_inputs}}}").scale(0.8).move_to(input_circles[3].get_center()),  
            )  

        input_circles.next_to(self.receive_layer, LEFT, buff=2.5)  
        return input_circles, input_labels  
    
    def _create_connections(self):  
        """创建输入层到接收层的连接箭头"""  
        connections = VGroup()  
        for obj in self.input_layer:  
            if isinstance(obj, Circle):  
                connection = Arrow(  
                    start=obj.get_right(),  
                    end=self.receive_layer.get_center(),  
                    color=self.input_line_color,  
                    stroke_opacity=self.input_line_opacity,  
                    tip_length=self.input_line_tip_length,  
                    buff=0.6  
                )  
                connections.add(connection)  
        return connections  
    
    def _create_output_arrow(self):  
        """创建输出箭头"""  
        output_arrow = Arrow(  
            start=self.receive_layer.get_center(),  
            end=self.output_layer.get_center(),  
            color=self.output_line_color,  
            stroke_opacity=self.output_line_opacity,  
            tip_length=self.output_line_tip_length,  
            buff=0.6  
        )  
        return output_arrow 
