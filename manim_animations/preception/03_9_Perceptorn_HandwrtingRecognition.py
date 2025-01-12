import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
from manim import *

# matplotlib 设置
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


# 加载 MNIST 数据集  
mnist = tf.keras.datasets.mnist  
(image_train, label_train), (image_test, label_test) = mnist.load_data()

# # 选择一张图片（例如第一张训练图片）  
# index_1 = 0  
# image_1 = x_train[index_1]  
# label_1 = y_train[index_1]  

# # 打印标签  
# print('训练集第一张图片的标签:')
# print(f'标签: {label_1}')  

# # 打印像素值  
# print('像素值:')  
# print(image_1)  

# # 可视化图片  
# plt.imshow(image_1, cmap='gray')  
# plt.title(f'标签: {label_1}')  
# plt.show()

class HandwritingVisualization(Scene):
    def construct(self):
        # 加载 MNIST 数据集并选择图片  
        image_1, label_1 = self.load_mnist_image(label_target=1, index=0)  
        image_0, label_0 = self.load_mnist_image(label_target=0, index=0)

        print(f'灰度值: {image_1}')
        # 创建像素组  
        pixel_group_1 = self.create_pixel_group(image_1)  
        pixel_group_0 = self.create_pixel_group(image_0)
        
        # 缩放像素组  
        pixel_group_1.scale(0.8).to_edge(LEFT, buff=1)
        pixel_group_0.scale(0.8).to_edge(RIGHT, buff=1)
        
        # 添加像素组到场景  
        self.play(Create(pixel_group_1), Create(pixel_group_0))  
        self.wait(2)  
        
        title = Text("如何用感知机识别手写数字").to_edge(UP, buff=1)
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        self.wait(1)
        self.play(FadeOut(pixel_group_0), pixel_group_1.animate.shift(RIGHT))
        self.wait(1)

        # 定义图片的长和宽  
        image_dimensions = {"width": 28, "height": 28}
        # 添加表示宽度的下方大括号  
        width_brace = Brace(pixel_group_1, DOWN)  
        width_label = width_brace.get_tex(f"W={image_dimensions['width']}")  
        width_group = VGroup(width_brace, width_label)  
        self.play(FadeIn(width_brace), Write(width_label))  
        self.wait(1)  
        
        # 添加表示高度的左侧大括号  
        height_brace = Brace(pixel_group_1, LEFT)  
        height_label = height_brace.get_tex(f"H={image_dimensions['height']}")  
        height_group = VGroup(height_brace, height_label)  
        self.play(FadeIn(height_brace), Write(height_label))  
        self.wait(1)  
        
        # 添加外框和替换像素为灰度值文本同时进行  
        frame = SurroundingRectangle(pixel_group_1, buff=0.1, color=BLUE)  
        text_group = VGroup()  
        for i, square in enumerate(pixel_group_1):  
            if i != 8 * 28 + 17:  # 排除选中的像素  
                row = i // 28  
                col = i % 28  
                gray_value = image_1[row, col] /  255
                text = Text(str(gray_value), font_size=8).move_to(square.get_center())  
                text_group.add(text)  
        
        # 为每个非选中像素创建同时动画：隐藏方块并显示文本  
        replacement_animations = []  
        for i, square in enumerate(pixel_group_1):  
            if i != 8 * 28 + 17:  
                replacement_animations.append(FadeOut(square))  
        replacement_animations.append(Create(frame))  
        replacement_animations.append(Write(text_group))  
        
        # 同时运行所有替换动画  
        self.play(*replacement_animations)  
        self.wait(1)  

        # 选择一个特定的像素点  
        selected_index = 8 * 28 + 17  # 第8行第17列（索引从0开始）  
        selected_pixel = pixel_group_1[selected_index]  
        selected_pixel_gray = image_1[8, 17]  
        
        # 创建显示灰度值的文本  
        gray_text = DecimalNumber(selected_pixel_gray, num_decimal_places=0, include_sign=False)  
        gray_text.set_color(RED)  
        gray_text.scale(0.5)  
        gray_text.next_to(selected_pixel, UP)  
        
        # 添加灰度值文本到场景但初始透明  
        self.add(gray_text)  
        gray_text.set_opacity(0)  
        
        # 动画：移动选中的像素并显示灰度值  
        self.play(  
            selected_pixel.animate.shift(RIGHT * 3).scale(2.5),  
            FadeIn(gray_text)  
        )  
        self.wait(1)  
        
        # 动画：灰度值先趋于1，再趋于0  
        # 使用 ValueTracker 控制动画进度  
        # 第一部分：趋于1  
        tracker_up = ValueTracker(0)  
        
        # 所有的变化都会基于 tracker_up 的值  
        def update_pixel_up(mob):  
            new_opacity = self.lerp(selected_pixel_gray / 255, 1.0, tracker_up.get_value())  
            mob.set_fill(opacity=new_opacity)  
            gray_text.set_value(int(new_opacity * 255))  
        
        selected_pixel.add_updater(update_pixel_up)  
        gray_text.add_updater(lambda mob: mob.set_value(int(selected_pixel.get_fill_opacity() * 255)))  
        
        # 执行趋于1的动画  
        self.play(tracker_up.animate.set_value(1), run_time=2, rate_func=linear)  
        self.wait(1)  
        
        # 第二部分：趋于0  
        tracker_down = ValueTracker(1)  
        
        def update_pixel_down(mob):  
            new_opacity = self.lerp(1.0, 0.0, tracker_down.get_value())  
            mob.set_fill(opacity=new_opacity)  
            gray_text.set_value(int(new_opacity * 255))  
        
        # 更新updater为趋于0  
        selected_pixel.remove_updater(update_pixel_up)  
        selected_pixel.add_updater(update_pixel_down)  
        gray_text.remove_updater(lambda mob: mob.set_value(int(selected_pixel.get_fill_opacity() * 255)))  
        gray_text.add_updater(lambda mob: mob.set_value(int(selected_pixel.get_fill_opacity() * 255)))  
        
        # 执行趋于0的动画  
        self.play(tracker_down.animate.set_value(1), run_time=2, rate_func=linear)  
        self.wait(1)  
        
        # 移除 updaters  
        selected_pixel.remove_updater(update_pixel_down)  
        gray_text.remove_updater(lambda mob: mob.set_value(int(selected_pixel.get_fill_opacity() * 255)))  
        
        # 可选：移除灰度值文本  
        self.play(FadeOut(gray_text))  
        self.wait(1)
        
        
    
    def load_mnist_image(self, label_target=1, index=0):  
        """  
        加载 MNIST 数据集并选择指定标签的图片。  

        参数:  
            label_target (int): 目标标签（0-9）。  
            index (int): 选择的图片在目标标签中的索引。  

        返回:  
            image (np.ndarray): 选中的图片数组。  
            label (int): 图片的标签。  
        """  
        # 加载 MNIST 数据集  
        mnist = tf.keras.datasets.mnist  
        (x_train, y_train), (x_test, y_test) = mnist.load_data()  
        images = np.concatenate((x_train, x_test), axis=0)  
        labels = np.concatenate((y_train, y_test), axis=0)  
        
        # 找到所有目标标签的索引  
        target_indices = np.where(labels == label_target)[0]  
        
        if index >= len(target_indices):  
            raise IndexError(f"标签为 {label_target} 的图片不足 {index + 1} 张。")  
        
        selected_index = target_indices[index]  
        image = images[selected_index]  
        label = labels[selected_index]  
        
        print(f'选择的图片索引: {selected_index}')  
        print(f'标签: {label}')  
        
        return image, label  
    
    def create_pixel_group(self, image, pixel_size=0.2, spacing=0.05):  
        """  
        根据输入的图片数组创建一个像素组。  

        参数:  
            image (np.ndarray): 28x28 的灰度图片数组。  
            pixel_size (float): 每个像素方块的边长。  
            spacing (float): 方块之间的间距。  

        返回:  
            pixel_group (VGroup): 包含所有像素方块的组。  
        """  
        pixel_group = VGroup()  
        
        for i in range(28):  
            for j in range(28):  
                # 获取像素值（0-255），转换为灰度颜色并反转  
                pixel_value = image[i, j]  
                gray = pixel_value / 255  # 0为黑色，1为白色  
                
                # 创建一个方块代表像素  
                square = Square(  
                    side_length=pixel_size,  
                    fill_color=WHITE,  
                    fill_opacity=gray,  # 根据灰度设置不透明度  
                    stroke_width=0.2,  
                    stroke_color=WHITE  
                )  
                square.shift(  
                    (j - 14) * (pixel_size + spacing) * RIGHT +  
                    (14 - i) * (pixel_size + spacing) * UP  
                )  
                pixel_group.add(square)  
        
        return pixel_group  
    
    @staticmethod  
    def lerp(start, end, alpha):  
        """线性插值函数"""  
        return start + (end - start) * alpha  

class Test(Scene):
    def construct(self):
        # 初始化灰度值，从0.5（灰色）开始  
        grayscale_tracker = ValueTracker(0.5)  

        # 创建一个方块，初始颜色为灰色  
        square = Square(side_length=2, fill_color=GRAY, fill_opacity=1)  
        square.to_edge(LEFT)  # 将方块移动到画面的左侧  

        # 创建一个显示灰度值的文本  
        grayscale_text = DecimalNumber(  
            grayscale_tracker.get_value(),  
            num_decimal_places=2,  
            include_sign=False  
        )  
        grayscale_text.add_updater(lambda d: d.set_value(grayscale_tracker.get_value()))  
        grayscale_text.move_to(square.get_center())  # 将文本放置在方块中心  
        
        # 添加方块和灰度文本到场景中  
        self.add(square, grayscale_text)  

        # 添加文本颜色更新器，确保对比度  
        grayscale_text.add_updater(  
            lambda d: d.set_color(  
                WHITE if grayscale_tracker.get_value() < 0.5 else BLACK  
            )  
        )  

        # 颜色由灰色 (0.5) 变为白色 (1.0)  
        self.play(  
            grayscale_tracker.animate.set_value(1.0),  
            square.animate.set_fill(WHITE, opacity=1),  
            run_time=2  
        )  

        # 颜色由白色 (1.0) 变为黑色 (0.0)  
        self.play(  
            grayscale_tracker.animate.set_value(0.0),  
            square.animate.set_fill(BLACK, opacity=1),  
            run_time=2  
        )  

        # 等待1秒以保持最终画面  
        self.wait(1)  


import tensorflow as tf  
import matplotlib.pyplot as plt  
import numpy as np  
from manim import *  

# matplotlib 设置  
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签  

# 加载 MNIST 数据集  
mnist = tf.keras.datasets.mnist  
(x_train, y_train), (x_test, y_test) = mnist.load_data()  

class HandwritingVisualization(Scene):  
    def construct(self):  
        # 加载 MNIST 数据集并选择图片  
        image_1, label_1 = self.load_mnist_image(label_target=1, index=0)  
        image_0, label_0 = self.load_mnist_image(label_target=0, index=0)  

        print(f'灰度值: {image_1}')  
        # 创建像素组  
        pixel_group_1 = self.create_pixel_group(image_1)  
        pixel_group_0 = self.create_pixel_group(image_0)  
        
        # 缩放像素组  
        pixel_group_1.scale(0.8).to_edge(LEFT, buff=1)  
        pixel_group_0.scale(0.8).to_edge(RIGHT, buff=1)  
        
        # 添加像素组到场景  
        self.play(Create(pixel_group_1), Create(pixel_group_0))  
        self.wait(2)  
        
        title = Text("如何用感知机识别手写数字").to_edge(UP, buff=1)  
        self.play(Write(title))  
        self.wait(1)  
        self.play(FadeOut(title))  
        self.wait(1)  
        self.play(FadeOut(pixel_group_0), pixel_group_1.animate.shift(RIGHT*2))  
        self.wait(1)  

        # 定义图片的长和宽  
        image_dimensions = {"width": 28, "height": 28}  
        # 添加表示宽度的下方大括号  
        width_brace = Brace(pixel_group_1, DOWN)  
        width_label = width_brace.get_tex(f"W={image_dimensions['width']}")  
        width_group = VGroup(width_brace, width_label)  
        self.play(FadeIn(width_brace), Write(width_label))  
        self.wait(1)  
        
        # 添加表示高度的左侧大括号  
        height_brace = Brace(pixel_group_1, LEFT)  
        height_label = height_brace.get_tex(f"H={image_dimensions['height']}")  
        height_group = VGroup(height_brace, height_label)  
        self.play(FadeIn(height_brace), Write(height_label))  
        self.wait(1)  
        
        # 添加外框和替换像素为灰度值文本同时进行  
        text_group = VGroup()  
        for i, square in enumerate(pixel_group_1):  
            row = i // 28  
            col = i % 28  
            gray_value = image_1[row, col] / 255  
            
            # 格式化灰度值为一位小数  
            formatted_gray_value = f"{gray_value:.1f}"  
            
            text = Text(formatted_gray_value, font_size=8).move_to(square.get_center())  
            
            if gray_value < 0.5:  
                text.set_color(WHITE)  
            else:  
                text.set_color(BLACK)  
            
            text_group.add(text)  
                
        self.play(FadeIn(text_group))  
        self.wait(1)  
        
        # 选择一个特定的像素点演示灰度变化  
        select_row = 8  
        select_col = 17  
        selected_index = select_row * 28 + select_col   
        selected_pixel = pixel_group_1[selected_index]  # 选择特定的像素点  
        selected_pixel_gray_value = ValueTracker(image_1[select_row, select_col] / 255)  # 创建一个值跟踪器来跟踪灰度值  
        selected_pixel_text = DecimalNumber(  
            selected_pixel_gray_value.get_value(),   
            num_decimal_places=1,  
            include_sign=False,  
            font_size=8  
        )   # 创建一个数字对象来显示灰度值  

        selected_pixel_text.add_updater(lambda m: m.set_value(selected_pixel_gray_value.get_value())) # 更新灰度值  
        selected_pixel_text.move_to(selected_pixel.get_center()+RIGHT*4+DOWN*1) # 将文本移动到选中的像素点上  
        selected_pixel_text.scale(2.5) # 将文本放大2.5倍与方块大小一致  
        self.play(selected_pixel.animate.shift(RIGHT*4 + DOWN*1).scale(2.5))  

        # 用新的文字来取代这个像素原本的文字便于后续动画演示  
        self.add(selected_pixel_text) # 添加文本到场景中  
        self.remove(text_group[selected_index]) # 移除原来的文本  
        

        # 添加文本颜色更新器，确保对比度  
        selected_pixel_text.add_updater(  
            lambda d: d.set_color(  
                WHITE if selected_pixel_gray_value.get_value() < 0.5 else BLACK  
            )  
        )  

        # 颜色由初始颜色（灰色）变为白色（1.0）  
        self.play(  
            selected_pixel_gray_value.animate.set_value(1.0),  
            selected_pixel.animate.set_fill(WHITE, opacity=1),  # 仅更改填充颜色  
            run_time=2,  
        )  

        # 颜色由白色（1.0）变为黑色（0.0）  
        self.play(  
            selected_pixel_gray_value.animate.set_value(0.0),  
            selected_pixel.animate.set_fill(BLACK, opacity=1),  # 仅更改填充颜色  
            run_time=2  
        )  

        self.wait(1)  
        
        # 新增部分：将除第一列外的每一列依次移动到第一列下面  
        # 获取每一列的像素  
        columns = VGroup(*[  
            VGroup(*[  
                pixel for pixel in pixel_group_1 if np.isclose(pixel.get_center()[0], col.get_center()[0], atol=1e-4)  
            ]) for col in pixel_group_1.submobjects[::28]  
        ])  

        # 排除第一列  
        columns_to_move = columns[1:]  

        # 设置每列移动的目标位置  
        for i, column in enumerate(columns_to_move):  
            # 计算每列向下移动的距离  
            shift_distance = DOWN * (i + 1) * (0.2 + 0.05) * 28  # pixel_size + spacing 乘以行数  
            self.play(column.animate.shift(shift_distance), run_time=0.5)  
            self.wait(0.2)  # 每列移动后稍作停顿  

        self.wait(1)  

        # 简化显示：保留前三个像素和最后一个像素，其余用省略号代替  
        # 获取移动后的总像素列  
        total_pixels = columns[0].submobjects  # 第一列，共784个像素  

        # 保留前三个和最后一个像素  
        simplified_pixels = VGroup(  
            total_pixels[:3],  # 前三个像素  
            Text("...", font_size=12).next_to(total_pixels[2], DOWN, buff=0.1),  # 省略号  
            total_pixels[-1].copy()  # 最后一个像素的拷贝  
        ).arrange(DOWN, aligned_edge=LEFT)  

        # 隐藏中间的像素  
        for pixel in total_pixels[3:-1]:  
            self.play(FadeOut(pixel), run_time=0.01)  # 快速淡出中间像素  

        # 添加简化后的像素和省略号  
        self.play(FadeIn(simplified_pixels), run_time=1)  
        self.wait(2)  

    def load_mnist_image(self, label_target=1, index=0):  
        """  
        加载 MNIST 数据集并选择指定标签的图片。  

        参数:  
            label_target (int): 目标标签（0-9）。  
            index (int): 选择的图片在目标标签中的索引。  

        返回:  
            image (np.ndarray): 选中的图片数组。  
            label (int): 图片的标签。  
        """  
        # 加载 MNIST 数据集  
        mnist = tf.keras.datasets.mnist  
        (x_train, y_train), (x_test, y_test) = mnist.load_data()  
        images = np.concatenate((x_train, x_test), axis=0)  
        labels = np.concatenate((y_train, y_test), axis=0)  
        
        # 找到所有目标标签的索引  
        target_indices = np.where(labels == label_target)[0]  
        
        if index >= len(target_indices):  
            raise IndexError(f"标签为 {label_target} 的图片不足 {index + 1} 张。")  
        
        selected_index = target_indices[index]  
        image = images[selected_index]  
        label = labels[selected_index]  
        
        print(f'选择的图片索引: {selected_index}')  
        print(f'标签: {label}')  
        
        return image, label  
    
    def create_pixel_group(self, image, pixel_size=0.2, spacing=0.05):  
        """  
        根据输入的图片数组创建一个像素组。  

        参数:  
            image (np.ndarray): 28x28 的灰度图片数组。  
            pixel_size (float): 每个像素方块的边长。  
            spacing (float): 方块之间的间距。  

        返回:  
            pixel_group (VGroup): 包含所有像素方块的组。  
        """  
        pixel_group = VGroup()  
        
        for i in range(28):  
            for j in range(28):  
                # 获取像素值（0-255），转换为灰度颜色并反转  
                pixel_value = image[i, j]  
                gray = pixel_value / 255  # 0为黑色，1为白色  
                
                # 创建一个方块代表像素  
                square = Square(  
                    side_length=pixel_size,  
                    fill_color=WHITE,  
                    fill_opacity=gray,  # 根据灰度设置不透明度  
                    stroke_width=0.2,  
                    stroke_color=WHITE  
                )  
                square.shift(  
                    (j - 14) * (pixel_size + spacing) * RIGHT +  
                    (14 - i) * (pixel_size + spacing) * UP  
                )  
                pixel_group.add(square)  
        
        return pixel_group  
    
    @staticmethod  
    def lerp(start, end, alpha):  
        """线性插值函数"""  
        return start + (end - start) * alpha  


if __name__ == "__main__":  
    config.pixel_height = 720  # 设置垂直分辨率  
    config.pixel_width = 1280  # 设置水平分辨率  
    config.frame_rate = 30  # 设置帧率  

    scene = HandwritingVisualization()  
    scene.render(preview=True)