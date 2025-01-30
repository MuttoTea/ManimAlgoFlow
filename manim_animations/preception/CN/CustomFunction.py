"""  
摘要：  
该代码使用 TensorFlow 和 Manim 库加载 MNIST 数据集中的手写数字图片，并将其可视化为像素方块。主要功能包括：  

1. 加载 MNIST 图片：通过 `load_mnist_image` 函数加载指定标签的手写数字图片，并返回图片数组和标签。  
2. 创建像素组：通过 `create_pixel_group` 函数将加载的 28x28 灰度图片转换为一组可视化的像素方块，每个方块的透明度根据对应的像素值设置。  

"""  

from manim import *
import tensorflow as tf


def load_mnist_image(label_target=1, index=0):  
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

def create_pixel_group(image, pixel_size=0.2, spacing=0.05):  
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

# 根据激活值设置颜色和透明度  
def set_activation(circles, activation_values):  
    animations = []  
    for circle, value in zip(circles, activation_values):  
        if value == -1:  
            continue  # 省略号跳过  
        color = interpolate_color(BLACK, WHITE, value)  
        opacity = value  
        animations.append(circle.animate.set_fill(color, opacity=opacity))  
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
    stroke_width=1.5  # 设置圆环厚度  
):  
    """  
    生成一列圆圈，超过16个时上下各八个，中间用垂直排列的省略号代替。  

    参数：  
    - n (int): 圆圈总数  
    - radius (float): 圆圈半径  
    - spacing (float): 圆圈之间的垂直间距  
    - color (str/Color): 圆圈和省略号的颜色  
    - ellipsis_dots (int): 省略号中点的数量，默认3  
    - ellipsis_dot_radius (float): 省略号中每个点的半径，默认radius/6  
    - ellipsis_buff (float): 省略号中点之间的垂直间距，默认0.1  
    - ellipsis_color (str/Color): 省略号中点的颜色，默认与圆圈颜色相同  
    - stroke_width (float): 圆圈边框的厚度，默认1.5  

    返回:  
    - elements (VGroup): 包含圆圈和省略号的组合  
    - circles (List[Circle]): 所有圆圈的列表  
    """  
    elements = VGroup()  
    circles = []  # 存储所有圆圈的列表  

    # 设置省略号点的半径和颜色  
    if ellipsis_dot_radius is None:  
        ellipsis_dot_radius = radius / 6  
    if ellipsis_color is None:  
        ellipsis_color = color  

    if n <= 16:  
        # 全部显示圆圈  
        for i in range(n):  
            circle = Circle(  
                radius=radius,   
                color=color,  
                stroke_width=stroke_width  # 设置圆环厚度  
            )  
            circle.number = i + 1  # 添加编号属性  
            circles.append(circle)  
            elements.add(circle)  
    else:  
        # 上八个圆圈  
        for i in range(8):  
            circle = Circle(  
                radius=radius,   
                color=color,  
                stroke_width=stroke_width  # 设置圆环厚度  
            )  
            circle.number = i + 1  # 添加编号属性  
            circles.append(circle)  
            elements.add(circle)  
        
        # 添加垂直排列的省略号  
        ellipsis = VGroup()  
        for _ in range(ellipsis_dots):  
            dot = Dot(  
                radius=ellipsis_dot_radius,   
                color=ellipsis_color  
            )  
            ellipsis.add(dot)  
        ellipsis.arrange(DOWN, buff=ellipsis_buff)  
        elements.add(ellipsis)  
        
        # 下八个圆圈  
        for i in range(n - 8, n):  
            circle = Circle(  
                radius=radius,   
                color=color,  
                stroke_width=stroke_width  # 设置圆环厚度  
            )  
            circle.number = i + 1  # 添加编号属性  
            circles.append(circle)  
            elements.add(circle)  

    # 设置圆圈之间的间距  
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
    在两组圆圈之间添加全连接。  

    参数和返回值同上。  

    """  
    lines = VGroup()  
    for circle1 in group1_circles:  
        for circle2 in group2_circles:  
            line = Line(  
                circle1.get_right() + RIGHT * buff,  
                circle2.get_left() + LEFT * buff,  
                color=connection_color,  
                stroke_width=stroke_width  
            )  
            lines.add(line)  
    return lines

def split_line_into_sublines(line, num_subsegments=100, overlap=0.01, stroke_width=2):  
    """  
    将线段分割成多个子线段，并增加重叠以减少分界线的明显性。  
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
    定义持续颜色波动的函数。  
    颜色从蓝色逐渐变化到黄色，再回到蓝色，循环往复。  
    """  
    phase = alpha * TAU  
    gradient_colors = color_gradient([BLUE, YELLOW, BLUE], 100)  
    num_subsegments = len(mob)  
    
    for i in range(num_subsegments):  
        t = i / num_subsegments  
        offset = (t * TAU + phase) % TAU  
        color_value = (np.sin(offset) + 1) / 2  # 范围 [0,1]  
        color_index = int(color_value * (len(gradient_colors) - 1))  
        current_color = gradient_colors[color_index]  
        mob[i].set_color(current_color)  

def single_pulse_wave(mob, alpha):  
        """  
        让脉冲中心从 -pulse_width 跑到 1 + pulse_width，  
        确保动画结束时脉冲尾部也能到达线段末端。  
        """  
        pulse_width = 0.05  
        num_subsegments = len(mob)  

        # 将 alpha 的 [0, 1] 映射到脉冲中心的 [-pulse_width, 1 + pulse_width]  
        current_pulse = -pulse_width + alpha * (1 + pulse_width - (-pulse_width))  
        # 或者直接： current_pulse = (alpha * (1 + 2 * pulse_width)) - pulse_width  

        for i in range(num_subsegments):  
            t = i / num_subsegments  
            if current_pulse - pulse_width < t < current_pulse + pulse_width:  
                mob[i].set_color(YELLOW)  
            else:  
                mob[i].set_color(BLUE)    

def animate_line_wave(line, wave_type="continuous",   
                        num_subsegments=100, overlap=0.01, stroke_width=2):  
    """  
    接受一条Line，将其分段并根据 wave_type 参数播放相应动画。  
    wave_type 支持 "continuous"（持续波动）或 "pulse"（脉冲波动）。  
    """  
    # 1. 拆分线段   
    sublines = split_line_into_sublines(  
        line, num_subsegments, overlap, stroke_width  
    )  
    
    # 2. 根据类型选择动画更新函数  
    if wave_type == "continuous":  
        wave_func = continuous_color_wave  
    elif wave_type == "pulse":  
        wave_func = single_pulse_wave  
    else:  
        raise ValueError("wave_type 只能是 'continuous' 或 'pulse'。")  
    
    # 3. 创建动画（UpdateFromAlphaFunc 会在播放时连续调用 wave_func）  
    line_wave_animation = UpdateFromAlphaFunc(  
        sublines,   
        lambda m, alpha: wave_func(m, alpha)  
    )  
    
    return sublines, line_wave_animation