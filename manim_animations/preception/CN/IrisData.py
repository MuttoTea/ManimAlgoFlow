"""  
摘要：  
该代码使用 Manim 库创建多个动画场景，展示鸢尾花数据集及其特征选择和数据可视化的过程。具体包括：  
1. LirsDataTitle：展示鸢尾花数据集的简介，包括不同种类的鸢尾花及其图片。  
2. FeatureSelcet：展示鸢尾花的特征（花萼长度和宽度）以及分类对象（山鸢尾和变色鸢尾）。  
3. LirdDataVisual：可视化鸢尾花数据集，训练感知器模型，并绘制决策边界。  
4. CoordinateSystem：创建坐标轴，展示鸢尾花数据点及其对应的超平面，并添加注释。  

"""  

from manim import *
from perceptron_models import SingleLayerPerceptron
from DataProcessor import IrisDataProcessor


class LirsDataTitle(Scene):
    """
    数据集简介
    """

    def construct(self):
        # 显示标题并设置字体大小和位置
        title = Text("鸢尾花数据集", color=RED).scale(1.2)

        # 加载图片
        ImageSetosa = ImageMobject(
            "C:\Project\Adobe\PR\Perceptron\DisposalCode\media\images\山鸢尾.jpg"
        )
        ImageVersicolor = ImageMobject(
            "C:\Project\Adobe\PR\Perceptron\DisposalCode\media\images\变色鸢尾.jpeg"
        )
        ImageVirginica = ImageMobject(
            "C:\Project\Adobe\PR\Perceptron\DisposalCode\media\images\维吉尼亚鸢尾png.png"
        )
        ImageP = ImageMobject(
            "C:\Project\Adobe\PR\Perceptron\DisposalCode\media\images\罗纳德·费舍尔.jpg"
        )

        # 创建文本标签
        TextSetosa = Paragraph(
            "山鸢尾", "(Iris Setosa)", color=BLUE, alignment="center", font_size=20
        )
        TextVersicolor = Paragraph(
            "变色鸢尾",
            "(Iris Versicolor)",
            color=BLUE,
            alignment="center",
            font_size=20,
        )
        TextVirginica = Paragraph(
            "维吉尼亚鸢尾",
            "(Iris Virginica)",
            color=BLUE,
            alignment="center",
            font_size=20,
        )
        TextP_title = Text("罗纳德·费舍尔", color=YELLOW, font_size=20)
        TextP_date = MathTex("1890-1962", color=YELLOW).scale(0.7)

        # 组合成一个组，并垂直排列
        TextP = (
            VGroup(TextP_title, TextP_date).arrange(DOWN, buff=0.2).set_color(YELLOW)
        )

        # 调整图片大小
        target_height = 2  # 目标高度
        ImageSetosa.height = target_height
        ImageVersicolor.height = target_height
        ImageVirginica.height = target_height
        ImageP.height = target_height * 1.5

        # 创建图片与文本的组合
        setosa_group = Group(ImageSetosa, TextSetosa).arrange(DOWN, buff=0.2)
        versicolor_group = Group(ImageVersicolor, TextVersicolor).arrange(
            DOWN, buff=0.2
        )
        virginica_group = Group(ImageVirginica, TextVirginica).arrange(DOWN, buff=0.2)
        P_group = Group(ImageP, TextP).arrange(DOWN, buff=0.2)

        # 显示标题
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.shift(UP * 2.5))
        self.wait(0.5)

        # 设置组合的位置
        versicolor_group.move_to(ORIGIN)
        setosa_group.next_to(versicolor_group, LEFT, buff=1)
        virginica_group.next_to(versicolor_group, RIGHT, buff=1)

        # 创建一个整体组
        full_group = Group(title, setosa_group, versicolor_group, virginica_group)

        # 添加图片和文本到场景
        self.play(
            FadeIn(setosa_group),
            FadeIn(versicolor_group),
            FadeIn(virginica_group),
            run_time=1,
        )
        self.wait(0.5)
        # 缩小并左移
        self.play(full_group.animate.scale(0.8).shift(LEFT * 2), run_time=1)

        # 将 P_group 的中轴线与 full_group 对齐
        P_group.align_to(full_group, UP)
        P_group.next_to(full_group, RIGHT, buff=1)

        self.play(FadeIn(P_group), run_time=1)
        self.wait(1)
        self.play(FadeOut(Group(title, full_group, P_group)))
        self.wait(1)


class FeatureSelcet(Scene):
    """
    数据选择
    """

    def construct(self):
        # 创建文本对象
        title1 = Text("特征", color=RED).scale(1.2)
        title2 = Text("分类对象", color=RED).scale(1.2)
        feature_l = Text("长", color=WHITE)
        feature_w = Text("宽", color=WHITE)
        object_1 = Text("山鸢尾", color=WHITE)
        object_2 = Text("变色鸢尾", color=WHITE)

        # 创建特征列
        feature_column = VGroup(title1, feature_l, feature_w).arrange(DOWN, buff=0.5)

        # 创建分类对象列
        object_column = VGroup(title2, object_1, object_2).arrange(DOWN, buff=0.5)

        # 将两列水平排列
        layout = VGroup(feature_column, object_column).arrange(RIGHT, buff=2)

        # 逐步显示文本对象
        self.play(Write(title1))
        self.play(Write(feature_l), Write(feature_w))
        self.play(Write(title2))
        self.play(Write(object_1), Write(object_2))

        self.wait(1)
        self.play(FadeOut(layout))
        self.wait(1)


class LirdDataVisual(Scene):
    """
    数据可视化
    """

    def construct(self):
        # 数据处理
        data_processor = IrisDataProcessor()
        X, y = data_processor.get_data()

        # 训练感知机
        model = SingleLayerPerceptron(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        # 获取直线信息
        w1, w2 = model.weights
        b = model.bias - 0.1
        slope = -w1 / w2
        intercept = -b / w2

        # 创建坐标轴
        axes = Axes(
            x_range=[4, 7.5, 0.5],  # x轴范围
            y_range=[1, 5, 1],  # y轴范围
            axis_config={"color": BLUE, "include_numbers": True},
        )

        # 添加标签
        x_label = axes.get_x_axis_label(Text("花萼长度 (cm)")).scale(0.8)
        y_label = axes.get_y_axis_label(Text("花萼宽度 (cm)")).scale(0.8)

        # 添加坐标轴和标签到场景
        self.play(Create(axes), Write(x_label), Write(y_label))

        # 使用 numpy 进行数据筛选
        setosa_indices = np.where(y == 0)[0]
        versicolor_indices = np.where(y == 1)[0]

        setosa_points = [(X[i, 0], X[i, 1]) for i in setosa_indices]
        versicolor_points = [(X[i, 0], X[i, 1]) for i in versicolor_indices]

        # 创建散点
        setosa_dots = [Dot(axes.c2p(x, y), color=BLUE) for x, y in setosa_points]
        versicolor_dots = [
            Dot(axes.c2p(x, y), color=ORANGE) for x, y in versicolor_points
        ]

        # 创建点的组合
        dots_group = VGroup(*setosa_dots, *versicolor_dots)

        # 添加散点到场景
        self.play(*[Create(dot) for dot in setosa_dots + versicolor_dots])

        # 绘制超平面
        hyperplane = axes.plot(
            lambda x: slope * x + intercept, color=WHITE, x_range=[4, 7]
        )

        # 添加超平面到场景
        self.play(Create(hyperplane))

        # 添加文字“如何求这条直线？”
        annotation_text = (
            Text("如何求这条直线？", font_size=24)
            .next_to(hyperplane, UP, buff=0.5)
            .shift(RIGHT * 1.5)
        )

        # 创建曲线箭头指向超平面上x为6.75的点
        x_target = 6.75
        y_target = slope * x_target + intercept
        end_point = axes.c2p(x_target, y_target)
        arrow = CurvedArrow(
            annotation_text.get_center() + [1.3, 0, 0],
            end_point,
            angle=-TAU / 8,
            color=YELLOW,
            stroke_width=4,
            tip_length=0.3,
        )
        self.play(Write(annotation_text), Create(arrow))

        # 等待
        self.wait(1)
        self.play(
            FadeOut(
                Group(
                    annotation_text,
                    arrow,
                    axes,
                    x_label,
                    y_label,
                    hyperplane,
                    dots_group,
                )
            )
        )
        self.wait(1)


class CoordinateSystem(Scene):
    def construct(self):
        # 创建坐标轴
        # 数据处理
        data_processor = IrisDataProcessor()
        X, y = data_processor.get_data()

        # 训练感知机
        model = SingleLayerPerceptron(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        # 获取直线信息
        w1, w2 = model.weights
        b = model.bias - 0.1
        slope = -w1 / w2
        intercept = -b / w2

        # 创建坐标轴
        axes = Axes(
            x_range=[4, 7.5, 0.5],  # x轴范围
            y_range=[1, 5, 1],  # y轴范围
            axis_config={"color": BLUE, "include_numbers": True},
        )

        # 添加标签
        x_label = axes.get_x_axis_label(Text("花萼长度 (cm)")).scale(0.8)
        y_label = axes.get_y_axis_label(Text("花萼宽度 (cm)")).scale(0.8)

        # 添加坐标轴和标签到场景
        self.play(Create(axes), Write(x_label), Write(y_label))

        # 使用 numpy 进行数据筛选
        setosa_indices = np.where(y == 0)[0]
        versicolor_indices = np.where(y == 1)[0]

        setosa_points = [(X[i, 0], X[i, 1]) for i in setosa_indices]
        versicolor_points = [(X[i, 0], X[i, 1]) for i in versicolor_indices]

        # 创建散点
        setosa_dots = [Dot(axes.c2p(x, y), color=BLUE) for x, y in setosa_points]
        versicolor_dots = [
            Dot(axes.c2p(x, y), color=ORANGE) for x, y in versicolor_points
        ]

        # 创建点的组合
        dots_group = VGroup(*setosa_dots, *versicolor_dots)

        # 添加散点到场景
        self.play(*[Create(dot) for dot in setosa_dots + versicolor_dots])

        # 绘制超平面
        hyperplane = axes.plot(
            lambda x: slope * x + intercept, color=WHITE, x_range=[4, 7]
        )

        # 添加超平面到场景
        self.play(Create(hyperplane))

        # 添加文字“如何求这条直线？”
        annotation_text = (
            Text("如何求这条直线？", font_size=24)
            .next_to(hyperplane, UP, buff=0.5)
            .shift(RIGHT * 1.5)
        )

        # 创建曲线箭头指向超平面上x为6.75的点
        x_target = 6.75
        y_target = slope * x_target + intercept
        end_point = axes.c2p(x_target, y_target)
        arrow = CurvedArrow(
            annotation_text.get_center() + [1.3, 0, 0],
            end_point,
            angle=-TAU / 8,
            color=YELLOW,
            stroke_width=4,
            tip_length=0.3,
        )
        self.play(Write(annotation_text), Create(arrow))

        # 等待
        self.wait(1)
        self.play(
            FadeOut(
                Group(
                    annotation_text,
                    arrow,
                    axes,
                    x_label,
                    y_label,
                    hyperplane,
                    dots_group,
                )
            )
        )
        self.wait(1)
