"""  
Summary:  
This code uses the Manim library to create multiple animation scenes that demonstrate the Iris dataset, feature selection, and data visualization processes. Specifically, it includes:  
1. LirsDataTitle: Introduces the Iris dataset, including different species of Iris flowers and their images.  
2. FeatureSelect: Displays the features (sepal length and width) of the Iris flowers and the classification targets (Iris-setosa and Iris-versicolor).  
3. IrisDataVisual: Visualizes the Iris dataset, trains a perceptron model, and plots the decision boundary.  
4. CoordinateSystem: Creates axes to display the Iris data points and their corresponding hyperplane, along with annotations.  
"""  

from manim import *  
from perceptron_models import SinglePerceptron  
from DataProcessor import IrisDataProcessor  


class IrisDataTitle(Scene):  
    """Scene to introduce the Iris dataset."""  

    def construct(self):  
        # Create and position the title  
        title_text = Text("鸢尾花数据集", color=RED).scale(1.2)  

        # Load images of different Iris species  
        image_setosa = ImageMobject(  
            "C:\\Project\\Adobe\\PR\\Perceptron\\DisposalCode\\media\\images\\山鸢尾.jpg"  
        )  
        image_versicolor = ImageMobject(  
            "C:\\Project\\Adobe\\PR\\Perceptron\\DisposalCode\\media\\images\\变色鸢尾.jpeg"  
        )  
        image_virginica = ImageMobject(  
            "C:\\Project\\Adobe\\PR\\Perceptron\\DisposalCode\\media\\images\\维吉尼亚鸢尾png.png"  
        )  
        image_fisher = ImageMobject(  
            "C:\\Project\\Adobe\\PR\\Perceptron\\DisposalCode\\media\\images\\罗纳德·费舍尔.jpg"  
        )  

        # Create labels for the images  
        label_setosa = Paragraph(  
            "山鸢尾", "(Iris Setosa)", color=BLUE, alignment="center", font_size=20  
        )  
        label_versicolor = Paragraph(  
            "变色鸢尾",  
            "(Iris Versicolor)",  
            color=BLUE,  
            alignment="center",  
            font_size=20,  
        )  
        label_virginica = Paragraph(  
            "维吉尼亚鸢尾",  
            "(Iris Virginica)",  
            color=BLUE,  
            alignment="center",  
            font_size=20,  
        )  
        label_fisher_title = Text("罗纳德·费舍尔", color=YELLOW, font_size=20)  
        label_fisher_years = MathTex("1890-1962", color=YELLOW).scale(0.7)  

        # Group the Fisher label components  
        label_fisher = (  
            VGroup(label_fisher_title, label_fisher_years).arrange(DOWN, buff=0.2).set_color(YELLOW)  
        )  

        # Resize images to a target height  
        target_height = 2  # Target height for images  
        image_setosa.height = target_height  
        image_versicolor.height = target_height  
        image_virginica.height = target_height  
        image_fisher.height = target_height * 1.5  

        # Create groups for each species and Fisher  
        group_setosa = Group(image_setosa, label_setosa).arrange(DOWN, buff=0.2)  
        group_versicolor = Group(image_versicolor, label_versicolor).arrange(DOWN, buff=0.2)  
        group_virginica = Group(image_virginica, label_virginica).arrange(DOWN, buff=0.2)  
        group_fisher = Group(image_fisher, label_fisher).arrange(DOWN, buff=0.2)  

        # Display the title  
        self.play(Write(title_text))  
        self.wait(0.5)  
        self.play(title_text.animate.shift(UP * 2.5))  
        self.wait(0.5)  

        # Position the species groups  
        group_versicolor.move_to(ORIGIN)  
        group_setosa.next_to(group_versicolor, LEFT, buff=1)  
        group_virginica.next_to(group_versicolor, RIGHT, buff=1)  

        # Create a full group for the title and species  
        full_group = Group(title_text, group_setosa, group_versicolor, group_virginica)  

        # Add images and labels to the scene  
        self.play(  
            FadeIn(group_setosa),  
            FadeIn(group_versicolor),  
            FadeIn(group_virginica),  
            run_time=1,  
        )  
        self.wait(0.5)  

        # Scale down and shift the full group  
        self.play(full_group.animate.scale(0.8).shift(LEFT * 2), run_time=1)  

        # Align Fisher group with the full group  
        group_fisher.align_to(full_group, UP)  
        group_fisher.next_to(full_group, RIGHT, buff=1)  

        self.play(FadeIn(group_fisher), run_time=1)  
        self.wait(1)  
        self.play(FadeOut(Group(title_text, full_group, group_fisher)))  
        self.wait(1)  


class FeatureSelect(Scene):  
    """Scene to display feature selection for the Iris dataset."""  

    def construct(self):  
        # Create title texts for features and classification objects  
        title_features = Text("特征", color=RED).scale(1.2)  
        title_objects = Text("分类对象", color=RED).scale(1.2)  
        feature_length = Text("长", color=WHITE)  
        feature_width = Text("宽", color=WHITE)  
        iris_setosa_label = Text("山鸢尾", color=WHITE)  
        iris_versicolor_label = Text("变色鸢尾", color=WHITE)  

        # Create feature and object columns  
        feature_column = VGroup(title_features, feature_length, feature_width).arrange(DOWN, buff=0.5)  
        object_column = VGroup(title_objects, iris_setosa_label, iris_versicolor_label).arrange(DOWN, buff=0.5)  

        # Arrange the columns horizontally  
        layout = VGroup(feature_column, object_column).arrange(RIGHT, buff=2)  

        # Sequentially display the text objects  
        self.play(Write(title_features))  
        self.play(Write(feature_length), Write(feature_width))  
        self.play(Write(title_objects))  
        self.play(Write(iris_setosa_label), Write(iris_versicolor_label))  

        self.wait(1)  
        self.play(FadeOut(layout))  
        self.wait(1)  


class IrisDataVisual(Scene):  
    """Scene to visualize the Iris dataset and decision boundary."""  

    def construct(self):  
        # Data processing  
        data_processor = IrisDataProcessor()  
        features, labels = data_processor.get_data()  

        # Train the perceptron model  
        perceptron_model = SinglePerceptron(learning_rate=0.1, n_iterations=1000)  
        perceptron_model.fit(features, labels)  

        # Retrieve line parameters  
        weight1, weight2 = perceptron_model.weights  
        bias = perceptron_model.bias - 0.1  
        slope = -weight1 / weight2  
        intercept = -bias / weight2  

        # Create axes for visualization  
        axes = Axes(  
            x_range=[4, 7.5, 0.5],  # X-axis range  
            y_range=[1, 5, 1],  # Y-axis range  
            axis_config={"color": BLUE, "include_numbers": True},  
        )  

        # Add axis labels  
        x_label = axes.get_x_axis_label(Text("花萼长度 (cm)")).scale(0.8)  
        y_label = axes.get_y_axis_label(Text("花萼宽度 (cm)")).scale(0.8)  

        # Add axes and labels to the scene  
        self.play(Create(axes), Write(x_label), Write(y_label))  

        # Use numpy to filter data points  
        setosa_indices = np.where(labels == 0)[0]  
        versicolor_indices = np.where(labels == 1)[0]  

        setosa_points = [(features[i, 0], features[i, 1]) for i in setosa_indices]  
        versicolor_points = [(features[i, 0], features[i, 1]) for i in versicolor_indices]  

        # Create scatter points for each species  
        setosa_dots = [Dot(axes.c2p(x, y), color=BLUE) for x, y in setosa_points]  
        versicolor_dots = [Dot(axes.c2p(x, y), color=ORANGE) for x, y in versicolor_points]  

        # Group all dots together  
        dots_group = VGroup(*setosa_dots, *versicolor_dots)  

        # Add scatter points to the scene  
        self.play(*[Create(dot) for dot in setosa_dots + versicolor_dots])  

        # Plot the decision boundary (hyperplane)  
        hyperplane = axes.plot(  
            lambda x: slope * x + intercept, color=WHITE, x_range=[4, 7]  
        )  

        # Add the hyperplane to the scene  
        self.play(Create(hyperplane))  

        # Add annotation for the hyperplane  
        annotation_text = (  
            Text("如何求这条直线？", font_size=24)  
            .next_to(hyperplane, UP, buff=0.5)  
            .shift(RIGHT * 1.5)  
        )  

        # Create a curved arrow pointing to a specific point on the hyperplane  
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

        # Wait before fading out  
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
    """Scene to create a coordinate system and visualize the Iris dataset."""  

    def construct(self):  
        # Data processing  
        data_processor = IrisDataProcessor()  
        features, labels = data_processor.get_data()  

        # Train the perceptron model  
        perceptron_model = SinglePerceptron(learning_rate=0.1, n_iterations=1000)  
        perceptron_model.fit(features, labels)  

        # Retrieve line parameters  
        weight1, weight2 = perceptron_model.weights  
        bias = perceptron_model.bias - 0.1  
        slope = -weight1 / weight2  
        intercept = -bias / weight2  

        # Create axes for visualization  
        axes = Axes(  
            x_range=[4, 7.5, 0.5],  # X-axis range  
            y_range=[1, 5, 1],  # Y-axis range  
            axis_config={"color": BLUE, "include_numbers": True},  
        )  

        # Add axis labels  
        x_label = axes.get_x_axis_label(Text("花萼长度 (cm)")).scale(0.8)  
        y_label = axes.get_y_axis_label(Text("花萼宽度 (cm)")).scale(0.8)  

        # Add axes and labels to the scene  
        self.play(Create(axes), Write(x_label), Write(y_label))  

        # Use numpy to filter data points  
        setosa_indices = np.where(labels == 0)[0]  
        versicolor_indices = np.where(labels == 1)[0]  

        setosa_points = [(features[i, 0], features[i, 1]) for i in setosa_indices]  
        versicolor_points = [(features[i, 0], features[i, 1]) for i in versicolor_indices]  

        # Create scatter points for each species  
        setosa_dots = [Dot(axes.c2p(x, y), color=BLUE) for x, y in setosa_points]  
        versicolor_dots = [Dot(axes.c2p(x, y), color=ORANGE) for x, y in versicolor_points]  

        # Group all dots together  
        dots_group = VGroup(*setosa_dots, *versicolor_dots)  

        # Add scatter points to the scene  
        self.play(*[Create(dot) for dot in setosa_dots + versicolor_dots])  

        # Plot the decision boundary (hyperplane)  
        hyperplane = axes.plot(  
            lambda x: slope * x + intercept, color=WHITE, x_range=[4, 7]  
        )  

        # Add the hyperplane to the scene  
        self.play(Create(hyperplane))  

        # Add annotation for the hyperplane  
        annotation_text = (  
            Text("如何求这条直线？", font_size=24)  
            .next_to(hyperplane, UP, buff=0.5)  
            .shift(RIGHT * 1.5)  
        )  

        # Create a curved arrow pointing to a specific point on the hyperplane  
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

        # Wait before fading out  
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