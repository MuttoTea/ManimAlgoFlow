"""  
Summary:  
This code uses the Manim library to create a simple animation scene that showcases four text elements, each representing a different theme:  
1. Data Classification  
2. Handwriting Recognition  
3. Tumor Detection  
4. Other Topics (represented by "......")  

The animation process includes:  
- Sequentially writing each text with different colors.  
- Moving the texts to different positions.  
- Finally, fading out all texts to conclude the scene.  
"""  

from manim import *  

class Prologue(Scene):  
    def construct(self):  
        """Construct the animation scene with four text elements."""  
        # Create text elements with specified colors  
        data_classification_text = Text("数据分类", color=RED)  
        handwriting_recognition_text = Text("字迹识别", color=YELLOW)  
        tumor_detection_text = Text("肿瘤识别", color=BLUE)  
        other_topics_text = Text("......", color=ORANGE)  

        # Animate the writing of the first text element  
        self.play(Write(data_classification_text), runtime=1.5)  
        self.play(data_classification_text.animate.shift(UP * 2))  
        self.wait(1)  

        # Position handwriting_recognition_text next to tumor_detection_text on the left side  
        handwriting_recognition_text.next_to(tumor_detection_text, LEFT, buff=1)  
        self.play(Write(handwriting_recognition_text), runtime=1)  
        self.wait(1)  

        # Animate the writing of the third text element  
        self.play(Write(tumor_detection_text), runtime=1)  
        self.wait(1)  

        # Position other_topics_text next to tumor_detection_text on the right side  
        other_topics_text.next_to(tumor_detection_text, RIGHT, buff=1)  
        self.play(Write(other_topics_text))  
        self.wait(1)  

        # Fade out all text elements together  
        self.play(FadeOut(Group(data_classification_text, handwriting_recognition_text, tumor_detection_text, other_topics_text)))  
        self.wait(1)