from manim import *

class OutputFunctionTransfrom(Scene):  
    def setup_deepness(self):
        pass

    def construct(self):  
        # 1) 配置并显示公式（以及检查索引）
        # substrings_to_isolate 用于指定要拆分的子部分
        formula_z = MathTex(
            r"z = w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + w_{4}x_{4} + b",
            substrings_to_isolate=[
                "z", "w_{1}x_{1}", "w_{2}x_{2}", "w_{3}x_{3}", "w_{4}x_{4}", "b",
            ],
        ).to_edge(UP)

            # 打印公式中每个部分的索引（调试用）  
        for i, part in enumerate(formula_z):  
            print(f"index = {i}, content = {part}")  

        # 2) 创建并显示花萼、花瓣相关的文本  
        textSepalLen = Text("花萼长度").set_color(BLUE)  
        textSepalWid = Text("花萼宽度").set_color(BLUE)  
        textPetalLen = Text("花瓣长度").set_color(BLUE)  
        textPetalWid = Text("花瓣宽度").set_color(BLUE)  

        groupLeft = VGroup(textSepalLen, textSepalWid).arrange(DOWN, buff=1)  
        groupRight = VGroup(textPetalLen, textPetalWid).arrange(DOWN, buff=1)  
        groupAll = VGroup(groupLeft, groupRight).arrange(RIGHT, buff=3).scale(0.8)  

        self.play(Write(groupLeft))  
        self.play(Write(groupRight))  
        self.wait(1)  

        # 3) 创建与特征对应的 x_i  
        feature1 = MathTex("x_{1}").set_color(RED).next_to(textSepalLen, RIGHT, buff=0.3)  
        feature2 = MathTex("x_{2}").set_color(RED).next_to(textSepalWid, RIGHT, buff=0.3)  
        feature3 = MathTex("x_{3}").set_color(RED).next_to(textPetalLen, RIGHT, buff=0.3)  
        feature4 = MathTex("x_{4}").set_color(RED).next_to(textPetalWid, RIGHT, buff=0.3)  
        groupFeatures = VGroup(feature1, feature2, feature3, feature4)  

        self.play(Write(groupFeatures))  
        self.wait(1)  

        # 4) 创建 w_i 并让它们替换原花萼/花瓣文本  
        weight1 = MathTex("w_{1}").set_color(BLUE).move_to(textSepalLen)  
        weight2 = MathTex("w_{2}").set_color(BLUE).move_to(textSepalWid)  
        weight3 = MathTex("w_{3}").set_color(BLUE).move_to(textPetalLen)  
        weight4 = MathTex("w_{4}").set_color(BLUE).move_to(textPetalWid)  

        self.play(  
            ReplacementTransform(textSepalLen, weight1),  
            ReplacementTransform(textSepalWid, weight2),  
            ReplacementTransform(textPetalLen, weight3),  
            ReplacementTransform(textPetalWid, weight4),  
            feature1.animate.next_to(weight1, RIGHT, buff=0.2),  
            feature2.animate.next_to(weight2, RIGHT, buff=0.2),  
            feature3.animate.next_to(weight3, RIGHT, buff=0.2),  
            feature4.animate.next_to(weight4, RIGHT, buff=0.2),  
        )  
        self.wait(1)  

        # 5) (w_i x_i) 组合，并将它们一起移动到公式中对应位置  
        groupWxPositions = [  
            (VGroup(weight1, feature1), formula_z[2].get_center()),  # w1x1 在公式索引2处  
            (VGroup(weight2, feature2), formula_z[4].get_center()),  # w2x2 在公式索引4处  
            (VGroup(weight3, feature3), formula_z[6].get_center()),  
            (VGroup(weight4, feature4), formula_z[8].get_center()),  
        ]  
        
        # 同时移动到目标位置  
        animations = []  
        for (wxGroup, targetPos) in groupWxPositions:  
            animations.append(wxGroup.animate.move_to(targetPos))  
        
        self.play(*animations)  
        self.wait(1)  

        # 6) 依次淡入公式的其余部分  
        # formula_z 包含: 0:z, 1:=, 2:w1x1, 3:+, 4:w2x2, 5:+, 6:w3x3, 7:+, 8:w4x4, 9:+, 10:b  
        part0 = formula_z[0]   # z  
        part1 = formula_z[1]   # =  
        part3 = formula_z[3]   # +  
        part5 = formula_z[5]   # +  
        part7 = formula_z[7]   # +  
        part9 = formula_z[9]   # +  
        part10 = formula_z[10] # b  

        self.play(  
            FadeIn(part0),  
            FadeIn(part1),  
            FadeIn(part3),  
            FadeIn(part5),  
            FadeIn(part7),  
            FadeIn(part9),  
            FadeIn(part10)  
        )  
        self.wait(1)  

        # 7) 将 w_i、x_i 恢复白色  
        self.play(  
            weight1.animate.set_color(WHITE),  
            weight2.animate.set_color(WHITE),  
            weight3.animate.set_color(WHITE),  
            weight4.animate.set_color(WHITE),  
            feature1.animate.set_color(WHITE),  
            feature2.animate.set_color(WHITE),  
            feature3.animate.set_color(WHITE),  
            feature4.animate.set_color(WHITE),  
        )  
        self.wait(1)  

        # 8) 显示列向量 W、X，再作后续动画  
        text_W = MathTex(  
            r"\mathbf{W} = \begin{bmatrix} w_{1} \\ w_{2} \\ w_{3} \\ w_{4} \end{bmatrix}"  
        ).to_edge(LEFT).scale(0.8)  
        text_X = MathTex(  
            r"\mathbf{X} = \begin{bmatrix} x_{1} \\ x_{2} \\ x_{3} \\ x_{4} \end{bmatrix}"  
        ).next_to(text_W, RIGHT, buff=0.3).scale(0.8)  

        self.play(Write(text_W), Write(text_X))  
        self.wait(1)  

        # 9) 创建等号和向量  
        text_WT = MathTex(r"\mathbf{W}^{T}", color=BLUE).next_to(text_X, RIGHT, buff=1).scale(0.8)  
        textX = MathTex(r"\mathbf{X}", color=GREEN).next_to(text_WT, RIGHT, buff=0.075).scale(0.8)  
        eq1 = MathTex("=", color=WHITE).next_to(textX, RIGHT, buff=0.2).scale(0.8)  
        eq2 = MathTex("=", color=WHITE).next_to(eq1, DOWN, buff=1.5).scale(0.8)  

        self.play(Write(text_WT), Write(textX), Write(eq1))  
        self.wait(1)  

        # 10) 创建矩阵对象  
        W_matrix = Matrix(  
            [["w_{1}", "w_{2}", "w_{3}", "w_{4}"]],  
            h_buff=0.8,  
            bracket_h_buff=SMALL_BUFF,  
            bracket_v_buff=SMALL_BUFF  
        ).next_to(eq1, RIGHT, buff=0.2).scale(0.8)  

        X_matrix = Matrix(  
            [  
                ["x_{1}"],  
                ["x_{2}"],  
                ["x_{3}"],  
                ["x_{4}"]  
            ],  
            v_buff=0.6,  
            bracket_h_buff=SMALL_BUFF,  
            bracket_v_buff=SMALL_BUFF  
        ).next_to(W_matrix, RIGHT, buff=0.2).scale(0.8)  

        self.play(Write(W_matrix), Write(X_matrix))  
        self.wait(1)  

        # 11) 创建展开表达式并设置要拆分的部分  
        expandedExpression = MathTex(  
            r"w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + w_{4}x_{4}",  
            substrings_to_isolate=[  
                "w_{1}", "x_{1}",  
                "w_{2}", "x_{2}",  
                "w_{3}", "x_{3}",  
                "w_{4}", "x_{4}"  
            ]  
        ).scale(0.8).next_to(eq2, RIGHT, buff=0.2)  

        self.play(Write(eq2))  
        self.wait(1)  

        # 12) 获取矩阵条目，用于 TransformFromCopy  
        wEntries = W_matrix.get_entries()  # [w1, w2, w3, w4]  
        xEntries = X_matrix.get_entries()  # [x1, x2, x3, x4]  

        # 配置对应关系 (w_i, x_i) 与 expandedExpression 里的对应索引  
        # expandedExpression 的可能拆分顺序：0:w1, 1:x1, 2:+, 3:w2, 4:x2, 5:+, 6:w3, 7:x3, 8:+, 9:w4, 10:x4  
        # 这里要注意 substring 是否完全匹配  
        pairData = [  
            (wEntries[0], xEntries[0], 0, 1, 2),  
            (wEntries[1], xEntries[1], 3, 4, 5),  
            (wEntries[2], xEntries[2], 6, 7, 8),  
            (wEntries[3], xEntries[3], 9, 10, None),  
        ]  

        # 13) 循环播放动画  
        for wVal, xVal, idx_w, idx_x, idx_plus in pairData:  
            # a) 先将当前的 w、x 设色  
            self.play(  
                wVal.animate.set_color(BLUE),  
                xVal.animate.set_color(RED),  
            )  
            self.wait(0.15)  

            # b) TransformFromCopy 到展开表达式  
            self.play(  
                TransformFromCopy(wVal, expandedExpression[idx_w]),  
                TransformFromCopy(xVal, expandedExpression[idx_x]),  
                wVal.animate.set_color(WHITE),  
                xVal.animate.set_color(WHITE),  
            )  
            self.wait(0.15)  

            # c) 加上加号（如果还有）  
            if idx_plus is not None:  
                self.play(Write(expandedExpression[idx_plus], run_time=0.25))  

        self.wait(1)  

        # 14) 用矩阵形式替换 (包含已显示的 w1, x1, w2, x2...)  
        # 避免 w1,x1... 残留  
        formulaMatrix = MathTex(r"z = \mathbf{w}^T \mathbf{x} + b").move_to(formula_z.get_center()).scale(0.8)
        self.play(  
            ReplacementTransform(  
                VGroup(formula_z, weight1, feature1, weight2, feature2,  
                    weight3, feature3, weight4, feature4),  
                formulaMatrix  
            )  
        )  
        self.wait(1)


class HyperplaneTransfrom(Scene):
    def construct(self):
        # 1. 顶部公式：z = w^T x + b
        z_equation = MathTex(r"z = \mathbf{w}^T \mathbf{x} + b")
        z_equation.to_edge(UP).scale(0.8)
        self.play(Write(z_equation))
        
        # 2. 二维超平面及损失函数  
        plane_two_var_equation = MathTex(r"w_{1} x_{1} + w_{2} x_{2} + b = 0")  
        loss_two_var_equation = MathTex(r"L = (w_{1} x_{1} + w_{2} x_{2} + b)\, y")  

        plane_two_var_equation.next_to(z_equation, DOWN, buff=1.5).scale(0.8)  
        loss_two_var_equation.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(  
            Write(plane_two_var_equation),  
            Write(loss_two_var_equation)  
        )  
        self.wait(1)  

        # 3. 四维超平面及损失函数  
        plane_four_var_equation = MathTex(r"w_{1} x_{1} + w_{2} x_{2} + w_{3} x_{3} + w_{4} x_{4} + b = 0")  
        loss_four_var_equation = MathTex(r"L = (w_{1} x_{1} + w_{2} x_{2} + w_{3} x_{3} + w_{4} x_{4} + b)\, y")  

        plane_four_var_equation.next_to(z_equation, DOWN, buff=1.5).scale(0.8)  
        loss_four_var_equation.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(  
            ReplacementTransform(plane_two_var_equation, plane_four_var_equation),  
            ReplacementTransform(loss_two_var_equation, loss_four_var_equation)  
        )  
        self.wait(1)  

        # 4. 向量化形式及相应损失函数  
        plane_matrix_equation = MathTex(r"\mathbf{w}^T \mathbf{x} + b = 0")  
        loss_vector_equation = MathTex(r"L = \mathbf{w}^T \mathbf{x} + b")  

        plane_matrix_equation.next_to(z_equation, DOWN, buff=1.5).scale(0.8)  
        loss_vector_equation.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(  
            ReplacementTransform(plane_four_var_equation, plane_matrix_equation),  
            ReplacementTransform(loss_four_var_equation, loss_vector_equation)  
        )  
        self.wait(1)  

        # 5. 引入感知机常用的损失函数 (hinge-like)  
        loss_function_new = MathTex(r"L = \max\left(0,\,-y\bigl(\mathbf{w}^T \mathbf{x} + b\bigr)\right)") 
        loss_function_new.next_to(z_equation, DOWN, buff=3).scale(0.8)  

        self.play(ReplacementTransform(loss_vector_equation, loss_function_new))  
        self.wait(1)  

        # # 6. 再次展示 z = w^T x + b（可视为重复强调或过渡）  
        # z_equation_updated = MathTex(r"z = \mathbf{w}^T \mathbf{x} + b")  
        # z_equation_updated.to_edge(UP).scale(0.8)  
        # self.play(Write(z_equation_updated))  

        # 7. 梯度下降更新公式  
        gradient_update_w = MathTex(r"\mathbf{w} \leftarrow \mathbf{w} - \eta \frac{\partial L}{\partial \mathbf{w}}")  
        gradient_update_b = MathTex(r"b \leftarrow b - \eta \frac{\partial L}{\partial b}")  

        # 将两个公式水平并列  
        gradient_updates = VGroup(gradient_update_w, gradient_update_b)  
        gradient_updates.arrange(RIGHT, buff=0.5).scale(0.8)  
        gradient_updates.next_to(z_equation, DOWN, buff=4.5)  

        self.play(Write(gradient_updates))  
        self.wait(1)  

        # # 8. 场景结束（可选择添加淡出动画）  
        # self.play(  
        #     FadeOut(gradient_updates),  
        #     FadeOut(loss_function_new),    
        #     FadeOut(plane_matrix_equation),  
        #     FadeOut(z_equation)  
        # )  
        # self.wait(1)