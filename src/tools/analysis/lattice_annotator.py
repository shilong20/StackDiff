"""
晶格标架标注工具

本工具用于研究人员手动标注2D材料图像中的晶格标架方向。
通过图形界面让用户在图像上点击三个关键点来定义dadb坐标系。

主要功能：
1. 图像加载和显示
2. 交互式点标注
3. 自动绘制晶格方向
4. 标架坐标系计算
5. 标注结果保存

使用说明：
1. 点击"选择图片"按钮加载要标注的图像
2. 在图像上依次点击三个点：
   - 第一个点：标架原点
   - 第二个点：da方向上的点
   - 第三个点：db方向上的点
3. 点击"确认标架方向"保存结果

作者: Moire项目组
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import json
import sys
from datetime import datetime


class LatticeAnnotator:
    """
    晶格标架标注工具类

    用于在2D材料图像上手动标注晶格标架的交互式工具。
    支持dadb坐标系的定义和保存。
    """

    def __init__(self, master):
        self.master = master
        self.master.title("晶格标架标注工具 - Moire项目")
        self.master.geometry("1400x900")  # 增加宽度以适应左右分栏
        self.master.configure(bg='#f0f0f0')

        # 设置样式
        style = ttk.Style()
        style.configure('TButton', font=('Arial', 10), padding=6)
        style.configure('TLabel', font=('Arial', 10))

        # 创建主框架
        self.main_frame = ttk.Frame(self.master, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 工具栏
        self.create_toolbar()

        # 状态栏
        self.create_status_bar()

        # 画布区域
        self.create_canvas_area()

        # 初始化变量
        self.reset_annotation()
        # 视图缩放与偏移（滚轮缩放支持）
        self.scale = 1.0
        self.offset_x = 10
        self.offset_y = 10
        self.min_scale = 0.1
        self.max_scale = 10.0

    def create_toolbar(self):
        """创建工具栏"""
        toolbar = ttk.Frame(self.main_frame)
        toolbar.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        # 文件操作
        file_frame = ttk.LabelFrame(toolbar, text="文件操作", padding="5")
        file_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.load_btn = ttk.Button(file_frame, text="📁 选择图片",
                                  command=self.load_image)
        self.load_btn.pack(side=tk.LEFT, padx=2)

        self.save_btn = ttk.Button(file_frame, text="💾 保存标注",
                                  command=self.save_annotation, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=2)

        # 标注操作
        annotation_frame = ttk.LabelFrame(toolbar, text="标注操作", padding="5")
        annotation_frame.pack(side=tk.LEFT, padx=(0, 10))

        self.confirm_btn = ttk.Button(annotation_frame, text="✅ 确认标架方向",
                                     command=self.confirm_annotation, state=tk.DISABLED)
        self.confirm_btn.pack(side=tk.LEFT, padx=2)

        self.clear_btn = ttk.Button(annotation_frame, text="🗑️ 清除标注",
                                   command=self.clear_annotation, state=tk.DISABLED)
        self.clear_btn.pack(side=tk.LEFT, padx=2)

        # 信息显示
        info_frame = ttk.LabelFrame(toolbar, text="标注信息", padding="5")
        info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.info_label = ttk.Label(info_frame, text="请先选择图片")
        self.info_label.pack(side=tk.LEFT)

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = ttk.Label(self.master, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def create_canvas_area(self):
        """创建画布区域"""
        # 创建左右分栏的主框架
        canvas_frame = ttk.Frame(self.main_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # 左侧标注区域
        left_frame = ttk.LabelFrame(canvas_frame, text="图像标注区域", padding="5")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # 创建左侧画布（加大默认尺寸，便于更大展示图像）
        self.canvas = tk.Canvas(left_frame, width=900, height=750,
                               bg='white', relief='sunken', borderwidth=2)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 绑定事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        # 鼠标滚轮缩放（Windows/macOS）
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        # 鼠标滚轮缩放（Linux X11）
        self.canvas.bind("<Button-4>", self.on_mouse_wheel_linux_up)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel_linux_down)

        # 创建说明文本
        instruction_text = """使用说明：
1. 选择要标注的图像
2. 在图像上依次点击三个点：
   • 蓝色点：标架原点
   • 绿色点：da方向参考点
   • 红色点：db方向参考点
3. 系统自动绘制标架方向
4. 确认并保存标注结果"""
        instruction_label = ttk.Label(left_frame, text=instruction_text,
                                    justify=tk.LEFT, font=('Arial', 9))
        instruction_label.pack(fill=tk.X, pady=(10, 0))

        # 右侧参考图片区域
        right_frame = ttk.LabelFrame(canvas_frame, text="晶格标架参考", padding="5")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # 创建右侧画布用于显示参考图片
        self.reference_canvas = tk.Canvas(right_frame, width=400, height=600,
                                        bg='#f8f8f8', relief='sunken', borderwidth=2)
        self.reference_canvas.pack(fill=tk.BOTH, expand=True)

        # 加载并显示参考图片
        self.load_reference_image()

        # 参考说明
        reference_text = """参考说明：
• da方向：晶格的a轴方向
• db方向：晶格的b轴方向
• 标架原点：晶格的参考原点

请参考右侧标准标架进行标注"""
        reference_label = ttk.Label(right_frame, text=reference_text,
                                  justify=tk.LEFT, font=('Arial', 9))
        reference_label.pack(fill=tk.X, pady=(10, 0))

    def load_reference_image(self):
        """加载并显示参考图片"""
        try:
            # 加载参考图片
            reference_path = "dadb_ref.jpg"
            if os.path.exists(reference_path):
                self.reference_image = Image.open(reference_path).convert('RGB')

                # 调整参考图片大小以适应右侧区域
                max_size = (380, 500)
                self.reference_image.thumbnail(max_size, Image.Resampling.LANCZOS)

                # 创建tkinter图像
                self.reference_image_tk = ImageTk.PhotoImage(self.reference_image)

                # 显示参考图片
                self.reference_canvas.create_image(
                    200, 250,  # 居中显示
                    anchor=tk.CENTER,
                    image=self.reference_image_tk
                )

                # 添加标题
                self.reference_canvas.create_text(
                    200, 20,
                    text="标准晶格标架参考",
                    font=('Arial', 12, 'bold'),
                    fill='#333333'
                )

                self.update_status("参考图片加载成功")
            else:
                # 如果找不到参考图片，显示提示信息
                self.reference_canvas.create_text(
                    200, 250,
                    text="未找到参考图片\ndadb_ref.jpg",
                    font=('Arial', 10),
                    fill='#666666',
                    justify=tk.CENTER
                )
                self.update_status("未找到参考图片")

        except Exception as e:
            print(f"加载参考图片时发生错误: {e}")
            self.reference_canvas.create_text(
                200, 250,
                text=f"加载参考图片失败:\n{str(e)}",
                font=('Arial', 10),
                fill='#ff0000',
                justify=tk.CENTER
            )

    def reset_annotation(self):
        """重置标注状态"""
        self.image = None
        self.image_tk = None
        self.file_name = None
        self.annotation_points = []
        self.canvas_items = []
        self.annotation_confirmed = False

    def load_image(self):
        """加载图像文件"""
        try:
            file_path = filedialog.askopenfilename(
                title="选择要标注的图像",
                filetypes=[
                    ("图像文件", "*.png *.jpg *.jpeg *.bmp *.tiff"),
                    ("PNG文件", "*.png"),
                    ("JPEG文件", "*.jpg *.jpeg"),
                    ("所有文件", "*.*")
                ]
            )

            if not file_path:
                return

            # 在加载新图像前重置标注状态，避免清空已创建的图像引用
            # 注意：必须在创建 PhotoImage 之前调用，否则会导致图像对象被释放后画布显示空白
            self.reset_annotation()

            # 解析文件名
            str_list = file_path.split('/')
            file_name_list = str_list[-1].split('.')
            base_name = file_name_list[0]
            self.file_name = base_name.split('_')[0]

            # 加载原始图像（保持原尺寸，显示时按scale缩放）
            self.image = Image.open(file_path).convert('L')

            # 计算适配画布的初始缩放，使图像尽可能大地显示（保留10px边距）
            self.canvas.update_idletasks()
            available_w = max(800, self.canvas.winfo_width() - 20)
            available_h = max(600, self.canvas.winfo_height() - 20)
            scale_w = available_w / self.image.width
            scale_h = available_h / self.image.height
            self.scale = min(scale_w, scale_h, 1.0)  # 初始不放大超过原图
            self.offset_x, self.offset_y = 10, 10

            # 初次绘制
            self.redraw_canvas()

            # 初始化标注状态（不再重置图像引用）
            self.annotation_points = []
            self.annotation_confirmed = False

            # 更新界面状态
            self.save_btn.config(state=tk.DISABLED)
            self.confirm_btn.config(state=tk.NORMAL)
            self.clear_btn.config(state=tk.NORMAL)
            self.info_label.config(text=f"已加载图片: {base_name} | 请在图像上点击三个点进行标注")

            self.update_status("图像加载成功，请开始标注")

        except Exception as e:
            messagebox.showerror("错误", f"加载图像时发生错误:\n{str(e)}")
            self.update_status("加载图像失败")

    def on_canvas_click(self, event):
        """处理画布点击事件"""
        if not self.image or self.annotation_confirmed:
            return

        # 获取画布坐标（考虑图像偏移）
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # 当前显示尺寸
        disp_w = int(self.image.width * self.scale)
        disp_h = int(self.image.height * self.scale)
        # 检查是否在图像范围内
        if not (self.offset_x <= canvas_x <= self.offset_x + disp_w and
                self.offset_y <= canvas_y <= self.offset_y + disp_h):
            return

        # 转换为原图坐标
        img_x = (canvas_x - self.offset_x) / self.scale
        img_y = (canvas_y - self.offset_y) / self.scale

        # 添加标注点
        self.annotation_points.append([img_x, img_y])

        # 绘制点和线
        self.draw_annotation()

        # 更新信息
        self.update_annotation_info()

    def draw_annotation(self):
        """绘制标注图形"""
        # 清除之前的标注图形
        for item in self.canvas_items:
            self.canvas.delete(item)
        self.canvas_items = []

        points = self.annotation_points
        colors = ['blue', 'green', 'red']  # 原点、da方向、db方向

        # 绘制点（半径=2，按缩放和偏移投影到画布）
        for i, (x, y) in enumerate(points):
            if i < len(colors):
                canvas_x = self.offset_x + x * self.scale
                canvas_y = self.offset_y + y * self.scale
                # 绘制点
                item = self.canvas.create_oval(
                    canvas_x-2, canvas_y-2, canvas_x+2, canvas_y+2,
                    fill=colors[i], outline=colors[i]
                )
                self.canvas_items.append(item)

                # 绘制标签
                item = self.canvas.create_text(
                    canvas_x+10, canvas_y-10, text=f"点{i+1}",
                    fill=colors[i], font=('Arial', 8, 'bold')
                )
                self.canvas_items.append(item)

        # 绘制连接线
        if len(points) >= 2:
            # da方向线（蓝色）
            canvas_x1 = self.offset_x + points[0][0] * self.scale
            canvas_y1 = self.offset_y + points[0][1] * self.scale
            canvas_x2 = self.offset_x + points[1][0] * self.scale
            canvas_y2 = self.offset_y + points[1][1] * self.scale
            item = self.canvas.create_line(
                canvas_x1, canvas_y1, canvas_x2, canvas_y2,
                fill='blue', width=3, arrow=tk.LAST, arrowshape=(8, 10, 3)
            )
            self.canvas_items.append(item)

            # 添加da方向标签
            mid_x = (canvas_x1 + canvas_x2) / 2
            mid_y = (canvas_y1 + canvas_y2) / 2
            item = self.canvas.create_text(
                mid_x, mid_y - 15, text="da方向",
                fill='blue', font=('Arial', 9, 'bold')
            )
            self.canvas_items.append(item)

        if len(points) >= 3:
            # db方向线（红色）
            canvas_x1 = self.offset_x + points[0][0] * self.scale
            canvas_y1 = self.offset_y + points[0][1] * self.scale
            canvas_x3 = self.offset_x + points[2][0] * self.scale
            canvas_y3 = self.offset_y + points[2][1] * self.scale
            item = self.canvas.create_line(
                canvas_x1, canvas_y1, canvas_x3, canvas_y3,
                fill='red', width=3, arrow=tk.LAST, arrowshape=(8, 10, 3)
            )
            self.canvas_items.append(item)

            # 添加db方向标签
            mid_x = (canvas_x1 + canvas_x3) / 2
            mid_y = (canvas_y1 + canvas_y3) / 2
            item = self.canvas.create_text(
                mid_x, mid_y + 15, text="db方向",
                fill='red', font=('Arial', 9, 'bold')
            )
            self.canvas_items.append(item)

    def update_annotation_info(self):
        """更新标注信息"""
        if len(self.annotation_points) == 0:
            self.info_label.config(text="请在图像上点击第一个点（标架原点）")
        elif len(self.annotation_points) == 1:
            self.info_label.config(text="请点击第二个点（da方向参考点）")
        elif len(self.annotation_points) == 2:
            self.info_label.config(text="请点击第三个点（db方向参考点）")
        else:
            self.info_label.config(text="标注完成，点击'确认标架方向'保存")

    def confirm_annotation(self):
        """确认标注"""
        if len(self.annotation_points) < 3:
            messagebox.showwarning("警告", "请先完成三个点的标注")
            return

        try:
            # 计算标架向量
            origin = np.array(self.annotation_points[0])
            da_point = np.array(self.annotation_points[1])
            db_point = np.array(self.annotation_points[2])

            da_direction = da_point - origin
            db_direction = db_point - origin

            # 计算角度验证
            angle = np.arccos(np.dot(da_direction, db_direction) /
                            (np.linalg.norm(da_direction) * np.linalg.norm(db_direction)))
            angle_deg = np.degrees(angle)

            # 显示确认对话框
            msg = f"""标架标注确认：

da方向向量: [{da_direction[0]:.1f}, {da_direction[1]:.1f}]
db方向向量: [{db_direction[0]:.1f}, {db_direction[1]:.1f}]
da-db角度: {angle_deg:.1f}°

确定要保存这个标架吗？"""

            if messagebox.askyesno("确认标注", msg):
                self.save_annotation_data(da_direction, db_direction)
                self.annotation_confirmed = True
                self.confirm_btn.config(state=tk.DISABLED)
                self.save_btn.config(state=tk.NORMAL)
                self.update_status("标架标注已确认，可选择保存")

        except Exception as e:
            messagebox.showerror("错误", f"确认标注时发生错误:\n{str(e)}")

    def save_annotation_data(self, da_direction, db_direction):
        """保存标注数据"""
        try:
            # 创建标注数据
            annotation_data = {
                self.file_name: [
                    da_direction.tolist(),
                    db_direction.tolist()
                ]
            }

            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"biaozhu_{self.file_name}_{timestamp}.txt"

            # 保存为JSON格式
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)

            messagebox.showinfo("保存成功", f"标架标注已保存到:\n{filename}")
            self.update_status(f"标注数据已保存到 {filename}")

        except Exception as e:
            messagebox.showerror("保存错误", f"保存标注数据时发生错误:\n{str(e)}")

    def save_annotation(self):
        """保存标注结果到标准文件"""
        try:
            if len(self.annotation_points) < 3:
                messagebox.showwarning("警告", "没有有效的标注数据")
                return

            origin = np.array(self.annotation_points[0])
            da_point = np.array(self.annotation_points[1])
            db_point = np.array(self.annotation_points[2])

            da_direction = da_point - origin
            db_direction = db_point - origin

            # 追加到biaozhu.txt文件
            annotation_data = {
                self.file_name: [
                    da_direction.tolist(),
                    db_direction.tolist()
                ]
            }

            with open('biaozhu.txt', 'a', encoding='utf-8') as f:
                f.write(str(annotation_data) + '\n')

            messagebox.showinfo("保存成功", "标架标注已追加到 biaozhu.txt")
            self.update_status("标注数据已保存到 biaozhu.txt")

        except Exception as e:
            messagebox.showerror("保存错误", f"保存到biaozhu.txt时发生错误:\n{str(e)}")

    def clear_annotation(self):
        """清除当前标注"""
        if messagebox.askyesno("确认清除", "确定要清除当前的标注吗？"):
            # 清除画布上的标注图形
            for item in self.canvas_items:
                self.canvas.delete(item)
            self.canvas_items = []

            # 重置标注状态
            self.annotation_points = []
            self.annotation_confirmed = False

            # 更新界面状态
            self.confirm_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)
            self.update_annotation_info()

            self.update_status("标注已清除")

    def on_mouse_move(self, event):
        """处理鼠标移动事件"""
        if self.image:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)

            disp_w = int(self.image.width * self.scale)
            disp_h = int(self.image.height * self.scale)
            if (self.offset_x <= canvas_x <= self.offset_x + disp_w and
                self.offset_y <= canvas_y <= self.offset_y + disp_h):
                img_x = (canvas_x - self.offset_x) / self.scale
                img_y = (canvas_y - self.offset_y) / self.scale
                self.status_bar.config(text=f"图像坐标: ({img_x:.0f}, {img_y:.0f})")
            else:
                self.status_bar.config(text="就绪")

    def update_status(self, message):
        """更新状态栏"""
        self.status_bar.config(text=message)
        self.master.update_idletasks()

    # ============== 画布与缩放相关 ==============
    def redraw_canvas(self):
        """根据当前缩放与偏移重绘图像和标注"""
        if not self.image:
            return
        # 生成缩放后的图像
        disp_w = max(1, int(self.image.width * self.scale))
        disp_h = max(1, int(self.image.height * self.scale))
        display_image = self.image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(display_image)

        # 全部清除并重绘
        self.canvas.delete("all")
        self.canvas_items = []
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.image_tk)
        # 叠加标注
        self.draw_annotation()

    def on_mouse_wheel(self, event):
        """鼠标滚轮缩放（Windows/macOS）"""
        if not self.image:
            return
        delta = 1 if event.delta > 0 else -1
        self._zoom(delta, event.x, event.y)

    def on_mouse_wheel_linux_up(self, event):
        if not self.image:
            return
        self._zoom(1, event.x, event.y)

    def on_mouse_wheel_linux_down(self, event):
        if not self.image:
            return
        self._zoom(-1, event.x, event.y)

    def _zoom(self, direction, cx, cy):
        """以鼠标位置为中心缩放 direction=1 放大，-1 缩小"""
        factor = 1.1 if direction > 0 else 0.9
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        if abs(new_scale - self.scale) < 1e-6:
            return

        canvas_x = self.canvas.canvasx(cx)
        canvas_y = self.canvas.canvasy(cy)
        # 转为图像坐标
        img_x = (canvas_x - self.offset_x) / self.scale
        img_y = (canvas_y - self.offset_y) / self.scale

        # 更新缩放
        self.scale = new_scale

        # 根据新缩放保持鼠标处图像点位置不变
        self.offset_x = canvas_x - img_x * self.scale
        self.offset_y = canvas_y - img_y * self.scale

        # 若图像比画布小，尽量保持10像素边距
        disp_w = int(self.image.width * self.scale)
        disp_h = int(self.image.height * self.scale)
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        if disp_w + 20 <= canvas_w:
            self.offset_x = max(10, min(self.offset_x, canvas_w - disp_w - 10))
        if disp_h + 20 <= canvas_h:
            self.offset_y = max(10, min(self.offset_y, canvas_h - disp_h - 10))

        self.redraw_canvas()


def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = LatticeAnnotator(root)
        root.mainloop()
    except Exception as e:
        print(f"启动应用程序时发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
