import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
from matplotlib.figure import Figure

# 设置matplotlib显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class HighPassFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("频率域高通滤波器交互界面")
        self.root.geometry("1400x900")
        
        # 读取图像
        self.load_image()
        
        # 创建界面
        self.create_widgets()
        
        # 初始化显示原始图像
        self.show_original_image()
    
    def load_image(self):
        """加载图像"""
        try:
            # 可选择的图片列表
            self.available_images = ['p3-01.tif', 'p3-03.tif', 'p3-04.tif', 'p3-05.tif', 'p3-06.tif', 'p3-08.tif']
            self.current_image_path = 'p3-01.tif'  # 默认选择
            
            self.image = cv2.imread(self.current_image_path)
            if self.image is None:
                # 尝试其他图片
                for img_path in self.available_images:
                    self.image = cv2.imread(img_path)
                    if self.image is not None:
                        self.current_image_path = img_path
                        break
                
                if self.image is None:
                    messagebox.showerror("错误", f"无法读取任何图像文件: {self.available_images}")
                    self.root.quit()
                    return
            
            # 转换为灰度图像
            if len(self.image.shape) == 3:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                self.gray_image = self.image.copy()
                
            print(f"成功加载图像 {self.current_image_path}，尺寸: {self.gray_image.shape}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图像时出错: {str(e)}")
            self.root.quit()
    
    def create_widgets(self):
        """创建界面控件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 控制面板
        control_frame = ttk.LabelFrame(main_frame, text="滤波器控制", padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        # 标题标签
        title_label = ttk.Label(control_frame, text="频率域高通滤波器", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # 图像选择框架
        image_frame = ttk.Frame(control_frame)
        image_frame.pack(pady=5)
        
        ttk.Label(image_frame, text="选择图像:").pack(side=tk.LEFT, padx=(0, 5))
        self.image_var = tk.StringVar(value=self.current_image_path)
        self.image_combo = ttk.Combobox(image_frame, textvariable=self.image_var, 
                                       values=self.available_images, state="readonly", width=15)
        self.image_combo.pack(side=tk.LEFT, padx=5)
        self.image_combo.bind('<<ComboboxSelected>>', self.change_image)
        
        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        # 三个高通滤波器按钮
        self.ideal_btn = ttk.Button(button_frame, text="理想高通滤波器", 
                                  command=self.apply_ideal_filter, width=20)
        self.ideal_btn.pack(side=tk.LEFT, padx=5)
        
        self.butterworth_btn = ttk.Button(button_frame, text="巴特沃斯高通滤波器", 
                                        command=self.apply_butterworth_filter, width=20)
        self.butterworth_btn.pack(side=tk.LEFT, padx=5)
        
        self.gaussian_btn = ttk.Button(button_frame, text="高斯高通滤波器", 
                                     command=self.apply_gaussian_filter, width=20)
        self.gaussian_btn.pack(side=tk.LEFT, padx=5)
        
        # 重置和边缘增强按钮
        self.reset_btn = ttk.Button(button_frame, text="显示原图", 
                                  command=self.show_original_image, width=15)
        self.reset_btn.pack(side=tk.LEFT, padx=(20, 5))
        
        self.enhance_btn = ttk.Button(button_frame, text="边缘增强", 
                                    command=self.apply_edge_enhancement, width=15)
        self.enhance_btn.pack(side=tk.LEFT, padx=5)
        
        # 参数调节框架
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding="5")
        param_frame.pack(pady=(10, 0), fill=tk.X)
        
        # 截止频率滑块
        ttk.Label(param_frame, text="截止频率:").pack(side=tk.LEFT, padx=(0, 5))
        self.cutoff_var = tk.DoubleVar(value=15)
        self.cutoff_scale = ttk.Scale(param_frame, from_=5, to=50, 
                                    variable=self.cutoff_var, orient=tk.HORIZONTAL, length=200)
        self.cutoff_scale.pack(side=tk.LEFT, padx=5)
        self.cutoff_label = ttk.Label(param_frame, text="15")
        self.cutoff_label.pack(side=tk.LEFT, padx=(5, 20))
        
        # 阶数滑块（用于巴特沃斯滤波器）
        ttk.Label(param_frame, text="阶数(巴氏):").pack(side=tk.LEFT, padx=(0, 5))
        self.order_var = tk.DoubleVar(value=2)
        self.order_scale = ttk.Scale(param_frame, from_=1, to=5, 
                                   variable=self.order_var, orient=tk.HORIZONTAL, length=150)
        self.order_scale.pack(side=tk.LEFT, padx=5)
        self.order_label = ttk.Label(param_frame, text="2")
        self.order_label.pack(side=tk.LEFT, padx=(5, 20))
        
        # 标准差滑块（用于高斯滤波器）
        ttk.Label(param_frame, text="标准差(高斯):").pack(side=tk.LEFT, padx=(0, 5))
        self.sigma_var = tk.DoubleVar(value=25)
        self.sigma_scale = ttk.Scale(param_frame, from_=10, to=60, 
                                   variable=self.sigma_var, orient=tk.HORIZONTAL, length=150)
        self.sigma_scale.pack(side=tk.LEFT, padx=5)
        self.sigma_label = ttk.Label(param_frame, text="25")
        self.sigma_label.pack(side=tk.LEFT, padx=(5, 20))
        
        # 增强权重滑块
        ttk.Label(param_frame, text="增强权重:").pack(side=tk.LEFT, padx=(10, 5))
        self.weight_var = tk.DoubleVar(value=0.3)
        self.weight_scale = ttk.Scale(param_frame, from_=0.1, to=0.8, 
                                    variable=self.weight_var, orient=tk.HORIZONTAL, length=120)
        self.weight_scale.pack(side=tk.LEFT, padx=5)
        self.weight_label = ttk.Label(param_frame, text="0.3")
        self.weight_label.pack(side=tk.LEFT)
        
        # 绑定滑块事件
        self.cutoff_scale.configure(command=self.update_cutoff_label)
        self.order_scale.configure(command=self.update_order_label)
        self.sigma_scale.configure(command=self.update_sigma_label)
        self.weight_scale.configure(command=self.update_weight_label)
        
        # 图像显示区域
        self.create_plot_area(main_frame)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请选择高通滤波器")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
        
        # 存储当前滤波结果
        self.current_filtered = None
        self.current_filter_name = ""
    
    def create_plot_area(self, parent):
        """创建图像显示区域"""
        # 创建matplotlib图形
        self.fig = Figure(figsize=(14, 8), dpi=80)
        self.fig.suptitle('频率域高通滤波结果对比', fontsize=16)
        
        # 创建子图
        self.axes = []
        positions = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4),
                    (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
        titles = ['原始图像', '高通滤波结果', '滤波器模板', '边缘增强图像',
                 '原始频谱', '滤波后频谱', '频谱对比', '分析统计']
        
        for i, (pos, title) in enumerate(zip(positions, titles)):
            ax = self.fig.add_subplot(*pos)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            self.axes.append(ax)
        
        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def update_cutoff_label(self, value):
        """更新截止频率标签"""
        self.cutoff_label.config(text=f"{float(value):.0f}")
    
    def update_order_label(self, value):
        """更新阶数标签"""
        self.order_label.config(text=f"{float(value):.0f}")
    
    def update_sigma_label(self, value):
        """更新标准差标签"""
        self.sigma_label.config(text=f"{float(value):.0f}")
    
    def update_weight_label(self, value):
        """更新增强权重标签"""
        self.weight_label.config(text=f"{float(value):.1f}")
    
    def change_image(self, event):
        """切换图像"""
        new_image_path = self.image_var.get()
        try:
            new_image = cv2.imread(new_image_path)
            if new_image is not None:
                self.image = new_image
                self.current_image_path = new_image_path
                if len(self.image.shape) == 3:
                    self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
                else:
                    self.gray_image = self.image.copy()
                
                self.show_original_image()
                self.status_var.set(f"已切换到图像: {new_image_path}")
                print(f"切换到图像 {new_image_path}，尺寸: {self.gray_image.shape}")
            else:
                messagebox.showerror("错误", f"无法读取图像: {new_image_path}")
        except Exception as e:
            messagebox.showerror("错误", f"切换图像时出错: {str(e)}")
    
    def ideal_highpass_filter(self, rows, cols, cutoff_freq):
        """理想高通滤波器"""
        u = np.arange(rows) - rows // 2
        v = np.arange(cols) - cols // 2
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        H = np.ones((rows, cols))
        H[D <= cutoff_freq] = 0
        return H
    
    def butterworth_highpass_filter(self, rows, cols, cutoff_freq, n=2):
        """巴特沃斯高通滤波器"""
        u = np.arange(rows) - rows // 2
        v = np.arange(cols) - cols // 2
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        D[D == 0] = 1e-10
        H = 1 / (1 + (cutoff_freq / D)**(2 * n))
        return H
    
    def gaussian_highpass_filter(self, rows, cols, sigma):
        """高斯高通滤波器"""
        u = np.arange(rows) - rows // 2
        v = np.arange(cols) - cols // 2
        U, V = np.meshgrid(v, u)
        D_squared = U**2 + V**2
        H = 1 - np.exp(-D_squared / (2 * sigma**2))
        return H
    
    def apply_filter(self, filter_func, *params, filter_name=""):
        """应用滤波器"""
        try:
            self.status_var.set(f"正在处理 {filter_name}...")
            self.root.update()
            
            # 获取图像尺寸
            rows, cols = self.gray_image.shape
            
            # FFT变换
            f_transform = np.fft.fft2(self.gray_image)
            f_shift = np.fft.fftshift(f_transform)
            
            # 创建滤波器
            H = filter_func(rows, cols, *params)
            
            # 频域滤波
            g_shift = f_shift * H
            
            # 逆FFT
            g_transform = np.fft.ifftshift(g_shift)
            filtered_image = np.fft.ifft2(g_transform)
            filtered_image = np.real(filtered_image)
            
            # 对于高通滤波，需要特殊处理
            filtered_image = np.abs(filtered_image)
            # 归一化到0-255范围
            if filtered_image.max() > filtered_image.min():
                filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min()) * 255
            filtered_image = filtered_image.astype(np.uint8)
            
            # 存储当前结果
            self.current_filtered = filtered_image
            self.current_filter_name = filter_name
            
            # 显示结果
            self.display_results(self.gray_image, filtered_image, f_shift, g_shift, H, filter_name, params)
            
            self.status_var.set(f"{filter_name} 处理完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"滤波处理时出错: {str(e)}")
            self.status_var.set("处理出错")
    
    def display_results(self, original, filtered, original_spectrum, filtered_spectrum, filter_h, filter_name, params):
        """显示滤波结果"""
        # 清除之前的图像
        for ax in self.axes:
            ax.clear()
        
        # 1. 原始图像
        self.axes[0].imshow(original, cmap='gray')
        self.axes[0].set_title(f'原始图像\n({self.current_image_path})')
        self.axes[0].axis('off')
        
        # 2. 高通滤波结果
        self.axes[1].imshow(filtered, cmap='gray')
        self.axes[1].set_title(f'高通滤波结果\n({filter_name})')
        self.axes[1].axis('off')
        
        # 3. 滤波器模板
        self.axes[2].imshow(filter_h, cmap='gray')
        param_str = f"参数: {params}"
        self.axes[2].set_title(f'滤波器模板\n{param_str}')
        self.axes[2].axis('off')
        
        # 4. 边缘增强图像
        weight = self.weight_var.get()
        enhanced_image = cv2.addWeighted(original, 1-weight, filtered, weight, 0)
        self.axes[3].imshow(enhanced_image, cmap='gray')
        self.axes[3].set_title(f'边缘增强图像\n(权重:{weight:.1f})')
        self.axes[3].axis('off')
        
        # 5. 原始频谱
        spectrum_magnitude = np.log(np.abs(original_spectrum) + 1)
        self.axes[4].imshow(spectrum_magnitude, cmap='gray')
        self.axes[4].set_title('原始频谱')
        self.axes[4].axis('off')
        
        # 6. 滤波后频谱
        filtered_spectrum_magnitude = np.log(np.abs(filtered_spectrum) + 1)
        self.axes[5].imshow(filtered_spectrum_magnitude, cmap='gray')
        self.axes[5].set_title('滤波后频谱')
        self.axes[5].axis('off')
        
        # 7. 频谱对比
        spectrum_diff = filtered_spectrum_magnitude - spectrum_magnitude
        self.axes[6].imshow(spectrum_diff, cmap='RdBu')
        self.axes[6].set_title('频谱变化\n(蓝色减少,红色增加)')
        self.axes[6].axis('off')
        
        # 8. 统计信息
        self.axes[7].axis('on')
        self.axes[7].set_xlim(0, 1)
        self.axes[7].set_ylim(0, 1)
        
        # 计算边缘强度
        edge_strength = np.std(filtered)
        original_std = np.std(original)
        edge_enhancement = edge_strength / original_std if original_std > 0 else 0
        
        stats_text = f"""统计分析：
        
原始图像：
均值: {np.mean(original):.1f}
标准差: {np.std(original):.1f}
        
高通滤波：
均值: {np.mean(filtered):.1f}
标准差: {np.std(filtered):.1f}
边缘强度: {edge_strength:.1f}
        
效果评估：
边缘增强比: {edge_enhancement:.2f}
对比度增强: {(np.std(enhanced_image)/original_std*100):.1f}%
        
当前设置：
滤波器: {filter_name}
参数: {params}
增强权重: {weight:.1f}"""
        
        self.axes[7].text(0.05, 0.95, stats_text, transform=self.axes[7].transAxes,
                         verticalalignment='top', fontsize=8, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        self.axes[7].set_xticks([])
        self.axes[7].set_yticks([])
        self.axes[7].set_title('分析统计')
        
        # 更新显示
        self.canvas.draw()
    
    def show_original_image(self):
        """显示原始图像"""
        for ax in self.axes:
            ax.clear()
        
        # 只显示原始图像和其频谱
        self.axes[0].imshow(self.gray_image, cmap='gray')
        self.axes[0].set_title(f'原始图像\n({self.current_image_path})')
        self.axes[0].axis('off')
        
        # 原始频谱
        f_transform = np.fft.fft2(self.gray_image)
        f_shift = np.fft.fftshift(f_transform)
        spectrum_magnitude = np.log(np.abs(f_shift) + 1)
        self.axes[4].imshow(spectrum_magnitude, cmap='gray')
        self.axes[4].set_title('原始频谱')
        self.axes[4].axis('off')
        
        # 图像信息
        self.axes[7].axis('on')
        self.axes[7].set_xlim(0, 1)
        self.axes[7].set_ylim(0, 1)
        
        info_text = f"""图像信息：
        
当前文件: {self.current_image_path}
尺寸: {self.gray_image.shape}
        
统计特征：
均值: {np.mean(self.gray_image):.1f}
标准差: {np.std(self.gray_image):.1f}
最小值: {np.min(self.gray_image)}
最大值: {np.max(self.gray_image)}
        
高通滤波说明：
• 突出边缘和细节
• 去除低频背景
• 用于边缘检测
• 可进行图像锐化
        
操作指南：
1. 选择不同的测试图像
2. 调节滤波器参数
3. 点击滤波器按钮
4. 观察边缘检测效果"""
        
        self.axes[7].text(0.05, 0.95, info_text, transform=self.axes[7].transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))
        self.axes[7].set_xticks([])
        self.axes[7].set_yticks([])
        self.axes[7].set_title('图像信息')
        
        self.canvas.draw()
        self.status_var.set(f"显示原始图像: {self.current_image_path}")
    
    def apply_ideal_filter(self):
        """应用理想高通滤波器"""
        cutoff = self.cutoff_var.get()
        self.apply_filter(self.ideal_highpass_filter, cutoff, 
                         filter_name="理想高通滤波器")
    
    def apply_butterworth_filter(self):
        """应用巴特沃斯高通滤波器"""
        cutoff = self.cutoff_var.get()
        order = int(self.order_var.get())
        self.apply_filter(self.butterworth_highpass_filter, cutoff, order,
                         filter_name="巴特沃斯高通滤波器")
    
    def apply_gaussian_filter(self):
        """应用高斯高通滤波器"""
        sigma = self.sigma_var.get()
        self.apply_filter(self.gaussian_highpass_filter, sigma,
                         filter_name="高斯高通滤波器")
    
    def apply_edge_enhancement(self):
        """应用边缘增强"""
        if self.current_filtered is None:
            messagebox.showwarning("提示", "请先选择一个高通滤波器进行滤波处理")
            return
        
        try:
            weight = self.weight_var.get()
            enhanced_image = cv2.addWeighted(self.gray_image, 1-weight, self.current_filtered, weight, 0)
            
            # 显示增强结果
            self.display_edge_enhancement_result(enhanced_image, weight)
            self.status_var.set(f"边缘增强完成 (权重: {weight:.1f})")
            
        except Exception as e:
            messagebox.showerror("错误", f"边缘增强时出错: {str(e)}")
    
    def display_edge_enhancement_result(self, enhanced_image, weight):
        """显示边缘增强结果"""
        # 只更新相关的显示区域
        self.axes[3].clear()
        self.axes[3].imshow(enhanced_image, cmap='gray')
        self.axes[3].set_title(f'边缘增强图像\n(权重:{weight:.1f})')
        self.axes[3].axis('off')
        
        # 更新统计信息
        self.axes[7].clear()
        self.axes[7].axis('on')
        self.axes[7].set_xlim(0, 1)
        self.axes[7].set_ylim(0, 1)
        
        original_std = np.std(self.gray_image)
        enhanced_std = np.std(enhanced_image)
        contrast_improvement = (enhanced_std / original_std * 100) if original_std > 0 else 100
        
        stats_text = f"""边缘增强分析：
        
原始图像：
均值: {np.mean(self.gray_image):.1f}
标准差: {np.std(self.gray_image):.1f}
        
增强图像：
均值: {np.mean(enhanced_image):.1f}
标准差: {np.std(enhanced_image):.1f}
        
增强效果：
对比度提升: {contrast_improvement:.1f}%
增强权重: {weight:.1f}
        
使用的滤波器：
{self.current_filter_name}
        
说明：
• 权重越大边缘越突出
• 保留原图的同时增强细节
• 适合图像锐化应用"""
        
        self.axes[7].text(0.05, 0.95, stats_text, transform=self.axes[7].transAxes,
                         verticalalignment='top', fontsize=8,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.axes[7].set_xticks([])
        self.axes[7].set_yticks([])
        self.axes[7].set_title('增强分析')
        
        self.canvas.draw()

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = HighPassFilterGUI(root)
        
        # 设置窗口图标和样式
        try:
            root.state('zoomed')  # Windows最大化
        except:
            root.attributes('-zoomed', True)  # Linux最大化
        
        root.mainloop()
        
    except Exception as e:
        print(f"程序启动错误: {e}")
        messagebox.showerror("错误", f"程序启动失败: {str(e)}")

if __name__ == "__main__":
    main()