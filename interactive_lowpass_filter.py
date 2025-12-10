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

class LowPassFilterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("频率域低通滤波器交互界面")
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
            self.image = cv2.imread('p3-02.bmp')
            if self.image is None:
                messagebox.showerror("错误", "无法读取图像 p3-02.bmp")
                self.root.quit()
                return
            
            # 转换为灰度图像
            if len(self.image.shape) == 3:
                self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            else:
                self.gray_image = self.image.copy()
                
            print(f"成功加载图像，尺寸: {self.gray_image.shape}")
            
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
        title_label = ttk.Label(control_frame, text="频率域低通滤波器", font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # 按钮框架
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(pady=10)
        
        # 三个滤波器按钮
        self.ideal_btn = ttk.Button(button_frame, text="理想低通滤波器", 
                                  command=self.apply_ideal_filter, width=20)
        self.ideal_btn.pack(side=tk.LEFT, padx=5)
        
        self.butterworth_btn = ttk.Button(button_frame, text="巴特沃斯滤波器", 
                                        command=self.apply_butterworth_filter, width=20)
        self.butterworth_btn.pack(side=tk.LEFT, padx=5)
        
        self.gaussian_btn = ttk.Button(button_frame, text="高斯滤波器", 
                                     command=self.apply_gaussian_filter, width=20)
        self.gaussian_btn.pack(side=tk.LEFT, padx=5)
        
        # 重置按钮
        self.reset_btn = ttk.Button(button_frame, text="显示原图", 
                                  command=self.show_original_image, width=15)
        self.reset_btn.pack(side=tk.LEFT, padx=(20, 5))
        
        # 参数调节框架
        param_frame = ttk.LabelFrame(control_frame, text="参数调节", padding="5")
        param_frame.pack(pady=(10, 0), fill=tk.X)
        
        # 截止频率滑块
        ttk.Label(param_frame, text="截止频率:").pack(side=tk.LEFT, padx=(0, 5))
        self.cutoff_var = tk.DoubleVar(value=30)
        self.cutoff_scale = ttk.Scale(param_frame, from_=5, to=100, 
                                    variable=self.cutoff_var, orient=tk.HORIZONTAL, length=200)
        self.cutoff_scale.pack(side=tk.LEFT, padx=5)
        self.cutoff_label = ttk.Label(param_frame, text="30")
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
        self.sigma_var = tk.DoubleVar(value=20)
        self.sigma_scale = ttk.Scale(param_frame, from_=5, to=50, 
                                   variable=self.sigma_var, orient=tk.HORIZONTAL, length=150)
        self.sigma_scale.pack(side=tk.LEFT, padx=5)
        self.sigma_label = ttk.Label(param_frame, text="20")
        self.sigma_label.pack(side=tk.LEFT)
        
        # 绑定滑块事件
        self.cutoff_scale.configure(command=self.update_cutoff_label)
        self.order_scale.configure(command=self.update_order_label)
        self.sigma_scale.configure(command=self.update_sigma_label)
        
        # 图像显示区域
        self.create_plot_area(main_frame)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请选择滤波器")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    
    def create_plot_area(self, parent):
        """创建图像显示区域"""
        # 创建matplotlib图形
        self.fig = Figure(figsize=(14, 8), dpi=80)
        self.fig.suptitle('频率域低通滤波结果对比', fontsize=16)
        
        # 创建子图
        self.axes = []
        positions = [(2, 4, 1), (2, 4, 2), (2, 4, 3), (2, 4, 4),
                    (2, 4, 5), (2, 4, 6), (2, 4, 7), (2, 4, 8)]
        titles = ['原始图像', '滤波后图像', '滤波器模板', '差值图像',
                 '原始频谱', '滤波后频谱', '频谱差值', '图像统计']
        
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
    
    def ideal_lowpass_filter(self, rows, cols, cutoff_freq):
        """理想低通滤波器"""
        u = np.arange(rows) - rows // 2
        v = np.arange(cols) - cols // 2
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        H = np.zeros((rows, cols))
        H[D <= cutoff_freq] = 1
        return H
    
    def butterworth_lowpass_filter(self, rows, cols, cutoff_freq, n=2):
        """巴特沃斯低通滤波器"""
        u = np.arange(rows) - rows // 2
        v = np.arange(cols) - cols // 2
        U, V = np.meshgrid(v, u)
        D = np.sqrt(U**2 + V**2)
        D[D == 0] = 1e-10
        H = 1 / (1 + (D / cutoff_freq)**(2 * n))
        return H
    
    def gaussian_lowpass_filter(self, rows, cols, sigma):
        """高斯低通滤波器"""
        u = np.arange(rows) - rows // 2
        v = np.arange(cols) - cols // 2
        U, V = np.meshgrid(v, u)
        D_squared = U**2 + V**2
        H = np.exp(-D_squared / (2 * sigma**2))
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
            filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)
            
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
        self.axes[0].set_title('原始图像')
        self.axes[0].axis('off')
        
        # 2. 滤波后图像
        self.axes[1].imshow(filtered, cmap='gray')
        self.axes[1].set_title(f'滤波后图像\n({filter_name})')
        self.axes[1].axis('off')
        
        # 3. 滤波器模板
        self.axes[2].imshow(filter_h, cmap='gray')
        param_str = f"参数: {params}"
        self.axes[2].set_title(f'滤波器模板\n{param_str}')
        self.axes[2].axis('off')
        
        # 4. 差值图像
        diff_image = np.abs(original.astype(float) - filtered.astype(float))
        self.axes[3].imshow(diff_image, cmap='hot')
        self.axes[3].set_title('差值图像\n(原图-滤波图)')
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
        
        # 7. 频谱差值
        spectrum_diff = spectrum_magnitude - filtered_spectrum_magnitude
        self.axes[6].imshow(spectrum_diff, cmap='hot')
        self.axes[6].set_title('频谱差值')
        self.axes[6].axis('off')
        
        # 8. 统计信息
        self.axes[7].axis('on')
        self.axes[7].set_xlim(0, 1)
        self.axes[7].set_ylim(0, 1)
        
        stats_text = f"""统计信息：
        
原始图像：
均值: {np.mean(original):.1f}
标准差: {np.std(original):.1f}
        
滤波后：
均值: {np.mean(filtered):.1f}
标准差: {np.std(filtered):.1f}
        
变化：
平滑度: {(np.std(filtered)/np.std(original)*100):.1f}%
        
滤波器: {filter_name}
参数: {params}"""
        
        self.axes[7].text(0.05, 0.95, stats_text, transform=self.axes[7].transAxes,
                         verticalalignment='top', fontsize=9, 
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.axes[7].set_xticks([])
        self.axes[7].set_yticks([])
        self.axes[7].set_title('统计信息')
        
        # 更新显示
        self.canvas.draw()
    
    def show_original_image(self):
        """显示原始图像"""
        for ax in self.axes:
            ax.clear()
        
        # 只显示原始图像和其频谱
        self.axes[0].imshow(self.gray_image, cmap='gray')
        self.axes[0].set_title('原始图像 (p3-02.bmp)')
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
        
文件: p3-02.bmp
尺寸: {self.gray_image.shape}
        
统计：
均值: {np.mean(self.gray_image):.1f}
标准差: {np.std(self.gray_image):.1f}
最小值: {np.min(self.gray_image)}
最大值: {np.max(self.gray_image)}
        
操作指南：
1. 调节参数滑块
2. 点击滤波器按钮
3. 观察滤波效果"""
        
        self.axes[7].text(0.05, 0.95, info_text, transform=self.axes[7].transAxes,
                         verticalalignment='top', fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.axes[7].set_xticks([])
        self.axes[7].set_yticks([])
        self.axes[7].set_title('图像信息')
        
        self.canvas.draw()
        self.status_var.set("显示原始图像")
    
    def apply_ideal_filter(self):
        """应用理想低通滤波器"""
        cutoff = self.cutoff_var.get()
        self.apply_filter(self.ideal_lowpass_filter, cutoff, 
                         filter_name="理想低通滤波器")
    
    def apply_butterworth_filter(self):
        """应用巴特沃斯滤波器"""
        cutoff = self.cutoff_var.get()
        order = int(self.order_var.get())
        self.apply_filter(self.butterworth_lowpass_filter, cutoff, order,
                         filter_name="巴特沃斯低通滤波器")
    
    def apply_gaussian_filter(self):
        """应用高斯滤波器"""
        sigma = self.sigma_var.get()
        self.apply_filter(self.gaussian_lowpass_filter, sigma,
                         filter_name="高斯低通滤波器")

def main():
    """主函数"""
    try:
        root = tk.Tk()
        app = LowPassFilterGUI(root)
        
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