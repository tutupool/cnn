import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import matplotlib

# 设置matplotlib显示中文
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def ideal_highpass_filter(rows, cols, cutoff_freq):
    """
    创建理想高通滤波器
    参数:
        rows: 图像行数
        cols: 图像列数
        cutoff_freq: 截止频率
    返回:
        滤波器矩阵
    """
    # 创建频率网格
    u = np.arange(rows)
    v = np.arange(cols)
    
    # 将频率原点移到中心
    u = u - rows // 2
    v = v - cols // 2
    
    # 创建频率矩阵
    U, V = np.meshgrid(v, u)
    
    # 计算距离矩阵
    D = np.sqrt(U**2 + V**2)
    
    # 创建理想高通滤波器 (与低通滤波器相反)
    H = np.ones((rows, cols))
    H[D <= cutoff_freq] = 0
    
    return H

def butterworth_highpass_filter(rows, cols, cutoff_freq, n=2):
    """
    创建巴特沃斯高通滤波器
    参数:
        rows: 图像行数
        cols: 图像列数
        cutoff_freq: 截止频率
        n: 滤波器阶数
    返回:
        滤波器矩阵
    """
    # 创建频率网格
    u = np.arange(rows)
    v = np.arange(cols)
    
    # 将频率原点移到中心
    u = u - rows // 2
    v = v - cols // 2
    
    # 创建频率矩阵
    U, V = np.meshgrid(v, u)
    
    # 计算距离矩阵
    D = np.sqrt(U**2 + V**2)
    
    # 避免除零错误
    D[D == 0] = 1e-10
    
    # 创建巴特沃斯高通滤波器
    H = 1 / (1 + (cutoff_freq / D)**(2 * n))
    
    return H

def gaussian_highpass_filter(rows, cols, sigma):
    """
    创建高斯高通滤波器
    参数:
        rows: 图像行数
        cols: 图像列数
        sigma: 标准差参数
    返回:
        滤波器矩阵
    """
    # 创建频率网格
    u = np.arange(rows)
    v = np.arange(cols)
    
    # 将频率原点移到中心
    u = u - rows // 2
    v = v - cols // 2
    
    # 创建频率矩阵
    U, V = np.meshgrid(v, u)
    
    # 计算距离矩阵的平方
    D_squared = U**2 + V**2
    
    # 创建高斯高通滤波器 (1 - 高斯低通)
    H = 1 - np.exp(-D_squared / (2 * sigma**2))
    
    return H

def frequency_domain_filter(image, filter_func, *filter_params):
    """
    频率域滤波主函数
    参数:
        image: 输入图像
        filter_func: 滤波器函数
        filter_params: 滤波器参数
    返回:
        滤波后的图像和相关频谱信息
    """
    # 如果是彩色图像，转换为灰度图像
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 获取图像尺寸
    rows, cols = image.shape
    
    # 步骤1: 进行FFT变换
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)  # 移动零频率到中心
    
    # 步骤2: 创建滤波器
    H = filter_func(rows, cols, *filter_params)
    
    # 步骤3: 在频率域进行滤波
    g_shift = f_shift * H
    
    # 步骤4: 逆FFT变换
    g_transform = np.fft.ifftshift(g_shift)
    filtered_image = np.fft.ifft2(g_transform)
    filtered_image = np.real(filtered_image)  # 取实部
    
    # 对于高通滤波，需要进行对比度增强
    filtered_image = np.abs(filtered_image)  # 取绝对值
    filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min()) * 255
    filtered_image = filtered_image.astype(np.uint8)
    
    return {
        'original_image': image,
        'filtered_image': filtered_image,
        'original_spectrum': f_shift,
        'filtered_spectrum': g_shift,
        'filter': H
    }

def display_results(results, filter_name, filter_params):
    """
    显示滤波结果
    """
    plt.figure(figsize=(15, 10))
    
    # 原始图像
    plt.subplot(2, 4, 1)
    plt.imshow(results['original_image'], cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    
    # 滤波后图像
    plt.subplot(2, 4, 2)
    plt.imshow(results['filtered_image'], cmap='gray')
    plt.title(f'滤波后图像\n({filter_name})')
    plt.axis('off')
    
    # 滤波器
    plt.subplot(2, 4, 3)
    plt.imshow(results['filter'], cmap='gray')
    plt.title(f'滤波器模板\n参数: {filter_params}')
    plt.axis('off')
    
    # 边缘增强图像
    enhanced_image = cv2.addWeighted(results['original_image'], 0.7, results['filtered_image'], 0.3, 0)
    plt.subplot(2, 4, 4)
    plt.imshow(enhanced_image, cmap='gray')
    plt.title('边缘增强图像\n(原图+高通结果)')
    plt.axis('off')
    
    # 原始频谱
    plt.subplot(2, 4, 5)
    spectrum_magnitude = np.log(np.abs(results['original_spectrum']) + 1)
    plt.imshow(spectrum_magnitude, cmap='gray')
    plt.title('原始频谱')
    plt.axis('off')
    
    # 滤波后频谱
    plt.subplot(2, 4, 6)
    filtered_spectrum_magnitude = np.log(np.abs(results['filtered_spectrum']) + 1)
    plt.imshow(filtered_spectrum_magnitude, cmap='gray')
    plt.title('滤波后频谱')
    plt.axis('off')
    
    # 频谱对比
    plt.subplot(2, 4, 7)
    spectrum_diff = filtered_spectrum_magnitude - spectrum_magnitude
    plt.imshow(spectrum_diff, cmap='hot')
    plt.title('频谱变化')
    plt.axis('off')
    
    # 滤波器3D视图
    plt.subplot(2, 4, 8)
    x = np.arange(results['filter'].shape[1])
    y = np.arange(results['filter'].shape[0])
    X, Y = np.meshgrid(x[::20], y[::20])  # 采样显示
    Z = results['filter'][::20, ::20]
    plt.contour(X, Y, Z, levels=10)
    plt.title('滤波器等高线图')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def analyze_image_properties(original, filtered, filter_name):
    """
    分析图像特性
    """
    print(f"\n=== {filter_name} 滤波分析结果 ===")
    print(f"原始图像:")
    print(f"  - 均值: {np.mean(original):.2f}")
    print(f"  - 标准差: {np.std(original):.2f}")
    print(f"  - 最小值: {np.min(original)}")
    print(f"  - 最大值: {np.max(original)}")
    
    print(f"\n滤波后图像:")
    print(f"  - 均值: {np.mean(filtered):.2f}")
    print(f"  - 标准差: {np.std(filtered):.2f}")
    print(f"  - 最小值: {np.min(filtered)}")
    print(f"  - 最大值: {np.max(filtered)}")
    
    # 计算边缘强度
    edge_strength = np.std(filtered)
    print(f"  - 边缘强度: {edge_strength:.2f}")
    
    # 计算对比度增强
    original_contrast = np.std(original)
    filtered_contrast = np.std(filtered)
    contrast_ratio = filtered_contrast / original_contrast if original_contrast > 0 else 0
    print(f"  - 对比度增强比: {contrast_ratio:.2f}")

def main():
    """
    主函数 - 对自选图片进行高通滤波处理
    """
    # 可选择的图片列表
    available_images = ['p3-01.tif', 'p3-03.tif', 'p3-04.tif', 'p3-05.tif', 'p3-06.tif', 'p3-08.tif']
    
    # 选择一张图片进行处理（这里选择p3-01.tif）
    image_path = 'p3-01.tif'
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"错误：无法读取图像 {image_path}")
        print(f"可用图片: {available_images}")
        return
    
    print("开始频率域高通滤波实验...")
    print(f"选择的图像: {image_path}")
    print(f"图像尺寸: {image.shape}")
    
    # 定义三种高通滤波器及其参数
    filters = [
        {
            'name': '理想高通滤波器',
            'func': ideal_highpass_filter,
            'params': (15,),  # 截止频率15
            'description': '截止频率=15像素'
        },
        {
            'name': '巴特沃斯高通滤波器',
            'func': butterworth_highpass_filter,
            'params': (15, 2),  # 截止频率15，阶数2
            'description': '截止频率=15像素，阶数=2'
        },
        {
            'name': '高斯高通滤波器',
            'func': gaussian_highpass_filter,
            'params': (25,),  # 标准差25
            'description': '标准差σ=25'
        }
    ]
    
    # 对每种滤波器进行处理和分析
    for i, filter_info in enumerate(filters):
        print(f"\n{'='*50}")
        print(f"正在处理: {filter_info['name']}")
        print(f"参数: {filter_info['description']}")
        
        # 进行滤波
        results = frequency_domain_filter(
            image.copy(), 
            filter_info['func'], 
            *filter_info['params']
        )
        
        # 显示结果
        display_results(results, filter_info['name'], filter_info['description'])
        
        # 分析图像特性
        analyze_image_properties(
            results['original_image'], 
            results['filtered_image'], 
            filter_info['name']
        )
        
        # 保存滤波后的图像
        output_filename = f'{image_path.split(".")[0]}_highpass_{i+1}_{filter_info["name"].replace(" ", "_")}.bmp'
        cv2.imwrite(output_filename, results['filtered_image'])
        print(f"高通滤波图像已保存为: {output_filename}")
        
        # 保存边缘增强图像
        enhanced_image = cv2.addWeighted(results['original_image'], 0.7, results['filtered_image'], 0.3, 0)
        enhanced_filename = f'{image_path.split(".")[0]}_enhanced_{i+1}_{filter_info["name"].replace(" ", "_")}.bmp'
        cv2.imwrite(enhanced_filename, enhanced_image)
        print(f"边缘增强图像已保存为: {enhanced_filename}")
    
    print(f"\n{'='*50}")
    print("频率域高通滤波实验完成！")
    print("\n实验总结:")
    print("1. 理想高通滤波器：能够突出边缘，但会产生振铃效应")
    print("2. 巴特沃斯高通滤波器：边缘检测效果好，振铃效应较小")
    print("3. 高斯高通滤波器：平滑的边缘检测，无振铃效应")
    print("4. 高通滤波主要用于边缘检测和图像锐化")

if __name__ == "__main__":
    main()