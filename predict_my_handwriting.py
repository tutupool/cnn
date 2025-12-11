"""
手写数字预测系统
用于预处理和预测自己的手写数字图片

核心问题：
- 您的手写图片：浅色背景、黑色数字、大尺寸、细线条
- MNIST格式：黑色背景、白色数字、28×28、较粗线条

预处理流程：
1. 颜色反转 (白底黑字 → 黑底白字)
2. 对比度增强
3. 去噪
4. 二值化
5. 笔画增粗 (膨胀操作)
6. 裁剪数字区域
7. 保持纵横比缩放到20×20
8. 质心居中到28×28画布
9. 高斯模糊平滑
10. 归一化到[0,1]
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class HandwritingPredictor:
    """手写数字预测器"""
    
    def __init__(self, model_path='mnist_cnn_model.h5', data_dir='my_handwriting_digits'):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的模型路径
            data_dir: 手写数字图片目录
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.model = None
        
    def load_model(self):
        """加载训练好的模型"""
        if not os.path.exists(self.model_path):
            print(f"❌ 模型文件不存在: {self.model_path}")
            return False
        
        try:
            print(f"📦 加载模型: {self.model_path}")
            self.model = tf.keras.models.load_model(self.model_path)
            print("✅ 模型加载成功!")
            return True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            return False
    
    def preprocess_image(self, image_path, show_steps=False):
        """
        预处理手写数字图片，转换为MNIST格式
        
        处理流程：
        1. 先在原图上找到数字轮廓位置
        2. 裁剪出数字区域
        3. 在裁剪区域内进行二值化和膨胀加粗
        4. 缩放并居中到28×28
        
        Args:
            image_path: 图片路径
            show_steps: 是否显示每一步的处理结果
            
        Returns:
            处理后的28×28归一化图像，或None（失败时）
        """
        # ========== 步骤1: 读取图像 ==========
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"⚠️ 无法读取: {image_path}")
            return None
        
        steps = {'1.原始图像': img.copy()} if show_steps else {}
        
        # ========== 步骤2: 先找数字轮廓位置 ==========
        # 使用高斯模糊减少噪声
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 判断是白底黑字还是黑底白字
        is_light_bg = np.mean(img) > 127
        
        # 使用Otsu二值化找轮廓（白底黑字用THRESH_BINARY_INV使数字变白）
        if is_light_bg:
            _, thresh_for_contour = cv2.threshold(blurred, 0, 255, 
                                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, thresh_for_contour = cv2.threshold(blurred, 0, 255,
                                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if show_steps:
            steps['2.轮廓检测用'] = thresh_for_contour.copy()
        
        # ========== 步骤3: 找到最大轮廓（数字） ==========
        contours, _ = cv2.findContours(thresh_for_contour, cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print(f"⚠️ 未找到轮廓: {image_path}")
            return None
        
        # 找最大轮廓（假设是数字）
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        
        # 添加边距
        margin = int(max(w, h) * 0.15)
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(img.shape[1] - x, w + 2 * margin)
        h = min(img.shape[0] - y, h + 2 * margin)
        
        if show_steps:
            # 在原图上画出检测到的区域
            img_with_rect = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(img_with_rect, (x, y), (x+w, y+h), (0, 255, 0), 3)
            steps['3.检测区域'] = img_with_rect
        
        # ========== 步骤4: 裁剪数字区域 ==========
        cropped = img[y:y+h, x:x+w]
        
        if show_steps:
            steps['4.裁剪区域'] = cropped.copy()
        
        # ========== 步骤5: 在裁剪区域内进行对比度增强 ==========
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(cropped)
        
        if show_steps:
            steps['5.对比度增强'] = enhanced.copy()
        
        # ========== 步骤6: 二值化（白底黑字→黑底白字） ==========
        # 对裁剪后的小区域使用Otsu
        if is_light_bg:
            # 白底黑字：使用THRESH_BINARY_INV，数字变白，背景变黑
            _, binary = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(enhanced, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if show_steps:
            steps['6.二值化'] = binary.copy()
        
        # ========== 步骤7: 膨胀加粗笔画 ==========
        # 这是关键！细笔画需要加粗才能匹配MNIST
        # 根据裁剪区域大小和纵横比动态调整膨胀强度
        crop_size = max(cropped.shape)
        crop_h, crop_w = cropped.shape
        crop_aspect = crop_w / crop_h if crop_h > 0 else 1.0
        
        # 极窄数字需要更多膨胀来保留特征
        if crop_aspect < 0.4:
            dilate_base = 5  # 极窄数字：更多膨胀
        elif crop_size > 500:
            dilate_base = 4  # 大图需要更多膨胀
        elif crop_size > 300:
            dilate_base = 3
        else:
            dilate_base = 2
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=dilate_base)
        
        if show_steps:
            steps['7.膨胀加粗'] = dilated.copy()
        
        # ========== 步骤8: 闭运算连接断笔 ==========
        # 对于极窄数字（如9），使用更大的闭运算核来连接顶部可能断开的圆形
        if crop_aspect < 0.4:
            close_size = 9  # 极窄数字需要更大的闭运算来连接9顶部圆圈
        else:
            close_size = 5
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
        
        if show_steps:
            steps['8.闭运算'] = closed.copy()
        
        # ========== 步骤9: 再次裁剪到数字边界 ==========
        coords = cv2.findNonZero(closed)
        if coords is None:
            print(f"⚠️ 处理后未检测到数字: {image_path}")
            return None
        
        x2, y2, w2, h2 = cv2.boundingRect(coords)
        margin2 = max(2, int(min(w2, h2) * 0.1))
        x2 = max(0, x2 - margin2)
        y2 = max(0, y2 - margin2)
        w2 = min(closed.shape[1] - x2, w2 + 2 * margin2)
        h2 = min(closed.shape[0] - y2, h2 + 2 * margin2)
        
        digit = closed[y2:y2+h2, x2:x2+w2]
        
        if show_steps:
            steps['9.精确裁剪'] = digit.copy()
        
        # ========== 步骤10: 缩放到合适尺寸 ==========
        # 问题分析：
        # - 您写的4、7、9比较瘦长，纵横比约0.3-0.55
        # - 缩放后只有8-9像素宽，和数字1(也是8像素宽)很像
        # - MNIST的数字通常更"胖"一些
        # 
        # 解决方案：对于瘦长数字，适当增加宽度，但要分情况处理
        
        h_d, w_d = digit.shape
        aspect_ratio = w_d / h_d
        
        # 根据纵横比决定目标尺寸
        if aspect_ratio < 0.35:
            # 极窄数字(如某些9)：需要更宽才能保持顶部圆形的可辨识性
            new_h = 20
            new_w = max(14, int(w_d * 20 / h_d))  # 极窄数字也用14像素最小宽度
            new_w = min(new_w, 20)
        elif aspect_ratio < 0.6:
            # 瘦长数字：固定高度20，宽度最小14像素（确保不会太窄像1）
            new_h = 20
            new_w = max(14, int(w_d * 20 / h_d))
            new_w = min(new_w, 20)
        elif aspect_ratio > 2.0:
            # 扁宽数字：固定宽度20，高度按比例
            new_w = 20
            new_h = max(10, int(h_d * 20 / w_d))
            new_h = min(new_h, 20)
        else:
            # 正常比例：保持纵横比缩放到20×20内
            scale = min(20 / w_d, 20 / h_d)
            new_w = int(w_d * scale)
            new_h = int(h_d * scale)
        
        resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        if show_steps:
            steps['10.缩放'] = resized.copy()
        
        # ========== 步骤11: 居中到28×28画布 ==========
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        if show_steps:
            steps['11.居中28x28'] = canvas.copy()
        
        # ========== 步骤12: 高斯模糊平滑边缘 ==========
        blurred = cv2.GaussianBlur(canvas, (3, 3), 0.5)
        
        # ========== 步骤13: 增强亮度使其匹配MNIST ==========
        # 问题：处理后像素值太低(均值0.02)，MNIST均值约0.1
        # 解决：对非零区域进行亮度增强
        if np.max(blurred) > 0:
            # 将最大值拉伸到200-255范围，模拟MNIST的亮度
            scale_factor = 220.0 / max(np.max(blurred), 1)
            final = np.clip(blurred * scale_factor, 0, 255).astype(np.uint8)
        else:
            final = blurred
        
        if show_steps:
            steps['12.亮度增强'] = final.copy()
        
        # ========== 归一化到[0,1] ==========
        normalized = final.astype('float32') / 255.0
        
        if show_steps:
            return normalized, steps
        
        return normalized
    
    def visualize_preprocessing(self, image_path):
        """可视化预处理的每一步"""
        result = self.preprocess_image(image_path, show_steps=True)
        
        if result is None:
            return
        
        normalized, steps = result
        
        # 创建可视化
        n_steps = len(steps)
        cols = 4
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = axes.flatten()
        
        for idx, (title, img) in enumerate(steps.items()):
            axes[idx].imshow(img, cmap='gray')
            axes[idx].set_title(title, fontsize=10)
            axes[idx].axis('off')
            
            # 显示尺寸
            h, w = img.shape[:2]
            axes[idx].text(0.02, 0.98, f'{w}×{h}', transform=axes[idx].transAxes,
                          fontsize=8, va='top', color='red',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 隐藏多余的子图
        for idx in range(n_steps, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'预处理步骤: {os.path.basename(image_path)}', fontsize=14)
        plt.tight_layout()
        plt.show()
        
        return normalized
    
    def predict_single(self, image_path, show_details=False):
        """
        预测单张图片
        
        Args:
            image_path: 图片路径
            show_details: 是否显示详细信息
            
        Returns:
            (预测标签, 置信度, 所有类别概率)
        """
        if self.model is None:
            print("❌ 请先加载模型")
            return None, None, None
        
        # 预处理
        processed = self.preprocess_image(image_path)
        if processed is None:
            return None, None, None
        
        # 预测
        img_input = processed.reshape(1, 28, 28, 1)
        predictions = self.model.predict(img_input, verbose=0)[0]
        
        pred_label = np.argmax(predictions)
        confidence = predictions[pred_label]
        
        if show_details:
            print(f"\n📊 预测详情: {os.path.basename(image_path)}")
            print(f"   预测结果: {pred_label}")
            print(f"   置信度: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"   各类别概率:")
            for i, prob in enumerate(predictions):
                bar = '█' * int(prob * 20)
                print(f"      {i}: {prob:.4f} |{bar}")
        
        return pred_label, confidence, predictions
    
    def predict_batch(self, show_results=True):
        """
        批量预测目录中的所有图片
        
        Returns:
            预测结果列表, 准确率
        """
        if self.model is None:
            print("❌ 请先加载模型")
            return None, None
        
        if not os.path.exists(self.data_dir):
            print(f"❌ 目录不存在: {self.data_dir}")
            return None, None
        
        # 获取所有图片
        image_files = sorted([
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        if not image_files:
            print(f"❌ 未找到图片文件")
            return None, None
        
        print(f"\n📁 找到 {len(image_files)} 张图片")
        print("=" * 60)
        
        results = []
        correct = 0
        
        for filename in image_files:
            # 从文件名提取真实标签 (格式: 数字_编号.jpg)
            try:
                true_label = int(filename.split('_')[0])
            except:
                print(f"⚠️ 跳过 {filename}: 无法解析标签")
                continue
            
            image_path = os.path.join(self.data_dir, filename)
            pred_label, confidence, probs = self.predict_single(image_path)
            
            if pred_label is None:
                continue
            
            is_correct = (pred_label == true_label)
            if is_correct:
                correct += 1
            
            results.append({
                'filename': filename,
                'true_label': true_label,
                'pred_label': pred_label,
                'confidence': confidence,
                'correct': is_correct,
                'probabilities': probs
            })
            
            # 打印结果
            status = '✓' if is_correct else '✗'
            print(f"{status} {filename}: 真实={true_label}, 预测={pred_label}, 置信度={confidence:.3f}")
        
        # 计算准确率
        total = len(results)
        accuracy = correct / total if total > 0 else 0
        
        print("\n" + "=" * 60)
        print(f"📊 预测结果汇总")
        print(f"   总样本: {total}")
        print(f"   正确数: {correct}")
        print(f"   准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("=" * 60)
        
        if show_results:
            self._show_results(results)
        
        return results, accuracy
    
    def _show_results(self, results):
        """可视化预测结果"""
        if not results:
            return
        
        # 统计每个数字的准确率
        digit_stats = {}
        for i in range(10):
            digit_results = [r for r in results if r['true_label'] == i]
            if digit_results:
                correct = sum(1 for r in digit_results if r['correct'])
                digit_stats[i] = {
                    'total': len(digit_results),
                    'correct': correct,
                    'accuracy': correct / len(digit_results)
                }
        
        # 绘制各数字准确率
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 柱状图
        digits = list(digit_stats.keys())
        accuracies = [digit_stats[d]['accuracy'] for d in digits]
        colors = ['green' if acc >= 0.8 else 'orange' if acc >= 0.5 else 'red' for acc in accuracies]
        
        axes[0].bar(digits, accuracies, color=colors, edgecolor='black')
        axes[0].set_xlabel('数字')
        axes[0].set_ylabel('准确率')
        axes[0].set_title('各数字识别准确率')
        axes[0].set_ylim(0, 1.1)
        axes[0].axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80%')
        axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='50%')
        
        for i, (d, acc) in enumerate(zip(digits, accuracies)):
            axes[0].text(d, acc + 0.02, f'{acc:.0%}', ha='center', fontsize=9)
        
        # 混淆矩阵
        true_labels = [r['true_label'] for r in results]
        pred_labels = [r['pred_label'] for r in results]
        
        cm = confusion_matrix(true_labels, pred_labels, labels=range(10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                   xticklabels=range(10), yticklabels=range(10))
        axes[1].set_xlabel('预测标签')
        axes[1].set_ylabel('真实标签')
        axes[1].set_title('混淆矩阵')
        
        plt.tight_layout()
        plt.show()
    
    def compare_with_mnist(self, image_path):
        """将预处理后的图片与MNIST样本对比"""
        result = self.preprocess_image(image_path, show_steps=False)
        if result is None:
            return
        
        # 获取真实标签
        filename = os.path.basename(image_path)
        try:
            true_label = int(filename.split('_')[0])
        except:
            true_label = 0
        
        # 加载MNIST样本
        (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # 找到相同数字的MNIST样本
        indices = np.where(y_test == true_label)[0][:5]
        
        # 可视化对比
        fig, axes = plt.subplots(2, 6, figsize=(15, 5))
        
        # 第一行: 原图和处理后
        original = cv2.imread(image_path)
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('原图')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(result, cmap='gray')
        axes[0, 1].set_title('预处理后', color='green', fontweight='bold')
        axes[0, 1].axis('off')
        
        # MNIST样本
        for i, idx in enumerate(indices[:4]):
            axes[0, 2 + i].imshow(x_test[idx], cmap='gray')
            axes[0, 2 + i].set_title(f'MNIST #{idx}')
            axes[0, 2 + i].axis('off')
        
        # 第二行: 差异分析
        axes[1, 0].text(0.5, 0.5, f'真实标签: {true_label}', ha='center', va='center', fontsize=14)
        axes[1, 0].axis('off')
        
        # 预测结果
        if self.model is not None:
            pred_label, confidence, _ = self.predict_single(image_path)
            color = 'green' if pred_label == true_label else 'red'
            axes[1, 1].text(0.5, 0.5, f'预测: {pred_label}\n置信度: {confidence:.2%}', 
                          ha='center', va='center', fontsize=12, color=color)
        axes[1, 1].axis('off')
        
        # 显示与MNIST的差异
        result_uint8 = (result * 255).astype(np.uint8)
        for i, idx in enumerate(indices[:4]):
            mnist_img = x_test[idx]
            diff = cv2.absdiff(result_uint8, mnist_img)
            axes[1, 2 + i].imshow(diff, cmap='hot')
            axes[1, 2 + i].set_title(f'差异 #{idx}')
            axes[1, 2 + i].axis('off')
        
        plt.suptitle(f'与MNIST样本对比 - {filename}', fontsize=14)
        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("🖐️  手写数字预测系统")
    print("=" * 60)
    
    # 创建预测器
    predictor = HandwritingPredictor(
        model_path='mnist_cnn_model.h5',
        data_dir='my_handwriting_digits'
    )
    
    # 加载模型
    if not predictor.load_model():
        return
    
    # 可视化一张数字9的预处理过程
    print("\n📊 预处理可视化...")
    test_file = '9_001.jpg'
    image_path = os.path.join(predictor.data_dir, test_file)
    if os.path.exists(image_path):
        print(f"\n处理: {test_file}")
        predictor.visualize_preprocessing(image_path)
        predictor.compare_with_mnist(image_path)
    
    # 批量预测
    print("\n" + "=" * 60)
    print("📊 批量预测...")
    print("=" * 60)
    
    results, accuracy = predictor.predict_batch()
    
    # 保存结果
    if results:
        with open('prediction_results.txt', 'w', encoding='utf-8') as f:
            f.write("手写数字预测结果\n")
            f.write("=" * 50 + "\n")
            f.write(f"总样本: {len(results)}\n")
            f.write(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
            f.write("=" * 50 + "\n\n")
            
            for r in results:
                status = '✓' if r['correct'] else '✗'
                f.write(f"{status} {r['filename']}: 真实={r['true_label']}, "
                       f"预测={r['pred_label']}, 置信度={r['confidence']:.3f}\n")
        
        print(f"\n✅ 结果已保存到: prediction_results.txt")
    
    print("\n🎊 预测完成!")


if __name__ == "__main__":
    main()
