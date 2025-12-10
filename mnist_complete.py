import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import time

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MNISTRecognition:
    def __init__(self):
        """初始化MNIST手写数字识别系统"""
        print("=== MNIST手写数字识别系统 ===")
        print("正在初始化...")
        
        # 设置随机种子确保结果可重现
        tf.random.set_seed(42)
        np.random.seed(42)
        
        # 初始化变量
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def download_and_load_data(self):
        """下载并加载MNIST数据集"""
        print("\n1. 正在下载MNIST数据集...")
        print("数据来源: http://yann.lecun.com/exdb/mnist/")
        
        try:
            # TensorFlow会自动从官方源下载MNIST数据
            (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
            
            print(f"✅ 数据下载成功！")
            print(f"训练集大小: {self.x_train.shape[0]} 张图片")
            print(f"测试集大小: {self.x_test.shape[0]} 张图片")
            print(f"图片尺寸: {self.x_train.shape[1]}×{self.x_train.shape[2]} 像素")
            
            return True
            
        except Exception as e:
            print(f"❌ 数据下载失败: {str(e)}")
            print("请检查网络连接或手动下载数据集")
            return False
    
    def preprocess_data(self):
        """数据预处理"""
        print("\n2. 正在进行数据预处理...")
        
        # 数据标准化: 将像素值从 [0,255] 缩放到 [0,1]
        self.x_train = self.x_train.astype('float32') / 255.0
        self.x_test = self.x_test.astype('float32') / 255.0
        
        # 对于CNN，需要添加通道维度
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 28, 28, 1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], 28, 28, 1)
        
        # 标签转换为one-hot编码
        self.y_train_categorical = tf.keras.utils.to_categorical(self.y_train, 10)
        self.y_test_categorical = tf.keras.utils.to_categorical(self.y_test, 10)
        
        print("✅ 数据预处理完成")
        print(f"训练数据形状: {self.x_train.shape}")
        print(f"测试数据形状: {self.x_test.shape}")
        print(f"标签类别数: {len(np.unique(self.y_train))}")
    
    def visualize_samples(self):
        """可视化数据样本"""
        print("\n3. 可视化数据样本...")
        
        # 创建图形
        plt.figure(figsize=(15, 8))
        
        # 显示前20个样本
        for i in range(20):
            plt.subplot(2, 10, i + 1)
            plt.imshow(self.x_train[i].reshape(28, 28), cmap='gray')
            plt.title(f'标签: {self.y_train[i]}', fontsize=10)
            plt.axis('off')
        
        plt.suptitle('MNIST数据集样本展示', fontsize=16)
        plt.tight_layout()
        plt.show()
        
        # 统计各类别数量
        unique, counts = np.unique(self.y_train, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar(unique, counts, color='skyblue', edgecolor='black')
        plt.xlabel('数字类别')
        plt.ylabel('样本数量')
        plt.title('训练集中各数字类别分布')
        plt.grid(True, alpha=0.3)
        
        # 添加数量标签
        for i, count in enumerate(counts):
            plt.text(i, count + 50, str(count), ha='center', va='bottom')
        
        plt.show()
    
    def build_cnn_model(self):
        """构建CNN模型（推荐，准确率最高）"""
        print("\n4. 构建CNN模型...")
        
        model = tf.keras.Sequential([
            # 第一个卷积层
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # 第二个卷积层
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            # 第三个卷积层
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            
            # 展平层
            tf.keras.layers.Flatten(),
            
            # 全连接层
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),  # 防止过拟合
            
            # 输出层
            tf.keras.layers.Dense(10, activation='softmax')  # 10个类别
        ])
        
        # 编译模型
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 显示模型结构
        model.summary()
        
        self.model = model
        print("✅ CNN模型构建完成")
    
    def train_model(self, epochs=12, batch_size=128):
        """训练模型"""
        print(f"\n5. 开始训练模型 (epochs={epochs}, batch_size={batch_size})...")
        
        # 设置回调函数
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        start_time = time.time()
        
        # 训练模型
        self.history = self.model.fit(
            self.x_train, self.y_train_categorical,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.x_test, self.y_test_categorical),
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        print(f"✅ 模型训练完成，耗时: {training_time:.2f} 秒")
        
        # 获取最佳准确率
        best_val_accuracy = max(self.history.history['val_accuracy'])
        print(f"🎯 最佳验证准确率: {best_val_accuracy:.4f} ({best_val_accuracy*100:.2f}%)")
        
        return best_val_accuracy
    
    def evaluate_model(self):
        """评估模型性能"""
        print("\n6. 评估模型性能...")
        
        # 在测试集上评估
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test_categorical, verbose=0)
        
        print(f"📊 测试集性能:")
        print(f"  - 损失值: {test_loss:.4f}")
        print(f"  - 准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # 检查是否达到要求
        if test_accuracy >= 0.96:
            print(f"🎉 恭喜！模型准确率 {test_accuracy*100:.2f}% 达到要求 (≥96%)")
        else:
            print(f"⚠️ 模型准确率 {test_accuracy*100:.2f}% 未达到要求 (≥96%)")
            print("建议：增加训练轮数或调整模型结构")
        
        return test_accuracy
    
    def detailed_analysis(self):
        """详细分析模型性能"""
        print("\n7. 详细性能分析...")
        
        # 预测测试集
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(self.y_test, y_pred_classes))
        
        # 混淆矩阵可视化
        cm = confusion_matrix(self.y_test, y_pred_classes)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()
        
        # 计算每个类别的准确率
        class_accuracy = cm.diagonal() / cm.sum(axis=1)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(10), class_accuracy, color='lightgreen', edgecolor='black')
        plt.xlabel('数字类别')
        plt.ylabel('准确率')
        plt.title('各数字类别识别准确率')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        plt.show()
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            print("没有训练历史数据")
            return
        
        print("\n8. 绘制训练历史...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 准确率曲线
        ax1.plot(self.history.history['accuracy'], label='训练准确率', marker='o')
        ax1.plot(self.history.history['val_accuracy'], label='验证准确率', marker='s')
        ax1.set_title('模型准确率变化')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 损失曲线
        ax2.plot(self.history.history['loss'], label='训练损失', marker='o')
        ax2.plot(self.history.history['val_loss'], label='验证损失', marker='s')
        ax2.set_title('模型损失变化')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def predict_samples(self, num_samples=10):
        """预测随机样本并可视化"""
        print(f"\n9. 预测 {num_samples} 个随机样本...")
        
        # 随机选择样本
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        plt.figure(figsize=(15, 6))
        
        for i, idx in enumerate(indices):
            # 预测
            img = self.x_test[idx:idx+1]
            pred = self.model.predict(img, verbose=0)
            pred_class = np.argmax(pred)
            confidence = np.max(pred)
            true_class = self.y_test[idx]
            
            # 显示图像
            plt.subplot(2, 5, i + 1)
            plt.imshow(img.reshape(28, 28), cmap='gray')
            
            # 设置标题颜色（正确为绿色，错误为红色）
            color = 'green' if pred_class == true_class else 'red'
            plt.title(f'真实:{true_class} 预测:{pred_class}\n置信度:{confidence:.3f}', 
                     color=color, fontsize=10)
            plt.axis('off')
        
        plt.suptitle('随机样本预测结果 (绿色=正确, 红色=错误)', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filename='mnist_model.h5'):
        """保存模型"""
        if self.model is None:
            print("没有训练好的模型可保存")
            return
        
        self.model.save(filename)
        print(f"✅ 模型已保存为: {filename}")
    
    def run_complete_pipeline(self):
        """运行完整的训练和评估流程"""
        print("🚀 开始MNIST手写数字识别完整流程...")
        
        # 1. 下载数据
        if not self.download_and_load_data():
            return False
        
        # 2. 预处理
        self.preprocess_data()
        
        # 3. 可视化样本
        self.visualize_samples()
        
        # 4. 构建模型
        self.build_cnn_model()
        
        # 5. 训练模型
        best_accuracy = self.train_model(epochs=12)
        
        # 6. 评估模型
        final_accuracy = self.evaluate_model()
        
        # 7. 详细分析
        self.detailed_analysis()
        
        # 8. 绘制训练历史
        self.plot_training_history()
        
        # 9. 预测样本
        self.predict_samples()
        
        # 10. 保存模型
        self.save_model()
        
        # 最终报告
        print("\n" + "="*50)
        print("📋 最终报告")
        print("="*50)
        print(f"🎯 最终测试准确率: {final_accuracy*100:.2f}%")
        print(f"📈 要求准确率: 96.00%")
        
        if final_accuracy >= 0.96:
            print("🎉 任务完成！模型达到要求")
        else:
            print("⚠️ 未达到要求，建议调整参数或模型结构")
        
        print("="*50)
        
        return final_accuracy >= 0.96

def main():
    """主函数"""
    # 创建识别系统
    mnist_system = MNISTRecognition()
    
    # 运行完整流程
    success = mnist_system.run_complete_pipeline()
    
    if success:
        print("\n🎊 MNIST手写数字识别任务成功完成！")
    else:
        print("\n🔄 如需提高准确率，建议增加训练轮数或调整模型")

if __name__ == "__main__":
    main()