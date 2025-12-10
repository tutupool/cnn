import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import time
import os

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class MNISTDigitRecognizer:
    def __init__(self):
        """初始化MNIST手写数字识别器"""
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        print("MNIST手写数字识别器初始化完成")
        print(f"TensorFlow版本: {tf.__version__}")
        
    def load_data(self):
        """
        加载MNIST数据集
        TensorFlow会自动下载MNIST数据
        """
        print("\n正在加载MNIST数据集...")
        try:
            # 加载数据集
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            
            print(f"训练集大小: {x_train.shape}")
            print(f"训练标签: {y_train.shape}")
            print(f"测试集大小: {x_test.shape}")
            print(f"测试标签: {y_test.shape}")
            
            # 数据预处理
            # 归一化到0-1范围
            self.x_train = x_train.astype('float32') / 255.0
            self.x_test = x_test.astype('float32') / 255.0
            
            # 保存原始标签用于后续分析
            self.y_train = y_train
            self.y_test = y_test
            
            print("数据预处理完成")
            return True
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            return False
    
    def visualize_samples(self, num_samples=10):
        """可视化样本数据"""
        if self.x_train is None:
            print("请先加载数据")
            return
            
        plt.figure(figsize=(15, 6))
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            plt.imshow(self.x_train[i], cmap='gray')
            plt.title(f'标签: {self.y_train[i]}')
            plt.axis('off')
        
        plt.suptitle('MNIST训练样本展示', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def build_model(self, model_type='cnn'):
        """
        构建模型
        model_type: 'simple', 'mlp', 'cnn'
        """
        print(f"\n正在构建{model_type}模型...")
        
        if model_type == 'simple':
            # 简单的全连接网络
            self.model = keras.Sequential([
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(10, activation='softmax')
            ])
            
        elif model_type == 'mlp':
            # 多层感知机
            self.model = keras.Sequential([
                layers.Flatten(input_shape=(28, 28)),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(10, activation='softmax')
            ])
            
        elif model_type == 'cnn':
            # 卷积神经网络（推荐，准确率最高）
            self.model = keras.Sequential([
                # 第一个卷积块
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # 第二个卷积块
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # 全连接层
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(10, activation='softmax')
            ])
        
        # 编译模型
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型构建完成")
        self.model.summary()
        
    def train_model(self, epochs=10, batch_size=128, validation_split=0.1):
        """训练模型"""
        if self.model is None:
            print("请先构建模型")
            return
            
        print(f"\n开始训练模型...")
        print(f"训练轮数: {epochs}")
        print(f"批次大小: {batch_size}")
        
        # 准备数据
        x_train = self.x_train
        y_train = self.y_train
        
        # 如果是CNN模型，需要reshape数据
        if hasattr(self.model.layers[0], 'filters'):  # CNN层有filters属性
            x_train = x_train.reshape(-1, 28, 28, 1)
        
        # 设置回调函数
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=3,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=0.0001
            )
        ]
        
        # 开始训练
        start_time = time.time()
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"\n训练完成！用时: {training_time:.2f}秒")
        
    def evaluate_model(self):
        """评估模型性能"""
        if self.model is None:
            print("请先训练模型")
            return
            
        print("\n正在评估模型性能...")
        
        # 准备测试数据
        x_test = self.x_test
        if hasattr(self.model.layers[0], 'filters'):  # CNN层有filters属性
            x_test = x_test.reshape(-1, 28, 28, 1)
        
        # 评估模型
        test_loss, test_accuracy = self.model.evaluate(x_test, self.y_test, verbose=0)
        
        print(f"测试集损失: {test_loss:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # 检查是否达到要求
        if test_accuracy >= 0.96:
            print("✅ 模型准确率达到要求（≥96%）")
        else:
            print("❌ 模型准确率未达到要求（<96%）")
            
        return test_accuracy
    
    def detailed_evaluation(self):
        """详细评估和可视化"""
        if self.model is None:
            print("请先训练模型")
            return
            
        # 准备测试数据
        x_test = self.x_test
        if hasattr(self.model.layers[0], 'filters'):
            x_test = x_test.reshape(-1, 28, 28, 1)
        
        # 预测
        print("正在进行预测...")
        predictions = self.model.predict(x_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(self.y_test, predicted_classes))
        
        # 混淆矩阵
        cm = confusion_matrix(self.y_test, predicted_classes)
        
        # 可视化结果
        self.plot_training_history()
        self.plot_confusion_matrix(cm)
        self.plot_prediction_examples(predicted_classes)
    
    def plot_training_history(self):
        """绘制训练历史"""
        if self.history is None:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 准确率曲线
        ax1.plot(self.history.history['accuracy'], label='训练准确率')
        ax1.plot(self.history.history['val_accuracy'], label='验证准确率')
        ax1.set_title('模型准确率')
        ax1.set_xlabel('轮次')
        ax1.set_ylabel('准确率')
        ax1.legend()
        ax1.grid(True)
        
        # 损失曲线
        ax2.plot(self.history.history['loss'], label='训练损失')
        ax2.plot(self.history.history['val_loss'], label='验证损失')
        ax2.set_title('模型损失')
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('损失')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, cm):
        """绘制混淆矩阵"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.show()
    
    def plot_prediction_examples(self, predicted_classes, num_examples=20):
        """显示预测例子"""
        # 找出正确和错误的预测
        correct = predicted_classes == self.y_test
        incorrect = predicted_classes != self.y_test
        
        # 显示正确预测的例子
        correct_indices = np.where(correct)[0][:num_examples//2]
        incorrect_indices = np.where(incorrect)[0][:num_examples//2]
        
        fig, axes = plt.subplots(2, num_examples//2, figsize=(15, 6))
        
        # 正确预测
        for i, idx in enumerate(correct_indices):
            axes[0, i].imshow(self.x_test[idx], cmap='gray')
            axes[0, i].set_title(f'✓ 真实:{self.y_test[idx]} 预测:{predicted_classes[idx]}')
            axes[0, i].axis('off')
        
        # 错误预测
        for i, idx in enumerate(incorrect_indices):
            axes[1, i].imshow(self.x_test[idx], cmap='gray')
            axes[1, i].set_title(f'✗ 真实:{self.y_test[idx]} 预测:{predicted_classes[idx]}')
            axes[1, i].axis('off')
        
        plt.suptitle('预测结果示例（上行：正确预测，下行：错误预测）', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath='mnist_model.h5'):
        """保存模型"""
        if self.model is None:
            print("没有可保存的模型")
            return
            
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath='mnist_model.h5'):
        """加载模型"""
        try:
            self.model = keras.models.load_model(filepath)
            print(f"模型已从 {filepath} 加载")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

def main():
    """主函数"""
    print("="*60)
    print("MNIST手写数字识别项目")
    print("目标：在MNIST测试数据上实现≥96%的识别率")
    print("="*60)
    
    # 创建识别器
    recognizer = MNISTDigitRecognizer()
    
    # 1. 加载数据
    if not recognizer.load_data():
        return
    
    # 2. 显示样本
    recognizer.visualize_samples()
    
    # 3. 构建模型（使用CNN获得最佳性能）
    recognizer.build_model(model_type='cnn')
    
    # 4. 训练模型
    recognizer.train_model(epochs=15, batch_size=128)
    
    # 5. 评估模型
    accuracy = recognizer.evaluate_model()
    
    # 6. 详细分析
    recognizer.detailed_evaluation()
    
    # 7. 保存模型
    recognizer.save_model('mnist_cnn_model.h5')
    
    print(f"\n项目完成！最终测试准确率: {accuracy*100:.2f}%")
    
    if accuracy >= 0.96:
        print("🎉 恭喜！模型成功达到96%以上的准确率要求！")
    else:
        print("💡 提示：如果准确率未达到96%，可以尝试：")
        print("   - 增加训练轮数")
        print("   - 调整模型结构")
        print("   - 使用数据增强")

if __name__ == "__main__":
    main()