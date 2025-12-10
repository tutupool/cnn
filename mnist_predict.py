"""
MNIST官方数据集预测脚本
使用已训练好的模型对官方测试集进行预测
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# 设置matplotlib显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_model(model_path='mnist_cnn_model.h5'):
    """加载已训练的模型"""
    print(f"正在加载模型: {model_path}")
    model = keras.models.load_model(model_path)
    print("模型加载成功！")
    model.summary()
    return model

def load_test_data():
    """加载MNIST测试数据"""
    print("\n正在加载MNIST测试数据...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 归一化
    x_test = x_test.astype('float32') / 255.0
    
    print(f"测试集大小: {x_test.shape}")
    print(f"测试标签数量: {y_test.shape}")
    return x_test, y_test

def predict_and_evaluate(model, x_test, y_test):
    """预测并评估模型性能"""
    print("\n正在进行预测...")
    
    # 预测
    predictions = model.predict(x_test, verbose=1)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 计算准确率
    accuracy = np.mean(predicted_classes == y_test)
    print(f"\n测试集准确率: {accuracy * 100:.2f}%")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, predicted_classes, 
                                target_names=[str(i) for i in range(10)]))
    
    return predictions, predicted_classes

def plot_confusion_matrix(y_test, predicted_classes):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_test, predicted_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel('预测值')
    plt.ylabel('真实值')
    plt.title('MNIST测试集混淆矩阵')
    plt.tight_layout()
    plt.show()

def show_sample_predictions(x_test, y_test, predictions, num_samples=20):
    """显示样本预测结果"""
    predicted_classes = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(15, 8))
    for i in range(num_samples):
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_test[i], cmap='gray')
        
        pred = predicted_classes[i]
        true = y_test[i]
        confidence = predictions[i][pred] * 100
        
        color = 'green' if pred == true else 'red'
        plt.title(f'预测:{pred} 真实:{true}\n置信度:{confidence:.1f}%', 
                  color=color, fontsize=9)
        plt.axis('off')
    
    plt.suptitle('MNIST测试集预测样本', fontsize=14)
    plt.tight_layout()
    plt.show()

def show_wrong_predictions(x_test, y_test, predictions, num_samples=20):
    """显示错误预测的样本"""
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 找到错误预测的索引
    wrong_indices = np.where(predicted_classes != y_test)[0]
    print(f"\n错误预测数量: {len(wrong_indices)}/{len(y_test)}")
    
    if len(wrong_indices) == 0:
        print("没有错误预测！")
        return
    
    # 显示部分错误预测
    num_to_show = min(num_samples, len(wrong_indices))
    
    plt.figure(figsize=(15, 8))
    for i in range(num_to_show):
        idx = wrong_indices[i]
        plt.subplot(4, 5, i + 1)
        plt.imshow(x_test[idx], cmap='gray')
        
        pred = predicted_classes[idx]
        true = y_test[idx]
        confidence = predictions[idx][pred] * 100
        
        plt.title(f'预测:{pred} 真实:{true}\n置信度:{confidence:.1f}%', 
                  color='red', fontsize=9)
        plt.axis('off')
    
    plt.suptitle('错误预测样本', fontsize=14)
    plt.tight_layout()
    plt.show()

def main():
    """主函数"""
    print("=" * 50)
    print("MNIST官方数据集预测")
    print("=" * 50)
    
    # 1. 加载模型
    model = load_model('mnist_cnn_model.h5')
    
    # 2. 加载测试数据
    x_test, y_test = load_test_data()
    
    # 3. 预测并评估
    predictions, predicted_classes = predict_and_evaluate(model, x_test, y_test)
    
    # 4. 显示样本预测
    show_sample_predictions(x_test, y_test, predictions)
    
    # 5. 显示混淆矩阵
    plot_confusion_matrix(y_test, predicted_classes)
    
    # 6. 显示错误预测
    show_wrong_predictions(x_test, y_test, predictions)
    
    print("\n预测完成！")

if __name__ == "__main__":
    main()
