import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp
import platform

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader


# ===================== 中文显示配置（新增，自动适配系统）=====================
def setup_chinese_font():
    """设置Matplotlib支持中文（Windows/Linux/Mac通用）"""
    try:
        system = platform.system()
        if system == "Windows":
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'FangSong', 'Arial Unicode MS']
        elif system == "Linux":
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'DejaVu Sans', 'Arial Unicode MS']
        elif system == "Darwin":
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题
        print("✅ 中文字体配置成功")
    except Exception as e:
        print(f"⚠️  中文字体配置失败，将使用英文标签：{str(e)[:50]}")
        # 降级为英文标签，避免警告
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False

# 初始化中文配置
setup_chinese_font()

# ===================== 全局配置（统一管理参数）=====================
class Config:
    def __init__(self):
        # 路径配置
        self.DATA_PATH = "resources/air_quality_data.csv"
        self.ROOT_SAVE_DIR = "./results/air_quality"
        self.MODEL_SAVE_PATH = osp.join(self.ROOT_SAVE_DIR, "model.pth")
        self.TRAIN_STATE_PATH = osp.join(self.ROOT_SAVE_DIR, "train_state.pth")  # 新增：训练状态保存路径
        self.LOSS_PLOT_PATH = osp.join(self.ROOT_SAVE_DIR, "loss_curve.png")
        self.PRED_COMPARE_PATH = osp.join(self.ROOT_SAVE_DIR, "pred_vs_true.png")
        
        # 训练参数
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 0
        self.EPOCHS = 10000 
        self.BATCH_SIZE = 256
        self.LEARNING_RATE = 0.01
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.interval = 500  # 日志打印及状态保存间隔（轮数）
        
        # 创建结果目录
        os.makedirs(self.ROOT_SAVE_DIR, exist_ok=True)

# ===================== 1. 数据处理函数（独立封装）=====================
def load_and_preprocess_data(config):
    """
    功能：加载数据、处理异常值、归一化、划分数据集、转换为Tensor
    参数：config - 全局配置对象
    返回：train_loader, x_test, y_test, scaler
    """
    # 加载数据
    dataset = pd.read_csv(config.DATA_PATH)
    print(f"数据集总样本数：{len(dataset)}")
    
    # 处理异常值
    dataset['PM2.5'] = dataset['PM2.5'].apply(lambda x: 300 if x > 300 else x)
    dataset['PM10'] = dataset['PM10'].apply(lambda x: 500 if x > 500 else x)
    dataset['AQI'] = dataset['AQI'].apply(lambda x: 500 if x > 500 else x)
    
    # 特征+标签一起归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # 分离特征和标签
    X_scaled = scaled_data[:, :-1]
    y_scaled = scaled_data[:, -1]
    
    # 划分训练集/测试集
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )
    
    # 转换为Tensor并移至设备
    x_train = torch.tensor(x_train, dtype=torch.float32).to(config.DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
    x_test = torch.tensor(x_test, dtype=torch.float32).to(config.DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(config.DEVICE)
    
    # 创建DataLoader
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    print(f"训练集：{len(x_train)}个样本，测试集：{len(x_test)}个样本")
    return train_loader, x_test, y_test, scaler

# ===================== 2. 模型构建函数（独立封装）=====================
def build_model(input_dim=6, config=None):
    """
    功能：构建全连接神经网络模型
    参数：input_dim - 输入特征数，config - 配置对象（指定设备）
    返回：model - PyTorch模型
    """
    class AirQualityNN(nn.Module):
        def __init__(self, input_dim):
            super(AirQualityNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.relu(out)
            out = self.fc3(out)
            return out
    
    model = AirQualityNN(input_dim).to(config.DEVICE)
    print(f"模型已初始化并移至设备：{config.DEVICE}")
    return model

# ===================== 3. 反归一化工具函数（独立封装）=====================
def inverse_transform(scaler, X_scaled, y_scaled):
    """
    功能：拼接特征和标签，反归一化还原原始AQI值
    参数：scaler - 归一化器，X_scaled - 归一化特征，y_scaled - 归一化标签
    返回：y_inv - 反归一化后的原始标签
    """
    combined = torch.cat([X_scaled, y_scaled], dim=1).cpu().numpy()
    combined_inv = scaler.inverse_transform(combined)
    y_inv = combined_inv[:, -1]
    return y_inv

# ===================== 4. 训练状态保存/加载函数（新增）=====================
def save_train_state(config, model, optimizer, start_epoch, train_losses, val_losses):
    """
    功能：保存训练状态（支持中断后继续训练）
    参数：config - 配置对象，model - 模型，optimizer - 优化器，start_epoch - 下次开始训练的轮数
          train_losses - 已记录的训练损失，val_losses - 已记录的验证损失
    """
    state = {
        "start_epoch": start_epoch,  # 下次训练开始的轮数（当前轮数+1）
        "model_state_dict": model.state_dict(),  # 模型权重
        "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态（学习率、动量等）
        "train_losses": train_losses,  # 历史训练损失
        "val_losses": val_losses  # 历史验证损失
    }
    torch.save(state, config.TRAIN_STATE_PATH)
    print(f"训练状态已保存至：{config.TRAIN_STATE_PATH}")

def load_train_state(config, model, optimizer):
    """
    功能：加载训练状态（中断后继续训练）
    参数：config - 配置对象，model - 初始化后的模型，optimizer - 初始化后的优化器
    返回：start_epoch - 下次开始训练的轮数，train_losses - 历史训练损失，val_losses - 历史验证损失
    """
    if not osp.exists(config.TRAIN_STATE_PATH):
        print("未找到训练状态文件，将从头开始训练")
        return 0, [], []  # 从第0轮开始，空损失记录
    
    # 加载状态
    state = torch.load(config.TRAIN_STATE_PATH, map_location=config.DEVICE)
    model.load_state_dict(state["model_state_dict"])  # 加载模型权重
    optimizer.load_state_dict(state["optimizer_state_dict"])  # 加载优化器状态
    start_epoch = state["start_epoch"]  # 上次中断时的下一轮
    train_losses = state["train_losses"]  # 历史训练损失
    val_losses = state["val_losses"]  # 历史验证损失
    
    print(f"已加载训练状态，将从第{start_epoch+1}轮开始训练（总轮数：{config.EPOCHS}）")
    print(f"已训练轮数：{start_epoch}，剩余轮数：{config.EPOCHS - start_epoch}")
    print(f"历史训练损失记录数：{len(train_losses)}，历史验证损失记录数：{len(val_losses)}")
    return start_epoch, train_losses, val_losses

# ===================== 5. 训练函数（修改：支持继续训练）=====================
def train_model(config, train_loader, x_test, y_test):
    """
    功能：模型训练（支持中断后继续训练）、损失记录、模型保存、绘制损失曲线
    参数：config - 配置对象，train_loader - 训练数据加载器，x_test/y_test - 测试数据
    返回：model - 训练好的模型， train_losses/val_losses - 完整损失记录
    """
    # 初始化模型、损失函数、优化器
    model = build_model(input_dim=6, config=config)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=config.LEARNING_RATE)
    
    # 加载训练状态（若存在）
    start_epoch, train_losses, val_losses = load_train_state(config, model, optimizer)
    
    print("\n" + "="*60)
    print("开始训练模型...")
    print("="*60)
    
    # 从上次中断的轮数开始训练（遍历到总轮数）
    for epoch in range(start_epoch, config.EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs, y_test).item()
            val_losses.append(val_loss)
        
        # 打印日志（每20轮打印一次）
        if (epoch + 1) % (config.interval) == 0:
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] | 训练损失：{avg_train_loss:.6f} | 验证损失：{val_loss:.6f}")
            # 每20轮保存一次训练状态（防止中断）
            save_train_state(config, model, optimizer, epoch+1, train_losses, val_losses)
    
    # 训练完成后，保存最终模型和状态
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    save_train_state(config, model, optimizer, config.EPOCHS, train_losses, val_losses)
    print(f"\n最终模型已保存至：{config.MODEL_SAVE_PATH}")
    
    return model, train_losses, val_losses

# ===================== 6. 测试与评估函数（独立封装）=====================
def test_model(config, model, scaler, x_test, y_test):
    """
    功能：模型预测、反归一化、指标计算、绘制预测对比图
    参数：config - 配置对象，model - 训练好的模型，scaler - 归一化器，x_test/y_test - 测试数据
    返回：metrics - 评估指标字典
    """
    print("\n" + "="*60)
    print("开始测试模型...")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(x_test)
        # 反归一化
        y_true = inverse_transform(scaler, x_test, y_test)
        y_pred = inverse_transform(scaler, x_test, y_pred_scaled)
    
    # 计算指标
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_pred + 1e-6))) * 100  # 避免除零
    
    # 打印指标
    print(f"\n测试集回归指标：")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    metrics = {
        "mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "mape": mape
    }
    return metrics, y_true, y_pred

# ===================== 7. 可视化函数（独立封装）=====================
def plot_results(config, train_losses, val_losses, y_true, y_pred):
    """
    功能：绘制损失曲线和预测对比图（统一可视化入口）
    参数：config - 配置对象，train_losses/val_losses - 损失记录，y_true/y_pred - 真实值/预测值
    """
    # 绘制损失曲线（包含历史损失）
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='训练损失', linewidth=2, color='#2E86AB')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='验证损失', linewidth=2, linestyle='--', color='#A23B72')
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('损失值（MSE）', fontsize=12)
    plt.title('AQI回归模型 - 损失曲线（含历史训练记录）', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.LOSS_PLOT_PATH, dpi=300, bbox_inches='tight')
    print(f"\n损失曲线已保存至：{config.LOSS_PLOT_PATH}")
    plt.show()
    
    # 绘制预测对比图（折线图+散点图）
    plt.figure(figsize=(12, 6))
    # 折线图（前200样本）
    plt.subplot(1, 2, 1)
    plt.plot(range(min(200, len(y_true))), y_true[:200], label='真实AQI', linewidth=2, color='#3F88C5')
    plt.plot(range(min(200, len(y_pred))), y_pred[:200], label='预测AQI', linewidth=2, linestyle='--', color='#F9C74F')
    plt.xlabel('样本序号', fontsize=12)
    plt.ylabel('AQI值', fontsize=12)
    plt.title('真实值vs预测值（前200样本）', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    # 散点图
    plt.subplot(1, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.6, color='#2E86AB', s=50)
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线（y=x）')
    plt.xlabel('真实AQI值', fontsize=12)
    plt.ylabel('预测AQI值', fontsize=12)
    plt.title('真实值vs预测值散点图', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    # 保存
    plt.tight_layout()
    plt.savefig(config.PRED_COMPARE_PATH, dpi=300, bbox_inches='tight')
    print(f"预测对比图已保存至：{config.PRED_COMPARE_PATH}")
    plt.show()

# ===================== 主控制函数（统一调度）=====================
def main():
    # 初始化配置
    config = Config()
    print(f"当前设备：{config.DEVICE}")
    print(f"结果保存目录：{config.ROOT_SAVE_DIR}")
    
    # 1. 数据处理
    train_loader, x_test, y_test, scaler = load_and_preprocess_data(config)
    
    # 2. 模型训练（支持中断后继续）
    model, train_losses, val_losses = train_model(config, train_loader, x_test, y_test)
    
    # 3. 模型测试
    metrics, y_true, y_pred = test_model(config, model, scaler, x_test, y_test)
    
    # 4. 可视化结果（含历史损失）
    plot_results(config, train_losses, val_losses, y_true, y_pred)
    
    print("\n" + "="*60)
    print("模型训练+测试+可视化全流程完成！")
    print("="*60)

# ===================== 执行入口 =====================
if __name__ == "__main__":
    main()