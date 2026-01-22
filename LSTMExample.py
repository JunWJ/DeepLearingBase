import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from sklearn.preprocessing import MinMaxScaler
import platform

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ===================== 全局环境配置 =====================
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('medium')

# ===================== 指标计算工具函数 =====================
def calculate_rmse(y_true, y_pred):
    """计算均方根误差 (RMSE)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_mape(y_true, y_pred, epsilon=1e-8):
    """计算平均绝对百分比误差 (MAPE)，返回百分比形式"""
    y_true = np.where(y_true == 0, epsilon, y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

def inverse_transform_predictions(y_pred, y_true, scaler):
    """反归一化预测值和真实值"""
    y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_true_inv = scaler.inverse_transform(y_true.reshape(-1, 1))
    return y_pred_inv.flatten(), y_true_inv.flatten()

# ===================== 数据预处理类 =====================
class GoldPriceDataProcessor:
    """黄金价格数据预处理类"""
    def __init__(self, data_path, time_step=10, train_test_split=200):
        self.data_path = data_path
        self.time_step = time_step
        self.train_test_split = train_test_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataSet = None
        self.train_data = None
        self.test_data = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self._setup_chinese_font()
    
    def _setup_chinese_font(self):
        """设置Matplotlib支持中文"""
        try:
            system = platform.system()
            font_configs = {
                "Windows": ['Microsoft YaHei', 'SimHei'],
                "Linux": ['WenQuanYi Zen Hei'],
                "Darwin": ['Arial Unicode MS', 'PingFang SC']
            }
            plt.rcParams['font.sans-serif'] = font_configs.get(system, ['DejaVu Sans'])
            plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
    
    def load_data(self):
        """加载CSV数据，增加异常处理"""
        try:
            self.dataSet = pd.read_csv(self.data_path, index_col=[0])
            print(f"✅ 数据加载成功，形状: {self.dataSet.shape}")
            return self.dataSet
        except FileNotFoundError:
            print(f"❌ 错误：未找到数据文件 {self.data_path}")
            raise
        except Exception as e:
            print(f"❌ 数据加载失败：{str(e)}")
            raise
    
    def split_train_test(self):
        """划分并归一化训练/测试集"""
        if self.dataSet is None:
            self.load_data()
        
        train_len = len(self.dataSet) - self.train_test_split
        train_set = self.dataSet.iloc[:train_len, [0]]
        test_set = self.dataSet.iloc[train_len:, [0]]
        
        self.train_data = self.scaler.fit_transform(train_set)
        self.test_data = self.scaler.transform(test_set)
        return self.train_data, self.test_data
    
    def _create_sequences(self, data):
        """创建时间序列数据"""
        x, y = [], []
        for i in range(self.time_step, len(data)):
            x.append(data[i-self.time_step:i, 0])
            y.append(data[i, 0])
        
        x = np.array(x).reshape(-1, self.time_step, 1)
        y = np.array(y).reshape(-1, 1)
        return x, y
    
    def get_processed_data(self):
        """获取所有预处理后的数据（转为Tensor）"""
        self.split_train_test()
        x_train, y_train = self._create_sequences(self.train_data)
        x_test, y_test = self._create_sequences(self.test_data)
        
        # 转换为PyTorch张量
        self.x_train = torch.tensor(x_train, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.float32)
        self.x_test = torch.tensor(x_test, dtype=torch.float32)
        self.y_test = torch.tensor(y_test, dtype=torch.float32)
        
        print(f"✅ 数据预处理完成")
        print(f"   - 训练集: {self.x_train.shape} | 测试集: {self.x_test.shape}")
        return self.x_train, self.y_train, self.x_test, self.y_test

# ===================== LSTM模型类（修复dropout警告） =====================
class GoldPriceLSTM(nn.Module):
    """黄金价格预测的LSTM模型（匹配经典双层结构）"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, dropout=0.1):
        super(GoldPriceLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 第一层LSTM：返回序列（return_sequences=True），单层不设置dropout
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # 第二层LSTM：不返回序列，单层不设置dropout（修复警告）
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)  # 单独的dropout层
    
    def forward(self, x):
        """前向传播"""
        # 第一层LSTM：输出 (batch_size, seq_len, hidden_size)
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout(lstm1_out)
        
        # 第二层LSTM：输出 (batch_size, 1, hidden_size)
        lstm2_out, _ = self.lstm2(lstm1_out)
        
        # 取最后一个时间步的输出
        out = lstm2_out[:, -1, :]
        
        # 全连接层输出
        out = self.fc(out)
        return out

# ===================== 训练器类（改用Adam优化器） =====================
class GoldPriceTrainer:
    """模型训练器（支持断点续训+RMSE/MAPE监控）"""
    def __init__(self, model, scaler, device, save_path="results/goldPredict/gold_price_model.pth"):
        self.model = model
        self.scaler = scaler  # 保存归一化器用于指标计算
        self.device = device
        self.save_path = save_path
        self.model.to(device)
        
        # 训练记录（扩展指标）
        self.train_losses = []
        self.val_losses = []
        self.train_rmse = []
        self.val_rmse = []
        self.train_mape = []
        self.val_mape = []
        
        # 创建模型保存目录
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
    
    def _save_checkpoint(self, epoch, optimizer, loss):
        """保存训练断点（包含所有指标）"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_rmse': self.train_rmse,
            'val_rmse': self.val_rmse,
            'train_mape': self.train_mape,
            'val_mape': self.val_mape,
            'loss': loss
        }
        torch.save(checkpoint, self.save_path)
        print(f"✅ 模型已保存至: {self.save_path} (Epoch: {epoch})")
    
    def _load_checkpoint(self, optimizer):
        """加载训练断点（手动处理设备问题）"""
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, map_location='cpu')  # 先加载到CPU
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)  # 再移到目标设备
            
            # 加载优化器状态到CPU，再移到目标设备（避免CUDA断言错误）
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            
            start_epoch = checkpoint['epoch'] + 1
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_rmse = checkpoint['train_rmse']
            self.val_rmse = checkpoint['val_rmse']
            self.train_mape = checkpoint['train_mape']
            self.val_mape = checkpoint['val_mape']
            
            print(f"✅ 加载断点成功，从Epoch {start_epoch} 继续训练")
            return start_epoch
        else:
            print("⚠️  未找到断点文件，从头开始训练")
            return 0
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001, patience=10):
        """
        模型训练（支持早停和断点续训，监控RMSE/MAPE）
        """
        # ========== 改用Adam优化器（移除capturable参数，适配旧版本PyTorch） ==========
        optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr,
            betas=(0.9, 0.999),  # 默认值，显式声明增加兼容性
            eps=1e-08
        )
        criterion = nn.MSELoss()
        
        # 加载断点
        start_epoch = self._load_checkpoint(optimizer)
        
        # 早停相关（用RMSE作为早停指标）
        best_val_rmse = float('inf')
        patience_counter = 0
        
        print(f"\n🚀 开始训练 (总轮数: {epochs}, 起始轮数: {start_epoch})")
        print("-" * 90)
        print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Train RMSE':<12} {'Val RMSE':<12} {'Train MAPE(%)':<12} {'Val MAPE(%)':<12} {'Time(s)':<8}")
        print("-" * 90)
        
        for epoch in range(start_epoch, epochs):
            start_time = time.time()
            
            # ========== 训练阶段 ==========
            self.model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * batch_x.size(0)
                # 收集预测值和真实值
                train_preds.extend(outputs.detach().cpu().numpy())
                train_targets.extend(batch_y.detach().cpu().numpy())
            
            # 计算训练集指标
            avg_train_loss = train_loss / len(train_loader.dataset)
            train_preds, train_targets = inverse_transform_predictions(
                np.array(train_preds), np.array(train_targets), self.scaler
            )
            train_rmse = calculate_rmse(train_targets, train_preds)
            train_mape = calculate_mape(train_targets, train_preds)
            
            # ========== 验证阶段 ==========
            self.model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    
                    val_loss += loss.item() * batch_x.size(0)
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # 计算验证集指标
            avg_val_loss = val_loss / len(val_loader.dataset)
            val_preds, val_targets = inverse_transform_predictions(
                np.array(val_preds), np.array(val_targets), self.scaler
            )
            val_rmse = calculate_rmse(val_targets, val_preds)
            val_mape = calculate_mape(val_targets, val_preds)
            
            # ========== 记录与打印 ==========
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_rmse.append(train_rmse)
            self.val_rmse.append(val_rmse)
            self.train_mape.append(train_mape)
            self.val_mape.append(val_mape)
            
            epoch_time = time.time() - start_time
            print(f"{epoch+1:<6} {avg_train_loss:<12.6f} {avg_val_loss:<12.6f} "
                  f"{train_rmse:<12.2f} {val_rmse:<12.2f} {train_mape:<12.2f} {val_mape:<12.2f} {epoch_time:<8.2f}")
            
            # 保存最佳模型（基于RMSE）
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                self._save_checkpoint(epoch, optimizer, avg_val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停
            if patience_counter >= patience:
                print(f"\n🛑 早停触发 (Patience: {patience})")
                break
        
        print("-" * 90)
        print("\n🏁 训练完成！")
    
    def plot_training_metrics(self):
        """可视化训练指标（Loss + RMSE + MAPE）"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. 损失曲线
        axes[0].plot(range(1, len(self.train_losses)+1), self.train_losses, label='训练损失', linewidth=2)
        axes[0].plot(range(1, len(self.val_losses)+1), self.val_losses, label='验证损失', linewidth=2)
        axes[0].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[0].set_ylabel('损失值 (MSE)', fontsize=12)
        axes[0].set_title('黄金价格LSTM模型 - 损失曲线', fontsize=14)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. RMSE曲线
        axes[1].plot(range(1, len(self.train_rmse)+1), self.train_rmse, label='训练RMSE', linewidth=2)
        axes[1].plot(range(1, len(self.val_rmse)+1), self.val_rmse, label='验证RMSE', linewidth=2)
        axes[1].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[1].set_ylabel('RMSE (USD)', fontsize=12)
        axes[1].set_title('黄金价格LSTM模型 - RMSE曲线', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 3. MAPE曲线
        axes[2].plot(range(1, len(self.train_mape)+1), self.train_mape, label='训练MAPE', linewidth=2)
        axes[2].plot(range(1, len(self.val_mape)+1), self.val_mape, label='验证MAPE', linewidth=2)
        axes[2].set_xlabel('训练轮数 (Epoch)', fontsize=12)
        axes[2].set_ylabel('MAPE (%)', fontsize=12)
        axes[2].set_title('黄金价格LSTM模型 - MAPE曲线', fontsize=14)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/goldPredict/training_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()

# ===================== 测试与可视化 =====================
def test_model(model, x_test, y_test, scaler, device):
    """模型测试与结果可视化（计算RMSE/MAPE）"""
    model.eval()
    test_preds = []
    test_targets = []
    
    with torch.no_grad():
        # 分批预测避免内存溢出
        for i in range(0, len(x_test), 32):
            batch_x = x_test[i:i+32].to(device)
            batch_y = y_test[i:i+32].cpu().numpy()
            
            outputs = model(batch_x)
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(batch_y)
    
    # 反归一化
    test_preds, test_targets = inverse_transform_predictions(
        np.array(test_preds), np.array(test_targets), scaler
    )
    
    # 计算评估指标
    mae = np.mean(np.abs(test_targets - test_preds))
    rmse = calculate_rmse(test_targets, test_preds)
    mape = calculate_mape(test_targets, test_preds)
    
    print(f"\n📊 测试集最终评估结果:")
    print(f"   - 平均绝对误差 (MAE): {mae:.2f} USD")
    print(f"   - 均方根误差 (RMSE): {rmse:.2f} USD")
    print(f"   - 平均绝对百分比误差 (MAPE): {mape:.2f} %")
    
    # 可视化预测结果
    plt.figure(figsize=(14, 7))
    plt.plot(test_targets, label='真实价格', linewidth=2)
    plt.plot(test_preds, label='预测价格', linewidth=2, alpha=0.8)
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('黄金价格 (USD)', fontsize=12)
    plt.title(f'黄金价格预测结果对比 (RMSE: {rmse:.2f} | MAPE: {mape:.2f}%)', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/goldPredict/prediction_result.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return test_preds, mae, rmse, mape

# ===================== 主函数 =====================
def main():
    # 1. 配置参数
    DATA_PATH = "resources/LBMA-GOLD.csv"
    TIME_STEP = 5               # 时间步长
    BATCH_SIZE = 32             # 批次大小
    EPOCHS = 1000               # 训练轮数
    LR = 1e-4                   # Adam推荐学习率
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 使用设备: {DEVICE}")
    
    # 2. 数据预处理
    processor = GoldPriceDataProcessor(DATA_PATH, time_step=TIME_STEP)
    x_train, y_train, x_test, y_test = processor.get_processed_data()
    
    # 3. 创建数据加载器
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_test, y_test)
    
    # 训练集shuffle=True（打乱样本间顺序，保留样本内时序）
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 验证集shuffle=False（保持时间顺序）
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    model = GoldPriceLSTM(hidden_size=50, num_layers=2)
    print(f"\n🧠 模型结构: {model}")
    
    # 5. 训练模型（传入scaler用于指标计算）
    trainer = GoldPriceTrainer(model, processor.scaler, DEVICE)
    trainer.train(train_loader, val_loader, epochs=EPOCHS, lr=LR, patience=20)
    
    # 6. 可视化训练指标
    trainer.plot_training_metrics()
    
    # 7. 测试模型
    test_model(model, x_test, y_test, processor.scaler, DEVICE)

if __name__ == "__main__":
    main()