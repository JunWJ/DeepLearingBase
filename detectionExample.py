import numpy as np
import matplotlib.pyplot as plt
import platform
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import os
import argparse  # 新增：解析命令行参数

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import time

# ===================== 全局配置（支持动态修改epoch）=====================
class Config:
    def __init__(self, epochs=50):  # 新增epochs参数，支持动态传入
        # 相对路径核心：基于当前脚本定位项目根目录
        self.SCRIPT_PATH = Path(__file__).resolve()  # 当前脚本绝对路径
        self.PROJECT_ROOT = self.SCRIPT_PATH.parent  # 项目根目录（脚本所在目录）

        # 数据路径（仅train/val，无test）
        self.DATA_ROOT = self.PROJECT_ROOT / "resources" / "detection_data"
        self.TRAIN_PATH = self.DATA_ROOT / "train"
        self.VAL_PATH = self.DATA_ROOT / "val"
        self.ROOT_SAVE_DIR = self.PROJECT_ROOT / "results" / "detection"

        # 结果保存路径（新增训练状态保存路径）
        self.BEST_MODEL_PATH = self.ROOT_SAVE_DIR / "best_model.pth"
        self.TRAIN_STATE_PATH = self.ROOT_SAVE_DIR / "train_state.pth"  # 断点续训状态
        self.LOSS_PLOT_PATH = self.ROOT_SAVE_DIR / "loss_curve.png"
        self.CONFUSION_MATRIX_PATH = self.ROOT_SAVE_DIR / "confusion_matrix.png"

        # 数据配置
        self.CLASS_NAMES = ["Cr", "In", "Pa", "PS", "Rs", "Sc"]  # 6类缺陷（与文件夹名一致）
        self.NUM_CLASSES = len(self.CLASS_NAMES)
        self.IMG_HEIGHT = 32
        self.IMG_WIDTH = 32
        self.IMAGE_CHANNELS = 3  # RGB=3/灰度图=1
        self.BATCH_SIZE = 128
        self.NUM_WORKERS = 0 if platform.system() == "Windows" else 4  # Windows禁用多线程
        self.USE_AUGMENTATION = True
        self.NORMALIZE_MEAN = [0.485, 0.456, 0.406] if self.IMAGE_CHANNELS == 3 else [0.5]
        self.NORMALIZE_STD = [0.229, 0.224, 0.225] if self.IMAGE_CHANNELS == 3 else [0.5]

        # 模型配置
        self.DROPOUT_RATE = 0.2

        # 训练配置（关键修改：从参数传入epochs，不再硬编码）
        self.EPOCHS = epochs  # 动态值，支持外部修改
        self.LEARNING_RATE = 1e-4
        self.WEIGHT_DECAY = 1e-5
        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.EARLY_STOPPING_PATIENCE = 10
        self.MONITOR_METRIC = "val_f1"
        self.SEED = 42

        # 初始化操作
        self._create_dirs()
        self._set_seed()
        self._setup_chinese_font()
        self._validate_paths()  # 验证train/val路径是否存在

    def _create_dirs(self):
        """创建结果目录"""
        self.ROOT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✅ 结果目录创建完成：{self.ROOT_SAVE_DIR}")

    def _set_seed(self):
        """固定随机种子"""
        import random
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.SEED)
            torch.cuda.manual_seed_all(self.SEED)
        torch.backends.cudnn.deterministic = True
        print(f"✅ 随机种子固定：{self.SEED}")

    def _setup_chinese_font(self):
        """配置中文字体"""
        try:
            system = platform.system()
            font_map = {
                "Windows": ['Microsoft YaHei', 'SimHei'],
                "Linux": ['WenQuanYi Zen Hei'],
                "Darwin": ['Arial Unicode MS']
            }
            plt.rcParams['font.sans-serif'] = font_map.get(system, ['DejaVu Sans'])
            plt.rcParams['axes.unicode_minus'] = False
            print("✅ 中文字体配置成功")
        except Exception as e:
            print(f"⚠️  中文字体配置失败：{str(e)[:30]}")

    def _validate_paths(self):
        """验证train/val路径是否存在（核心检查）"""
        required_paths = [self.TRAIN_PATH, self.VAL_PATH]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(f"❌ 关键数据路径不存在：{path}\n请检查路径是否正确！")
        print(f"\n📌 路径验证通过：")
        print(f"训练集路径：{self.TRAIN_PATH}")
        print(f"验证集路径：{self.VAL_PATH}")

# ===================== 数据集类（兼容两种目录结构）=====================
class DefectDataset(Dataset):
    def __init__(self, data_dir: Path, class_to_idx: Dict[str, int], transform=None, image_channels: int = 3):
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_channels = image_channels
        self.image_paths, self.labels = self._load_data()

        # 关键检查：确保加载到数据
        if len(self.image_paths) == 0:
            raise ValueError(f"❌ 在 {data_dir} 中未找到任何图像文件！\n请检查：1.图像格式（jpg/png/bmp等）2.是否按类别分文件夹")

    def _load_data(self) -> Tuple[list, list]:
        """加载按类别分文件夹的图像（推荐结构：train/Cr/xxx.jpg）"""
        image_paths, labels = [], []
        for cls_name, cls_idx in self.class_to_idx.items():
            cls_dir = self.data_dir / cls_name
            if not cls_dir.exists():
                print(f"⚠️  类别文件夹不存在：{cls_dir}（跳过该类别）")
                continue
            # 支持多种图像格式
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif"]:
                cls_images = list(cls_dir.glob(ext))
                if cls_images:
                    image_paths.extend([str(p) for p in cls_images])
                    labels.extend([cls_idx] * len(cls_images))
        return image_paths, labels

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """读取单张图像并返回（image, label）"""
        img_path = self.image_paths[idx]
        try:
            # 根据通道数选择图像模式（RGB/灰度）
            mode = "RGB" if self.image_channels == 3 else "L"
            image = Image.open(img_path).convert(mode)
            if self.transform:
                image = self.transform(image)
            return image, self.labels[idx]
        except Exception as e:
            raise RuntimeError(f"❌ 读取图像失败：{img_path}\n错误原因：{str(e)}")

# ===================== 数据加载函数（无test集，返回train/val_loader）=====================
def create_dataloaders(config: Config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练/验证/测试DataLoader（测试集用val集替代）"""
    class_to_idx = {cls: idx for idx, cls in enumerate(config.CLASS_NAMES)}

    # 训练集变换（含数据增强）
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=(-5, 5)),  # 轻微旋转，避免过度增强
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ]) if config.USE_AUGMENTATION else transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    # 验证集/测试集变换（无增强，仅归一化）
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])

    # 创建数据集
    train_dataset = DefectDataset(config.TRAIN_PATH, class_to_idx, train_transform, config.IMAGE_CHANNELS)
    val_dataset = DefectDataset(config.VAL_PATH, class_to_idx, val_test_transform, config.IMAGE_CHANNELS)
    test_dataset = val_dataset  # 🔥 无test集：用val集替代测试集

    # 创建DataLoader（Windows强制num_workers=0）
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=False
    )

    # 打印数据加载信息
    print(f"\n✅ 数据加载完成：")
    print(f"训练集：{len(train_dataset)} 样本")
    print(f"验证集：{len(val_dataset)} 样本")
    print(f"测试集：使用验证集替代（{len(test_dataset)} 样本）")
    return train_loader, val_loader, test_loader

# ===================== CNN模型（轻量型，适配小图像）=====================
class DefectCNN(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        in_channels = config.IMAGE_CHANNELS
        num_classes = config.NUM_CLASSES
        dropout = config.DROPOUT_RATE

        # 卷积特征提取（32x32→4x4）
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32→16

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16→8

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 8→4
        )

        # 分类头（全连接层）
        self.fc_layers = nn.Sequential(
            nn.Flatten(),  # 64*4*4 = 1024
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        """前向传播"""
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ===================== 工具函数（指标计算+可视化）=====================
def calculate_metrics(outputs: torch.Tensor, labels: torch.Tensor) -> Tuple[float, float]:
    """计算准确率和F1分数"""
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    return accuracy_score(labels, preds), f1_score(labels, preds, average="macro")

def plot_curves(train_losses: List[float], val_losses: List[float],
                train_accs: List[float], val_accs: List[float],
                train_f1s: List[float], val_f1s: List[float], save_path: Path):
    """绘制损失和指标曲线"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="训练损失", marker="o", markersize=4)
    plt.plot(epochs, val_losses, label="验证损失", marker="s", markersize=4)
    plt.xlabel("训练轮数（Epoch）")
    plt.ylabel("损失值")
    plt.legend()
    plt.grid(alpha=0.3)

    # 指标曲线（准确率+F1）
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="训练准确率", marker="o", markersize=4)
    plt.plot(epochs, val_accs, label="验证准确率", marker="s", markersize=4)
    plt.plot(epochs, train_f1s, label="训练F1分数", marker="^", markersize=4)
    plt.plot(epochs, val_f1s, label="验证F1分数", marker="d", markersize=4)
    plt.xlabel("训练轮数（Epoch）")
    plt.ylabel("指标值（0-1）")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 损失/指标曲线已保存：{save_path}")

def plot_confusion_matrix(labels: np.ndarray, preds: np.ndarray, class_names: List[str], save_path: Path):
    """绘制混淆矩阵（基于验证集，因无test集）"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        annot_kws={"fontsize": 10}
    )
    plt.xlabel("预测类别", fontsize=12)
    plt.ylabel("真实类别", fontsize=12)
    plt.title("缺陷检测混淆矩阵（验证集）", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"✅ 混淆矩阵已保存：{save_path}")

# ===================== 训练流程（支持动态epoch+断点续训）=====================
def train_model(config: Config, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    """训练模型（含早停、最佳模型保存、断点续训）"""
    # 确保模型在正确设备上
    model = model.to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()  # 多分类损失
    
    # 优化器使用动态配置的超参数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    # 调度器使用动态的EPOCHS值
    scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)

    # ========== 断点续训核心逻辑 ==========
    start_epoch = 0
    best_metric = 0.0
    early_stop_counter = 0
    # 训练记录初始化
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    # 检查是否有保存的训练状态
    if config.TRAIN_STATE_PATH.exists():
        print(f"\n🔄 发现断点续训文件：{config.TRAIN_STATE_PATH}")
        # 加载训练状态（先加载到CPU，再迁移到目标设备）
        checkpoint = torch.load(config.TRAIN_STATE_PATH, map_location='cpu')
        
        # 1. 加载模型权重（确保在目标设备）
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(config.DEVICE)
        
        # 2. 重新初始化优化器（保留超参数）
        optimizer = optim.AdamW(
            model.parameters(),
            lr=checkpoint.get('lr', config.LEARNING_RATE),
            weight_decay=checkpoint.get('weight_decay', config.WEIGHT_DECAY)
        )
        
        # 3. 重新初始化调度器（使用新的EPOCHS值）
        scheduler = CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
        try:
            # 尝试加载调度器状态（兼容旧的断点文件）
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except:
            print(f"⚠️  调度器状态加载失败，已使用新的epoch数（{config.EPOCHS}）重新初始化")
        
        # 4. 加载其他训练状态
        start_epoch = checkpoint['epoch'] + 1
        best_metric = checkpoint['best_metric']
        early_stop_counter = checkpoint['early_stop_counter']
        # 加载历史指标
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        train_accs = checkpoint['train_accs']
        val_accs = checkpoint['val_accs']
        train_f1s = checkpoint['train_f1s']
        val_f1s = checkpoint['val_f1s']

        print(f"✅ 成功加载断点状态：")
        print(f"  - 上次训练到第 {checkpoint['epoch']} 轮")
        print(f"  - 本次训练总轮数：{config.EPOCHS}（从{start_epoch}开始）")
        print(f"  - 最佳{config.MONITOR_METRIC}：{best_metric:.4f}")
        print(f"  - 早停计数器：{early_stop_counter}/{config.EARLY_STOPPING_PATIENCE}")
    else:
        print(f"\n🚀 未发现断点文件，开始全新训练（总轮数：{config.EPOCHS}）")

    print("\n" + "="*60 + " 开始训练 " + "="*60 + "\n")
    start_time = time.time()

    # 关键修改：循环上限是config.EPOCHS（动态值）
    for epoch in range(start_epoch, config.EPOCHS):
        # ---------------------- 训练阶段 ----------------------
        model.train()
        train_total_loss = 0.0
        train_all_preds, train_all_labels = [], []

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播+优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累计损失和预测结果
            train_total_loss += loss.item() * images.size(0)
            train_all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_all_labels.extend(labels.cpu().numpy())

            # 打印批次日志（每10个batch打印一次）
            if (batch_idx + 1) % 10 == 0:
                batch_acc = accuracy_score(labels.cpu().numpy(), torch.argmax(outputs, dim=1).cpu().numpy())
                print(f"Epoch [{epoch+1}/{config.EPOCHS}] | Batch [{batch_idx+1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | Acc: {batch_acc:.4f}")

        # 计算训练集指标
        train_avg_loss = train_total_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_all_labels, train_all_preds)
        train_f1 = f1_score(train_all_labels, train_all_preds, average="macro")

        # ---------------------- 验证阶段 ----------------------
        model.eval()
        val_total_loss = 0.0
        val_all_preds, val_all_labels = [], []

        with torch.no_grad():  # 禁用梯度计算，加速验证
            for images, labels in val_loader:
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_total_loss += loss.item() * images.size(0)
                val_all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_all_labels.extend(labels.cpu().numpy())

        # 计算验证集指标
        val_avg_loss = val_total_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_all_labels, val_all_preds)
        val_f1 = f1_score(val_all_labels, val_all_preds, average="macro")

        # ---------------------- 记录与保存 ----------------------
        # 保存指标
        train_losses.append(train_avg_loss)
        val_losses.append(val_avg_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        # 打印轮次日志
        print(f"\n📊 Epoch [{epoch+1}/{config.EPOCHS}] 总结：")
        print(f"训练集 - 损失：{train_avg_loss:.4f} | 准确率：{train_acc:.4f} | F1：{train_f1:.4f}")
        print(f"验证集 - 损失：{val_avg_loss:.4f} | 准确率：{val_acc:.4f} | F1：{val_f1:.4f}\n")

        # 学习率调度
        scheduler.step()

        # 保存最佳模型（基于监控指标）
        current_metric = val_f1 if config.MONITOR_METRIC == "val_f1" else val_acc
        if current_metric > best_metric:
            best_metric = current_metric
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"🏆 保存最佳模型（{config.MONITOR_METRIC}：{best_metric:.4f}）\n")
            early_stop_counter = 0  # 重置早停计数器
        else:
            early_stop_counter += 1
            print(f"⚠️  早停计数器：{early_stop_counter}/{config.EARLY_STOPPING_PATIENCE}\n")
            # 早停触发
            if early_stop_counter >= config.EARLY_STOPPING_PATIENCE:
                print(f"✅ 早停触发！验证集{config.MONITOR_METRIC}已{config.EARLY_STOPPING_PATIENCE}轮无提升")
                break

        # ========== 保存训练状态（包含当前epoch数） ==========
        train_state = {
            'epoch': epoch,  # 当前训练到的轮数
            'model_state_dict': model.state_dict(),  # 模型权重（核心）
            'scheduler_state_dict': scheduler.state_dict(),  # 调度器状态
            'best_metric': best_metric,  # 最佳指标
            'early_stop_counter': early_stop_counter,  # 早停计数器
            'lr': config.LEARNING_RATE,  # 学习率
            'weight_decay': config.WEIGHT_DECAY,  # 权重衰减
            'total_epochs': config.EPOCHS,  # 新增：记录本次训练的总轮数
            # 历史指标记录
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'train_f1s': train_f1s,
            'val_f1s': val_f1s
        }
        # 保存到CPU，避免CUDA张量问题
        torch.save(train_state, config.TRAIN_STATE_PATH)
        print(f"💾 已保存训练状态：{config.TRAIN_STATE_PATH}\n")

    # 训练结束：绘制曲线
    total_train_time = (time.time() - start_time) / 60
    print(f"\n" + "="*60 + " 训练完成 " + "="*60)
    print(f"总训练时间：{total_train_time:.2f} 分钟")
    print(f"实际训练轮数：{epoch+1 - start_epoch}（从{start_epoch}到{epoch+1}）")
    print(f"最佳{config.MONITOR_METRIC}：{best_metric:.4f}")
    print(f"最佳模型路径：{config.BEST_MODEL_PATH}")
    plot_curves(train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s, config.LOSS_PLOT_PATH)

# ===================== 测试流程（用val集替代test集）=====================
def test_model(config: Config, model: nn.Module, test_loader: DataLoader):
    """测试模型（基于验证集，因无test集）"""
    model = model.to(config.DEVICE)
    # 加载最佳模型
    if not config.BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ 未找到最佳模型：{config.BEST_MODEL_PATH}")
    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model.eval()  # 切换到评估模式

    criterion = nn.CrossEntropyLoss()
    test_total_loss = 0.0
    all_preds, all_labels = [], []

    print("\n" + "="*60 + " 开始测试（使用验证集替代） " + "="*60 + "\n")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_total_loss += loss.item() * images.size(0)
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算核心指标（避免除零错误）
    test_avg_loss = test_total_loss / len(test_loader.dataset)
    test_acc = accuracy_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    test_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    test_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # 打印测试结果
    print(f"📋 测试结果（验证集）：")
    print(f"平均损失：{test_avg_loss:.4f}")
    print(f"准确率（Accuracy）：{test_acc:.4f}")
    print(f"精确率（Precision）：{test_precision:.4f}")
    print(f"召回率（Recall）：{test_recall:.4f}")
    print(f"F1分数（F1-Score）：{test_f1:.4f}\n")

    # 打印详细分类报告
    print("📋 分类详细报告：")
    print(classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    # 绘制混淆矩阵
    plot_confusion_matrix(np.array(all_labels), np.array(all_preds), config.CLASS_NAMES, config.CONFUSION_MATRIX_PATH)

# ===================== 单张图片预测函数=====================
def predict_single_image(
    config: Config,
    model: nn.Module,
    image_path: str or Path,
    show_image: bool = True,
    save_result: bool = True
) -> Dict[str, any]:
    """
    用训练好的模型预测单张图片的类别
    
    参数：
        config: 配置类实例（包含类别名、图像尺寸等）
        model: 训练好的模型实例
        image_path: 测试图片的路径（字符串/Path对象）
        show_image: 是否显示预测结果图片（含类别+置信度）
        save_result: 是否保存预测结果图片到结果目录
    
    返回：
        预测结果字典，包含：
            - pred_class: 预测类别名称（如 "Cr"）
            - pred_idx: 预测类别索引（0-5）
            - confidence: 预测置信度（0-1）
            - all_confidences: 所有类别的置信度列表
            - image_path: 输入图片路径
    """
    # 1. 路径校验
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"❌ 图片路径不存在：{image_path}")
    
    # 2. 模型准备（切换到评估模式，禁用梯度）
    model = model.to(config.DEVICE)
    model.eval()
    
    # 3. 图片预处理（与训练时的验证集预处理保持一致）
    transform = transforms.Compose([
        transforms.Resize((config.IMG_HEIGHT, config.IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(config.NORMALIZE_MEAN, config.NORMALIZE_STD)
    ])
    
    try:
        # 加载图片（兼容RGB/灰度，与训练时的通道数一致）
        mode = "RGB" if config.IMAGE_CHANNELS == 3 else "L"
        image = Image.open(image_path).convert(mode)
        original_image = image.copy()  # 保存原始图片用于可视化
    except Exception as e:
        raise RuntimeError(f"❌ 加载图片失败：{str(e)}")
    
    # 预处理并添加batch维度（模型要求输入是[batch, channel, h, w]）
    input_tensor = transform(image).unsqueeze(0).to(config.DEVICE)
    
    # 4. 模型推理（禁用梯度计算，加速推理）
    with torch.no_grad():
        outputs = model(input_tensor)  # 输出为logits
        probabilities = torch.softmax(outputs, dim=1)  # 转换为概率（0-1）
        pred_idx = torch.argmax(probabilities, dim=1).item()  # 预测类别索引
        pred_confidence = probabilities[0][pred_idx].item()  # 预测置信度
        all_confidences = probabilities[0].cpu().numpy().tolist()  # 所有类别置信度
    
    # 5. 结果解析
    pred_class = config.CLASS_NAMES[pred_idx]
    result = {
        "pred_class": pred_class,
        "pred_idx": pred_idx,
        "confidence": round(pred_confidence, 4),
        "all_confidences": [round(c, 4) for c in all_confidences],
        "image_path": str(image_path)
    }
    
    # 6. 打印清晰的预测结果
    print("\n" + "="*50 + " 单张图片预测结果 " + "="*50)
    print(f"图片路径：{image_path}")
    print(f"预测类别：{pred_class} (索引：{pred_idx})")
    print(f"预测置信度：{result['confidence'] * 100:.2f}%")
    print("\n所有类别置信度：")
    for cls_name, conf in zip(config.CLASS_NAMES, result['all_confidences']):
        print(f"  - {cls_name}: {conf * 100:.2f}%")
    print("="*110 + "\n")
    
    # 7. 可视化结果（可选）
    if show_image or save_result:
        plt.figure(figsize=(8, 6))
        plt.imshow(original_image)
        plt.axis('off')  # 隐藏坐标轴
        # 添加预测结果文本
        text = f"预测类别：{pred_class}\n置信度：{pred_confidence * 100:.2f}%"
        plt.text(
            10, 10, text, 
            fontsize=12, color='white', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='red', alpha=0.8)
        )
        plt.title(f"缺陷检测结果：{pred_class}", fontsize=14, fontweight='bold')
        
        # 保存结果图片
        if save_result:
            save_path = config.ROOT_SAVE_DIR / f"single_pred_{image_path.stem}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 预测结果图片已保存：{save_path}")
        
        # 显示图片（可选）
        if show_image:
            plt.show()
        plt.close()
    
    return result

# ===================== 主函数（支持命令行传参修改epoch）=====================
def main():
    # 新增：解析命令行参数（保留原有参数，新增单张图片测试参数）
    parser = argparse.ArgumentParser(description='缺陷检测训练/测试脚本')
    parser.add_argument('--epochs', type=int, default=50, help='训练总轮数（默认50）')
    parser.add_argument('--predict', type=str, default=None, help='单张图片测试路径（如：./test_img.png）')
    args = parser.parse_args()

    try:
        # 1. 初始化配置
        config = Config(epochs=args.epochs)
        print(f"\n📌 训练配置：")
        print(f"设备：{config.DEVICE} | 类别数：{config.NUM_CLASSES} | 批次大小：{config.BATCH_SIZE}")
        print(f"训练轮数：{config.EPOCHS} | 学习率：{config.LEARNING_RATE}")

        # 2. 初始化模型
        model = DefectCNN(config)
        print(f"\n📌 模型信息：")
        print(f"模型参数总数：{sum(p.numel() for p in model.parameters()):,}")

        # ====== 新增：单张图片测试逻辑 ======
        if args.predict:
            # 加载训练好的最佳模型
            if not config.BEST_MODEL_PATH.exists():
                raise FileNotFoundError(f"❌ 未找到训练好的模型：{config.BEST_MODEL_PATH}\n请先训练模型！")
            
            # 加载模型权重
            model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
            print(f"✅ 成功加载最佳模型：{config.BEST_MODEL_PATH}")
            
            # 调用单张图片预测函数
            predict_single_image(
                config=config,
                model=model,
                image_path=args.predict,
                show_image=True,  # 显示图片
                save_result=True   # 保存结果
            )
            return  # 仅测试单张图片，不执行训练/验证集测试
        
        # ====== 原有逻辑（训练+验证集测试） ======
        # 3. 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(config)

        # 4. 训练模型
        train_model(config, model, train_loader, val_loader)

        # 5. 测试模型（验证集）
        test_model(config, model, test_loader)

        print("\n" + "="*60 + " 所有任务完成！ " + "="*60)
        print(f"结果保存目录：{config.ROOT_SAVE_DIR}")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{str(e)}")
        raise

if __name__ == "__main__":
    main()