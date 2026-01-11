import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import os.path as osp

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ===================== 1. 参数设置模块 =====================
def get_config():
    root_save_dir = "./results/BreastCancer"
    config = {
        "data_path": "./resources/breast_cancer_data.csv",
        "test_size": 0.2,
        "random_state": 22,
        "batch_size": 32,
        "epochs": 1000,
        "learning_rate": 0.001,
        "input_dim": 30,
        "hidden1_dim": 16,
        "hidden2_dim": 8,
        "output_dim": 2,
        "model_save_dir": osp.join(root_save_dir, "model_checkpoints"),
        "scaler_save_dir": osp.join(root_save_dir, "scaler"),
        "loss_curve_path": osp.join(root_save_dir, "loss_curve.png"),
        "acc_curve_path": osp.join(root_save_dir, "accuracy_curve.png"),
        "confusion_matrix_path": osp.join(root_save_dir, "confusion_matrix.png"),
        "full_model_name": "breast_cancer_full_model.pth",
        "scaler_name": "minmax_scaler.npy",
        "log_interval": 200
    }
    return config


# ===================== 2. 工具函数：自动创建目录 =====================
def create_dir_if_not_exist(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)
        print(f"已创建目录：{dir_path}")


# ===================== 3. 数据处理模块 =====================
def load_and_preprocess_data(config, load_scaler=False, return_full_data=False):
    create_dir_if_not_exist(config["scaler_save_dir"])
    
    dataSet = pd.read_csv(config["data_path"])
    x_full = dataSet.iloc[:, :-1].values
    y_full = dataSet['target'].values

    if not return_full_data:
        x_train, x_test, y_train, y_test = train_test_split(
            x_full, y_full, test_size=config["test_size"], random_state=config["random_state"]
        )
    else:
        x_train, x_test, y_train, y_test = None, x_full, None, y_full

    scaler_save_path = osp.join(config["scaler_save_dir"], config["scaler_name"])
    sc = MinMaxScaler(feature_range=(0, 1))
    if load_scaler and osp.exists(scaler_save_path):
        scaler_params = np.load(scaler_save_path, allow_pickle=True).item()
        sc.min_ = scaler_params["min_"]
        sc.scale_ = scaler_params["scale_"]
        sc.data_min_ = scaler_params["data_min_"]
        sc.data_max_ = scaler_params["data_max_"]
        sc.data_range_ = scaler_params["data_range_"]
        
        if not return_full_data:
            x_train = sc.transform(x_train)
        x_test = sc.transform(x_test)
        print(f"已加载归一化器：{scaler_save_path}")
    else:
        if not return_full_data and not load_scaler:
            x_train = sc.fit_transform(x_train)
            x_test = sc.transform(x_test)
            
            scaler_params = {
                "min_": sc.min_,
                "scale_": sc.scale_,
                "data_min_": sc.data_min_,
                "data_max_": sc.data_max_,
                "data_range_": sc.data_range_
            }
            np.save(scaler_save_path, scaler_params)
            print(f"归一化器已保存至：{scaler_save_path}")

    if not return_full_data:
        x_train = torch.tensor(x_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
    
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = None
    if not return_full_data:
        train_dataset = TensorDataset(x_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )

    if return_full_data:
        return x_test, y_test, sc
    else:
        return x_train, x_test, y_train, y_test, train_loader, sc


# ===================== 4. 模型定义模块 =====================
class BreastCancerModel(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden1_dim),
            nn.ReLU(),
            nn.Linear(hidden1_dim, hidden2_dim),
            nn.ReLU(),
            nn.Linear(hidden2_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)


# ===================== 5. 模型保存/加载模块（修复首次训练报错）=====================
def save_full_model(config, model, optimizer, epoch, train_loss, val_loss, train_acc, val_acc):
    create_dir_if_not_exist(config["model_save_dir"])
    model_save_path = osp.join(config["model_save_dir"], config["full_model_name"])
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "learning_rate": config["learning_rate"]
    }
    torch.save(checkpoint, model_save_path)
    print(f"\n模型已保存至：{model_save_path}（第{epoch}轮）")


def load_full_model(config, model, optimizer=None, only_model=False):
    """
    首次训练时未找到模型文件，返回初始状态而非报错
    """
    model_save_path = osp.join(config["model_save_dir"], config["full_model_name"])
    
    if not osp.exists(model_save_path):
        if only_model:
            # 仅加载模型时，未找到文件则报错（测试时必须有模型）
            raise FileNotFoundError(f"未找到模型文件：{model_save_path}（请先训练模型）")
        else:
            # 训练/续训时，未找到文件则返回初始状态
            print(f"未找到模型文件：{model_save_path}，将从头开始训练")
            return model, optimizer, 0, [], [], [], []
    
    # 找到模型文件，正常加载
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"已加载模型权重：{model_save_path}")

    if only_model:
        return model
    
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    train_loss = checkpoint.get("train_loss", [])
    val_loss = checkpoint.get("val_loss", [])
    train_acc = checkpoint.get("train_acc", [])
    val_acc = checkpoint.get("val_acc", [])
    
    print(f"已加载训练状态，从第{start_epoch}轮开始训练")
    return model, optimizer, start_epoch, train_loss, val_loss, train_acc, val_acc


# ===================== 6. 训练流程模块 =====================
def train_model(model, train_loader, x_test, y_test, loss_fn, optimizer, config):
    model, optimizer, start_epoch, train_loss, val_loss, train_acc, val_acc = load_full_model(
        config, model, optimizer, only_model=False
    )

    model.train()
    for epoch in range(start_epoch, config["epochs"]):
        total_train_loss = 0.0
        train_preds = []
        train_labels = []

        for batch_x, batch_y in train_loader:
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_x.size(0)
            train_pred_labels = torch.argmax(y_pred, dim=1)
            train_preds.extend(train_pred_labels.numpy())
            train_labels.extend(batch_y.numpy())

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_loss.append(avg_train_loss)
        train_acc.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            y_test_pred = model(x_test)
            val_loss_val = loss_fn(y_test_pred, y_test).item()
            val_pred_labels = torch.argmax(y_test_pred, dim=1)
            val_accuracy = accuracy_score(y_test.numpy(), val_pred_labels.numpy())
            
            val_loss.append(val_loss_val)
            val_acc.append(val_accuracy)

        if (epoch + 1) % config["log_interval"] == 0:
            print(f'Epoch [{epoch + 1}/{config["epochs"]}]')
            print(f'Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f}')
            print(f'Val Loss: {val_loss_val:.4f} | Val Acc: {val_accuracy:.4f}')
            save_full_model(
                config, model, optimizer, epoch + 1,
                train_loss, val_loss, train_acc, val_acc
            )

        model.train()

    save_full_model(
        config, model, optimizer, config["epochs"],
        train_loss, val_loss, train_acc, val_acc
    )

    return train_loss, val_loss, train_acc, val_acc


# ===================== 7. 可视化模块 =====================
def plot_loss_curve(train_loss, val_loss, config):
    create_dir_if_not_exist(osp.dirname(config["loss_curve_path"]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, 
             label='训练损失', linewidth=2, color='#2E86AB')
    plt.plot(range(1, len(val_loss) + 1), val_loss, 
             label='验证损失', linewidth=2, linestyle='--', color='#A23B72')
    
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.title('乳腺癌分类模型 - 损失曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config["loss_curve_path"], dpi=300, bbox_inches='tight')
    print(f'\n损失曲线已保存至：{config["loss_curve_path"]}')


def plot_accuracy_curve(train_acc, val_acc, config):
    create_dir_if_not_exist(osp.dirname(config["acc_curve_path"]))
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_acc) + 1), train_acc, 
             label='训练准确率', linewidth=2, color='#F18F01')
    plt.plot(range(1, len(val_acc) + 1), val_acc, 
             label='验证准确率', linewidth=2, linestyle='--', color='#C73E1D')
    
    plt.xlabel('训练轮数', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.title('乳腺癌分类模型 - 准确率曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.8, 1.0)
    plt.tight_layout()
    plt.savefig(config["acc_curve_path"], dpi=300, bbox_inches='tight')
    print(f'准确率曲线已保存至：{config["acc_curve_path"]}')


def plot_confusion_matrix(y_true, y_pred, config):
    create_dir_if_not_exist(osp.dirname(config["confusion_matrix_path"]))
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['良性', '恶性'],
        yticklabels=['良性', '恶性']
    )
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('乳腺癌分类模型 - 混淆矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(config["confusion_matrix_path"], dpi=300, bbox_inches='tight')
    print(f'混淆矩阵已保存至：{config["confusion_matrix_path"]}')


# ===================== 8. 核心测试函数 =====================
def test_model(config, use_full_data=False):
    print("\n" + "="*60)
    print("开始测试训练好的模型...")
    print("="*60)

    if use_full_data:
        x_test, y_test, sc = load_and_preprocess_data(config, load_scaler=True, return_full_data=True)
        print(f"测试数据：完整数据集（{len(x_test)}个样本）")
    else:
        _, x_test, _, y_test, _, sc = load_and_preprocess_data(config, load_scaler=True, return_full_data=False)
        print(f"测试数据：训练集划分的测试集（{len(x_test)}个样本）")

    model = BreastCancerModel(
        input_dim=config["input_dim"],
        hidden1_dim=config["hidden1_dim"],
        hidden2_dim=config["hidden2_dim"],
        output_dim=config["output_dim"]
    )
    model = load_full_model(config, model, only_model=True)
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test)
        y_pred_labels = torch.argmax(y_pred, dim=1)

    test_acc = accuracy_score(y_test.numpy(), y_pred_labels.numpy())
    test_report = classification_report(
        y_test.numpy(), y_pred_labels.numpy(),
        labels=[0, 1],
        target_names=['良性', '恶性'],
        output_dict=False
    )

    print(f"\n测试集准确率：{test_acc:.4f}")
    print("\n分类报告：")
    print(test_report)

    plot_confusion_matrix(y_test.numpy(), y_pred_labels.numpy(), config)

    print("\n" + "="*60)
    print("模型测试完成！")
    print("="*60)

    return test_acc, test_report


# ===================== 主函数 =====================
def main():
    config = get_config()
    print("配置参数加载完成")

    x_train, x_test, y_train, y_test, train_loader, sc = load_and_preprocess_data(config, load_scaler=False)
    print(f"数据预处理完成（训练集：{len(x_train)}样本，测试集：{len(x_test)}样本）")

    model = BreastCancerModel(
        input_dim=config["input_dim"],
        hidden1_dim=config["hidden1_dim"],
        hidden2_dim=config["hidden2_dim"],
        output_dim=config["output_dim"]
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    print("模型初始化完成")

    print("\n开始训练...")
    train_loss, val_loss, train_acc, val_acc = train_model(
        model, train_loader, x_test, y_test, loss_fn, optimizer, config
    )

    plot_loss_curve(train_loss, val_loss, config)
    plot_accuracy_curve(train_acc, val_acc, config)

    test_model(config, use_full_data=False)

    print("\n整个流程（训练+测试）全部完成！")


if __name__ == "__main__":
    main()