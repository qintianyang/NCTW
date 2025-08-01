import yaml
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# 导入公共模块
from common.utils import set_seed, get_device, print_device_info
from common.trainer import ClassifierTrainer
from common.attacks import hard_label_attack
from common.fine_tuning import FTAL  # 使用不同的微调策略

# 导入模型定义
from model_train.model import get_model

# 自定义数据集（与exp4_tes类似，但参数不同）
class CNNDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=10, image_size=32):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.image_size = image_size
        # 生成随机图像数据 (通道数, 高, 宽)
        self.data = torch.randn(num_samples, 3, image_size, image_size)
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_data_loaders(config):
    """获取数据加载器"""
    # 创建数据集
    train_dataset = CNNDataset(
        num_samples=10000, 
        num_classes=config['num_classes'],
        image_size=32
    )
    val_dataset = CNNDataset(
        num_samples=2000, 
        num_classes=config['num_classes'],
        image_size=32
    )
    pre_dataset = CNNDataset(
        num_samples=5000, 
        num_classes=config['num_classes'],
        image_size=32
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    pre_loader = DataLoader(
        pre_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    
    return train_loader, val_loader, pre_loader

def main():
    # 加载配置文件
    config_path = os.path.join("../../configs", "exp1_CCNN.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    print("实验配置:", config)
    
    # 设置随机种子
    set_seed(42)
    
    # 获取设备
    device = get_device()
    print_device_info(device)
    
    # 创建模型保存目录
    result_dir = os.path.join("./results", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(result_dir, exist_ok=True)
    
    # 1. 训练目标模型
    print("\n===== 开始训练目标模型 =====")
    # 获取模型 - 使用CCNN架构
    target_model = get_model(
        architecture="CCNN",
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim']
    )
    
    # 使用FTAL微调策略（微调所有层）
    target_model = FTAL(target_model)
    
    # 定义优化器和学习率调度器
    optimizer = optim.Adam(
        target_model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['scheduler_step_size'],
        gamma=config['scheduler_gamma']
    )
    
    # 获取数据加载器
    train_loader, val_loader, pre_loader = get_data_loaders(config)
    
    # 初始化训练器并开始训练
    trainer = ClassifierTrainer(
        model=target_model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        config=config
    )
    
    # 开始训练
    best_acc = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        pre_loader=pre_loader,
        epochs=config['epochs'],
        save_path=os.path.join(result_dir, "target_model")
    )
    print(f"目标模型最佳验证准确率: {best_acc:.2f}%")
    
    # 2. 执行模型攻击
    print("\n===== 开始执行硬标签攻击 =====")
    # 创建替代模型
    surrogate_model = get_model(
        architecture="CCNN",
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim']
    )
    
    # 执行攻击
    attack_results_dir = os.path.join(result_dir, "attack_results")
    _, best_attack_acc, _ = hard_label_attack(
        target_model=target_model,
        surrogate_model=surrogate_model,
        train_loader=train_loader,
        aux_loader=pre_loader,
        val_loader=val_loader,
        epochs=config['epochs'],
        lr=config['learning_rate'],
        result_model_path=attack_results_dir,
        device=device,
        config=config
    )
    print(f"攻击模型最佳验证准确率: {best_attack_acc:.2f}%")
    
    print("\n===== 所有实验步骤完成 =====")

if __name__ == "__main__":
    from datetime import datetime  # 局部导入
    main()