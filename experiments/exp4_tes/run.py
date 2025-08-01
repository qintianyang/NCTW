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
from common.fine_tuning import FTLL  # 导入微调策略

# 导入模型定义
from model_train.model import get_model

# 自定义数据集示例（实际使用时替换为您的数据集）
class SimpleDataset(Dataset):
    def __init__(self, num_samples=1000, num_classes=2, feature_dim=20):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        # 生成随机数据
        self.data = torch.randn(num_samples, 1, feature_dim)  # 假设输入形状 [1, feature_dim]
        self.labels = torch.randint(0, num_classes, (num_samples,))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_data_loaders(config):
    """获取数据加载器"""
    # 创建数据集（实际使用时替换为您的真实数据加载逻辑）
    train_dataset = SimpleDataset(
        num_samples=5000, 
        num_classes=config['num_classes'],
        feature_dim=20  # 输入特征维度
    )
    val_dataset = SimpleDataset(
        num_samples=1000, 
        num_classes=config['num_classes'],
        feature_dim=20
    )
    pre_dataset = SimpleDataset(
        num_samples=2000, 
        num_classes=config['num_classes'],
        feature_dim=20
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2
    )
    pre_loader = DataLoader(
        pre_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    
    return train_loader, val_loader, pre_loader

def main():
    # 加载配置文件
    config_path = os.path.join("../../configs", "exp4_tes.yaml")
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
    # 获取模型
    target_model = get_model(
        architecture="TSCeption",
        num_classes=config['num_classes'],
        feature_dim=config['feature_dim']
    )
    
    # 使用FTLL微调策略
    target_model = FTLL(target_model)
    
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
        architecture="TSCeption",
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
    from datetime import datetime  # 局部导入，避免在其他脚本中不必要的导入
    main()