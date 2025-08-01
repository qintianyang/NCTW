import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from datetime import datetime
from .utils import get_device

def hard_label_attack(target_model, surrogate_model, train_loader, aux_loader, val_loader, 
                     epochs=200, lr=0.001, result_model_path="./attack_results", 
                     device=None, config=None):
    """
    硬标签攻击实现：使用目标模型的输出标签来训练替代模型
    
    参数:
        target_model: 目标模型（被攻击模型）
        surrogate_model: 替代模型（攻击模型）
        train_loader: 主要训练数据加载器
        aux_loader: 辅助训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        result_model_path: 模型保存路径
        device: 计算设备
        config: 配置字典
        
    返回:
        surrogate_model: 训练好的替代模型
        best_val_acc: 最佳验证准确率
        total_time: 总训练时间
    """
    device = device or get_device()
    config = config or {}
    os.makedirs(result_model_path, exist_ok=True)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(surrogate_model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 模型移至设备
    target_model.to(device)
    surrogate_model.to(device)
    target_model.eval()  # 目标模型保持评估模式
    
    best_val_acc = 0.0
    start_time = datetime.now()
    print(f"硬标签攻击开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        surrogate_model.train()
        
        # 每3个epoch切换数据加载器
        current_loader = aux_loader if (epoch+1) % 3 == 0 else train_loader
        pbar = tqdm(current_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for inputs, true_labels in pbar:
            inputs = inputs.to(device)
            true_labels = true_labels.to(device)
            optimizer.zero_grad()
            
            # 获取目标模型的硬标签
            with torch.no_grad():
                target_outputs = target_model(inputs)
                _, target_labels = torch.max(target_outputs[0], 1)  # 取预测概率最高的类别
            
            # 前100个epoch使用真实标签，之后使用目标模型标签
            if epoch < 100:
                labels = true_labels
            else:
                labels = target_labels
            
            # 替代模型前向传播
            surrogate_outputs = surrogate_model(inputs)
            loss = criterion(surrogate_outputs[0], labels)
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(surrogate_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # 统计指标
            total_loss += loss.item()
            _, surrogate_preds = torch.max(surrogate_outputs[0], 1)
            correct += (surrogate_preds == labels).sum().item()
            total += inputs.size(0)
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}', 
                'Acc': f'{100 * correct/total:.2f}%'
            })
        
        # 调整学习率
        scheduler.step()
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(current_loader)
        avg_acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.2f}%')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 验证模型
        val_acc = validate_attack(surrogate_model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': surrogate_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
            }, os.path.join(result_model_path, 'best_attack_model.pth'))
            print(f"保存最佳攻击模型 (准确率: {best_val_acc:.2f}%)")
    
    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': surrogate_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_acc': best_val_acc,
    }, os.path.join(result_model_path, 'final_attack_model.pth'))
    
    # 计算总时间
    end_time = datetime.now()
    total_time = end_time - start_time
    print(f"\n硬标签攻击完成! 总耗时: {total_time}")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    return surrogate_model, best_val_acc, total_time

def validate_attack(model, val_loader, device):
    """验证攻击模型性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs[0], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f'攻击模型验证准确率: {acc:.2f}%')
    return acc