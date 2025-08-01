import torch
import torch.nn as nn
import pandas as pd
import os
from datetime import datetime
from .features import update_class_means
from .losses import CombinedLoss

class graphs:
    """训练过程中的指标记录器"""
    def __init__(self):
        self.accuracy = []
        self.loss = []
        self.reg_loss = []
        self.val_accuracy = []
        self.val_loss = []
        self.epochs = []

    def add_train_metrics(self, epoch, acc, loss, reg_loss=None):
        """添加训练指标"""
        self.epochs.append(epoch)
        self.accuracy.append(acc)
        self.loss.append(loss)
        self.reg_loss.append(reg_loss)

    def add_val_metrics(self, acc, loss):
        """添加验证指标"""
        self.val_accuracy.append(acc)
        self.val_loss.append(loss)

    def to_csv(self, filename, append=False):
        """将指标保存到CSV文件"""
        data = {
            'epoch': self.epochs,
            'train_accuracy': self.accuracy,
            'train_loss': self.loss,
            'train_reg_loss': self.reg_loss,
            'val_accuracy': self.val_accuracy,
            'val_loss': self.val_loss
        }
        
        # 确保所有列表长度一致
        max_length = max(len(v) for v in data.values()) if data else 0
        for key in data:
            data[key].extend([None] * (max_length - len(data[key])))
        
        df = pd.DataFrame(data)
        # 创建目录（如果不存在）
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        # 保存CSV
        df.to_csv(filename, mode='a' if append else 'w', header=not append, index=False)
        print(f"指标已保存到 {filename}")

class ClassifierTrainer:
    """分类器训练器，封装通用训练逻辑"""
    def __init__(self, model, optimizer, device, scheduler=None, config=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config or {}
        self.best_val_acc = 0.0
        self.graphs = graphs()
        
        # 初始化损失函数
        self.criterion = CombinedLoss(
            cls_weight=self.config.get('cls_weight', 1.0),
            collapse_weight=self.config.get('collapse_weight', 0.1),
            collapse_epsilon=self.config.get('collapse_epsilon', 5.0)
        )

    def train_one_epoch(self, train_loader, epoch):
        """训练单个epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 更新类别均值
        class_means = update_class_means(
            self.model, train_loader, 
            num_classes=self.config.get('num_classes', 2),
            device=self.device,
            feature_dim=self.config.get('feature_dim', 1024)
        )
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(data)
            logits, features = outputs[0], outputs[1]
            
            # 计算损失
            loss = self.criterion(logits, features, class_means, target)
            
            # 反向传播和优化
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # 打印批次信息
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        print(f'Epoch {epoch+1}, Train Loss: {avg_loss:.6f}, Accuracy: {acc:.2f}%')
        
        return avg_loss, acc

    def validate(self, val_loader):
        """验证模型性能"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                logits = outputs[0]
                
                # 计算损失
                loss = nn.CrossEntropyLoss()(logits, target)
                val_loss += loss.item()
                
                # 统计准确率
                _, predicted = torch.max(logits.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = val_loss / len(val_loader)
        acc = 100 * correct / total
        print(f'Validation Loss: {avg_loss:.6f}, Accuracy: {acc:.2f}%')
        
        return avg_loss, acc

    def fit(self, train_loader, val_loader, pre_loader=None, epochs=None, save_path="./results"):
        """完整训练流程"""
        epochs = epochs or self.config.get('epochs', 100)
        os.makedirs(save_path, exist_ok=True)
        
        # 记录开始时间
        start_time = datetime.now()
        print(f"训练开始于: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # 特殊epoch处理（如使用pre_loader）
            if pre_loader and (epoch + 1) % self.config.get('special_epoch_interval', 5) == 0:
                print("使用pre_loader进行训练...")
                train_loss, train_acc = self.train_one_epoch(pre_loader, epoch)
            else:
                train_loss, train_acc = self.train_one_epoch(train_loader, epoch)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
                print(f"学习率调整为: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 记录指标
            self.graphs.add_train_metrics(epoch+1, train_acc, train_loss)
            self.graphs.add_val_metrics(val_acc, val_loss)
            
            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': self.best_val_acc,
                }, os.path.join(save_path, 'best_model.pth'))
                print(f"保存最佳模型 (准确率: {self.best_val_acc:.2f}%)")
        
        # 训练结束，保存最终模型和指标
        torch.save({
            'epoch': epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
        }, os.path.join(save_path, 'final_model.pth'))
        
        # 保存指标
        self.graphs.to_csv(os.path.join(save_path, 'training_metrics.csv'))
        
        # 计算总训练时间
        end_time = datetime.now()
        total_time = end_time - start_time
        print(f"\n训练完成! 总耗时: {total_time}")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        
        return self.best_val_acc