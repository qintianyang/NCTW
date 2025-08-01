import torch
import torch.nn as nn

class NeuralCollapseLoss(nn.Module):
    """神经坍缩损失函数实现"""
    def __init__(self, epsilon=5.0):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, features, target_means, target_labels):
        """
        计算神经坍缩损失
        
        参数:
            features: 样本特征 [B, D]
            target_means: 类别均值字典 {class_idx: mean_vector}
            target_labels: 标签 [B]
        """
        losses = []
        for feat, label in zip(features, target_labels):
            mean = target_means[label.item()]
            dist = torch.norm(feat - mean, p=2)  # L2距离计算
            losses.append(torch.clamp(self.epsilon - dist, min=0))  # Hinge loss
        return torch.mean(torch.stack(losses))

class CombinedLoss(nn.Module):
    """组合损失函数：交叉熵损失 + 神经坍缩损失"""
    def __init__(self, cls_weight=1.0, collapse_weight=0.1, collapse_epsilon=5.0):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss()
        self.collapse_criterion = NeuralCollapseLoss(epsilon=collapse_epsilon)
        self.cls_weight = cls_weight
        self.collapse_weight = collapse_weight

    def forward(self, outputs, features, target_means, target_labels):
        """计算组合损失"""
        cls_loss = self.cls_criterion(outputs, target_labels)
        collapse_loss = self.collapse_criterion(features, target_means, target_labels)
        return self.cls_weight * cls_loss + self.collapse_weight * collapse_loss