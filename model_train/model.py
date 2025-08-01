import torch
import torch.nn as nn
import torch.nn.functional as F

class TSCeption(nn.Module):
    """时间序列ception模型"""
    def __init__(self, num_classes=2, feature_dim=2):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 卷积层
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(64 * 5, 128)  # 假设输入长度为20
        self.fc2 = nn.Linear(128, feature_dim)  # 特征层
        self.fc3 = nn.Linear(feature_dim, num_classes)  # 分类层
        
        # 批归一化和激活函数
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入形状: [batch_size, 1, seq_len]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)  # [batch_size, features]
        
        x = self.fc1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 特征层
        features = self.fc2(x)
        features = self.relu(features)
        
        # 分类层
        logits = self.fc3(features)
        
        return logits, features

class CCNN(nn.Module):
    """紧凑型卷积神经网络"""
    def __init__(self, num_classes=10, feature_dim=1024):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        
        # 卷积块1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 卷积块2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 卷积块3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 1024)  # 假设输入图像32x32
        self.fc2 = nn.Linear(1024, feature_dim)  # 特征层
        self.fc3 = nn.Linear(feature_dim, num_classes)  # 分类层
        
        # 其他层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 输入形状: [batch_size, 3, 32, 32]
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # 展平特征
        x = x.view(x.size(0), -1)  # [batch_size, features]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # 特征层
        features = self.fc2(x)
        features = self.relu(features)
        
        # 分类层
        logits = self.fc3(features)
        
        return logits, features

def get_model(architecture, num_classes=2, feature_dim=None):
    """
    获取指定架构的模型
    
    参数:
        architecture: 模型架构名称 ("TSCeption" 或 "CCNN")
        num_classes: 类别数量
        feature_dim: 特征维度，为None时使用默认值
        
    返回:
        初始化的模型
    """
    if architecture == "TSCeption":
        # TSCeption默认特征维度为2
        feature_dim = feature_dim if feature_dim is not None else 2
        return TSCeption(num_classes=num_classes, feature_dim=feature_dim)
    elif architecture == "CCNN":
        # CCNN默认特征维度为1024
        feature_dim = feature_dim if feature_dim is not None else 1024
        return CCNN(num_classes=num_classes, feature_dim=feature_dim)
    else:
        raise ValueError(f"不支持的模型架构: {architecture}")