from torch import nn

class FTLL(nn.Module):
    """
    Fine-Tune Last Layer (FTLL) 微调策略:
    仅解冻并训练模型的最后一层
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False
        # 解冻最后一层
        self._unfreeze_last_layer()

    def _unfreeze_last_layer(self):
        """解冻模型的最后一层"""
        # 尝试多种方式获取最后一层
        if hasattr(self.model, 'fc'):  # 对于有fc层的模型
            last_layer = self.model.fc
        elif hasattr(self.model, 'classifier'):  # 对于有classifier的模型
            last_layer = self.model.classifier
        else:  # 尝试获取最后一个子模块
            children = list(self.model.named_children())
            if children:
                last_layer = children[-1][1]
            else:
                raise ValueError("无法确定模型的最后一层")
        
        # 解冻最后一层
        for param in last_layer.parameters():
            param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class FTAL(nn.Module):
    """
    Fine-Tune All Layers (FTAL) 微调策略:
    解冻并训练模型的所有层
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 解冻所有层
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class RTLL(nn.Module):
    """
    Retrain Last Layer (RTLL) 策略:
    重新初始化并训练最后一层，冻结其他层
    """
    def __init__(self, model, num_classes=None):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        
        # 冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 重新初始化最后一层
        self._reinit_last_layer()

    def _reinit_last_layer(self):
        """重新初始化最后一层"""
        # 获取最后一层输入特征维度
        if hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            # 创建新的全连接层
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif hasattr(self.model, 'classifier'):
            # 处理可能的分类器序列
            if isinstance(self.model.classifier, nn.Sequential) and \
               isinstance(self.model.classifier[-1], nn.Linear):
                in_features = self.model.classifier[-1].in_features
                self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
            else:
                in_features = self.model.classifier.in_features
                self.model.classifier = nn.Linear(in_features, self.num_classes)
        else:
            raise ValueError("无法确定模型的最后一层")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

class RTAL(nn.Module):
    """
    Retrain All Layers (RTAL) 策略:
    重新初始化并训练所有层
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        # 重新初始化所有层
        self._reinit_all_layers()

    def _reinit_all_layers(self):
        """重新初始化所有层"""
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)