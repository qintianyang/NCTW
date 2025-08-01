import torch
from torch.no_grad import no_grad

@no_grad()
def update_class_means(model, train_loader, num_classes, device, feature_dim):
    """
    计算并更新各类别的特征均值
    
    参数:
        model: 训练好的模型
        train_loader: 训练数据加载器
        num_classes: 类别数量
        device: 计算设备
        feature_dim: 特征维度
        
    返回:
        class_means: 类别均值字典 {class_idx: mean_vector}
    """
    model.eval()  # 设置为评估模式
    # 初始化类别特征和计数器
    class_sums = {i: torch.zeros(feature_dim).to(device) for i in range(num_classes)}
    class_counts = {i: 0 for i in range(num_classes)}
    
    for x, y in train_loader:
        # 获取模型输出，假设特征是模型输出的第二个返回值
        outputs = model(x.to(device))
        z = outputs[1]  # 特征向量
        
        for i in range(num_classes):
            mask = (y == i)
            if mask.any():
                class_sums[i] += z[mask].sum(dim=0)
                class_counts[i] += mask.sum().item()
    
    # 计算均值，处理可能的零计数情况
    class_means = {}
    for i in range(num_classes):
        if class_counts[i] > 0:
            class_means[i] = class_sums[i] / class_counts[i]
        else:
            # 如果某类别没有样本，使用随机向量或零向量
            class_means[i] = torch.zeros(feature_dim).to(device)
    
    return class_means