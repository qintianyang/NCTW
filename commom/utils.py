import random
import numpy as np
import torch
from torch import cuda

def set_seed(seed):
    """设置所有随机种子，确保实验可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed(seed)
        cuda.manual_seed_all(seed)  # 多GPU时使用
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device():
    """获取可用的计算设备（GPU优先）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_device_info(device):
    """打印设备信息"""
    if device.type == "cuda":
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    else:
        print("使用CPU进行计算")