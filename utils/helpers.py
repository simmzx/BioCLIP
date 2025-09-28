# utils/helpers.py
import os
import torch
import random
import numpy as np
from typing import Dict, Any, Optional
import json
import yaml
from datetime import datetime

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    path: str,
    **kwargs
):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }
    
    # 添加额外参数
    for key, value in kwargs.items():
        checkpoint[key] = value
    
    # 创建目录
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 保存
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(
    path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = 'cpu'
) -> Dict:
    """加载检查点"""
    checkpoint = torch.load(path, map_location=map_location)
    
    if model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Optimizer loaded from {path}")
    
    return checkpoint

def count_parameters(model: torch.nn.Module) -> int:
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_device() -> torch.device:
    """获取最佳设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def create_exp_dir(base_dir: str = 'experiments') -> str:
    """创建实验目录"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f'exp_{timestamp}')
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    
    return exp_dir

def save_config(config: Any, path: str):
    """保存配置"""
    if hasattr(config, 'to_dict'):
        config_dict = config.to_dict()
    else:
        config_dict = config.__dict__
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

def load_config(path: str) -> Dict:
    """加载配置"""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config