import os
import yaml
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict

@dataclass
class BioCLIPConfig:
    """BioCLIP配置类 - 完整版本"""
    
    # 数据配置
    data_path: str = "./data"
    complete_trimodal: int = 1327
    partial_bimodal: int = 30000
    single_modal: int = 100000
    pseudo_samples: int = 5000
    batch_size: int = 32
    num_workers: int = 8
    
    # 模型配置
    hidden_dim: int = 256
    projection_dim: int = 128
    temperature: float = 0.07
    
    # 编码器配置
    mol_encoder_type: str = "gin_pretrained"  # gin_pretrained, molformer, chemberta
    gene_encoder_type: str = "performer"  # performer, scgpt, tab_transformer
    morph_encoder_type: str = "vit"  # vit, convnext, efficientnet, swin
    
    # 训练配置
    stage1_epochs: int = 50
    stage2_epochs: int = 100
    stage3_epochs: int = 150
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation: int = 4
    warmup_epochs: int = 10
    
    # 伪标签配置
    confidence_threshold: float = 0.8
    tanimoto_threshold: float = 0.8
    pathway_consistency_weight: float = 0.3
    
    # 损失配置
    use_vicreg: bool = True
    vicreg_sim_coeff: float = 25.0
    vicreg_std_coeff: float = 25.0
    vicreg_cov_coeff: float = 1.0
    
    # 日志配置
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    save_frequency: int = 10
    
    # 其他
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    seed: int = 42
    
    def __init__(self, config_path: Optional[str] = None):
        """从配置文件初始化"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
                self._update_from_dict(config_dict)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建必要的目录
        os.makedirs(self.data_path, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _update_from_dict(self, config_dict: Dict):
        """从字典更新配置"""
        for section, params in config_dict.items():
            if isinstance(params, dict):
                for key, value in params.items():
                    # 将嵌套的键转换为扁平化
                    attr_name = key
                    if hasattr(self, attr_name):
                        setattr(self, attr_name, value)
                    else:
                        # 处理嵌套配置
                        full_key = f"{section}_{key}"
                        if hasattr(self, full_key):
                            setattr(self, full_key, value)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'device'
        }
    
    def save(self, path: str):
        """保存配置到文件"""
        config_dict = self.to_dict()
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        """从文件加载配置"""
        return cls(config_path=path)