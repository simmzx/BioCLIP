# data/dataloader.py
"""确保数据在各模块间正确流动"""

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Optional, Tuple

class BioCLIPDataCollator:
    """自定义数据整理器，确保数据格式一致"""
    
    def __init__(self, config):
        self.config = config
        
    def __call__(self, batch: List[Dict]) -> Dict:
        """
        整理批次数据，确保格式正确
        """
        collated = {
            'batch_size': len(batch),
            'available_modalities': [],
            'is_complete': [],
            'is_pseudo': []
        }
        
        # 收集各模态数据
        mol_data = []
        gene_data = []
        morph_data = []
        
        for sample in batch:
            # 分子数据
            if 'mol' in sample:
                if isinstance(sample['mol'], torch.Tensor):
                    mol_data.append(sample['mol'])
                else:
                    # 确保是tensor
                    mol_data.append(torch.tensor(sample['mol'], dtype=torch.float32))
            
            # 基因数据
            if 'gene' in sample:
                if isinstance(sample['gene'], torch.Tensor):
                    gene_data.append(sample['gene'])
                else:
                    gene_data.append(torch.tensor(sample['gene'], dtype=torch.float32))
            
            # 形态数据
            if 'morph' in sample:
                if isinstance(sample['morph'], torch.Tensor):
                    morph_data.append(sample['morph'])
                else:
                    morph_data.append(torch.tensor(sample['morph'], dtype=torch.float32))
            
            # 元数据
            collated['available_modalities'].append(sample.get('available_modalities', []))
            collated['is_complete'].append(sample.get('is_complete', False))
            collated['is_pseudo'].append(sample.get('is_pseudo', False))
        
        # Stack数据
        if mol_data:
            collated['mol'] = torch.stack(mol_data)
        if gene_data:
            collated['gene'] = torch.stack(gene_data)
        if morph_data:
            collated['morph'] = torch.stack(morph_data)
        
        # 确保维度正确
        if 'mol' in collated:
            assert collated['mol'].dim() == 2 and collated['mol'].shape[1] == 2048
        if 'gene' in collated:
            assert collated['gene'].dim() == 2 and collated['gene'].shape[1] == 978
        if 'morph' in collated:
            assert collated['morph'].dim() == 4 and collated['morph'].shape[1] == 6
            
        return collated

def create_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    from data.dataset import BioCLIPDataset
    
    collator = BioCLIPDataCollator(config)
    
    # 训练集
    train_dataset = BioCLIPDataset(
        data_path=config.data_path,
        mode='train',
        data_type='complete'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    # 验证集
    val_dataset = BioCLIPDataset(
        data_path=config.data_path,
        mode='val',
        data_type='complete'
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    # 测试集
    test_dataset = BioCLIPDataset(
        data_path=config.data_path,
        mode='test',
        data_type='complete'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader