# data/augmentation.py
"""
基于最新文献的数据增强方法：
- 分子: MolCLR (Wang et al., Nature Machine Intelligence 2022)
- 基因: scVI-tools (Lopez et al., Nature Methods 2018) & scGPT (Cui et al., Nature Methods 2024)  
- 形态: Cell Painting (Bray et al., Nature Protocols 2016) & CellProfiler (Stirling et al., 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdFingerprintGenerator
import cv2
import albumentations as A
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

class MolecularAugmenter:
    """
    分子增强基于MolCLR和SMILES-BERT文献
    References:
    - MolCLR: https://github.com/yuyangw/MolCLR
    - SMILES-BERT: https://github.com/uta-smile/SMILES-BERT
    """
    
    def __init__(self, aug_prob: float = 0.5):
        self.aug_prob = aug_prob
        self.augmentation_methods = [
            self.atom_masking,
            self.bond_deletion,
            self.substructure_removal,
            self.smiles_randomization
        ]
        
    def augment_batch(self, smiles_list: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量增强SMILES
        返回: (原始特征, 增强特征)
        """
        original_features = []
        augmented_features = []
        
        for smiles in smiles_list:
            # 原始特征
            orig_feat = self.smiles_to_features(smiles)
            original_features.append(orig_feat)
            
            # 增强
            if random.random() < self.aug_prob:
                # 随机选择增强方法
                aug_method = random.choice(self.augmentation_methods)
                aug_smiles = aug_method(smiles)
                aug_feat = self.smiles_to_features(aug_smiles)
            else:
                aug_feat = orig_feat.clone()
                
            augmented_features.append(aug_feat)
            
        return torch.stack(original_features), torch.stack(augmented_features)
    
    def atom_masking(self, smiles: str, mask_rate: float = 0.15) -> str:
        """原子掩码 - MolCLR方法"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
            
        atoms = list(range(mol.GetNumAtoms()))
        num_mask = max(1, int(len(atoms) * mask_rate))
        masked_atoms = random.sample(atoms, num_mask)
        
        for atom_idx in masked_atoms:
            atom = mol.GetAtomWithIdx(atom_idx)
            atom.SetAtomicNum(0)  # 设置为通配符
            
        return Chem.MolToSmiles(mol)
    
    def bond_deletion(self, smiles: str, del_rate: float = 0.1) -> str:
        """键删除 - MolCLR方法"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumBonds() == 0:
            return smiles
            
        bonds = list(range(mol.GetNumBonds()))
        num_del = max(1, int(len(bonds) * del_rate))
        
        # 选择非环键优先删除
        non_ring_bonds = [b for b in bonds if not mol.GetBondWithIdx(b).IsInRing()]
        
        if non_ring_bonds:
            bonds_to_del = random.sample(non_ring_bonds, min(num_del, len(non_ring_bonds)))
            edit_mol = Chem.RWMol(mol)
            for bond_idx in sorted(bonds_to_del, reverse=True):
                bond = mol.GetBondWithIdx(bond_idx)
                edit_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            return Chem.MolToSmiles(edit_mol)
            
        return smiles
    
    def substructure_removal(self, smiles: str) -> str:
        """子结构移除 - 基于BRICS分解"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
            
        # BRICS分解
        from rdkit.Chem import BRICS
        fragments = BRICS.BRICSDecompose(mol)
        
        if len(fragments) > 1:
            # 随机移除一个片段
            remaining = random.sample(list(fragments), len(fragments) - 1)
            if remaining:
                return remaining[0]
                
        return smiles
    
    def smiles_randomization(self, smiles: str) -> str:
        """SMILES随机化 - SMILES-BERT方法"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
            
        # 生成多个等价SMILES
        random_smiles = []
        for _ in range(10):
            random_mol = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
            random_smiles.append(random_mol)
            
        return random.choice(random_smiles)
    
    def smiles_to_features(self, smiles: str) -> torch.Tensor:
        """将SMILES转换为特征向量"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(2048)
            
        # 使用Morgan指纹 (ECFP4)
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = fpgen.GetFingerprint(mol)
        arr = np.zeros(2048)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, arr)
        
        return torch.tensor(arr, dtype=torch.float32)

class GeneExpressionAugmenter:
    """
    基因表达增强基于scVI和scGPT文献
    References:
    - scVI: https://github.com/scverse/scvi-tools
    - scGPT: https://github.com/bowang-lab/scGPT
    - AutoClass: https://github.com/jlakkis/AutoClass
    """
    
    def __init__(self, n_genes: int = 978):
        self.n_genes = n_genes
        self.noise_model = self._build_noise_model()
        
        # 加载基因互作网络（简化版）
        self.gene_network = self._load_gene_network()
        
    def augment_batch(self, gene_expr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量增强基因表达
        输入: [batch_size, 978]
        返回: (原始表达, 增强表达)
        """
        batch_size = gene_expr.shape[0]
        augmented = []
        
        for i in range(batch_size):
            expr = gene_expr[i]
            
            # 组合多种增强
            aug_expr = expr.clone()
            
            # 1. 技术噪声 (scVI方法)
            if random.random() < 0.5:
                aug_expr = self.add_technical_noise_scvi(aug_expr)
            
            # 2. Dropout (scGPT方法)
            if random.random() < 0.5:
                aug_expr = self.add_dropout_scgpt(aug_expr)
                
            # 3. 批次效应 (Seurat方法)
            if random.random() < 0.3:
                aug_expr = self.add_batch_effect(aug_expr)
                
            # 4. 基因网络扰动
            if random.random() < 0.3:
                aug_expr = self.network_perturbation(aug_expr)
                
            augmented.append(aug_expr)
            
        return gene_expr, torch.stack(augmented)
    
    def add_technical_noise_scvi(self, expr: torch.Tensor) -> torch.Tensor:
        """scVI风格的技术噪声模型"""
        # 负二项分布噪声
        mean = expr
        dispersion = 0.1
        
        # 使用Gamma-Poisson混合模拟负二项
        theta = 1.0 / dispersion
        gamma_noise = torch.distributions.Gamma(theta, theta / (mean + 1e-8)).sample()
        noisy_expr = torch.distributions.Poisson(gamma_noise).sample()
        
        return noisy_expr.float()
    
    def add_dropout_scgpt(self, expr: torch.Tensor) -> torch.Tensor:
        """scGPT风格的dropout"""
        # 基于表达水平的dropout概率
        dropout_prob = torch.sigmoid(-expr * 2) * 0.3  # 低表达更容易dropout
        
        # Bernoulli采样
        mask = torch.bernoulli(1 - dropout_prob)
        
        return expr * mask
    
    def add_batch_effect(self, expr: torch.Tensor) -> torch.Tensor:
        """Seurat风格的批次效应"""
        # 全局偏移
        global_shift = torch.randn(1) * 0.1
        
        # 基因特异性缩放
        gene_scaling = 1 + torch.randn(self.n_genes) * 0.05
        
        return expr * gene_scaling + global_shift
    
    def network_perturbation(self, expr: torch.Tensor) -> torch.Tensor:
        """基于基因网络的扰动"""
        if not hasattr(self, '_network_matrix'):
            # 创建简单的网络矩阵
            self._network_matrix = torch.eye(self.n_genes) + torch.randn(self.n_genes, self.n_genes) * 0.01
            
        # 网络传播
        perturbed = torch.matmul(expr.unsqueeze(0), self._network_matrix).squeeze(0)
        
        # 保持原始尺度
        perturbed = perturbed * (expr.std() / perturbed.std())
        
        return perturbed
    
    def _build_noise_model(self):
        """构建噪声模型"""
        return nn.Sequential(
            nn.Linear(self.n_genes, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_genes)
        )
    
    def _load_gene_network(self):
        """加载基因网络（简化版）"""
        # 实际应该从STRING或KEGG数据库加载
        return {}

class CellMorphologyAugmenter:
    """
    细胞形态增强基于Cell Painting最佳实践
    References:
    - Cell Painting: https://github.com/broadinstitute/imaging-platform-pipelines
    - CellProfiler: https://github.com/CellProfiler/CellProfiler
    - https://www.nature.com/articles/s41596-016-0339-8
    """
    
    def __init__(self):
        # Cell Painting标准增强pipeline
        self.geometric_augment = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.15,
                rotate_limit=45,
                p=0.5
            )
        ])
        
        # 强度增强（Cell Painting特定）
        self.intensity_augment = A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.RandomGamma(
                gamma_limit=(80, 120),
                p=0.3
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=0.3
            )
        ])
        
        self.channel_names = [
            'Nuclei',      # Hoechst
            'ER',          # Concanavalin A  
            'RNA',         # SYTO 14
            'AGP',         # Phalloidin (actin)
            'Mito',        # MitoTracker
            'Membrane'     # WGA
        ]
        
    def augment_batch(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        批量增强Cell Painting图像
        输入: [batch_size, 6, 224, 224]
        返回: (原始图像, 增强图像)
        """
        batch_size = images.shape[0]
        augmented = []
        
        for i in range(batch_size):
            img = images[i].cpu().numpy()  # [6, 224, 224]
            
            # 转换为HWC格式用于augmentation
            img_hwc = np.transpose(img, (1, 2, 0))  # [224, 224, 6]
            
            # 应用增强
            aug_img = self.apply_augmentation(img_hwc)
            
            # 转回CHW格式
            aug_img = np.transpose(aug_img, (2, 0, 1))  # [6, 224, 224]
            
            augmented.append(torch.tensor(aug_img, dtype=torch.float32))
            
        return images, torch.stack(augmented)
    
    def apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """应用Cell Painting特定增强"""
        
        # 1. 几何变换
        if random.random() < 0.5:
            # 分离通道以保持Cell Painting特性
            channels = [image[:,:,i] for i in range(6)]
            
            # 对每个通道独立应用几何变换
            aug_channels = []
            for ch_idx, channel in enumerate(channels):
                # 转为uint8用于albumentation
                ch_uint8 = (channel * 255).astype(np.uint8)
                augmented = self.geometric_augment(image=ch_uint8)['image']
                aug_channels.append(augmented.astype(np.float32) / 255.0)
                
            image = np.stack(aug_channels, axis=2)
        
        # 2. 通道特异性强度调整
        if random.random() < 0.5:
            image = self.channel_specific_intensity(image)
        
        # 3. 细胞特异性增强
        if random.random() < 0.3:
            image = self.cell_specific_augmentation(image)
            
        # 4. 染色串扰模拟
        if random.random() < 0.2:
            image = self.simulate_channel_crosstalk(image)
            
        return image
    
    def channel_specific_intensity(self, image: np.ndarray) -> np.ndarray:
        """通道特异性强度调整"""
        aug_image = image.copy()
        
        for ch_idx in range(6):
            if random.random() < 0.3:
                # 通道特定的强度因子
                if ch_idx == 0:  # 核通道
                    factor = np.random.uniform(0.8, 1.2)
                elif ch_idx == 4:  # 线粒体通道
                    factor = np.random.uniform(0.7, 1.3)  # 线粒体染色变化更大
                else:
                    factor = np.random.uniform(0.85, 1.15)
                    
                aug_image[:,:,ch_idx] *= factor
                
        return np.clip(aug_image, 0, 1)
    
    def cell_specific_augmentation(self, image: np.ndarray) -> np.ndarray:
        """细胞生物学特异性增强"""
        aug_image = image.copy()
        
        # 模拟细胞密度变化
        if random.random() < 0.5:
            # 使用形态学操作模拟细胞聚集
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            # 主要影响核通道
            aug_image[:,:,0] = cv2.morphologyEx(
                aug_image[:,:,0], 
                cv2.MORPH_CLOSE if random.random() < 0.5 else cv2.MORPH_OPEN,
                kernel
            )
            
        # 模拟细胞周期变化
        if random.random() < 0.3:
            # G2/M期细胞核更亮
            if random.random() < 0.2:
                aug_image[:,:,0] *= np.random.uniform(1.1, 1.3)
                
        return np.clip(aug_image, 0, 1)
    
    def simulate_channel_crosstalk(self, image: np.ndarray) -> np.ndarray:
        """模拟染色通道间的串扰"""
        # Cell Painting中常见的串扰模式
        crosstalk_matrix = np.eye(6)
        
        # Hoechst可能影响SYTO
        crosstalk_matrix[0, 2] = random.uniform(0, 0.05)
        
        # MitoTracker可能影响ER
        crosstalk_matrix[4, 1] = random.uniform(0, 0.03)
        
        # 应用串扰
        result = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                pixel = image[i, j, :]
                result[i, j, :] = crosstalk_matrix @ pixel
                
        return np.clip(result, 0, 1)