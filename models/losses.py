# models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class ContrastiveLoss(nn.Module):
    """标准对比损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        计算对比损失
        Args:
            z1, z2: [batch_size, dim] 归一化的特征
        """
        batch_size = z1.shape[0]
        
        # 相似度矩阵
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # 标签（对角线为正样本）
        labels = torch.arange(batch_size).to(z1.device)
        
        # 双向损失
        loss_12 = F.cross_entropy(sim_matrix, labels)
        loss_21 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_12 + loss_21) / 2

class VICRegLoss(nn.Module):
    """VICReg损失 - 避免表征坍塌"""
    
    def __init__(self, 
                 sim_coeff: float = 25.0,
                 std_coeff: float = 25.0,
                 cov_coeff: float = 1.0):
        super().__init__()
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        VICReg损失计算
        Args:
            z1, z2: [batch_size, dim] 特征
        """
        batch_size = z1.shape[0]
        dim = z1.shape[1]
        
        # 1. Invariance loss
        sim_loss = F.mse_loss(z1, z2)
        
        # 2. Variance loss
        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))
        
        # 3. Covariance loss
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        
        cov_z1 = (z1.T @ z1) / (batch_size - 1)
        cov_z2 = (z2.T @ z2) / (batch_size - 1)
        
        # 只惩罚非对角元素
        off_diag_mask = ~torch.eye(dim, dtype=torch.bool, device=z1.device)
        cov_loss = cov_z1[off_diag_mask].pow(2).sum() / dim + \
                   cov_z2[off_diag_mask].pow(2).sum() / dim
        
        loss = self.sim_coeff * sim_loss + \
               self.std_coeff * std_loss + \
               self.cov_coeff * cov_loss
        
        return loss

class TripleContrastiveLoss(nn.Module):
    """三模态对比损失"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, 
                mol: torch.Tensor,
                gene: torch.Tensor, 
                morph: torch.Tensor,
                mol_queue: Optional[torch.Tensor] = None,
                gene_queue: Optional[torch.Tensor] = None,
                morph_queue: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        三模态对比损失
        """
        batch_size = mol.shape[0]
        
        # 计算成对相似度
        sim_mg = mol @ gene.T / self.temperature
        sim_mm = mol @ morph.T / self.temperature
        sim_gm = gene @ morph.T / self.temperature
        
        labels = torch.arange(batch_size).to(mol.device)
        
        # 如果有队列，添加负样本
        if mol_queue is not None:
            sim_mol_neg = mol @ mol_queue / self.temperature
            sim_mg = torch.cat([sim_mg, sim_mol_neg], dim=1)
        
        if gene_queue is not None:
            sim_gene_neg = gene @ gene_queue / self.temperature
            sim_mm = torch.cat([sim_mm, sim_gene_neg], dim=1)
        
        if morph_queue is not None:
            sim_morph_neg = morph @ morph_queue / self.temperature
            sim_gm = torch.cat([sim_gm, sim_morph_neg], dim=1)
        
        # 计算损失
        loss_mg = F.cross_entropy(sim_mg, labels)
        loss_mm = F.cross_entropy(sim_mm, labels)
        loss_gm = F.cross_entropy(sim_gm, labels)
        
        return (loss_mg + loss_mm + loss_gm) / 3

class PseudoLabelLoss(nn.Module):
    """伪标签损失 - 带置信度加权"""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.base_loss = TripleContrastiveLoss(temperature)
    
    def forward(self,
                mol: torch.Tensor,
                gene: torch.Tensor,
                morph: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        """
        置信度加权的伪标签损失
        Args:
            confidence: [batch_size] 每个样本的置信度
        """
        # 基础三模态损失
        base_loss = self.base_loss(mol, gene, morph)
        
        # 置信度加权
        weighted_loss = base_loss * confidence.mean()
        
        return weighted_loss