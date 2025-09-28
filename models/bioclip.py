import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np

# 导入编码器
from models.encoders import MolecularEncoder, GeneEncoder, MorphologyEncoder
# 导入损失函数
from models.losses import ContrastiveLoss, VICRegLoss, TripleContrastiveLoss

class BioCLIP(nn.Module):
    """
    BioCLIP主模型 - 集成所有组件
    实现Progressive Multi-modal Bootstrapping (PMB)
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ========== 编码器 ==========
        self.mol_encoder = MolecularEncoder(
            model_type=config.mol_encoder_type,
            hidden_dim=config.hidden_dim
        )
        
        self.gene_encoder = GeneEncoder(
            model_type=config.gene_encoder_type,
            input_dim=978,
            hidden_dim=config.hidden_dim
        )
        
        self.morph_encoder = MorphologyEncoder(
            model_type=config.morph_encoder_type,
            hidden_dim=config.hidden_dim
        )
        
        # ========== 投影头 ==========
        self.mol_proj = self._build_projection_head(
            config.hidden_dim, config.projection_dim
        )
        self.gene_proj = self._build_projection_head(
            config.hidden_dim, config.projection_dim
        )
        self.morph_proj = self._build_projection_head(
            config.hidden_dim, config.projection_dim
        )
        
        # ========== 跨模态预测器 ==========
        self.mol2gene = self._build_cross_modal_predictor(
            config.hidden_dim, config.hidden_dim
        )
        self.mol2morph = self._build_cross_modal_predictor(
            config.hidden_dim, config.hidden_dim
        )
        self.gene2mol = self._build_cross_modal_predictor(
            config.hidden_dim, config.hidden_dim
        )
        self.gene2morph = self._build_cross_modal_predictor(
            config.hidden_dim, config.hidden_dim
        )
        self.morph2mol = self._build_cross_modal_predictor(
            config.hidden_dim, config.hidden_dim
        )
        self.morph2gene = self._build_cross_modal_predictor(
            config.hidden_dim, config.hidden_dim
        )
        
        # ========== 动量编码器 ==========
        self.momentum = 0.999
        self._build_momentum_encoders()
        
        # ========== 内存队列 ==========
        queue_size = 65536
        self.register_buffer(
            "mol_queue", 
            F.normalize(torch.randn(config.projection_dim, queue_size), dim=0)
        )
        self.register_buffer(
            "gene_queue",
            F.normalize(torch.randn(config.projection_dim, queue_size), dim=0)
        )
        self.register_buffer(
            "morph_queue",
            F.normalize(torch.randn(config.projection_dim, queue_size), dim=0)
        )
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
        # ========== 损失函数 ==========
        self.contrastive_loss = ContrastiveLoss(config.temperature)
        self.vicreg_loss = VICRegLoss(
            sim_coeff=config.vicreg_sim_coeff,
            std_coeff=config.vicreg_std_coeff,
            cov_coeff=config.vicreg_cov_coeff
        ) if config.use_vicreg else None
        self.triple_loss = TripleContrastiveLoss(config.temperature)
        
        self.temperature = config.temperature
    
    def forward(self, batch: Dict, stage: str = 'stage1') -> Dict:
        """
        前向传播 - 根据训练阶段选择不同策略
        """
        outputs = {}
        
        if stage == 'stage1':
            outputs = self._stage1_forward(batch)
        elif stage == 'stage2':
            outputs = self._stage2_forward(batch)
        elif stage == 'stage3':
            outputs = self._stage3_forward(batch)
        else:
            # 推理模式
            outputs = self._inference_forward(batch)
        
        # 更新动量编码器
        if self.training:
            self._momentum_update()
        
        return outputs
    
    def _stage1_forward(self, batch: Dict) -> Dict:
        """Stage 1: 单模态自监督学习"""
        outputs = {}
        
        # 分子模态
        if 'mol' in batch and 'mol_aug' in batch:
            mol_feat = self.mol_encoder(batch['mol'])
            mol_aug_feat = self.mol_encoder(batch['mol_aug'])
            
            mol_proj = F.normalize(self.mol_proj(mol_feat), dim=1)
            mol_aug_proj = F.normalize(self.mol_proj(mol_aug_feat), dim=1)
            
            # 使用VICReg或标准对比损失
            if self.vicreg_loss is not None:
                outputs['mol_ssl_loss'] = self.vicreg_loss(mol_proj, mol_aug_proj)
            else:
                outputs['mol_ssl_loss'] = self.contrastive_loss(mol_proj, mol_aug_proj)
            
            outputs['mol_features'] = mol_feat
        
        # 基因模态
        if 'gene' in batch and 'gene_aug' in batch:
            gene_feat = self.gene_encoder(batch['gene'])
            gene_aug_feat = self.gene_encoder(batch['gene_aug'])
            
            gene_proj = F.normalize(self.gene_proj(gene_feat), dim=1)
            gene_aug_proj = F.normalize(self.gene_proj(gene_aug_feat), dim=1)
            
            if self.vicreg_loss is not None:
                outputs['gene_ssl_loss'] = self.vicreg_loss(gene_proj, gene_aug_proj)
            else:
                outputs['gene_ssl_loss'] = self.contrastive_loss(gene_proj, gene_aug_proj)
            
            outputs['gene_features'] = gene_feat
        
        # 形态模态
        if 'morph' in batch and 'morph_aug' in batch:
            morph_feat = self.morph_encoder(batch['morph'])
            morph_aug_feat = self.morph_encoder(batch['morph_aug'])
            
            morph_proj = F.normalize(self.morph_proj(morph_feat), dim=1)
            morph_aug_proj = F.normalize(self.morph_proj(morph_aug_feat), dim=1)
            
            if self.vicreg_loss is not None:
                outputs['morph_ssl_loss'] = self.vicreg_loss(morph_proj, morph_aug_proj)
            else:
                outputs['morph_ssl_loss'] = self.contrastive_loss(morph_proj, morph_aug_proj)
            
            outputs['morph_features'] = morph_feat
        
        return outputs
    
    def _stage2_forward(self, batch: Dict) -> Dict:
        """Stage 2: 双模态对齐"""
        outputs = {}
        available = batch.get('available_modalities', [])
        
        # 编码所有可用模态
        features = {}
        projections = {}
        
        if 'mol' in batch:
            features['mol'] = self.mol_encoder(batch['mol'])
            projections['mol'] = F.normalize(self.mol_proj(features['mol']), dim=1)
        
        if 'gene' in batch:
            features['gene'] = self.gene_encoder(batch['gene'])
            projections['gene'] = F.normalize(self.gene_proj(features['gene']), dim=1)
        
        if 'morph' in batch:
            features['morph'] = self.morph_encoder(batch['morph'])
            projections['morph'] = F.normalize(self.morph_proj(features['morph']), dim=1)
        
        # 计算成对损失
        if 'mol' in projections and 'gene' in projections:
            outputs['mol_gene_loss'] = self.contrastive_loss(
                projections['mol'], projections['gene']
            )
        
        if 'mol' in projections and 'morph' in projections:
            outputs['mol_morph_loss'] = self.contrastive_loss(
                projections['mol'], projections['morph']
            )
        
        if 'gene' in projections and 'morph' in projections:
            outputs['gene_morph_loss'] = self.contrastive_loss(
                projections['gene'], projections['morph']
            )
        
        # 跨模态预测（用于伪标签生成）
        if 'mol' in features:
            if 'gene' not in features:
                outputs['predicted_gene'] = self.mol2gene(features['mol'])
            if 'morph' not in features:
                outputs['predicted_morph'] = self.mol2morph(features['mol'])
        
        outputs.update({
            'features': features,
            'projections': projections
        })
        
        return outputs
    
    def _stage3_forward(self, batch: Dict) -> Dict:
        """Stage 3: 三模态融合"""
        outputs = {}
        
        # 处理完整三模态数据
        if batch.get('is_complete', False) or all(m in batch for m in ['mol', 'gene', 'morph']):
            # 编码
            mol_feat = self.mol_encoder(batch['mol'])
            gene_feat = self.gene_encoder(batch['gene'])
            morph_feat = self.morph_encoder(batch['morph'])
            
            # 投影
            mol_proj = F.normalize(self.mol_proj(mol_feat), dim=1)
            gene_proj = F.normalize(self.gene_proj(gene_feat), dim=1)
            morph_proj = F.normalize(self.morph_proj(morph_feat), dim=1)
            
            # 三模态损失（使用队列）
            outputs['triple_loss'] = self.triple_loss(
                mol_proj, gene_proj, morph_proj,
                self.mol_queue.clone().detach(),
                self.gene_queue.clone().detach(),
                self.morph_queue.clone().detach()
            )
            
            # 更新队列（使用动量编码器）
            if self.training:
                with torch.no_grad():
                    mol_proj_m = F.normalize(
                        self.mol_proj_m(self.mol_encoder_m(batch['mol'])), dim=1
                    )
                    gene_proj_m = F.normalize(
                        self.gene_proj_m(self.gene_encoder_m(batch['gene'])), dim=1
                    )
                    morph_proj_m = F.normalize(
                        self.morph_proj_m(self.morph_encoder_m(batch['morph'])), dim=1
                    )
                    
                    self._dequeue_and_enqueue(mol_proj_m, gene_proj_m, morph_proj_m)
            
            outputs.update({
                'mol_features': mol_feat,
                'gene_features': gene_feat,
                'morph_features': morph_feat
            })
        
        # 处理伪标签数据
        elif batch.get('is_pseudo', False):
            confidence = batch.get('confidence', torch.ones(1).to(batch['mol'].device))
            
            mol_feat = self.mol_encoder(batch['mol'])
            gene_feat = self.gene_encoder(batch['gene'])
            morph_feat = self.morph_encoder(batch['morph'])
            
            mol_proj = F.normalize(self.mol_proj(mol_feat), dim=1)
            gene_proj = F.normalize(self.gene_proj(gene_feat), dim=1)
            morph_proj = F.normalize(self.morph_proj(morph_feat), dim=1)
            
            # 软标签损失（温度调整）
            base_loss = self.triple_loss(
                mol_proj, gene_proj, morph_proj
            )
            
            outputs['pseudo_loss'] = base_loss * confidence.mean()
        
        # 处理部分模态
        else:
            outputs = self._stage2_forward(batch)
        
        return outputs
    
    def _inference_forward(self, batch: Dict) -> Dict:
        """推理模式"""
        outputs = {}
        
        if 'mol' in batch:
            outputs['mol_features'] = self.mol_encoder(batch['mol'])
        
        if 'gene' in batch:
            outputs['gene_features'] = self.gene_encoder(batch['gene'])
        
        if 'morph' in batch:
            outputs['morph_features'] = self.morph_encoder(batch['morph'])
        
        return outputs
    
    def _build_projection_head(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建投影头"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.BatchNorm1d(input_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim, affine=False)
        )
    
    def _build_cross_modal_predictor(self, input_dim: int, output_dim: int) -> nn.Module:
        """构建跨模态预测器"""
        return nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def _build_momentum_encoders(self):
        """构建动量编码器"""
        # 创建动量编码器副本
        self.mol_encoder_m = MolecularEncoder(
            model_type=self.config.mol_encoder_type,
            hidden_dim=self.config.hidden_dim
        )
        self.gene_encoder_m = GeneEncoder(
            model_type=self.config.gene_encoder_type,
            input_dim=978,
            hidden_dim=self.config.hidden_dim
        )
        self.morph_encoder_m = MorphologyEncoder(
            model_type=self.config.morph_encoder_type,
            hidden_dim=self.config.hidden_dim
        )
        
        # 动量投影头
        self.mol_proj_m = self._build_projection_head(
            self.config.hidden_dim, self.config.projection_dim
        )
        self.gene_proj_m = self._build_projection_head(
            self.config.hidden_dim, self.config.projection_dim
        )
        self.morph_proj_m = self._build_projection_head(
            self.config.hidden_dim, self.config.projection_dim
        )
        
        # 初始化为相同权重
        for param_q, param_k in zip(self.mol_encoder.parameters(),
                                    self.mol_encoder_m.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.gene_encoder.parameters(),
                                    self.gene_encoder_m.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        
        for param_q, param_k in zip(self.morph_encoder.parameters(),
                                    self.morph_encoder_m.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
    
    @torch.no_grad()
    def _momentum_update(self):
        """更新动量编码器"""
        for param_q, param_k in zip(self.mol_encoder.parameters(),
                                    self.mol_encoder_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.gene_encoder.parameters(),
                                    self.gene_encoder_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
        
        for param_q, param_k in zip(self.morph_encoder.parameters(),
                                    self.morph_encoder_m.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, mol_keys, gene_keys, morph_keys):
        """更新内存队列"""
        batch_size = mol_keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # 更新队列
        self.mol_queue[:, ptr:ptr + batch_size] = mol_keys.T
        self.gene_queue[:, ptr:ptr + batch_size] = gene_keys.T
        self.morph_queue[:, ptr:ptr + batch_size] = morph_keys.T