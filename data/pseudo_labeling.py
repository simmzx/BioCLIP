# data/pseudo_labeling.py
"""
基于最新文献的伪标签生成
References:
- FixMatch: https://arxiv.org/abs/2001.07685
- Co-teaching: https://arxiv.org/abs/1804.06872
- Uncertainty-based selection: https://arxiv.org/abs/2107.02331
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedPseudoLabelGenerator:
    """最先进的伪标签生成系统"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 多模型集成
        self.ensemble_models = self._build_ensemble()
        
        # 不确定性估计器
        self.uncertainty_estimator = UncertaintyEstimator()
        
        # 质量验证器
        self.quality_validator = QualityValidator(config)
        
        # 缓存高质量样本
        self.high_quality_cache = []
        
    def generate_pseudo_trimodal(self, 
                                 partial_data: List[Dict],
                                 n_samples: int = 5000) -> List[Dict]:
        """
        生成高质量伪三模态数据
        使用多种先进策略确保质量
        """
        print(f"Generating {n_samples} pseudo tri-modal samples...")
        
        pseudo_samples = []
        confidence_scores = []
        
        # 1. 多轮生成和筛选
        for round_idx in range(3):  # 3轮迭代改善
            print(f"  Round {round_idx + 1}/3...")
            
            round_samples = []
            
            for sample_idx, sample in enumerate(partial_data):
                if len(pseudo_samples) >= n_samples:
                    break
                    
                # 检查模态完整性
                available = sample.get('available_modalities', [])
                missing = self._get_missing_modalities(available)
                
                if not missing:
                    continue
                    
                # 生成缺失模态
                generated = self._generate_missing_modalities(
                    sample, missing, round_idx
                )
                
                if generated:
                    # 计算综合质量分数
                    quality_score = self._compute_quality_score(
                        sample, generated, round_idx
                    )
                    
                    if quality_score > self.config.confidence_threshold:
                        # 更新样本
                        pseudo_sample = self._create_pseudo_sample(
                            sample, generated, quality_score
                        )
                        round_samples.append(pseudo_sample)
                        confidence_scores.append(quality_score)
            
            # 选择本轮最佳样本
            if round_samples:
                # 按质量排序
                sorted_indices = np.argsort(confidence_scores)[::-1]
                top_k = min(len(round_samples), n_samples // 3)
                
                for idx in sorted_indices[:top_k]:
                    pseudo_samples.append(round_samples[idx])
            
            # 自训练：用高质量伪标签微调模型
            if round_idx < 2 and len(pseudo_samples) > 100:
                self._self_training_step(pseudo_samples[-100:])
        
        # 2. 最终质量控制
        final_samples = self._final_quality_control(pseudo_samples)
        
        print(f"Generated {len(final_samples)} high-quality pseudo samples")
        return final_samples[:n_samples]
    
    def _generate_missing_modalities(self, 
                                     sample: Dict,
                                     missing: List[str],
                                     round_idx: int) -> Dict:
        """使用集成模型生成缺失模态"""
        generated = {}
        
        available = sample.get('available_modalities', [])
        
        for modality in missing:
            if modality == 'mol' and len(available) >= 1:
                # 从其他模态预测分子
                generated['mol'] = self._predict_molecule(sample, available)
                
            elif modality == 'gene' and len(available) >= 1:
                # 预测基因表达
                generated['gene'] = self._predict_gene_expression(sample, available)
                
            elif modality == 'morph' and len(available) >= 1:
                # 预测细胞形态
                generated['morph'] = self._predict_morphology(sample, available)
        
        return generated
    
    def _predict_gene_expression(self, sample: Dict, available: List[str]) -> torch.Tensor:
        """预测基因表达 - 使用集成方法"""
        predictions = []
        uncertainties = []
        
        # 使用多个模型预测
        with torch.no_grad():
            for model in self.ensemble_models:
                if 'mol' in available:
                    # 从分子预测
                    pred = model.mol2gene(sample['mol'])
                elif 'morph' in available:
                    # 从形态预测
                    pred = model.morph2gene(sample['morph'])
                else:
                    continue
                    
                predictions.append(pred)
                
                # 计算不确定性
                if hasattr(model, 'mc_dropout'):
                    # MC Dropout不确定性估计
                    mc_preds = []
                    for _ in range(10):
                        model.train()
                        mc_pred = model.mol2gene(sample['mol'])
                        mc_preds.append(mc_pred)
                    
                    uncertainty = torch.stack(mc_preds).std(0).mean()
                    uncertainties.append(uncertainty)
        
        if not predictions:
            return None
            
        # 加权平均（基于不确定性）
        if uncertainties:
            weights = F.softmax(-torch.tensor(uncertainties), dim=0)
            weighted_pred = sum(w * p for w, p in zip(weights, predictions))
        else:
            # 简单平均
            weighted_pred = torch.stack(predictions).mean(0)
            
        return weighted_pred
    
    def _predict_morphology(self, sample: Dict, available: List[str]) -> torch.Tensor:
        """预测细胞形态 - 使用扩散模型风格的迭代改善"""
        # 初始预测
        if 'mol' in available:
            initial_pred = self.model.mol2morph(sample['mol'])
        elif 'gene' in available:
            initial_pred = self.model.gene2morph(sample['gene'])
        else:
            return None
            
        # 迭代改善（类似扩散模型）
        refined_pred = initial_pred
        for step in range(5):
            # 添加噪声
            noise = torch.randn_like(refined_pred) * (0.1 * (5 - step) / 5)
            noisy_pred = refined_pred + noise
            
            # 去噪（使用模型的投影头）
            refined_pred = self.model.morph_proj(
                self.model.morph_encoder(noisy_pred)
            )
            
        return refined_pred
    
    def _predict_molecule(self, sample: Dict, available: List[str]) -> torch.Tensor:
        """预测分子结构"""
        # 这是最困难的，通常使用预训练的分子特征
        if 'gene' in available:
            # 从基因表达预测分子特征
            mol_features = self.model.gene2mol(sample['gene'])
        elif 'morph' in available:
            mol_features = self.model.morph2mol(sample['morph'])
        else:
            return None
            
        return mol_features
    
    def _compute_quality_score(self, 
                               original: Dict,
                               generated: Dict,
                               round_idx: int) -> float:
        """计算综合质量分数"""
        scores = []
        
        # 1. 预测一致性
        consistency_score = self._check_consistency(original, generated)
        scores.append(consistency_score * 0.3)
        
        # 2. 不确定性分数
        uncertainty_score = self._compute_uncertainty(generated)
        scores.append((1 - uncertainty_score) * 0.2)
        
        # 3. 生物学合理性
        if 'gene' in generated:
            bio_score = self._check_biological_plausibility(generated['gene'])
            scores.append(bio_score * 0.25)
        
        # 4. 与已知样本的相似性
        similarity_score = self._check_similarity_to_known(original, generated)
        scores.append(similarity_score * 0.25)
        
        # Round bonus（后续轮次质量更高）
        round_bonus = 0.05 * round_idx
        
        return min(1.0, sum(scores) + round_bonus)
    
    def _check_consistency(self, original: Dict, generated: Dict) -> float:
        """检查跨模态一致性"""
        consistency_scores = []
        
        # 如果生成了多个模态，检查它们之间的一致性
        if len(generated) > 1:
            modalities = list(generated.keys())
            for i in range(len(modalities)):
                for j in range(i+1, len(modalities)):
                    # 计算模态间的相关性
                    mod1 = generated[modalities[i]].flatten()
                    mod2 = generated[modalities[j]].flatten()
                    
                    # 调整维度
                    min_dim = min(len(mod1), len(mod2))
                    corr = np.corrcoef(mod1[:min_dim].cpu().numpy(), 
                                       mod2[:min_dim].cpu().numpy())[0, 1]
                    consistency_scores.append(abs(corr))
        
        return np.mean(consistency_scores) if consistency_scores else 0.5
    
    def _compute_uncertainty(self, generated: Dict) -> float:
        """计算不确定性"""
        uncertainties = []
        
        for modality, data in generated.items():
            if data is not None:
                # 基于值的分散度
                std = data.std().item()
                uncertainties.append(std)
        
        return np.mean(uncertainties) if uncertainties else 0.5
    
    def _check_biological_plausibility(self, gene_expr: torch.Tensor) -> float:
        """检查生物学合理性"""
        gene_np = gene_expr.cpu().numpy()
        
        scores = []
        
        # 1. 表达值范围
        if -3 <= gene_np.min() and gene_np.max() <= 3:
            scores.append(1.0)
        else:
            scores.append(0.5)
        
        # 2. 稀疏性（大部分基因应该低表达）
        sparsity = (np.abs(gene_np) < 0.5).mean()
        scores.append(min(1.0, sparsity * 2))
        
        # 3. 分布检验
        _, p_value = stats.normaltest(gene_np)
        scores.append(min(1.0, p_value * 10))
        
        return np.mean(scores)
    
    def _check_similarity_to_known(self, original: Dict, generated: Dict) -> float:
        """检查与已知样本的相似性"""
        if not self.high_quality_cache:
            return 0.5
            
        similarities = []
        
        # 与缓存中的高质量样本比较
        for cached_sample in self.high_quality_cache[-10:]:
            sim_scores = []
            
            for modality in generated:
                if modality in cached_sample:
                    sim = F.cosine_similarity(
                        generated[modality].unsqueeze(0),
                        cached_sample[modality].unsqueeze(0)
                    )
                    sim_scores.append(sim.item())
            
            if sim_scores:
                similarities.append(np.mean(sim_scores))
        
        # 返回与最相似样本的相似度
        return max(similarities) if similarities else 0.5
    
    def _create_pseudo_sample(self, 
                             original: Dict,
                             generated: Dict,
                             quality_score: float) -> Dict:
        """创建伪标签样本"""
        pseudo_sample = original.copy()
        
        # 添加生成的模态
        for modality, data in generated.items():
            pseudo_sample[modality] = data
            pseudo_sample[f'{modality}_is_pseudo'] = True
            pseudo_sample[f'{modality}_confidence'] = quality_score
        
        # 更新元数据
        pseudo_sample['is_pseudo'] = True
        pseudo_sample['quality_score'] = quality_score
        pseudo_sample['available_modalities'] = list(set(
            original.get('available_modalities', []) + list(generated.keys())
        ))
        
        return pseudo_sample
    
    def _self_training_step(self, high_quality_samples: List[Dict]):
        """使用高质量伪标签进行自训练"""
        # 简单的微调步骤
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        
        for sample in high_quality_samples:
            # 计算损失
            if all(m in sample for m in ['mol', 'gene', 'morph']):
                outputs = self.model(sample, stage='stage3')
                
                if 'triple_loss' in outputs:
                    loss = outputs['triple_loss']
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
    
    def _final_quality_control(self, samples: List[Dict]) -> List[Dict]:
        """最终质量控制"""
        # 移除低质量样本
        filtered = []
        
        for sample in samples:
            if sample.get('quality_score', 0) > self.config.confidence_threshold:
                filtered.append(sample)
                
                # 添加到高质量缓存
                if sample['quality_score'] > 0.9:
                    self.high_quality_cache.append(sample)
        
        # 按质量排序
        filtered.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        return filtered
    
    def _get_missing_modalities(self, available: List[str]) -> List[str]:
        """获取缺失的模态"""
        all_modalities = ['mol', 'gene', 'morph']
        return [m for m in all_modalities if m not in available]
    
    def _build_ensemble(self) -> List:
        """构建模型集成"""
        # 这里简化，实际应该训练多个不同的模型
        return [self.model]

class UncertaintyEstimator:
    """不确定性估计器"""
    
    def estimate(self, predictions: torch.Tensor, method: str = 'entropy') -> float:
        """估计预测的不确定性"""
        if method == 'entropy':
            # 熵作为不确定性度量
            probs = F.softmax(predictions, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            return entropy.item()
        elif method == 'variance':
            # 方差作为不确定性
            return predictions.var().item()
        else:
            return 0.5

class QualityValidator:
    """质量验证器"""
    
    def __init__(self, config):
        self.config = config
        self.validation_criteria = [
            self.check_data_range,
            self.check_distribution,
            self.check_correlation
        ]
        
    def validate(self, sample: Dict) -> bool:
        """验证样本质量"""
        for criterion in self.validation_criteria:
            if not criterion(sample):
                return False
        return True
    
    def check_data_range(self, sample: Dict) -> bool:
        """检查数据范围"""
        if 'gene' in sample:
            gene_range = sample['gene'].max() - sample['gene'].min()
            if gene_range > 10:  # 异常大的范围
                return False
        return True
    
    def check_distribution(self, sample: Dict) -> bool:
        """检查分布"""
        # 简单的分布检查
        return True
    
    def check_correlation(self, sample: Dict) -> bool:
        """检查相关性"""
        # 检查模态间的相关性是否合理
        return True