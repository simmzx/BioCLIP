import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, mean_squared_error
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

class Evaluator:
    """模型评估器"""
    
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        
    @torch.no_grad()
    def evaluate(self, dataloader, stage: str = 'stage3') -> Dict:
        """
        评估模型性能
        """
        self.model.eval()
        
        all_embeddings = {
            'mol': [],
            'gene': [],
            'morph': []
        }
        
        all_losses = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 移到设备
            batch = self._to_device(batch)
            
            # 前向传播
            outputs = self.model(batch, stage=stage)
            
            # 收集损失
            for key in outputs:
                if 'loss' in key and outputs[key] is not None:
                    all_losses.append(outputs[key].item())
            
            # 收集嵌入
            if 'mol_features' in outputs:
                all_embeddings['mol'].append(outputs['mol_features'].cpu())
            if 'gene_features' in outputs:
                all_embeddings['gene'].append(outputs['gene_features'].cpu())
            if 'morph_features' in outputs:
                all_embeddings['morph'].append(outputs['morph_features'].cpu())
        
        # 计算指标
        metrics = {
            'avg_loss': np.mean(all_losses) if all_losses else 0
        }
        
        # 计算嵌入质量指标
        for modality in all_embeddings:
            if all_embeddings[modality]:
                embeddings = torch.cat(all_embeddings[modality], dim=0)
                
                # 计算嵌入统计
                metrics[f'{modality}_mean_norm'] = embeddings.norm(dim=1).mean().item()
                metrics[f'{modality}_std'] = embeddings.std().item()
                
                # 检查坍塌
                singular_values = torch.svd(embeddings).S
                metrics[f'{modality}_rank'] = (singular_values > 0.01).sum().item()
        
        # 计算检索指标
        if all(len(all_embeddings[m]) > 0 for m in all_embeddings):
            retrieval_metrics = self._compute_retrieval_metrics(all_embeddings)
            metrics.update(retrieval_metrics)
        
        return metrics
    
    def evaluate_downstream(self, 
                           model,
                           train_loader, 
                           test_loader,
                           task: str = 'toxicity') -> Dict:
        """
        评估下游任务性能
        """
        # 提取特征
        train_features, train_labels = self._extract_features(model, train_loader)
        test_features, test_labels = self._extract_features(model, test_loader)
        
        # 训练简单分类器
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(train_features, train_labels)
        
        # 预测
        test_pred = clf.predict_proba(test_features)[:, 1]
        
        # 计算指标
        metrics = {
            f'{task}_auroc': roc_auc_score(test_labels, test_pred),
            f'{task}_auprc': average_precision_score(test_labels, test_pred)
        }
        
        return metrics
    
    def _compute_retrieval_metrics(self, embeddings: Dict) -> Dict:
        """计算跨模态检索指标"""
        metrics = {}
        
        # 转换为numpy
        mol_emb = torch.cat(embeddings['mol']).numpy()
        gene_emb = torch.cat(embeddings['gene']).numpy()
        morph_emb = torch.cat(embeddings['morph']).numpy()
        
        # 计算相似度矩阵
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Mol -> Gene检索
        sim_mg = cosine_similarity(mol_emb, gene_emb)
        metrics['mol2gene_top1'] = self._compute_recall_at_k(sim_mg, k=1)
        metrics['mol2gene_top5'] = self._compute_recall_at_k(sim_mg, k=5)
        
        # Mol -> Morph检索
        sim_mm = cosine_similarity(mol_emb, morph_emb)
        metrics['mol2morph_top1'] = self._compute_recall_at_k(sim_mm, k=1)
        metrics['mol2morph_top5'] = self._compute_recall_at_k(sim_mm, k=5)
        
        # Gene -> Morph检索
        sim_gm = cosine_similarity(gene_emb, morph_emb)
        metrics['gene2morph_top1'] = self._compute_recall_at_k(sim_gm, k=1)
        metrics['gene2morph_top5'] = self._compute_recall_at_k(sim_gm, k=5)
        
        return metrics
    
    def _compute_recall_at_k(self, sim_matrix: np.ndarray, k: int = 5) -> float:
        """计算Recall@K"""
        n = sim_matrix.shape[0]
        correct = 0
        
        for i in range(n):
            # 获取top-k预测
            top_k_idx = np.argsort(sim_matrix[i])[::-1][:k]
            
            # 检查正确匹配是否在top-k中
            if i in top_k_idx:
                correct += 1
        
        return correct / n
    
    def _extract_features(self, model, dataloader) -> Tuple[np.ndarray, np.ndarray]:
        """提取特征用于下游任务"""
        model.eval()
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                batch = self._to_device(batch)
                
                # 获取特征
                features = []
                if 'mol' in batch:
                    mol_feat = model.mol_encoder(batch['mol'])
                    features.append(mol_feat)
                if 'gene' in batch:
                    gene_feat = model.gene_encoder(batch['gene'])
                    features.append(gene_feat)
                if 'morph' in batch:
                    morph_feat = model.morph_encoder(batch['morph'])
                    features.append(morph_feat)
                
                # 拼接所有可用特征
                if features:
                    combined = torch.cat(features, dim=-1)
                    all_features.append(combined.cpu().numpy())
                
                # 收集标签
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu().numpy())
        
        features = np.vstack(all_features) if all_features else np.array([])
        labels = np.hstack(all_labels) if all_labels else np.array([])
        
        return features, labels
    
    def _to_device(self, batch: Dict) -> Dict:
        """将数据移到设备"""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch