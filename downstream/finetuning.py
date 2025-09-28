# downstream/finetuning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, Optional, List
import numpy as np
from tqdm import tqdm

from models.bioclip import BioCLIP
from downstream.datasets import DownstreamDataset
from downstream.tasks import DOWNSTREAM_TASKS
from utils.metrics import compute_metrics

class FineTuner:
    """BioCLIP微调框架"""
    
    def __init__(self, 
                 pretrained_model: BioCLIP,
                 task_name: str,
                 config):
        
        self.model = pretrained_model
        self.task_name = task_name
        self.task_config = DOWNSTREAM_TASKS[task_name]
        self.config = config
        self.device = config.device
        
        # 冻结预训练模型的部分层
        self._freeze_pretrained_layers()
        
        # 添加任务特定的头
        self.task_head = self._build_task_head()
        
        # 优化器（只优化任务头和部分层）
        self.optimizer = self._setup_optimizer()
        
        # 损失函数
        self.criterion = self._get_loss_function()
        
    def _freeze_pretrained_layers(self):
        """冻结预训练层（可选）"""
        if self.config.freeze_encoders:
            # 冻结编码器
            for param in self.model.mol_encoder.parameters():
                param.requires_grad = False
            for param in self.model.gene_encoder.parameters():
                param.requires_grad = False
            for param in self.model.morph_encoder.parameters():
                param.requires_grad = False
        
        # 只冻结前几层
        elif self.config.freeze_n_layers > 0:
            # 实现部分冻结逻辑
            pass
    
    def _build_task_head(self) -> nn.Module:
        """构建任务特定的预测头"""
        
        input_dim = self.config.hidden_dim * len(self.config.use_features)
        
        if self.task_config.task_type == 'classification':
            return nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.task_config.n_classes)
            )
        
        elif self.task_config.task_type == 'regression':
            return nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1)
            )
        
        elif self.task_config.task_type == 'multilabel_classification':
            return nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, self.task_config.n_classes)
            )
        
        else:
            raise ValueError(f"Unknown task type: {self.task_config.task_type}")
    
    def _get_loss_function(self):
        """获取损失函数"""
        if self.task_config.task_type == 'classification':
            return nn.CrossEntropyLoss()
        elif self.task_config.task_type == 'regression':
            return nn.MSELoss()
        elif self.task_config.task_type == 'multilabel_classification':
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown task type: {self.task_config.task_type}")
    
    def _setup_optimizer(self):
        """设置优化器"""
        # 获取需要优化的参数
        params = []
        
        # 任务头参数
        params.append({
            'params': self.task_head.parameters(),
            'lr': self.config.finetune_lr
        })
        
        # 投影头参数（可选）
        if not self.config.freeze_projections:
            params.append({
                'params': self.model.mol_proj.parameters(),
                'lr': self.config.finetune_lr * 0.1
            })
            params.append({
                'params': self.model.gene_proj.parameters(),
                'lr': self.config.finetune_lr * 0.1
            })
            params.append({
                'params': self.model.morph_proj.parameters(),
                'lr': self.config.finetune_lr * 0.1
            })
        
        # 编码器参数（如果未冻结）
        if not self.config.freeze_encoders:
            params.append({
                'params': self.model.mol_encoder.parameters(),
                'lr': self.config.finetune_lr * 0.01
            })
            params.append({
                'params': self.model.gene_encoder.parameters(),
                'lr': self.config.finetune_lr * 0.01
            })
            params.append({
                'params': self.model.morph_encoder.parameters(),
                'lr': self.config.finetune_lr * 0.01
            })
        
        return torch.optim.AdamW(params, weight_decay=1e-5)
    
    def finetune(self, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 epochs: int = 50):
        """微调模型"""
        
        print(f"\nFinetuning BioCLIP for {self.task_name}")
        print("="*60)
        
        best_val_metric = -float('inf') if 'auc' in self.task_config.metrics[0] else float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_loss = self._train_epoch(train_loader, epoch, epochs)
            
            # 验证
            val_metrics = self._validate(val_loader)
            
            # 打印结果
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            for metric_name, value in val_metrics.items():
                print(f"  Val {metric_name}: {value:.4f}")
            
            # 早停
            primary_metric = val_metrics[self.task_config.metrics[0]]
            
            if 'auc' in self.task_config.metrics[0] or 'accuracy' in self.task_config.metrics[0]:
                is_better = primary_metric > best_val_metric
            else:
                is_better = primary_metric < best_val_metric
            
            if is_better:
                best_val_metric = primary_metric
                patience_counter = 0
                # 保存模型
                self._save_checkpoint(epoch, val_metrics)
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        return best_val_metric
    
    def _train_epoch(self, dataloader: DataLoader, epoch: int, total_epochs: int) -> float:
        """训练一个epoch"""
        self.model.train()
        self.task_head.train()
        
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training {epoch+1}/{total_epochs}")
        
        for batch in pbar:
            # 移到设备
            batch = self._to_device(batch)
            
            # 获取特征
            features = self._extract_features(batch)
            
            # 通过任务头
            logits = self.task_head(features)
            
            # 计算损失
            if self.task_config.task_type == 'regression':
                logits = logits.squeeze(-1)
            
            loss = self.criterion(logits, batch['labels'])
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.task_head.parameters()),
                1.0
            )
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> Dict:
        """验证"""
        self.model.eval()
        self.task_head.eval()
        
        all_preds = []
        all_labels = []
        
        for batch in dataloader:
            batch = self._to_device(batch)
            
            # 获取特征
            features = self._extract_features(batch)
            
            # 预测
            logits = self.task_head(features)
            
            if self.task_config.task_type == 'classification':
                preds = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            elif self.task_config.task_type == 'regression':
                preds = logits.squeeze(-1).cpu().numpy()
            elif self.task_config.task_type == 'multilabel_classification':
                preds = torch.sigmoid(logits).cpu().numpy()
            else:
                preds = logits.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(batch['labels'].cpu().numpy())
        
        # 合并预测
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # 计算指标
        metrics = compute_metrics(
            all_labels,
            all_preds,
            task_type=self.task_config.task_type,
            metrics_list=self.task_config.metrics
        )
        
        return metrics
    
    def _extract_features(self, batch: Dict) -> torch.Tensor:
        """提取特征"""
        features = []
        
        # 获取各模态特征
        with torch.no_grad() if self.config.freeze_encoders else torch.enable_grad():
            if 'mol' in batch:
                mol_feat = self.model.mol_encoder(batch['mol'])
                features.append(mol_feat)
            
            if 'gene' in batch:
                gene_feat = self.model.gene_encoder(batch['gene'])
                features.append(gene_feat)
            
            if 'morph' in batch:
                morph_feat = self.model.morph_encoder(batch['morph'])
                features.append(morph_feat)
        
        # 拼接特征
        combined_features = torch.cat(features, dim=-1)
        
        return combined_features
    
    def _to_device(self, batch: Dict) -> Dict:
        """移到设备"""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
        return batch
    
    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'task_head_state_dict': self.task_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'task_name': self.task_name
        }
        
        path = os.path.join(
            self.config.checkpoint_dir,
            f'{self.task_name}_best.pth'
        )
        torch.save(checkpoint, path)