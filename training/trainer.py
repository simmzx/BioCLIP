# training/trainer.py
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
import wandb

# 导入所有必要模块
from config.config import BioCLIPConfig
from models.bioclip import BioCLIP
from data.dataset import BioCLIPDataset, CombinedDataset
from data.augmentation import (
    MolecularAugmenter,
    GeneExpressionAugmenter,
    CellMorphologyAugmenter
)
from data.pseudo_labeling import AdvancedPseudoLabelGenerator
from data.dataloader import BioCLIPDataCollator, create_dataloaders
from training.evaluator import Evaluator
from utils.helpers import save_checkpoint, load_checkpoint, set_seed

class Trainer:
    """BioCLIP训练器 - 完整版本"""
    
    def __init__(self, config: BioCLIPConfig):
        self.config = config
        self.device = config.device
        
        # 设置随机种子
        set_seed(config.seed)
        
        # 初始化增强器
        print("Initializing augmenters...")
        self.mol_augmenter = MolecularAugmenter()
        self.gene_augmenter = GeneExpressionAugmenter()
        self.morph_augmenter = CellMorphologyAugmenter()
        
        # 初始化模型
        print("Initializing model...")
        self.model = BioCLIP(config).to(self.device)
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 创建数据加载器
        print("Creating data loaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(config)
        
        # 初始化伪标签生成器
        print("Initializing pseudo-label generator...")
        self.pseudo_generator = AdvancedPseudoLabelGenerator(self.model, config)
        
        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50,
            T_mult=2,
            eta_min=1e-6
        )
        
        # 评估器
        self.evaluator = Evaluator(self.model, config, self.device)
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 初始化wandb（可选）
        self.use_wandb = False  # 设为True以使用wandb
        if self.use_wandb:
            wandb.init(project="bioclip", config=config.to_dict())
        
        print("Trainer initialized successfully!")
    
    def train(self):
        """完整的PMB训练流程"""
        print("\n" + "="*60)
        print("Starting BioCLIP Progressive Multi-modal Bootstrapping")
        print("="*60)
        
        # Stage 1: 单模态预训练
        print("\n[Stage 1] Single-modal Self-supervised Pretraining")
        print("-"*60)
        self._train_stage1()
        
        # Stage 2: 双模态对齐
        print("\n[Stage 2] Bi-modal Alignment")
        print("-"*60)
        self._train_stage2()
        
        # 生成伪标签
        print("\n[Pseudo-labeling] Generating pseudo tri-modal samples")
        print("-"*60)
        pseudo_samples = self._generate_pseudo_labels()
        
        # Stage 3: 三模态融合
        print("\n[Stage 3] Tri-modal Fusion with Pseudo-labels")
        print("-"*60)
        self._train_stage3(pseudo_samples)
        
        # 最终评估
        print("\n[Final Evaluation]")
        print("-"*60)
        self._final_evaluation()
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print("="*60)
    
    def _train_stage1(self):
        """Stage 1: 单模态自监督预训练"""
        
        for epoch in range(self.config.stage1_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            epoch_losses = {'mol': [], 'gene': [], 'morph': []}
            
            pbar = tqdm(self.train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{self.config.stage1_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                # 数据增强和预处理
                batch = self._prepare_batch_stage1(batch)
                batch = self._to_device(batch)
                
                # 前向传播
                outputs = self.model(batch, stage='stage1')
                
                # 计算总损失
                total_loss = 0
                loss_dict = {}
                
                for modality in ['mol', 'gene', 'morph']:
                    loss_key = f'{modality}_ssl_loss'
                    if loss_key in outputs and outputs[loss_key] is not None:
                        loss = outputs[loss_key]
                        total_loss += loss
                        epoch_losses[modality].append(loss.item())
                        loss_dict[modality] = loss.item()
                
                # 反向传播
                if total_loss > 0:
                    # 梯度累积
                    total_loss = total_loss / self.config.gradient_accumulation
                    total_loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # 更新进度条
                pbar.set_postfix({
                    k: f"{np.mean(v[-100:]):.4f}" if v else 0
                    for k, v in epoch_losses.items()
                })
                
                # 记录到wandb
                if self.use_wandb and batch_idx % 100 == 0:
                    wandb.log({
                        f'stage1/{k}_loss': v for k, v in loss_dict.items()
                    })
            
            # 学习率调度
            self.scheduler.step()
            
            # 验证和保存
            if (epoch + 1) % self.config.save_frequency == 0:
                val_metrics = self._validate('stage1')
                print(f"\n  Epoch {epoch+1} - Validation metrics:")
                for k, v in val_metrics.items():
                    print(f"    {k}: {v:.4f}")
                
                if val_metrics['avg_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['avg_loss']
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        os.path.join(self.config.checkpoint_dir, 'best_stage1.pth'),
                        val_loss=self.best_val_loss
                    )
    
    def _prepare_batch_stage1(self, batch: Dict) -> Dict:
        """Stage 1批次准备 - 应用增强"""
        # 对每个模态应用增强
        if 'mol' in batch:
            mol_original = batch['mol']
            mol_augmented, mol_aug = self.mol_augmenter.augment_batch(
                batch.get('smiles', [])
            )
            batch['mol'] = mol_original if torch.is_tensor(mol_original) else mol_augmented
            batch['mol_aug'] = mol_aug
        
        if 'gene' in batch:
            gene_original = batch['gene']
            gene_original, gene_aug = self.gene_augmenter.augment_batch(gene_original)
            batch['gene'] = gene_original
            batch['gene_aug'] = gene_aug
        
        if 'morph' in batch:
            morph_original = batch['morph']
            morph_original, morph_aug = self.morph_augmenter.augment_batch(morph_original)
            batch['morph'] = morph_original
            batch['morph_aug'] = morph_aug
        
        return batch
    
    def _train_stage2(self):
        """Stage 2: 双模态对齐"""
        
        # 创建部分模态数据集
        partial_dataset = BioCLIPDataset(
            self.config.data_path,
            mode='train',
            data_type='partial'
        )
        
        partial_loader = partial_dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        for epoch in range(self.config.stage2_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            epoch_losses = {}
            pbar = tqdm(partial_loader, desc=f"Stage 2 - Epoch {epoch+1}/{self.config.stage2_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                batch = self._to_device(batch)
                
                # 前向传播
                outputs = self.model(batch, stage='stage2')
                
                # 计算损失
                total_loss = 0
                loss_dict = {}
                
                for key in outputs:
                    if 'loss' in key and outputs[key] is not None:
                        loss = outputs[key]
                        total_loss += loss
                        
                        loss_name = key.replace('_loss', '')
                        if loss_name not in epoch_losses:
                            epoch_losses[loss_name] = []
                        epoch_losses[loss_name].append(loss.item())
                        loss_dict[loss_name] = loss.item()
                
                # 反向传播
                if total_loss > 0:
                    total_loss = total_loss / self.config.gradient_accumulation
                    total_loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # 更新进度条
                pbar.set_postfix({
                    k: f"{np.mean(v[-100:]):.4f}" if v else 0
                    for k, v in epoch_losses.items()
                })
            
            self.scheduler.step()
            
            # 验证
            if (epoch + 1) % self.config.save_frequency == 0:
                val_metrics = self._validate('stage2')
                print(f"\n  Epoch {epoch+1} - Validation metrics:")
                for k, v in val_metrics.items():
                    print(f"    {k}: {v:.4f}")
    
    def _generate_pseudo_labels(self) -> List[Dict]:
        """生成伪标签"""
        # 收集部分模态数据
        partial_dataset = BioCLIPDataset(
            self.config.data_path,
            mode='train',
            data_type='partial'
        )
        
        partial_loader = partial_dataset.get_dataloader(
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        partial_data = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(partial_loader, desc="Collecting partial data"):
                batch = self._to_device(batch)
                
                # 转换批次为样本列表
                batch_size = len(batch.get('compound_id', []))
                for i in range(batch_size):
                    sample = {}
                    
                    for key in ['mol', 'gene', 'morph']:
                        if key in batch:
                            sample[key] = batch[key][i]
                    
                    sample['available_modalities'] = batch['available_modalities'][i] \
                        if 'available_modalities' in batch else []
                    sample['compound_id'] = batch['compound_id'][i] \
                        if 'compound_id' in batch else f'sample_{i}'
                    
                    partial_data.append(sample)
        
        # 生成伪标签
        pseudo_samples = self.pseudo_generator.generate_pseudo_trimodal(
            partial_data,
            n_samples=self.config.pseudo_samples
        )
        
        print(f"Generated {len(pseudo_samples)} pseudo tri-modal samples")
        
        return pseudo_samples
    
    def _train_stage3(self, pseudo_samples: List[Dict]):
        """Stage 3: 三模态融合训练"""
        
        # 创建组合数据集
        complete_dataset = BioCLIPDataset(
            self.config.data_path,
            mode='train',
            data_type='complete'
        )
        
        # 创建伪标签数据集（简化版本，实际需要更复杂的实现）
        from torch.utils.data import TensorDataset
        
        # 这里简化处理，实际应该创建专门的PseudoDataset
        combined_dataset = CombinedDataset([complete_dataset])
        
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=BioCLIPDataCollator(self.config)
        )
        
        for epoch in range(self.config.stage3_epochs):
            self.current_epoch = epoch
            self.model.train()
            
            epoch_losses = {'triple': [], 'pseudo': []}
            pbar = tqdm(combined_loader, desc=f"Stage 3 - Epoch {epoch+1}/{self.config.stage3_epochs}")
            
            for batch_idx, batch in enumerate(pbar):
                batch = self._to_device(batch)
                
                # 前向传播
                outputs = self.model(batch, stage='stage3')
                
                # 计算损失
                total_loss = 0
                
                if 'triple_loss' in outputs:
                    total_loss += outputs['triple_loss']
                    epoch_losses['triple'].append(outputs['triple_loss'].item())
                
                if 'pseudo_loss' in outputs:
                    total_loss += outputs['pseudo_loss'] * 0.5
                    epoch_losses['pseudo'].append(outputs['pseudo_loss'].item())
                
                # 反向传播
                if total_loss > 0:
                    total_loss = total_loss / self.config.gradient_accumulation
                    total_loss.backward()
                    
                    if (batch_idx + 1) % self.config.gradient_accumulation == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                
                # 更新进度条
                pbar.set_postfix({
                    k: f"{np.mean(v[-100:]):.4f}" if v else 0
                    for k, v in epoch_losses.items()
                })
            
            self.scheduler.step()
            
            # 验证和保存
            if (epoch + 1) % self.config.save_frequency == 0:
                val_metrics = self._validate('stage3')
                print(f"\n  Epoch {epoch+1} - Validation metrics:")
                for k, v in val_metrics.items():
                    print(f"    {k}: {v:.4f}")
                
                if val_metrics['avg_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['avg_loss']
                    save_checkpoint(
                        self.model, self.optimizer, epoch,
                        os.path.join(self.config.checkpoint_dir, 'best_model.pth'),
                        val_loss=self.best_val_loss
                    )
    
    def _validate(self, stage: str) -> Dict:
        """验证模型"""
        return self.evaluator.evaluate(self.val_loader, stage=stage)
    
    def _final_evaluation(self):
        """最终评估"""
        print("\nFinal evaluation on test set...")
        test_metrics = self.evaluator.evaluate(self.test_loader, stage='stage3')
        
        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        # 保存结果
        import json
        with open(os.path.join(self.config.log_dir, 'final_metrics.json'), 'w') as f:
            json.dump(test_metrics, f, indent=2)
    
    def _to_device(self, batch: Dict) -> Dict:
        """将批次数据移到设备"""
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(self.device)
            elif isinstance(batch[key], list):
                if len(batch[key]) > 0 and isinstance(batch[key][0], torch.Tensor):
                    batch[key] = [t.to(self.device) for t in batch[key]]
        return batch
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = load_checkpoint(
            checkpoint_path,
            self.model,
            self.optimizer
        )
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Loaded checkpoint from epoch {self.current_epoch}")