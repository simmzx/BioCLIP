# main.py - 更新版
#!/usr/bin/env python
"""
BioCLIP主程序入口
支持预训练、微调和数据预处理
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import torch
from config.config import BioCLIPConfig
from training.trainer import Trainer
from training.evaluator import Evaluator
from utils.helpers import set_seed, create_exp_dir, save_config

def main():
    parser = argparse.ArgumentParser(description='BioCLIP - Multi-modal Drug Discovery')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['preprocess', 'train', 'finetune', 'eval', 'predict'],
                       help='Running mode')
    parser.add_argument('--task', type=str, default=None,
                       help='Downstream task name (for finetune mode)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    if args.mode == 'preprocess':
        # 数据预处理模式
        from data.preprocessing import JUMPCPPreprocessor
        
        print("Starting data preprocessing...")
        preprocessor = JUMPCPPreprocessor(
            raw_data_path='./datasets/raw',
            processed_data_path='./datasets/jump_cp/processed'
        )
        preprocessor.preprocess_all()
        
    elif args.mode == 'train':
        # 预训练模式
        config = BioCLIPConfig(config_path=args.config)
        
        # 创建实验目录
        exp_dir = create_exp_dir()
        config.log_dir = os.path.join(exp_dir, 'logs')
        config.checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
        
        # 保存配置
        save_config(config, os.path.join(exp_dir, 'config.yaml'))
        
        # 训练
        trainer = Trainer(config)
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        trainer.train()
        
    elif args.mode == 'finetune':
        # 微调模式
        if not args.task:
            raise ValueError("Task name required for finetuning")
        if not args.checkpoint:
            raise ValueError("Checkpoint required for finetuning")
        
        from models.bioclip import BioCLIP
        from downstream.finetuning import FineTuner
        from downstream.datasets import DownstreamDataModule
        
        config = BioCLIPConfig(config_path=args.config)
        
        # 加载预训练模型
        model = BioCLIP(config).to(config.device)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建数据模块
        data_module = DownstreamDataModule(args.task)
        train_idx, val_idx, test_idx = data_module.get_splits()
        
        # 创建数据加载器
        from downstream.datasets import DownstreamDataset
        train_dataset = DownstreamDataset(args.task, split='train')
        val_dataset = DownstreamDataset(args.task, split='val')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=32, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=32, shuffle=False
        )
        
        # 微调
        finetuner = FineTuner(model, args.task, config)
        finetuner.finetune(train_loader, val_loader, epochs=50)
        
    elif args.mode == 'eval':
        # 评估模式
        if not args.checkpoint:
            raise ValueError("Checkpoint required for evaluation")
        
        from models.bioclip import BioCLIP
        from data.dataloader import create_dataloaders
        
        config = BioCLIPConfig(config_path=args.config)
        
        # 加载模型
        model = BioCLIP(config).to(config.device)
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建数据加载器
        _, val_loader, test_loader = create_dataloaders(config)
        
        # 评估
        evaluator = Evaluator(model, config, config.device)
        
        val_metrics = evaluator.evaluate(val_loader)
        print("\nValidation Metrics:")
        for k, v in val_metrics.items():
            print(f"  {k}: {v:.4f}")
        
        test_metrics = evaluator.evaluate(test_loader)
        print("\nTest Metrics:")
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.4f}")
    
    print("\nDone!")

if __name__ == '__main__':
    main()