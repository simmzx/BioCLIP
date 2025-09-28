# downstream/datasets.py
"""
下游任务数据集类
支持所有药物发现相关的下游任务
"""

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split, StratifiedKFold
import json
import warnings
warnings.filterwarnings('ignore')

from downstream.tasks import DOWNSTREAM_TASKS, get_task_info
from downstream.real_datasets import RealDownstreamDatasets

class DownstreamDataset(Dataset):
    """
    下游任务数据集
    支持多种药物发现任务的数据加载和处理
    """
    
    def __init__(self,
                 task_name: str,
                 data_root: str = './datasets/downstream',
                 split: str = 'train',
                 use_features: List[str] = ['mol'],
                 transform: Optional[Any] = None,
                 precomputed_features: Optional[str] = None):
        """
        Args:
            task_name: 任务名称（如'tox21', 'davis'等）
            data_root: 数据根目录
            split: 数据集划分 ('train', 'val', 'test')
            use_features: 使用的特征模态 ['mol', 'gene', 'morph']
            transform: 数据变换
            precomputed_features: 预计算特征的路径
        """
        super().__init__()
        
        self.task_name = task_name
        self.task_info = get_task_info(task_name)
        self.data_root = data_root
        self.split = split
        self.use_features = use_features
        self.transform = transform
        self.precomputed_features = precomputed_features
        
        # 加载数据
        self.data = self._load_data()
        
        # 如果有预计算特征，加载它们
        if precomputed_features:
            self._load_precomputed_features()
        
        print(f"Loaded {task_name} {split} dataset: {len(self.data)} samples")
        self._print_statistics()
    
    def _load_data(self) -> pd.DataFrame:
        """加载任务数据"""
        
        # 首先尝试加载已处理的数据
        processed_file = os.path.join(
            self.data_root,
            self.task_name,
            f'{self.split}.csv'
        )
        
        if os.path.exists(processed_file):
            return pd.read_csv(processed_file)
        
        # 如果没有处理好的数据，从原始数据加载
        print(f"Processed file not found, loading raw data for {self.task_name}")
        
        # 使用RealDownstreamDatasets加载原始数据
        data_loader = RealDownstreamDatasets(self.data_root)
        
        # 根据不同任务加载数据
        if self.task_name == 'tox21':
            raw_data = data_loader.load_tox21()
            return self._process_tox21_data(raw_data)
            
        elif self.task_name == 'davis':
            raw_data = data_loader.load_davis()
            return self._process_davis_data(raw_data)
            
        elif self.task_name == 'sider':
            raw_data = data_loader.load_sider()
            return self._process_sider_data(raw_data)
            
        elif self.task_name == 'bbbp':
            raw_data = data_loader.download_dataset('bbbp')
            return self._process_classification_data(raw_data)
            
        elif self.task_name == 'esol':
            raw_data = data_loader.download_dataset('esol')
            return self._process_regression_data(raw_data, 'measured log solubility in mols per litre')
            
        elif self.task_name == 'lipophilicity':
            raw_data = data_loader.download_dataset('lipophilicity')
            return self._process_regression_data(raw_data, 'exp')
            
        elif self.task_name == 'clintox':
            raw_data = data_loader.download_dataset('clintox')
            return self._process_multitask_data(raw_data, ['FDA_APPROVED', 'CT_TOX'])
            
        else:
            raise ValueError(f"Unsupported task: {self.task_name}")
    
    def _process_tox21_data(self, raw_data: Dict) -> pd.DataFrame:
        """处理Tox21数据"""
        
        # 创建DataFrame
        df = pd.DataFrame({
            'smiles': raw_data['smiles'],
            'compound_id': [f'TOX_{i:06d}' for i in range(len(raw_data['smiles']))]
        })
        
        # 添加标签（12个毒性终点）
        for i, task in enumerate(raw_data['task_names']):
            df[task] = raw_data['labels'][:, i]
        
        # 划分数据集
        return self._split_data(df)
    
    def _process_davis_data(self, raw_data: Dict) -> pd.DataFrame:
        """处理Davis数据（DTI）"""
        
        # Davis数据是一个亲和力矩阵
        affinity_matrix = raw_data['affinity_matrix']
        drug_smiles = raw_data['drug_smiles']
        target_sequences = raw_data['target_sequences']
        
        # 展开为成对数据
        data = []
        for i, smiles in enumerate(drug_smiles):
            for j, target in enumerate(target_sequences):
                if not np.isnan(affinity_matrix[i, j]):
                    data.append({
                        'smiles': smiles,
                        'target_sequence': target,
                        'affinity': affinity_matrix[i, j],
                        'drug_id': f'DRUG_{i:04d}',
                        'target_id': f'TARGET_{j:04d}'
                    })
        
        df = pd.DataFrame(data)
        return self._split_data(df)
    
    def _process_sider_data(self, raw_data: Dict) -> pd.DataFrame:
        """处理SIDER副作用数据"""
        
        df = pd.DataFrame({
            'smiles': raw_data['smiles'],
            'compound_id': [f'SIDER_{i:06d}' for i in range(len(raw_data['smiles']))]
        })
        
        # 添加27个副作用标签
        for i, side_effect in enumerate(raw_data['task_names']):
            df[side_effect] = raw_data['labels'][:, i]
        
        return self._split_data(df)
    
    def _process_classification_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理二分类数据"""
        
        # 重命名列
        if 'p_np' in df.columns:
            df['label'] = df['p_np']  # BBBP数据集
        elif 'y' in df.columns:
            df['label'] = df['y']
        
        # 添加compound_id
        df['compound_id'] = [f'{self.task_name.upper()}_{i:06d}' for i in range(len(df))]
        
        return self._split_data(df)
    
    def _process_regression_data(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """处理回归数据"""
        
        df['label'] = df[target_col]
        df['compound_id'] = [f'{self.task_name.upper()}_{i:06d}' for i in range(len(df))]
        
        return self._split_data(df)
    
    def _process_multitask_data(self, df: pd.DataFrame, task_cols: List[str]) -> pd.DataFrame:
        """处理多任务数据"""
        
        for col in task_cols:
            df[f'label_{col}'] = df[col]
        
        df['compound_id'] = [f'{self.task_name.upper()}_{i:06d}' for i in range(len(df))]
        
        return self._split_data(df)
    
    def _split_data(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1) -> pd.DataFrame:
        """划分数据集"""
        
        # 如果已经有划分的索引文件，使用它
        split_file = os.path.join(
            self.data_root,
            self.task_name,
            'splits.json'
        )
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            
            if self.split == 'train':
                return df.iloc[splits['train_idx']]
            elif self.split == 'val':
                return df.iloc[splits['val_idx']]
            else:
                return df.iloc[splits['test_idx']]
        
        # 否则创建新的划分
        n_samples = len(df)
        indices = np.arange(n_samples)
        
        # 使用scaffold split（基于分子骨架的划分）更符合药物发现实际
        if 'smiles' in df.columns:
            scaffolds = self._generate_scaffolds(df['smiles'].values)
            train_idx, test_idx = self._scaffold_split(scaffolds, test_size)
            train_idx, val_idx = self._scaffold_split(
                scaffolds[train_idx], 
                val_size / (1 - test_size)
            )
        else:
            # 随机划分
            train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
            train_idx, val_idx = train_test_split(train_idx, test_size=val_size/(1-test_size), random_state=42)
        
        # 保存划分
        os.makedirs(os.path.dirname(split_file), exist_ok=True)
        with open(split_file, 'w') as f:
            json.dump({
                'train_idx': train_idx.tolist(),
                'val_idx': val_idx.tolist(),
                'test_idx': test_idx.tolist()
            }, f)
        
        # 返回相应的数据
        if self.split == 'train':
            return df.iloc[train_idx]
        elif self.split == 'val':
            return df.iloc[val_idx]
        else:
            return df.iloc[test_idx]
    
    def _generate_scaffolds(self, smiles_list: List[str]) -> np.ndarray:
        """生成分子骨架用于scaffold split"""
        from rdkit.Chem import Scaffolds
        
        scaffolds = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                scaffold = Scaffolds.MurckoScaffold.GetScaffoldForMol(mol)
                scaffolds.append(Chem.MolToSmiles(scaffold))
            else:
                scaffolds.append(smiles)
        
        # 将骨架转换为数字标签
        unique_scaffolds = list(set(scaffolds))
        scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
        scaffold_ids = np.array([scaffold_to_id[s] for s in scaffolds])
        
        return scaffold_ids
    
    def _scaffold_split(self, scaffolds: np.ndarray, test_size: float) -> Tuple[np.ndarray, np.ndarray]:
        """基于骨架的数据划分"""
        unique_scaffolds = np.unique(scaffolds)
        np.random.shuffle(unique_scaffolds)
        
        # 按骨架分组
        scaffold_groups = {}
        for idx, scaffold in enumerate(scaffolds):
            if scaffold not in scaffold_groups:
                scaffold_groups[scaffold] = []
            scaffold_groups[scaffold].append(idx)
        
        # 分配到训练集和测试集
        train_idx = []
        test_idx = []
        test_count = 0
        total_count = len(scaffolds)
        
        for scaffold in unique_scaffolds:
            group = scaffold_groups[scaffold]
            if test_count + len(group) <= test_size * total_count:
                test_idx.extend(group)
                test_count += len(group)
            else:
                train_idx.extend(group)
        
        return np.array(train_idx), np.array(test_idx)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个样本"""
        
        row = self.data.iloc[idx]
        sample = {
            'idx': idx,
            'compound_id': row.get('compound_id', f'COMPOUND_{idx}')
        }
        
        # 加载分子特征
        if 'mol' in self.use_features and 'smiles' in row:
            sample['smiles'] = row['smiles']
            sample['mol'] = self._get_mol_features(row['smiles'])
        
        # 加载基因表达（如果有）
        if 'gene' in self.use_features:
            sample['gene'] = self._get_gene_features(row)
        
        # 加载细胞形态（如果有）
        if 'morph' in self.use_features:
            sample['morph'] = self._get_morph_features(row)
        
        # 加载靶点序列（对于DTI任务）
        if 'target_sequence' in row:
            sample['target'] = self._get_target_features(row['target_sequence'])
        
        # 加载标签
        sample['labels'] = self._get_labels(row)
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _get_mol_features(self, smiles: str) -> torch.Tensor:
        """获取分子特征（Morgan指纹）"""
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(2048, dtype=torch.float32)
        
        # 计算Morgan指纹
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fp_arr = np.zeros(2048)
        from rdkit.DataStructs import ConvertToNumpyArray
        ConvertToNumpyArray(fp, fp_arr)
        
        return torch.tensor(fp_arr, dtype=torch.float32)
    
    def _get_gene_features(self, row: pd.Series) -> torch.Tensor:
        """获取基因表达特征"""
        
        # 检查是否有基因表达数据
        gene_file = os.path.join(
            self.data_root,
            self.task_name,
            'gene_expression',
            f'{row["compound_id"]}.npy'
        )
        
        if os.path.exists(gene_file):
            gene_expr = np.load(gene_file)
        else:
            # 如果没有，返回零向量
            gene_expr = np.zeros(978)  # L1000的978个landmark基因
        
        return torch.tensor(gene_expr, dtype=torch.float32)
    
    def _get_morph_features(self, row: pd.Series) -> torch.Tensor:
        """获取细胞形态特征"""
        
        # 检查是否有细胞图像数据
        image_file = os.path.join(
            self.data_root,
            self.task_name,
            'cell_images',
            f'{row["compound_id"]}.npy'
        )
        
        if os.path.exists(image_file):
            cell_image = np.load(image_file)
        else:
            # 如果没有，返回零张量
            cell_image = np.zeros((6, 224, 224))
        
        return torch.tensor(cell_image, dtype=torch.float32)
    
    def _get_target_features(self, sequence: str) -> torch.Tensor:
        """获取蛋白质靶点特征（用于DTI任务）"""
        
        # 使用简单的氨基酸组成特征
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
        
        # 计算氨基酸频率
        features = np.zeros(len(amino_acids))
        for aa in sequence:
            if aa in aa_dict:
                features[aa_dict[aa]] += 1
        
        # 归一化
        if features.sum() > 0:
            features = features / features.sum()
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _get_labels(self, row: pd.Series) -> torch.Tensor:
        """获取标签"""
        
        task_type = self.task_info.task_type
        
        if task_type == 'binary_classification':
            label = row.get('label', 0)
            return torch.tensor(label, dtype=torch.long)
            
        elif task_type == 'regression':
            label = row.get('label', row.get('affinity', 0))
            return torch.tensor(label, dtype=torch.float32)
            
        elif task_type in ['multilabel_classification', 'multitask_classification']:
            # 收集所有标签列
            label_cols = [col for col in row.index if col.startswith('label_') or 
                         col in self.task_info.metrics]
            
            if not label_cols:
                # 对于Tox21等数据集，使用任务名作为列名
                if hasattr(self, 'task_names'):
                    label_cols = self.task_names
                else:
                    # 从数据中推断
                    label_cols = [col for col in row.index if col not in 
                                 ['smiles', 'compound_id', 'idx']]
            
            labels = []
            for col in label_cols:
                if col in row:
                    val = row[col]
                    # 处理缺失值（-1表示未测试）
                    if pd.isna(val) or val == -1:
                        labels.append(0)  # 或者使用特殊标记
                    else:
                        labels.append(val)
                else:
                    labels.append(0)
            
            return torch.tensor(labels, dtype=torch.float32)
            
        elif task_type == 'multitask_regression':
            # QM9等多任务回归
            label_cols = [col for col in row.index if col.startswith('target_')]
            labels = [row[col] for col in label_cols]
            return torch.tensor(labels, dtype=torch.float32)
            
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    
    def _print_statistics(self):
        """打印数据集统计信息"""
        
        print(f"  Task type: {self.task_info.task_type}")
        print(f"  Number of samples: {len(self.data)}")
        
        if 'smiles' in self.data.columns:
            # 计算分子多样性
            unique_smiles = self.data['smiles'].nunique()
            print(f"  Unique molecules: {unique_smiles}")
        
        # 标签分布
        if self.task_info.task_type == 'binary_classification':
            if 'label' in self.data.columns:
                pos_ratio = self.data['label'].mean()
                print(f"  Positive ratio: {pos_ratio:.2%}")
        
        elif self.task_info.task_type == 'regression':
            if 'label' in self.data.columns:
                mean_val = self.data['label'].mean()
                std_val = self.data['label'].std()
                print(f"  Label mean: {mean_val:.3f}, std: {std_val:.3f}")
    
    def _load_precomputed_features(self):
        """加载预计算的特征"""
        
        feature_file = os.path.join(self.precomputed_features, f'{self.split}.pt')
        
        if os.path.exists(feature_file):
            self.precomputed = torch.load(feature_file)
            print(f"Loaded precomputed features from {feature_file}")
        else:
            print(f"Precomputed features not found at {feature_file}")
            self.precomputed = None
    
    def collate_fn(self, batch: List[Dict]) -> Dict:
        """自定义批处理函数"""
        
        collated = {
            'idx': torch.tensor([x['idx'] for x in batch]),
            'compound_id': [x['compound_id'] for x in batch]
        }
        
        # 处理分子特征
        if 'mol' in batch[0]:
            collated['mol'] = torch.stack([x['mol'] for x in batch])
        
        # 处理基因表达
        if 'gene' in batch[0]:
            collated['gene'] = torch.stack([x['gene'] for x in batch])
        
        # 处理细胞形态
        if 'morph' in batch[0]:
            collated['morph'] = torch.stack([x['morph'] for x in batch])
        
        # 处理靶点
        if 'target' in batch[0]:
            collated['target'] = torch.stack([x['target'] for x in batch])
        
        # 处理标签
        collated['labels'] = torch.stack([x['labels'] for x in batch])
        
        return collated


class DownstreamDataModule:
    """
    下游任务数据模块
    管理数据集的创建、划分和加载
    """
    
    def __init__(self,
                 task_name: str,
                 data_root: str = './datasets/downstream',
                 batch_size: int = 32,
                 num_workers: int = 4,
                 use_features: List[str] = ['mol']):
        
        self.task_name = task_name
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_features = use_features
        
        # 任务信息
        self.task_info = get_task_info(task_name)
        
    def setup(self):
        """设置数据集"""
        
        # 创建数据集
        self.train_dataset = DownstreamDataset(
            self.task_name,
            self.data_root,
            split='train',
            use_features=self.use_features
        )
        
        self.val_dataset = DownstreamDataset(
            self.task_name,
            self.data_root,
            split='val',
            use_features=self.use_features
        )
        
        self.test_dataset = DownstreamDataset(
            self.task_name,
            self.data_root,
            split='test',
            use_features=self.use_features
        )
    
    def train_dataloader(self) -> DataLoader:
        """创建训练数据加载器"""
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """创建验证数据加载器"""
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """创建测试数据加载器"""
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.test_dataset.collate_fn,
            pin_memory=True
        )
    
    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取数据划分索引"""
        
        # 加载或创建划分
        split_file = os.path.join(
            self.data_root,
            self.task_name,
            'splits.json'
        )
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                splits = json.load(f)
            
            return (np.array(splits['train_idx']),
                   np.array(splits['val_idx']),
                   np.array(splits['test_idx']))
        
        # 如果没有，创建新的划分
        total_size = len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset)
        indices = np.arange(total_size)
        
        train_size = len(self.train_dataset)
        val_size = len(self.val_dataset)
        
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size+val_size]
        test_idx = indices[train_size+val_size:]
        
        return train_idx, val_idx, test_idx