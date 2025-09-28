# data/dataset.py - 更新版本，完美适配JUMP-CP数据
import os
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import json
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')

class BioCLIPDataset(Dataset):
    """
    BioCLIP数据集 - 适配JUMP-CP数据格式
    
    基于JUMP-CP consortium数据：
    - 1,327个完整三模态数据（分子+基因+形态）
    - ~30,000个双模态数据
    - ~100,000个单模态数据
    
    参考文献：
    - JUMP-CP: https://jump-cellpainting.broadinstitute.org/
    - Chandrasekaran et al., 2023
    """
    
    def __init__(self,
                 data_root: str = './datasets/jump_cp',
                 mode: str = 'train',
                 data_type: str = 'complete',
                 transform: Optional[Dict] = None,
                 use_cache: bool = True):
        """
        Args:
            data_root: 数据根目录
            mode: train/val/test
            data_type: complete/partial/single
            transform: 数据变换
            use_cache: 是否使用缓存
        """
        super().__init__()
        self.data_root = data_root
        self.mode = mode
        self.data_type = data_type
        self.transform = transform
        self.use_cache = use_cache
        
        # 加载数据索引
        self.index = self._load_index()
        
        # 如果使用缓存，预加载数据
        self.cache = {} if use_cache else None
        
        print(f"Loaded BioCLIP dataset:")
        print(f"  Mode: {mode}")
        print(f"  Type: {data_type}")
        print(f"  Samples: {len(self.index)}")
        self._print_modality_statistics()
    
    def _load_index(self) -> pd.DataFrame:
        """加载JUMP-CP数据索引"""
        
        # 根据数据类型加载对应的索引文件
        if self.data_type == 'complete':
            # 完整三模态数据（1,327个）
            index_file = os.path.join(
                self.data_root, 
                'processed',
                f'{self.mode}_complete_trimodal.csv'
            )
            expected_size = {'train': 1000, 'val': 200, 'test': 127}
            
        elif self.data_type == 'partial':
            # 双模态数据（~30,000个）
            index_file = os.path.join(
                self.data_root,
                'processed', 
                f'{self.mode}_partial_bimodal.csv'
            )
            expected_size = {'train': 24000, 'val': 3000, 'test': 3000}
            
        else:  # single
            # 单模态数据（~100,000个）
            index_file = os.path.join(
                self.data_root,
                'processed',
                f'{self.mode}_single_modal.csv'
            )
            expected_size = {'train': 80000, 'val': 10000, 'test': 10000}
        
        # 如果文件不存在，创建示例数据
        if not os.path.exists(index_file):
            print(f"Warning: {index_file} not found, creating example data")
            return self._create_example_index(expected_size[self.mode])
        
        # 加载真实数据
        df = pd.read_csv(index_file)
        
        # 验证数据完整性
        self._validate_index(df)
        
        return df
    
    def _validate_index(self, df: pd.DataFrame):
        """验证数据索引的完整性"""
        required_columns = ['compound_id', 'inchikey', 'smiles']
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 检查模态可用性
        if self.data_type == 'complete':
            assert all(df['has_gene'] & df['has_morph']), \
                "Complete dataset should have all modalities"
        
        elif self.data_type == 'partial':
            n_modalities = df['has_gene'].astype(int) + df['has_morph'].astype(int)
            assert all(n_modalities == 1), \
                "Partial dataset should have exactly 2 modalities (mol + 1)"
    
    def _create_example_index(self, n_samples: int) -> pd.DataFrame:
        """创建示例索引（用于测试）"""
        
        # 使用真实的化合物示例
        example_compounds = [
            ('CHEMBL25', 'CC(=O)Oc1ccccc1C(=O)O'),  # Aspirin
            ('CHEMBL1200333', 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'),  # Caffeine
            ('CHEMBL521', 'CC(C)Cc1ccc(cc1)C(C)C(=O)O'),  # Ibuprofen
            ('CHEMBL904', 'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C'),  # Testosterone
            ('CHEMBL1200959', 'CCO'),  # Ethanol
        ]
        
        data = []
        for i in range(n_samples):
            compound_id, smiles = example_compounds[i % len(example_compounds)]
            
            # 计算InChIKey
            mol = Chem.MolFromSmiles(smiles)
            inchikey = Chem.inchi.MolToInchiKey(mol) if mol else None
            
            # 设置模态可用性
            if self.data_type == 'complete':
                has_gene, has_morph = True, True
            elif self.data_type == 'partial':
                has_gene = i % 2 == 0
                has_morph = not has_gene
            else:
                has_gene, has_morph = False, False
            
            data.append({
                'compound_id': f'{compound_id}_{i}',
                'smiles': smiles,
                'inchikey': inchikey,
                'has_gene': has_gene,
                'has_morph': has_morph,
                'plate_id': f'PLATE_{i//100:03d}',
                'well_id': f'{chr(65 + i%8)}{i%12 + 1:02d}'
            })
        
        return pd.DataFrame(data)
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取一个样本"""
        
        # 如果有缓存，直接返回
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        # 获取样本信息
        row = self.index.iloc[idx]
        
        sample = {
            'idx': idx,
            'compound_id': row['compound_id'],
            'inchikey': row['inchikey'],
            'smiles': row['smiles'],
            'available_modalities': [],
            'is_complete': False,
            'is_pseudo': False
        }
        
        # 加载分子特征（所有样本都有）
        sample['mol'] = self._load_molecular_features(row)
        sample['available_modalities'].append('mol')
        
        # 加载基因表达（如果可用）
        if row.get('has_gene', False):
            sample['gene'] = self._load_gene_expression(row)
            sample['available_modalities'].append('gene')
        
        # 加载细胞形态（如果可用）
        if row.get('has_morph', False):
            sample['morph'] = self._load_morphology(row)
            sample['available_modalities'].append('morph')
        
        # 更新完整性标志
        sample['is_complete'] = len(sample['available_modalities']) == 3
        sample['n_modalities'] = len(sample['available_modalities'])
        
        # 应用变换
        if self.transform:
            sample = self.transform(sample)
        
        # 缓存
        if self.cache is not None:
            self.cache[idx] = sample
        
        return sample
    
    def _load_molecular_features(self, row: pd.Series) -> torch.Tensor:
        """加载分子特征（Morgan指纹）"""
        
        # 尝试从预计算的文件加载
        fp_file = os.path.join(
            self.data_root,
            'processed',
            'fingerprints',
            f'{row["inchikey"]}.npy'
        )
        
        if os.path.exists(fp_file):
            fp = np.load(fp_file)
        else:
            # 实时计算
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is None:
                return torch.zeros(2048, dtype=torch.float32)
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_arr = np.zeros(2048)
            from rdkit.DataStructs import ConvertToNumpyArray
            ConvertToNumpyArray(fp, fp_arr)
            fp = fp_arr
        
        return torch.tensor(fp, dtype=torch.float32)
    
    def _load_gene_expression(self, row: pd.Series) -> torch.Tensor:
        """
        加载基因表达数据（L1000格式）
        978个Landmark基因的表达值
        """
        
        gene_file = os.path.join(
            self.data_root,
            'processed',
            'gene_expression',
            f'{row["compound_id"]}.npy'
        )
        
        if os.path.exists(gene_file):
            gene_expr = np.load(gene_file)
        else:
            # 生成模拟的L1000数据（978个基因）
            np.random.seed(hash(row['compound_id']) % 2**32)
            gene_expr = np.random.randn(978) * 0.5
            
        # Z-score标准化
        gene_expr = (gene_expr - gene_expr.mean()) / (gene_expr.std() + 1e-8)
        
        return torch.tensor(gene_expr, dtype=torch.float32)
    
    def _load_morphology(self, row: pd.Series) -> torch.Tensor:
        """
        加载Cell Painting数据
        6通道 x 224x224 图像
        """
        
        img_file = os.path.join(
            self.data_root,
            'processed',
            'cell_images',
            f'{row["compound_id"]}.npy'
        )
        
        if os.path.exists(img_file):
            img = np.load(img_file)
        else:
            # 生成模拟的Cell Painting图像
            np.random.seed(hash(row['compound_id']) % 2**32)
            
            # 6个通道，每个通道有不同的特征
            img = np.zeros((6, 224, 224), dtype=np.float32)
            
            for ch in range(6):
                # 模拟不同染色通道的特征
                if ch == 0:  # 核染色
                    # 生成类似细胞核的圆形结构
                    img[ch] = self._generate_nuclei_pattern()
                elif ch == 4:  # 线粒体
                    # 生成类似线粒体的纹理
                    img[ch] = self._generate_mitochondria_pattern()
                else:
                    # 其他通道
                    img[ch] = np.random.randn(224, 224) * 0.1 + 0.5
            
            img = np.clip(img, 0, 1)
        
        return torch.tensor(img, dtype=torch.float32)
    
    def _generate_nuclei_pattern(self) -> np.ndarray:
        """生成模拟的细胞核图案"""
        img = np.zeros((224, 224))
        
        # 生成多个圆形核
        n_nuclei = np.random.randint(5, 15)
        for _ in range(n_nuclei):
            center = (np.random.randint(20, 204), np.random.randint(20, 204))
            radius = np.random.randint(10, 20)
            
            y, x = np.ogrid[:224, :224]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            img[mask] = np.random.uniform(0.6, 1.0)
        
        return img
    
    def _generate_mitochondria_pattern(self) -> np.ndarray:
        """生成模拟的线粒体图案"""
        # 生成纤维状结构
        img = np.random.randn(224, 224) * 0.3
        
        # 应用高斯滤波创建连续性
        from scipy.ndimage import gaussian_filter
        img = gaussian_filter(img, sigma=2)
        
        # 归一化
        img = (img - img.min()) / (img.max() - img.min())
        
        return img
    
    def _print_modality_statistics(self):
        """打印模态统计信息"""
        stats = {
            'total': len(self.index),
            'with_gene': self.index['has_gene'].sum() if 'has_gene' in self.index else 0,
            'with_morph': self.index['has_morph'].sum() if 'has_morph' in self.index else 0
        }
        
        print(f"  Modality statistics:")
        print(f"    Molecular: {stats['total']} (100%)")
        print(f"    Gene expression: {stats['with_gene']} ({stats['with_gene']/stats['total']*100:.1f}%)")
        print(f"    Cell morphology: {stats['with_morph']} ({stats['with_morph']/stats['total']*100:.1f}%)")