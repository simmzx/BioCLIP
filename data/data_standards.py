"""
BioCLIP数据标准定义
基于JUMP-CP联盟的标准协议
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

@dataclass
class DataStandards:
    """
    数据标准定义
    参考文献:
    - JUMP-CP: https://jump-cellpainting.broadinstitute.org/
    - L1000: https://clue.io/
    - Cell Painting: https://www.nature.com/articles/nprot.2016.105
    """
    
    # ========== 化合物标准 ==========
    COMPOUND_STANDARDS = {
        'identifier': 'inchikey',  # 使用InChIKey作为唯一标识
        'fingerprint_type': 'morgan',  # Morgan/ECFP指纹
        'fingerprint_bits': 2048,
        'fingerprint_radius': 2,
        'properties': [
            'mol_weight',
            'logp',
            'n_rings',
            'n_heteroatoms',
            'tpsa',  # 拓扑极性表面积
            'qed'    # 药物相似性评分
        ]
    }
    
    # ========== L1000基因表达标准 ==========
    L1000_STANDARDS = {
        'n_landmark_genes': 978,
        'n_best_inferred_genes': 11350,
        'cell_lines': {
            'core': ['A549', 'MCF7', 'PC3', 'VCAP', 'A375', 
                    'HA1E', 'HCC515', 'HEPG2', 'HT29'],
            'extended': ['BT20', 'BT474', 'BT549', 'COLO205', 'DU145',
                        'EFO21', 'H1299', 'H1437', 'H1975', 'HELA']
        },
        'doses': {
            'units': 'um',
            'standard': [0.04, 0.12, 0.37, 1.11, 3.33, 10.0]
        },
        'timepoints': {
            'units': 'hours',
            'standard': [6, 24]
        },
        'normalization': 'z-score',
        'quality_metrics': {
            'iqr': 0.5,  # 四分位距阈值
            'is_gold': True,  # 是否为金标准数据
            'distil_cc_q75': 0.15  # 相关性阈值
        }
    }
    
    # ========== Cell Painting标准（JUMP-CP协议）==========
    CELL_PAINTING_STANDARDS = {
        'channels': [
            {'name': 'Hoechst 33342', 'target': 'Nuclei', 'wavelength': 405},
            {'name': 'Concanavalin A', 'target': 'ER', 'wavelength': 488},
            {'name': 'SYTO 14', 'target': 'RNA', 'wavelength': 488},
            {'name': 'Phalloidin', 'target': 'Actin', 'wavelength': 568},
            {'name': 'WGA', 'target': 'Golgi/Membrane', 'wavelength': 568},
            {'name': 'MitoTracker', 'target': 'Mitochondria', 'wavelength': 647}
        ],
        'image_specs': {
            'bit_depth': 16,
            'pixel_size_um': 0.656,
            'objective': '20x',
            'binning': '2x2'
        },
        'preprocessing': {
            'illumination_correction': True,
            'background_subtraction': True,
            'normalization': 'percentile',
            'percentiles': [1, 99]
        },
        'quality_control': {
            'min_cells': 50,
            'max_cells': 2000,
            'min_confluence': 0.1,
            'max_confluence': 0.9,
            'focus_score_threshold': 0.7
        }
    }
    
    # ========== JUMP-CP板子布局 ==========
    PLATE_LAYOUT = {
        'format': '384-well',
        'rows': 16,
        'cols': 24,
        'control_wells': {
            'negative': ['A01', 'A02', 'P23', 'P24'],  # DMSO
            'positive': ['A23', 'A24', 'P01', 'P02']   # 阳性对照
        },
        'treatment_wells': 'B01-O22',
        'edge_exclusion': True  # 排除边缘孔
    }
    
    # ========== 数据质量标准 ==========
    QUALITY_STANDARDS = {
        'compound': {
            'min_purity': 0.9,
            'max_molecular_weight': 900,
            'min_molecular_weight': 150
        },
        'gene_expression': {
            'min_replicate_correlation': 0.7,
            'max_missing_genes': 50,
            'min_dynamic_range': 2.0
        },
        'cell_painting': {
            'min_snr': 3.0,  # 信噪比
            'max_saturation': 0.01,  # 饱和像素比例
            'min_focus_score': 0.7
        }
    }
    
    # ========== 数据集成标准 ==========
    INTEGRATION_STANDARDS = {
        'compound_id_format': 'BRD-[A-Z][0-9]{8}',  # Broad ID格式
        'batch_size': {
            'complete': 32,
            'partial': 64,
            'single': 128
        },
        'train_val_test_split': [0.8, 0.1, 0.1],
        'stratification': 'scaffold'  # 基于分子骨架的分层
    }
    
    @staticmethod
    def get_landmark_genes() -> List[str]:
        """获取L1000的978个Landmark基因列表"""
        # 这里只列出部分示例，实际应从L1000官方列表加载
        landmark_genes = [
            'AARS', 'ABCB1', 'ABCC1', 'ABHD2', 'ACLY', 'ACSL1', 'ACSS2', 'ACTB',
            'ACTN1', 'ACTR3', 'ADAM9', 'ADH1B', 'ADM', 'ADRA2A', 'ADRB2', 'AFF1',
            # ... 省略其他基因
        ]
        return landmark_genes[:978]  # 返回978个基因
    
    @staticmethod
    def get_cell_painting_features() -> List[str]:
        """获取Cell Painting提取的特征列表"""
        features = []
        
        # 每个通道的特征
        channels = ['Nuclei', 'ER', 'RNA', 'Actin', 'Golgi', 'Mito']
        
        # 形态学特征类型
        feature_types = [
            'Intensity_Mean', 'Intensity_StdDev', 'Intensity_Median',
            'Texture_Contrast', 'Texture_Correlation', 'Texture_Entropy',
            'Shape_Area', 'Shape_Perimeter', 'Shape_Eccentricity',
            'Shape_Solidity', 'Shape_Compactness'
        ]
        
        for channel in channels:
            for feat in feature_types:
                features.append(f'{channel}_{feat}')
        
        return features
    
    @staticmethod
    def validate_compound(smiles: str) -> bool:
        """验证化合物是否符合标准"""
        from rdkit import Chem
        from rdkit.Chem import Descriptors
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        
        # 检查分子量
        mw = Descriptors.MolWt(mol)
        if mw < 150 or mw > 900:
            return False
        
        # 检查Lipinski规则
        if Descriptors.MolLogP(mol) > 5:
            return False
        
        if Descriptors.NumHDonors(mol) > 5:
            return False
        
        if Descriptors.NumHAcceptors(mol) > 10:
            return False
        
        return True
    
    @staticmethod
    def normalize_gene_expression(expr: np.ndarray, method: str = 'z-score') -> np.ndarray:
        """标准化基因表达数据"""
        if method == 'z-score':
            mean = expr.mean(axis=0)
            std = expr.std(axis=0)
            return (expr - mean) / (std + 1e-8)
        
        elif method == 'robust_z-score':
            median = np.median(expr, axis=0)
            mad = np.median(np.abs(expr - median), axis=0)
            return (expr - median) / (mad * 1.4826 + 1e-8)
        
        elif method == 'quantile':
            from scipy import stats
            return stats.rankdata(expr, axis=0) / expr.shape[0]
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")