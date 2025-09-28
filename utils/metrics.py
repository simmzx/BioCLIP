import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    accuracy_score,
    f1_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from typing import Dict, List, Optional, Union

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = 'classification',
    metrics_list: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    计算各种指标
    """
    results = {}
    
    if task_type == 'classification':
        # 二分类指标
        if len(np.unique(y_true)) == 2:
            results['auroc'] = roc_auc_score(y_true, y_pred)
            results['auprc'] = average_precision_score(y_true, y_pred)
            
            # 如果是概率，转换为类别
            y_pred_class = (y_pred > 0.5).astype(int)
            results['accuracy'] = accuracy_score(y_true, y_pred_class)
            results['f1'] = f1_score(y_true, y_pred_class)
        
        # 多分类
        else:
            y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred
            results['accuracy'] = accuracy_score(y_true, y_pred_class)
            results['f1_macro'] = f1_score(y_true, y_pred_class, average='macro')
            results['f1_weighted'] = f1_score(y_true, y_pred_class, average='weighted')
    
    elif task_type == 'regression':
        results['mse'] = mean_squared_error(y_true, y_pred)
        results['rmse'] = np.sqrt(results['mse'])
        results['mae'] = mean_absolute_error(y_true, y_pred)
        results['r2'] = r2_score(y_true, y_pred)
        
        # Pearson相关系数
        results['pearson'] = np.corrcoef(y_true, y_pred)[0, 1]
        
        # Spearman相关系数
        from scipy.stats import spearmanr
        results['spearman'] = spearmanr(y_true, y_pred)[0]
    
    return results

def compute_embedding_metrics(embeddings: torch.Tensor) -> Dict[str, float]:
    """
    计算嵌入质量指标
    """
    metrics = {}
    
    # 基本统计
    metrics['mean_norm'] = embeddings.norm(dim=1).mean().item()
    metrics['std_norm'] = embeddings.norm(dim=1).std().item()
    
    # 检查坍塌
    singular_values = torch.svd(embeddings).S
    metrics['effective_rank'] = (singular_values > 0.01).sum().item()
    
    # 计算嵌入的均匀性
    distances = torch.cdist(embeddings, embeddings)
    metrics['mean_distance'] = distances[distances > 0].mean().item()
    metrics['std_distance'] = distances[distances > 0].std().item()
    
    # 计算alignment和uniformity (Wang & Isola, 2020)
    # Alignment: 正样本对应该接近
    # Uniformity: 特征应该均匀分布在超球面上
    
    return metrics