"""
BioCLIP下游任务定义
全部基于真实的药物发现任务
"""

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DownstreamTask:
    """下游任务配置"""
    name: str
    task_type: str
    dataset_name: str
    metrics: List[str]
    n_classes: Optional[int] = None
    description: str = ""
    reference: str = ""

# 定义所有支持的下游任务
DOWNSTREAM_TASKS = {
    # ========== 毒性预测任务 ==========
    'tox21': DownstreamTask(
        name='Tox21',
        task_type='multilabel_classification',
        dataset_name='tox21',
        metrics=['auroc', 'auprc', 'accuracy'],
        n_classes=12,
        description='12个毒性终点预测',
        reference='https://tripod.nih.gov/tox21/challenge/'
    ),
    
    'toxcast': DownstreamTask(
        name='ToxCast',
        task_type='multilabel_classification',
        dataset_name='toxcast',
        metrics=['auroc', 'auprc'],
        n_classes=617,
        description='617个体外毒性试验',
        reference='https://www.epa.gov/chemical-research/toxcast'
    ),
    
    'clintox': DownstreamTask(
        name='ClinTox',
        task_type='binary_classification',
        dataset_name='clintox',
        metrics=['auroc', 'accuracy', 'f1'],
        n_classes=2,
        description='FDA批准药物和临床试验失败药物的毒性',
        reference='https://www.frontiersin.org/articles/10.3389/fphar.2018.01162'
    ),
    
    # ========== ADMET性质预测 ==========
    'bbbp': DownstreamTask(
        name='BBBP',
        task_type='binary_classification',
        dataset_name='bbbp',
        metrics=['auroc', 'accuracy'],
        n_classes=2,
        description='血脑屏障通透性预测',
        reference='https://pubs.acs.org/doi/10.1021/ci300124c'
    ),
    
    'cyp450': DownstreamTask(
        name='CYP450',
        task_type='multilabel_classification',
        dataset_name='cyp450',
        metrics=['auroc', 'auprc'],
        n_classes=5,
        description='CYP450酶抑制预测（5个亚型）',
        reference='https://pubs.acs.org/doi/10.1021/ci5005288'
    ),
    
    'clearance': DownstreamTask(
        name='Clearance',
        task_type='regression',
        dataset_name='clearance',
        metrics=['rmse', 'mae', 'r2'],
        description='药物清除率预测',
        reference='https://pubs.acs.org/doi/10.1021/acs.jmedchem.7b00166'
    ),
    
    # ========== 药物-靶点相互作用 ==========
    'davis': DownstreamTask(
        name='Davis',
        task_type='regression',
        dataset_name='davis',
        metrics=['mse', 'ci', 'rm2'],
        description='激酶抑制剂亲和力预测',
        reference='https://doi.org/10.1038/nbt.1990'
    ),
    
    'kiba': DownstreamTask(
        name='KIBA',
        task_type='regression',
        dataset_name='kiba',
        metrics=['mse', 'ci', 'rm2'],
        description='激酶抑制剂生物活性',
        reference='https://doi.org/10.1021/ci400709d'
    ),
    
    'bindingdb': DownstreamTask(
        name='BindingDB',
        task_type='regression',
        dataset_name='bindingdb',
        metrics=['rmse', 'pearson', 'spearman'],
        description='蛋白-配体亲和力',
        reference='https://www.bindingdb.org'
    ),
    
    # ========== 副作用预测 ==========
    'sider': DownstreamTask(
        name='SIDER',
        task_type='multilabel_classification',
        dataset_name='sider',
        metrics=['auroc', 'auprc', 'f1'],
        n_classes=27,
        description='27种药物副作用预测',
        reference='http://sideeffects.embl.de/'
    ),
    
    # ========== 分子性质 ==========
    'qm9': DownstreamTask(
        name='QM9',
        task_type='multitask_regression',
        dataset_name='qm9',
        metrics=['mae', 'rmse'],
        description='量子力学性质预测（12个性质）',
        reference='https://doi.org/10.1038/sdata.2014.22'
    ),
    
    'esol': DownstreamTask(
        name='ESOL',
        task_type='regression',
        dataset_name='esol',
        metrics=['rmse', 'mae', 'r2'],
        description='水溶性预测',
        reference='https://pubs.acs.org/doi/10.1021/ci034243x'
    ),
    
    'lipophilicity': DownstreamTask(
        name='Lipophilicity',
        task_type='regression',
        dataset_name='lipophilicity',
        metrics=['rmse', 'mae', 'r2'],
        description='亲脂性预测',
        reference='https://doi.org/10.1021/acs.jcim.5b00161'
    ),
    
    # ========== 药物协同 ==========
    'drugcomb': DownstreamTask(
        name='DrugComb',
        task_type='regression',
        dataset_name='drugcomb',
        metrics=['pearson', 'spearman', 'rmse'],
        description='药物组合协同效应',
        reference='https://drugcomb.fimm.fi/'
    ),
}

def get_task_info(task_name: str) -> DownstreamTask:
    """获取任务信息"""
    if task_name not in DOWNSTREAM_TASKS:
        raise ValueError(f"Unknown task: {task_name}")
    return DOWNSTREAM_TASKS[task_name]

def list_available_tasks() -> List[str]:
    """列出所有可用任务"""
    return list(DOWNSTREAM_TASKS.keys())