# run_complete_pipeline.sh
#!/bin/bash

# BioCLIP完整训练流程

echo "===================="
echo "BioCLIP Pipeline"
echo "===================="

# 1. 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

# 2. 预处理数据
echo "Preprocessing JUMP-CP data..."
python main.py --mode preprocess

# 3. 预训练BioCLIP
echo "Starting BioCLIP pretraining..."
python main.py --mode train --config config/config.yaml

# 4. 在下游任务上微调（示例：Tox21）
echo "Finetuning on downstream tasks..."
python main.py --mode finetune \
    --task tox21 \
    --checkpoint checkpoints/best_model.pth

# 5. 评估
echo "Evaluating model..."
python main.py --mode eval \
    --checkpoint checkpoints/best_model.pth

echo "Pipeline completed!"