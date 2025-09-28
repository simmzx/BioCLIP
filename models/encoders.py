import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from transformers import AutoModel, AutoTokenizer
import timm

class MolecularEncoder(nn.Module):
    """分子编码器 - 支持多种架构"""
    
    def __init__(self, model_type: str = 'gin_pretrained', hidden_dim: int = 256):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        if model_type == 'gin_pretrained':
            self.encoder = self._build_gin()
        elif model_type == 'molformer':
            self.encoder = self._build_molformer()
        elif model_type == 'chemberta':
            self.encoder = self._build_chemberta()
        else:
            self.encoder = self._build_mlp()
    
    def _build_gin(self):
        """Graph Isomorphism Network"""
        return nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _build_molformer(self):
        """MolFormer - 需要额外安装"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            class MolFormerWrapper(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        "ibm/MoLFormer-XL-both-10pct",
                        trust_remote_code=True
                    )
                    self.model = AutoModel.from_pretrained(
                        "ibm/MoLFormer-XL-both-10pct",
                        trust_remote_code=True
                    )
                    # 冻结预训练权重
                    for param in self.model.parameters():
                        param.requires_grad = False
                    
                    self.projection = nn.Sequential(
                        nn.Linear(768, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                
                def forward(self, x):
                    # 如果输入是SMILES列表
                    if isinstance(x, list):
                        inputs = self.tokenizer(x, return_tensors="pt", 
                                               padding=True, truncation=True)
                        with torch.no_grad():
                            outputs = self.model(**inputs)
                        x = outputs.pooler_output
                    # 如果已经是特征向量，直接投影
                    elif x.shape[-1] == 768:
                        pass
                    else:
                        # 使用MLP处理特征向量
                        return self.projection(x)
                    
                    return self.projection(x)
            
            return MolFormerWrapper(self.hidden_dim)
            
        except ImportError:
            print("MolFormer not available, using MLP instead")
            return self._build_mlp()
    
    def _build_chemberta(self):
        """ChemBERTa编码器"""
        try:
            from transformers import AutoModel, AutoTokenizer
            
            class ChemBertaWrapper(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    self.tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                    self.model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
                    
                    # 微调最后几层
                    for param in self.model.embeddings.parameters():
                        param.requires_grad = False
                    
                    self.projection = nn.Sequential(
                        nn.Linear(768, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                
                def forward(self, x):
                    if isinstance(x, list):
                        inputs = self.tokenizer(x, return_tensors="pt",
                                               padding=True, truncation=True)
                        outputs = self.model(**inputs)
                        x = outputs.last_hidden_state.mean(dim=1)
                    elif x.shape[-1] != 768:
                        # 先投影到768维
                        x = F.linear(x, torch.randn(768, x.shape[-1]).to(x.device))
                    
                    return self.projection(x)
            
            return ChemBertaWrapper(self.hidden_dim)
            
        except ImportError:
            print("ChemBERTa not available, using MLP instead")
            return self._build_mlp()
    
    def _build_mlp(self):
        """简单MLP作为后备"""
        return nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class GeneEncoder(nn.Module):
    """基因表达编码器"""
    
    def __init__(self, model_type: str = 'transformer', 
                 input_dim: int = 978, hidden_dim: int = 256):
        super().__init__()
        self.model_type = model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        if model_type == 'transformer':
            self.encoder = self._build_transformer()
        elif model_type == 'performer':
            self.encoder = self._build_performer()
        elif model_type == 'tab_transformer':
            self.encoder = self._build_tab_transformer()
        else:
            self.encoder = self._build_mlp()
    
    def _build_transformer(self):
        """标准Transformer"""
        class GeneTransformer(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                self.embedding = nn.Linear(input_dim, hidden_dim)
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=8,
                    dim_feedforward=hidden_dim * 4,
                    dropout=0.1,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
                self.norm = nn.LayerNorm(hidden_dim)
            
            def forward(self, x):
                # x: [B, 978]
                x = self.embedding(x).unsqueeze(1)  # [B, 1, hidden]
                x = self.transformer(x)
                x = x.squeeze(1)
                return self.norm(x)
        
        return GeneTransformer(self.input_dim, self.hidden_dim)
    
    def _build_performer(self):
        """Performer - 线性复杂度Transformer"""
        # 简化版Performer
        return self._build_transformer()  # 如果没有安装performer-pytorch
    
    def _build_tab_transformer(self):
        """TabTransformer for tabular data"""
        class TabTransformer(nn.Module):
            def __init__(self, input_dim, hidden_dim):
                super().__init__()
                # 将基因分组
                self.n_groups = 32
                self.group_size = input_dim // self.n_groups + 1
                
                self.group_embeddings = nn.ModuleList([
                    nn.Linear(self.group_size, 64) for _ in range(self.n_groups)
                ])
                
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        d_model=64,
                        nhead=8,
                        dim_feedforward=256,
                        batch_first=True
                    ),
                    num_layers=4
                )
                
                self.output = nn.Sequential(
                    nn.Linear(64 * self.n_groups, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
            
            def forward(self, x):
                batch_size = x.shape[0]
                groups = []
                
                for i in range(self.n_groups):
                    start = i * self.group_size
                    end = min(start + self.group_size, x.shape[1])
                    
                    if start < x.shape[1]:
                        group_data = x[:, start:end]
                        
                        # Padding if needed
                        if group_data.shape[1] < self.group_size:
                            padding = torch.zeros(
                                batch_size,
                                self.group_size - group_data.shape[1]
                            ).to(x.device)
                            group_data = torch.cat([group_data, padding], dim=1)
                        
                        group_emb = self.group_embeddings[i](group_data)
                        groups.append(group_emb)
                
                x = torch.stack(groups, dim=1)
                x = self.transformer(x)
                x = x.view(batch_size, -1)
                return self.output(x)
        
        return TabTransformer(self.input_dim, self.hidden_dim)
    
    def _build_mlp(self):
        """MLP baseline"""
        return nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class MorphologyEncoder(nn.Module):
    """细胞形态编码器"""
    
    def __init__(self, model_type: str = 'vit', hidden_dim: int = 256):
        super().__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        
        if model_type == 'vit':
            self.encoder = self._build_vit()
        elif model_type == 'efficientnet':
            self.encoder = self._build_efficientnet()
        elif model_type == 'convnext':
            self.encoder = self._build_convnext()
        else:
            self.encoder = self._build_resnet()
    
    def _build_vit(self):
        """Vision Transformer"""
        try:
            import timm
            
            class ViTEncoder(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    self.vit = timm.create_model(
                        'vit_small_patch16_224',
                        pretrained=True,
                        in_chans=6,  # Cell Painting 6 channels
                        num_classes=0
                    )
                    
                    self.projection = nn.Sequential(
                        nn.Linear(self.vit.embed_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                
                def forward(self, x):
                    features = self.vit(x)
                    return self.projection(features)
            
            return ViTEncoder(self.hidden_dim)
            
        except ImportError:
            print("timm not available, using ResNet instead")
            return self._build_resnet()
    
    def _build_efficientnet(self):
        """EfficientNet"""
        try:
            import timm
            
            class EfficientNetEncoder(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    self.efficientnet = timm.create_model(
                        'efficientnet_b3',
                        pretrained=True,
                        in_chans=6,
                        num_classes=0
                    )
                    
                    feature_dim = self.efficientnet.num_features
                    self.projection = nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                
                def forward(self, x):
                    features = self.efficientnet(x)
                    return self.projection(features)
            
            return EfficientNetEncoder(self.hidden_dim)
            
        except ImportError:
            return self._build_resnet()
    
    def _build_convnext(self):
        """ConvNeXt"""
        try:
            import timm
            
            class ConvNeXtEncoder(nn.Module):
                def __init__(self, hidden_dim):
                    super().__init__()
                    self.convnext = timm.create_model(
                        'convnext_tiny',
                        pretrained=True,
                        in_chans=6,
                        num_classes=0
                    )
                    
                    # 动态获取特征维度
                    with torch.no_grad():
                        dummy = torch.randn(1, 6, 224, 224)
                        feat_dim = self.convnext(dummy).shape[1]
                    
                    self.projection = nn.Sequential(
                        nn.Linear(feat_dim, hidden_dim),
                        nn.LayerNorm(hidden_dim)
                    )
                
                def forward(self, x):
                    features = self.convnext(x)
                    return self.projection(features)
            
            return ConvNeXtEncoder(self.hidden_dim)
            
        except ImportError:
            return self._build_resnet()
    
    def _build_resnet(self):
        """ResNet - 基础CNN"""
        import torchvision.models as models
        
        class ResNetEncoder(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                resnet = models.resnet50(pretrained=True)
                
                # 修改第一层支持6通道
                resnet.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
                
                # 移除分类层
                self.features = nn.Sequential(*list(resnet.children())[:-1])
                
                self.projection = nn.Sequential(
                    nn.Linear(2048, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
            
            def forward(self, x):
                features = self.features(x).flatten(1)
                return self.projection(features)
        
        return ResNetEncoder(self.hidden_dim)
    
    def forward(self, x):
        return self.encoder(x)