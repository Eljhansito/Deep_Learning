"""
Arquitectura del modelo multimodal para predicción de engagement de POIs.

Combina:
- Rama visual: ResNet18 preentrenado con fine-tuning
- Rama tabular: MLP con regularización
- Clasificador: Fusión de ambas ramas

"""

import torch
import torch.nn as nn
from torchvision import models


class MejoradoModel(nn.Module):
    """
    Modelo multimodal que combina información visual y tabular
    para predecir el engagement de POIs turísticos.
    """
    
    def __init__(self, n_tab_feat):
        """
        Inicializa el modelo.
        
        Args:
            n_tab_feat (int): Número de características tabulares
        """
        super(MejoradoModel, self).__init__()
        
        # ========== RAMA IMAGEN (ResNet18 Pre-entrenado) ==========
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Fine-tuning: Congelar todas las capas excepto layer4
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        for param in self.cnn.layer4.parameters():
            param.requires_grad = True
        
        # Extraer features antes de la capa FC
        n_img_feat = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()  # Remover clasificador original
        
        # ========== RAMA TABULAR ==========
        self.tab_net = nn.Sequential(
            nn.Linear(n_tab_feat, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # ========== CLASIFICADOR (Fusión de ambas ramas) ==========
        self.classifier = nn.Sequential(
            nn.Linear(n_img_feat + 64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 2 clases: Bajo/Alto engagement
        )
    
    def forward(self, img, tab):
        """
        Forward pass del modelo.
        
        Args:
            img (torch.Tensor): Batch de imágenes [B, 3, 224, 224]
            tab (torch.Tensor): Batch de features tabulares [B, n_tab_feat]
            
        Returns:
            torch.Tensor: Logits de salida [B, 2]
        """
        # Extraer features de imagen
        feat_img = self.cnn(img)
        
        # Extraer features tabulares
        feat_tab = self.tab_net(tab)
        
        # Fusión por concatenación
        combined = torch.cat((feat_img, feat_tab), dim=1)
        
        # Clasificación
        output = self.classifier(combined)
        
        return output
