"""
Dataset personalizado para cargar datos de POIs turísticos (imágenes + datos tabulares).

"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


class POIDataset(Dataset):
    """
    Dataset personalizado para POIs turísticos que combina:
    - Imágenes (RGB 224x224)
    - Datos tabulares (features numéricas y categóricas)
    - Target (engagement bajo/alto)
    """
    
    def __init__(self, dataframe, transform=None):
        """
        Inicializa el dataset.
        
        Args:
            dataframe (pd.DataFrame): DataFrame con columnas 'main_image_path', 
                                     features tabulares y 'target'
            transform (torchvision.transforms): Transformaciones a aplicar a las imágenes
        """
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        """Retorna el tamaño del dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Obtiene un elemento del dataset.
        
        Args:
            idx (int): Índice del elemento
            
        Returns:
            tuple: (imagen_tensor, features_tabulares, target)
        """
        row = self.df.iloc[idx]
        
        # Cargar imagen
        img_path = row['main_image_path']
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            # Imagen de respaldo en caso de error
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        # Features tabulares (todas las columnas excepto metadatos)
        tabular_cols = [c for c in self.df.columns 
                       if c not in ['poi_id', 'name', 'main_image_path', 'target', 
                                   'categories', 'engagement_ratio']]
        tabular_data = torch.tensor(row[tabular_cols].values, dtype=torch.float32)
        
        # Target
        target = torch.tensor(row['target'], dtype=torch.long)
        
        return image, tabular_data, target
