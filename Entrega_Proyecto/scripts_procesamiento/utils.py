"""
Utilidades y funciones auxiliares para el proyecto de predicción de engagement
de POIs turísticos usando Deep Learning Multimodal.

"""

import os
from PIL import Image
import torch


def verificar_imagen_valida(path):
    """
    Verifica si una imagen es válida y puede ser cargada correctamente.
    
    Args:
        path (str): Ruta al archivo de imagen
        
    Returns:
        bool: True si la imagen es válida, False en caso contrario
    """
    try:
        if not os.path.exists(path):
            return False
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


def get_device():
    """
    Detecta y retorna el dispositivo disponible (CUDA, MPS o CPU).
    
    Returns:
        torch.device: Dispositivo a utilizar para los tensores
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def contar_parametros_modelo(model):
    """
    Cuenta el número total de parámetros entrenables y no entrenables en un modelo.
    
    Args:
        model (nn.Module): Modelo de PyTorch
        
    Returns:
        tuple: (parametros_totales, parametros_entrenables)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def formato_tiempo(segundos):
    """
    Formatea segundos a formato legible (HH:MM:SS).
    
    Args:
        segundos (float): Tiempo en segundos
        
    Returns:
        str: Tiempo formateado
    """
    h = int(segundos // 3600)
    m = int((segundos % 3600) // 60)
    s = int(segundos % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
