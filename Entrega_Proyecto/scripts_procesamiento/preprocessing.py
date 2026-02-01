"""
Scripts de preprocesamiento de datos para el proyecto.

Incluye:
- Filtrado de imágenes válidas
- Creación de variable target
- Split estratificado
- Normalización de features
- Codificación de categorías

"""

import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from utils import verificar_imagen_valida


def crear_target_engagement(df, likes_col='Likes', bookmarks_col='Bookmarks', 
                            visits_col='Visits', umbral=0.5):
    """
    Crea la variable target basada en el ratio de engagement.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        likes_col (str): Nombre de columna de likes
        bookmarks_col (str): Nombre de columna de bookmarks
        visits_col (str): Nombre de columna de visitas
        umbral (float): Umbral para clasificar como alto engagement
        
    Returns:
        pd.DataFrame: DataFrame con columna 'target' y 'engagement_ratio'
    """
    # Calcular ratio de engagement
    df['engagement_ratio'] = (df[likes_col] + df[bookmarks_col]) / (df[visits_col] + 1)
    
    # Crear target binario
    median_engagement = df['engagement_ratio'].median()
    df['target'] = (df['engagement_ratio'] > median_engagement).astype(int)
    
    print(f"✓ Target creado:")
    print(f"  - Engagement ratio mediano: {median_engagement:.6f}")
    print(f"  - Bajo engagement (0): {(df['target'] == 0).sum()} POIs")
    print(f"  - Alto engagement (1): {(df['target'] == 1).sum()} POIs")
    
    return df


def filtrar_imagenes_validas(df, img_path_col='main_image_path'):
    """
    Filtra el DataFrame manteniendo solo filas con imágenes válidas.
    
    Args:
        df (pd.DataFrame): DataFrame original
        img_path_col (str): Nombre de columna con rutas de imágenes
        
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    print(f"Verificando imágenes válidas...")
    df_valido = df[df[img_path_col].apply(verificar_imagen_valida)].copy()
    
    print(f"✓ Imágenes filtradas:")
    print(f"  - Total original: {len(df)}")
    print(f"  - Válidas: {len(df_valido)}")
    print(f"  - Descartadas: {len(df) - len(df_valido)}")
    
    return df_valido


def split_train_test(df, test_size=0.2, random_state=42, stratify_col='target'):
    """
    Divide el dataset en train y test de forma estratificada.
    
    Args:
        df (pd.DataFrame): DataFrame completo
        test_size (float): Proporción para test
        random_state (int): Semilla aleatoria
        stratify_col (str): Columna para estratificación
        
    Returns:
        tuple: (train_df, test_df)
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df[stratify_col]
    )
    
    print(f"✓ Split completado:")
    print(f"  - Train: {len(train_df)} muestras ({(1-test_size)*100:.0f}%)")
    print(f"  - Test:  {len(test_df)} muestras ({test_size*100:.0f}%)")
    
    return train_df, test_df


def normalizar_features(train_df, test_df, feature_cols):
    """
    Normaliza las features numéricas usando StandardScaler.
    FIT solo en train, TRANSFORM en ambos (previene data leakage).
    
    Args:
        train_df (pd.DataFrame): DataFrame de entrenamiento
        test_df (pd.DataFrame): DataFrame de test
        feature_cols (list): Lista de columnas a normalizar
        
    Returns:
        tuple: (train_df, test_df, scaler)
    """
    scaler = StandardScaler()
    
    # Fit SOLO en train (evita data leakage)
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    # Transform en test
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    print(f"✓ Features normalizadas: {len(feature_cols)} columnas")
    print(f"  - Media train: {train_df[feature_cols].mean().mean():.6f}")
    print(f"  - Std train:   {train_df[feature_cols].std().mean():.6f}")
    
    return train_df, test_df, scaler


def codificar_categorias(train_df, test_df, cat_col='categories'):
    """
    Codifica categorías usando MultiLabelBinarizer.
    FIT solo en train, TRANSFORM en ambos (previene data leakage).
    
    Args:
        train_df (pd.DataFrame): DataFrame de entrenamiento
        test_df (pd.DataFrame): DataFrame de test
        cat_col (str): Nombre de columna con categorías
        
    Returns:
        tuple: (train_df, test_df, mlb)
    """
    # Convertir strings a listas
    train_df[cat_col] = train_df[cat_col].apply(ast.literal_eval)
    test_df[cat_col] = test_df[cat_col].apply(ast.literal_eval)
    
    # Codificar
    mlb = MultiLabelBinarizer()
    cat_encoded_train = mlb.fit_transform(train_df[cat_col])
    cat_encoded_test = mlb.transform(test_df[cat_col])
    
    # Crear DataFrames con categorías
    cat_cols = [f'cat_{i}' for i in range(cat_encoded_train.shape[1])]
    df_cats_train = pd.DataFrame(cat_encoded_train, columns=cat_cols, index=train_df.index)
    df_cats_test = pd.DataFrame(cat_encoded_test, columns=cat_cols, index=test_df.index)
    
    # Concatenar
    train_df = pd.concat([train_df.reset_index(drop=True), 
                         df_cats_train.reset_index(drop=True)], axis=1)
    test_df = pd.concat([test_df.reset_index(drop=True), 
                        df_cats_test.reset_index(drop=True)], axis=1)
    
    print(f"✓ Categorías codificadas: {len(cat_cols)} features one-hot")
    print(f"  - Categorías únicas: {len(mlb.classes_)}")
    
    return train_df, test_df, mlb


def preparar_datos_completo(csv_path, random_state=42):
    """
    Pipeline completo de preprocesamiento de datos.
    
    Args:
        csv_path (str): Ruta al archivo CSV
        random_state (int): Semilla aleatoria
        
    Returns:
        dict: Diccionario con train_df, test_df y objetos de transformación
    """
    print("="*60)
    print("INICIANDO PREPROCESAMIENTO DE DATOS")
    print("="*60)
    
    # 1. Cargar datos
    print("\n1. Cargando datos...")
    df = pd.read_csv(csv_path)
    print(f"   ✓ Cargadas {len(df)} filas")
    
    # 2. Crear target
    print("\n2. Creando variable target...")
    df = crear_target_engagement(df)
    
    # 3. Filtrar imágenes válidas
    print("\n3. Filtrando imágenes válidas...")
    df = filtrar_imagenes_validas(df)
    
    # 4. Split train/test (ANTES de transformaciones)
    print("\n4. Dividiendo en train/test...")
    train_df, test_df = split_train_test(df, random_state=random_state)
    
    # 5. Normalizar features numéricas
    print("\n5. Normalizando features numéricas...")
    numeric_cols = ['locationLat', 'locationLon', 'xps', 'Visits']
    train_df, test_df, scaler = normalizar_features(train_df, test_df, numeric_cols)
    
    # 6. Codificar categorías
    print("\n6. Codificando categorías...")
    train_df, test_df, mlb = codificar_categorias(train_df, test_df)
    
    print("\n" + "="*60)
    print("✓ PREPROCESAMIENTO COMPLETADO")
    print("="*60)
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'scaler': scaler,
        'mlb': mlb
    }
