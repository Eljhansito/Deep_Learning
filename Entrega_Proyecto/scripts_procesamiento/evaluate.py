"""
Funciones de evaluación, análisis de errores e interpretabilidad del modelo.

Incluye:
- Evaluación completa en test set
- Análisis de errores (FP, FN)
- Importancia de características
- Visualización de predicciones

"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            ConfusionMatrixDisplay, accuracy_score, 
                            precision_score, recall_score, f1_score,
                            roc_curve, auc, roc_auc_score)
from PIL import Image


def evaluar_final_robusto(model, dataframe_test, dataset_class, test_transform, device):
    """
    Evaluación completa del modelo en el conjunto de test.
    
    Args:
        model (nn.Module): Modelo entrenado
        dataframe_test (pd.DataFrame): DataFrame de test
        dataset_class: Clase del Dataset personalizado
        test_transform: Transformaciones para test
        device: Dispositivo (cuda/cpu)
        
    Returns:
        tuple: (predicciones, targets, probabilidades)
    """
    model.eval()
    
    test_ds = dataset_class(dataframe_test, test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("Evaluando en el conjunto de test...")
    with torch.no_grad():
        for images, tabs, targets in test_loader:
            images = images.to(device)
            tabs = tabs.to(device)
            targets = targets.to(device)
            
            outputs = model(images, tabs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
    
    # Métricas profesionales
    print("\n" + "="*60)
    print("   RESULTADOS FINALES EN TEST")
    print("   (Sin Data Leakage - Métricas Confiables)")
    print("="*60)
    print(classification_report(all_targets, all_preds, 
                               target_names=['Bajo Engagement', 'Alto Engagement']))
    
    print(f"Accuracy:  {accuracy_score(all_targets, all_preds):.4f}")
    print(f"Precision: {precision_score(all_targets, all_preds):.4f}")
    print(f"Recall:    {recall_score(all_targets, all_preds):.4f}")
    print(f"F1-Score:  {f1_score(all_targets, all_preds):.4f}")
    
    # Matriz de Confusión
    cm = confusion_matrix(all_targets, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Bajo Engagement', 'Alto Engagement'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Matriz de Confusión - Test Set")
    plt.tight_layout()
    plt.show()
    
    return all_preds, all_targets, all_probs


def analizar_errores_modelo(model, dataframe_test, dataset_class, test_transform, device):
    """
    Analiza en detalle los errores de predicción del modelo.
    
    Args:
        model (nn.Module): Modelo entrenado
        dataframe_test (pd.DataFrame): DataFrame de test
        dataset_class: Clase del Dataset
        test_transform: Transformaciones
        device: Dispositivo
    """
    model.eval()
    test_ds = dataset_class(dataframe_test, test_transform)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    all_preds = []
    all_probs = []
    all_targets = []
    
    print("\n✓ Evaluando modelo en test set...")
    with torch.no_grad():
        for images, tabs, targets in test_loader:
            images = images.to(device)
            tabs = tabs.to(device)
            targets = targets.to(device)
            
            outputs = model(images, tabs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])
            all_targets.extend(targets.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Identificar tipos de errores
    fp_mask = (all_preds == 1) & (all_targets == 0)
    fn_mask = (all_preds == 0) & (all_targets == 1)
    tp_mask = (all_preds == 1) & (all_targets == 1)
    tn_mask = (all_preds == 0) & (all_targets == 0)
    
    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()
    n_tp = tp_mask.sum()
    n_tn = tn_mask.sum()
    
    print(f"\n📊 DISTRIBUCIÓN DE PREDICCIONES:")
    print(f"  • True Positives (TP):   {n_tp:4d} - Predijo correctamente ALTO engagement")
    print(f"  • True Negatives (TN):   {n_tn:4d} - Predijo correctamente BAJO engagement")
    print(f"  • False Positives (FP):  {n_fp:4d} - Predijo ALTO pero era BAJO")
    print(f"  • False Negatives (FN):  {n_fn:4d} - Predijo BAJO pero era ALTO")
    
    # Análisis de confianza
    print(f"\n🎯 ANÁLISIS DE CONFIANZA EN PREDICCIONES:")
    if (tp_mask | tn_mask).sum() > 0:
        correct_probs = all_probs[(tp_mask | tn_mask)]
        print(f"  Predicciones correctas:")
        print(f"    - Promedio: {correct_probs.mean():.4f}")
        print(f"    - Mín-Máx:  {correct_probs.min():.4f} - {correct_probs.max():.4f}")
    
    if (fp_mask | fn_mask).sum() > 0:
        incorrect_probs = all_probs[(fp_mask | fn_mask)]
        print(f"  Predicciones incorrectas:")
        print(f"    - Promedio: {incorrect_probs.mean():.4f}")
        print(f"    - Mín-Máx:  {incorrect_probs.min():.4f} - {incorrect_probs.max():.4f}")
    
    # Mostrar ejemplos de errores
    if n_fp > 0:
        print(f"\n❌ FALSOS POSITIVOS (Top 3 con mayor confianza):")
        fp_indices = np.where(fp_mask)[0]
        fp_data = dataframe_test.iloc[fp_indices]
        fp_confidences = all_probs[fp_mask]
        
        sorted_fp = np.argsort(fp_confidences)[::-1][:3]
        for rank, fp_idx in enumerate(sorted_fp, 1):
            poi = fp_data.iloc[fp_idx]
            print(f"  {rank}. {poi['name'][:40]}...")
            print(f"     Confianza: {fp_confidences[fp_idx]:.4f}")
            print(f"     Visitas: {poi['Visits']}, Likes: {poi['Likes']}")
    
    if n_fn > 0:
        print(f"\n❌ FALSOS NEGATIVOS (Top 3 con mayor confianza):")
        fn_indices = np.where(fn_mask)[0]
        fn_data = dataframe_test.iloc[fn_indices]
        fn_confidences = 1 - all_probs[fn_mask]
        
        sorted_fn = np.argsort(fn_confidences)[::-1][:3]
        for rank, fn_idx in enumerate(sorted_fn, 1):
            poi = fn_data.iloc[fn_idx]
            print(f"  {rank}. {poi['name'][:40]}...")
            print(f"     Confianza: {fn_confidences[fn_idx]:.4f}")
            print(f"     Visitas: {poi['Visits']}, Likes: {poi['Likes']}")


def visualizar_predicciones_ejemplos(model, test_df, dataset_class, test_transform, 
                                    device, n_examples=6):
    """
    Visualiza ejemplos de predicciones con sus imágenes.
    
    Args:
        model (nn.Module): Modelo entrenado
        test_df (pd.DataFrame): DataFrame de test
        dataset_class: Clase del Dataset
        test_transform: Transformaciones
        device: Dispositivo
        n_examples (int): Número de ejemplos a mostrar
    """
    model.eval()
    
    # Seleccionar ejemplos aleatorios
    indices = np.random.choice(len(test_df), n_examples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, sample_idx in enumerate(indices):
        row = test_df.iloc[sample_idx]
        
        # Cargar imagen
        img = Image.open(row['main_image_path']).convert('RGB')
        img_tensor = test_transform(img).unsqueeze(0).to(device)
        
        # Features tabulares
        tabular_cols = [c for c in test_df.columns 
                       if c not in ['poi_id', 'name', 'main_image_path', 'target', 
                                   'categories', 'engagement_ratio']]
        tab_tensor = torch.tensor(row[tabular_cols].values, 
                                 dtype=torch.float32).unsqueeze(0).to(device)
        
        # Predicción
        with torch.no_grad():
            output = model(img_tensor, tab_tensor)
            prob = F.softmax(output, dim=1)
            pred_label = output.argmax(1).item()
            confidence = prob[0, pred_label].item()
        
        true_label = row['target']
        
        # Visualizar
        axes[idx].imshow(img)
        axes[idx].axis('off')
        
        true_text = "Alto" if true_label == 1 else "Bajo"
        pred_text = "Alto" if pred_label == 1 else "Bajo"
        color = 'green' if true_label == pred_label else 'red'
        
        title = f"Real: {true_text} | Pred: {pred_text}\nConfianza: {confidence:.2%}"
        axes[idx].set_title(title, fontsize=10, fontweight='bold', color=color)
        
        info_text = f"{row['name'][:30]}...\nVisits: {row['Visits']}, Likes: {row['Likes']}"
        axes[idx].text(0.5, -0.15, info_text, transform=axes[idx].transAxes,
                      ha='center', fontsize=8, style='italic')
    
    plt.suptitle('Ejemplos de Predicciones del Modelo', 
                fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
