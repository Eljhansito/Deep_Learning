# 🏰 Predicción de Engagement Turístico con Deep Learning Multimodal

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Preprocessing-F7931E)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Descripción del Proyecto

Este proyecto implementa una solución de **Deep Learning Multimodal** para predecir el nivel de éxito (*engagement*) de Puntos de Interés Turístico (POIs). 

A diferencia de los clasificadores de imágenes tradicionales, este modelo utiliza una arquitectura de **Fusión Tardía (Late Fusion)** que combina:
1.  **Información Visual:** Imágenes de los monumentos procesadas mediante CNNs.
2.  **Información Tabular:** Metadatos (ubicación, categorías, etiquetas) procesados mediante redes densas.

El objetivo es clasificar si un lugar turístico tendrá un **Alto** o **Bajo** impacto en los usuarios, ayudando a entender qué factores (estéticos o contextuales) influyen más en su popularidad.

---

## 🧠 Arquitectura del Modelo

El modelo `MejoradoModel` combina dos ramas de procesamiento que convergen en una cabeza de clasificación final:

* **📸 Rama Visual (CNN):**
    * Utiliza **ResNet18** pre-entrenada en ImageNet.
    * Estrategia de **Transfer Learning**: Se congelaron las capas iniciales para preservar la extracción de características básicas y se realizó *Fine-Tuning* en las últimas capas convolucionales (`layer4`).
    * Incluye *Global Average Pooling* implícito.

* **📊 Rama Tabular (MLP):**
    * Procesa vectores de características numéricas (normalizadas) y categóricas (One-Hot/MultiLabel).
    * Arquitectura: `Linear` -> `ReLU` -> `Dropout`.

* **🔗 Fusión:**
    * Concatenación de los vectores de salida de ambas ramas.
    * Clasificador final con `BatchNormalization` y `Dropout (0.3)` para prevenir overfitting.

---

## 🛠️ Tecnologías y Metodología

### Prevención de Data Leakage 🛡️
Uno de los pilares de este proyecto fue el rigor metodológico.
* **Split Previo:** La división Train/Test se realizó **antes** de cualquier transformación.
* **Fit/Transform:** Los escaladores (`StandardScaler`) y codificadores (`MultiLabelBinarizer`) se ajustaron (`fit`) **exclusivamente** con el conjunto de entrenamiento para evitar filtrar información del futuro al modelo.

### Ingeniería de Características
* **Target:** Creación de la métrica compuesta `engagement_ratio` (Likes + Bookmarks / Visits).
* **Balanceo:** Binarización del target utilizando la **mediana** como umbral dinámico, garantizando un dataset balanceado (50/50).

---

## 📂 Estructura del Repositorio

```text
├── data/                   # Carpeta con el dataset y las imágenes 
├── EDA_exploración_datos.ipynb  # Notebook de Análisis Exploratorio (visualización y limpieza)
├── practica_final.ipynb    # Notebook Principal: Preprocesamiento, Entrenamiento y Evaluación
├── MEMORIA TÉCNICA.pdf     # Documentación detallada del proyecto
└── README.md               # Este archivo
