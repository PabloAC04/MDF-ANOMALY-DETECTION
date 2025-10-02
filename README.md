# Detección de Anomalías en Ficheros MDF

Este repositorio contiene el código, experimentos y documentación asociados a mi **Trabajo de Fin de Grado (TFG)**, cuyo objetivo es el **desarrollo e implementación de técnicas de detección de anomalías en datos industriales** almacenados en **ficheros MDF (Measurement Data Format)**.

## Objetivo del proyecto
El propósito del proyecto es **explorar, implementar y comparar diferentes algoritmos de detección de anomalías**, aplicados a series temporales industriales.  
Se busca evaluar su rendimiento en términos de precisión, recall y otras métricas específicas, así como su escalabilidad en entornos con grandes volúmenes de datos.

## Algoritmos implementados
Hasta el momento, el proyecto incluye las siguientes técnicas:

- **PCA (Principal Component Analysis)**  
- **Isolation Forest**  
- **One-Class SVM**  
- **Autoencoders**  
- **Transformers Autoencoders**

## Tecnologías utilizadas
- **Python 3.10+**
- **PyTorch** para modelos de Deep Learning  
- **Scikit-learn** para algoritmos clásicos de Machine Learning  
- **Dask / RAPIDS (cuDF, cuML)** para paralelización y ejecución en GPU  
- **Pandas / NumPy** para procesamiento de datos


## Resultados esperados
- Comparativa entre algoritmos clásicos y modelos basados en Deep Learning.  
- Evaluación del rendimiento en **datasets industriales** (ej. BATADAL, SWaT, u otros).  
- Análisis de **ventajas y limitaciones** de cada técnica.

## Autor
Este proyecto ha sido desarrollado por **Pablo Arquellada Cebrián** y tutorizado por **Pedro Luis Galindo Riaño** como parte del **Trabajo de Fin de Grado del Grado en Ingeniería Informática en la Universidad de Cádiz**.

---
