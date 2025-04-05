# Semana 2: Configuración Experimental y Preprocesamiento de Datos

## Objetivo de la Semana
Entender cómo preparar y organizar un experimento de Machine Learning para asegurar:
- **Reproducibilidad**
- **Confiabilidad**
- **Generalización de resultados**

---

## ¿Qué es una Configuración Experimental?

Una **configuración experimental** define cómo se estructuran y ejecutan los experimentos en ML. Incluye:

- Organización de datos
- Elección de modelos
- Evaluación con métricas
- Uso de herramientas y entornos adecuados

**Meta**: generar resultados significativos, comparables y repetibles.

---

## Flujo de trabajo en Machine Learning

1. **Obtención de datos**
2. **Preprocesamiento**
3. **División del conjunto de datos** (entrenamiento, validación, prueba)
4. **Entrenamiento del modelo**
5. **Evaluación**
6. **Ajuste y validación**
7. **Despliegue**

---

## Componentes Clave de la Configuración Experimental

- **Datos**: Preparación adecuada de los conjuntos de entrenamiento, validación y prueba.
- **Modelos**: Selección de algoritmos, configuración de hiperparámetros.
- **Métricas**: Exactitud, precisión, recall, F1-score, etc.
- **Infraestructura**: Herramientas como Google Colab, Jupyter, etc.

---

## Preprocesamiento de Datos

### 1. Exploración
- Identificar:
  - Valores atípicos (outliers)
  - Valores perdidos
  - Errores de formato
  - Distribuciones de datos

### 2. Limpieza
- Manejo de valores nulos (media, mediana, modo)
- Eliminación de duplicados
- Corrección de errores tipográficos o de tipo

### 3. Normalización y Estandarización
- **Normalización**: Escala valores a un rango (ej. 0 a 1)
- **Estandarización**: Media 0, desviación estándar 1

### 4. Codificación de Variables Categóricas
- **One-Hot Encoding**
- **Label Encoding**
- **Embeddings** (para deep learning)

### 5. Reducción de Dimensionalidad
- Selección de características (feature selection)
- PCA (Análisis de Componentes Principales)
- t-SNE (para visualización)

### 6. Manejo de Datos Desbalanceados
- **Oversampling** (SMOTE)
- **Undersampling**
- Asignación de pesos a clases

### 7. División del Dataset
- **Train/Test Split**
- **Validación cruzada (k-fold)**

### 8. Ingeniería de Características
- Crear nuevas variables a partir de otras
- Combinaciones lógicas, estadísticas, etc.

---

## Conclusión del Preprocesamiento
- El preprocesamiento es una etapa **crítica**.
- Mejora la calidad de los modelos.
- Debe adaptarse a cada conjunto de datos y problema.
- Facilita la generalización y evita errores como el sobreajuste.

---

## Ejercicio de la Semana

### Dataset: **Boston Housing**
- Objetivo: predecir el precio de viviendas
- Pasos propuestos:
  1. Cargar y explorar el dataset.
  2. Identificar filas, columnas y características.
  3. Aplicar limpieza de datos (valores nulos, outliers).
  4. Normalizar o estandarizar los datos.

**Enlace del dataset:**  
https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv

**Referencias prácticas en Colab:**
- [Ejemplo de preprocesamiento](https://colab.research.google.com/drive/1Ur2QVOg15cbstVJlt8k243R_VdhFRRIq?usp=sharing)
- [Notebook adicional](https://colab.research.google.com/drive/1WyUqHKOIEDTq3gYQIlXbjDnotzid7xqe?usp=sharing)

---

## Taller de Semana
Realizar preprocesamiento a datasets de toy datasets de scikit-learn:

**Link:**  
https://scikit-learn.org/stable/datasets/toy_dataset.html

---

## Conclusiones Generales
- Una buena configuración experimental mejora la validez de los resultados.
- El preprocesamiento es esencial y se adapta a cada problema.
- Se introdujo el uso de herramientas y datasets clásicos para comenzar a experimentar.

