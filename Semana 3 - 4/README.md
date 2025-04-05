# Semana 3 - 4: - Métricas de Desempeño y Evaluación de Modelos

## Propósito
Aprender a **evaluar modelos de aprendizaje automático**, tanto en clasificación como regresión, usando métricas adecuadas según el tipo de problema. También se introducen fundamentos del procesamiento de lenguaje natural (NLP).

---

## Clasificación - Métricas Clave

### 1. Accuracy (Exactitud)
- Proporción de predicciones correctas sobre el total.
- **Problema**: Puede ser engañosa en datos desbalanceados.

### 2. Precision (Precisión o Valor Predictivo Positivo)
- De los positivos predichos, ¿cuántos fueron correctos?
- Importante cuando los **falsos positivos son costosos** (ej. diagnóstico médico).

### 3. Recall (Sensibilidad)
- De los positivos reales, ¿cuántos fueron detectados?
- Crítico cuando los **falsos negativos son peligrosos** (ej. fraude o cáncer).

### 4. F1-Score
- Media armónica entre precisión y recall.
- Útil cuando hay **desbalance de clases**.

### 5. Matriz de Confusión
- Tabla con verdaderos positivos, negativos, falsos positivos y negativos.
- Permite observar **errores específicos** del modelo.

### 6. ROC-AUC
- Curva que muestra sensibilidad vs. tasa de falsos positivos.
- AUC mide el área bajo la curva: cuanto más cerca de 1, mejor.

---

## Regresión - Métricas Clave

### 1. MAE (Error Absoluto Medio)
- Promedio del valor absoluto de los errores.
- Interpretable en las **mismas unidades que la variable de salida**.

### 2. MSE (Error Cuadrático Medio)
- Penaliza más los errores grandes. Bueno para modelos que deben ser precisos en todos los rangos.

### 3. RMSE (Raíz del MSE)
- Como MSE, pero vuelve a las **unidades originales**.

### 4. R² (Coeficiente de Determinación)
- Indica **qué proporción de la varianza** está explicada por el modelo.
- Valor entre 0 y 1. Cuanto más alto, mejor.

---

## Conceptos Extra

- **Bias vs Variance**: sesgo y varianza como fuentes de error.
- **Overfitting**: cuando el modelo aprende demasiado del entrenamiento y no generaliza.
- **Underfitting**: cuando el modelo es demasiado simple.

---

## Introducción a NLP
- **Procesamiento de lenguaje natural** para análisis de texto.
- Representación con **BOW (Bag of Words)** y **TF-IDF**.
- Introducción a embeddings como **Word2Vec** y **Transformers**.

