# Semana 5: Procesamiento de Lenguaje Natural (NLP)

## ¿Qué es el Lenguaje?

Un lenguaje es un conjunto potencialmente infinito de oraciones y sentencias construidas con reglas gramaticales, fonéticas y semánticas. En el contexto del NLP, se trabaja con:

- **Lenguaje Natural**: Idiomas hablados (español, inglés, etc.)
- **Lenguaje Formal**: Usado en matemáticas, lógica, programación.
- **Lenguaje Artificial**: Mezcla de natural y formal, creado con un propósito.

---

## ¿Qué es el NLP?

Es un campo interdisciplinario que une la informática, IA y lingüística para permitir la interacción entre humanos y máquinas usando lenguaje natural.

### Objetivo:
Transformar texto en un formato adecuado para su análisis computacional.

---

## Aplicaciones del NLP

- **Recuperación de información**
- **Extracción y categorización de información**
- **Análisis de sentimientos**
- **Traducción automática**
- **Generación de lenguaje**
- **Chatbots y preguntas-respuestas**

---

## Conceptos Clave del Preprocesamiento en NLP

### Corpus
Conjunto de textos recopilados para análisis, como artículos, libros, tweets, etc.

### Normalización
- Convertir a minúsculas
- Eliminar puntuación
- Convertir números a texto
- Eliminar palabras irrelevantes (stop words)

### Tokenización
Separar el texto en unidades pequeñas como palabras, símbolos o frases.

### Segmentación
Dividir el texto en oraciones o párrafos.

### Stemming
Reducir las palabras a su raíz al eliminar afijos.
- Ejemplo: *caminando → caminar*

### Lematización
Transformar una palabra flexionada a su forma base válida en el idioma.
- Ejemplo: *niños → niño*

### Stop Words
Palabras que no aportan al significado, como artículos, preposiciones, etc.

### POS Tagging (Etiquetado gramatical)
Asignar una etiqueta gramatical a cada palabra:
- Sustantivo
- Verbo
- Adjetivo
- Adverbio
- Preposición, etc.

### n-gramas
Secuencias contiguas de N elementos (palabras o caracteres) que permiten capturar contexto:
- Unigrama, Bigramas, Trigramas...

---

## Actividades y Recursos

- Google Colabs para práctica con procesamiento de texto y exploración de técnicas básicas:
  - [Conceptos Básicos de NLP](https://colab.research.google.com/drive/1ayXlZpI0-SIHJRsLA4Z2XurPGIsVKhpA#scrollTo=MtnaMFuXHuSI)
  - [Normalización de textos y Bolsa de Palabras](https://colab.research.google.com/drive/1pZ0BsKXKekqlY-GVd1vEIK9BuxRU5Osn#scrollTo=WXv3dTcIFIFl)
  - [Clasificación de textos con Scikit-Learn](https://colab.research.google.com/drive/1yuH4eCOJJbGHmo7IagxNVnys5EHf1lrt)

---

> Esta semana marca el inicio del trabajo con datos de texto, enfocándose en cómo prepararlos antes de construir modelos de IA para tareas de comprensión y generación de lenguaje.
