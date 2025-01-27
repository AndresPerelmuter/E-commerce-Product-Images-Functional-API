# E-commerce Product Images Functional API

**Descripción del Proyecto**
Este proyecto implementa un modelo de clasificación multiuple basado en TensorFlow y Keras, diseñado para predecir múltiples etiquetas (como género, categoría, subcategoría, tipo de producto y color) a partir de imágenes. El modelo fue construido utilizando **Keras Functional API**, lo que permitió diseñar una arquitectura más flexible y personalizable para abordar las tareas multi-salida de forma simultánea.
Los datos fueron extraidos de Kaggle: https://www.kaggle.com/datasets/vikashrajluhaniwal/fashion-images.

**Objetivo**
El objetivo principal de este proyecto fue entrenar un modelo de deep learning capaz de realizar múltiples tareas de clasificación de forma simultánea.

**Pasos Realizados**

**1. Preparación del Dataset**
Etiquetas Categóricas: Se convirtieron columnas categóricas (Gender, Category, SubCategory, ProductType, Colour) en valores numéricos utilizando astype('category').cat.codes.
Imágenes: Cada ruta de imagen se vinculó con las etiquetas correspondientes para crear un pipeline de datos.

**2. División de Datos**
Se dividió el dataset en conjuntos de entrenamiento (80%) y prueba (20%) utilizando train_test_split de sklearn.

**3. Pipeline de Datos con TensorFlow**
Se creó un pipeline de TensorFlow (tf.data.Dataset) para leer imágenes, redimensionarlas a 128x128 píxeles y normalizarlas (valores entre 0 y 1).
Cada imagen fue vinculada con múltiples etiquetas de salida.

**4. Construcción del Modelo con Keras Functional API**
El modelo fue construido utilizando la Functional API de Keras, lo que permitió diseñar múltiples flujos de datos para diferentes salidas:

Capa Base Convolucional: Se usó EfficientNetB0 como extractor de características preentrenado.
Cabezas de Salida: A partir de la salida del extractor de características, se añadieron cinco ramas independientes (una para cada tarea de clasificación). Cada rama incluye capas densas con activación softmax.
La Functional API permitió conectar estas ramas de salida con el modelo base de manera eficiente, manteniendo una estructura clara y modular.

**5. Compilación del Modelo**
El modelo fue compilado con:

- Pérdida: SparseCategoricalCrossentropy para cada salida.
- Optimizador: Adam.
- Métricas: Accuracy para cada salida.
  
**6. Entrenamiento**
Se entrenó el modelo durante 10 épocas con un tamaño de batch de 32.
Se incluyó un conjunto de validación para monitorizar las métricas en datos no vistos.

**7. Evaluación**
Se evaluó el modelo en el conjunto de prueba.
