# Repositorio Proyecto Humedales
Aquí colocaremos los archivos que utilizamos para el desarrollo de el proyecto de Integrando Humedales.

Para hacer uso del repositorio es necesario clonar el repositorio de ultralytics de v8, para ello solo basta con ejecutar: 
```bash
pip install ultralytics
```
```bash
pip install opencv-python
```

# Modulos .py 
Los modulos Losses.py, Network.py, Trainer.py y datasets.py son archivos adapatados a este proyecto y uno de años anteriores. Los modulos originales se encuentran en el siguiento github: https://github.com/adambielski/siamese-triplet. El modulo duckies_dataset es un modulo creado especificamente para el proyecto anterior (link más abajo). 

Por otro lado, el módulo Deteccion_Camera.py fue desarrollado por nosotros, este realiza un reconocimiento a tiempo real de las aves. Al ejecutarse, recorta la detección realizada por YOLOv8, la pasa a través de nuestro modelo "VisionAlada", para entregar el nombre de la especie e imprimirlo 

# Carpetas models, detection y dataset-humedal 
En primer lugar, en la carpeta models hay archivos necesarios para la ejecución de nuestra red (VisionAlada), el archivo tipo .pkl, y otras redes necesarias para el funcionamiento del código. Por otro lado, dataset-humedal es la carpeta con todas la sfotos utilizadas para el entrenamiento del modelo y el archivo embedding. La carpeta detection es la carpeta en la cual se almacenan todos los archivos de texto con las detecciones realizadas durante el tiempo que este encendida nuestra cámara.

# Patimetria_2
Aquí se encuentra el archivo Jupyter para entrenar nuestro proyecto. Este archivo pertenece a un proyeccto de años anteriores (link repositorio: https://github.com/Mar3334/Patimetria.git), el cual toma las fotos (en nuestro caso desde Drive) y genera un dataset para entrenar el modelo de detección.
Durante el proyecto, primero utilizamos un dataset publico de la plataforma Kaggle(https://www.kaggle.com/datasets/gpiosenka/100-bird-species), con el cual entrenamos este modelo para aumentar la precisión de su funcionamiento. Más adelante, utilizamos los archivo enviados por la organizacion "Integrando Humedales" (las fotos se encuentran en la carpeta dataset-humedal).
