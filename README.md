# Repositorio Proyecto Humedales
Para hacer uso del código, en necesario instalar el la libreria de ultralytics en nuestro computador, junto con otras librerias como OpenCV, PyTorch, etc. Para esto es necesario ejecutar las siguiente lineas de codigo en la temrinal de la computadora:
```bash
pip install ultralytics
```
```bash
pip install opencv-python
```
Una vez instalados, dentro de la carpeta en la que quieran almacenar el repositorio, es necesario ejecutar la siguiente línea de código, 
```bash
git clone https://github.com/sensxciones/Proyecto_Humedales
```
Una vez realizado, sera posible acceder desde tu dispositivo a los archivos del proyecto, e incluisve modificarlos para adaptarlos a tus propios proyectos.
# Ultralytics
Para entender mejor el funcionamiento de YOLO, se tiene la siguiente documentación (<aYOLO href="https://docs.ultralytics.com/">Docs</a>), donde se muestran detalles acerca de como entrenar la detección de YOLO, además de utilizar otras funciones de este, como Pose Estimation o Segmentation.

# Modulos:
Los modulos Losses.py, Network.py, Trainer.py y datasets.py son archivos adapatados para un proyecto de años anteriores. Los modulos originales se encuentran en el siguiento github: https://github.com/adambielski/siamese-triplet
El modulo duckies_dataset es un modulo creado especificamente para este proyecto. 

Por otro lado, el módulo detection_camera.py fue desarrollado por nosotros, este realiza un reconocimiento a tiempo real de las aves, integrando nuestra red neuronal con las detecciones de YOLO. 

Por otro lado, una de las tareas que implementa nuestro proyecto es escribir un archivo de texto mencionando las detecciones realizadas y una estimación del tiempo en el que se realizo.
# Patimetria_2 y Embedding-aves-humedal
Aquí se encuentra el archivo Jupyter adpatado para nuestro proyecto, junto con el archivo que nos permite generar los embeddings de las aves de humedal y guardarlos en tuplas con su nombre. El primer archivo nos genera un modelo para la clasificación de aves como tal, mientras que el segundo nos genera un archivo especifico para las especies de aves presentes en el humedal.

Durante el proyecto, primero utilizamos un dataset publico de la plataforma <a href="https://docs.ultralytics.com/">Kaggle</a> con el cual entrenamos este modelo para aumentar la precisión de su funcionamiento. Más adeltante utilizamos el repositorio de imagenes de Reintegrando Humedales.
