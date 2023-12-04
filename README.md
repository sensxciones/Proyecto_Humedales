# Repositorio Proyecto Humedales
Aquí colocaremos los archivos que utilizamos para el desarrollo de el proyecto de Integrando Humedales.

Para hacer uso del repositorio es necesario clonar el repositorio de ultralytics de v8, para ello solo basta con ejecutar "pip install ultralytics" en el terminal, o bien, ejecutar "git clone+https://github.com/ultralytics/ultralytics.git"
Tambien es necesario contar con OpenCV.
# Train
Este es el dataset utilizado, aquí esta el data set generico de distintas especies de aves y las fotos enviadas por Integrando Humedales (hay que descargar y colocarlas en un archivo zip)
  Set de aves en general: --
  Set de aves "Integrando Humedales": --
# Modulos:
Los modulos Losses.py, Network.py, Trainer.py y datasets.py son archivos adapatados a este proyecto y uno de años anteriores. Los modulos originales se encuentran en el siguiento github: https://github.com/adambielski/siamese-triplet
El modulo duckies_dataset es un modulo creado especificamente para este proyecto.
Por otro lado, el módulo Deteccion_Camara1.py fue desarrollado por nosotros, este realiza un reconocimiento a tiempo real de las aves. (recordar agregar el archivo .py para despues)
# Patimetria_2
Aquí se encuentra el archivo Jupyter para entrenar nuestro proyecto.
Durante el proyecto, primero utilizamos un dataset publico de la plataforma Kaggle(ver link) con el cual entrenamos este modelo para aumentar la precisión de su funcionamiento. Más adelante, utilizamos los archivo enviados por la organizacion "Integrando Humedales"(fotos y videos recopilados por sus integrantes) para entrenar a nuestra red para la clasificación de las aves a detectar.
(explicar en mayor detalle como funciona)
