#El siguiente archivo permite la deteccion de las aves a tiempo real, cuando este archivo esta corriendo
from ultralytics import YOLO
import cv2
import math
import time
import datetime
import os
import pickle

# import funciones para cargar modelo
import torch
from networks import VGGEmbeddingNet, TripletNet
from torchvision import transforms
from PIL import Image
from duckies_dataset import Rescale
import numpy as np

def load_embeddings():
    # carga embeddings de aves
    data = pickle.loads(open("models/embeddings.pkl", "rb").read())
    # data = list[tuple[emb, label]]
    embeddings = []
    labels = []
    for emb, label in data:
        embeddings.append(emb)
        labels.append(label)
    return embeddings, labels

def load_model(model_path: str = 'models/epoch-10.pt'):
    # carga modelo para reconocimiento de aves
    vgg_model = VGGEmbeddingNet()
    model = TripletNet(vgg_model)
    cuda = torch.cuda.is_available()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if cuda:
        model.cuda()

    return model

def closest_image(emb_inf, embeddings):
    # encuentra la imagen mas cercana a la imagen de entrada
    # embeddings: lista de embeddings
    # emb_inf: embedding de la imagen de entrada
    # calcular distancia entre emb_inf y cada embedding
    dists = []
    for emb in embeddings:
        dist = np.linalg.norm(emb_inf - emb)
        dists.append(dist)
    # encontrar indice de la imagen mas cercana
    label_idx = np.argmin(dists)
    return label_idx

embeddings, labels = load_embeddings()
model_ave = load_model()
trans = transforms.Compose([Rescale(224),
                    transforms.ToTensor()])
def reconocer_ave(img):
    cv2_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    pil_image = Image.fromarray(cv2_image)

    img = trans(pil_image)
    img = img.cuda()
    emb_img = model_ave.get_embedding(img.unsqueeze(0))
    
    embedding_img = emb_img.data.cpu().numpy()

    idx = closest_image(embedding_img, embeddings)
    return labels[idx]
    
# se enciende la camara
cap = cv2.VideoCapture(0)
# cap.set(3, 640)
# cap.set(4, 480)

#coloque esta variable para contar el número de detecciones realizadas por
#mientras este encendida la camara y el tiempo donde comienza la deteccion
n = 0

#las siguiente variables marcan el momento en que el programa arranca, el nombre del archivo que se generara con la fecha
#se genera un nuevo archivo cada vez que el programa comienza y se guarda en la carpeta Detecciones
tiempo_inicio = time.time()
carpeta_destino = "detection"
fecha_inicio = datetime.datetime.now().strftime('%Y_%m_%d')
nombre_archivo = str(f"deteccion_{fecha_inicio}.txt")
ruta_archivo = os.path.join(carpeta_destino, nombre_archivo)


# se importa el modelo de YOLO, con algunos de los pesos. Ahora este esta
#entrenado para la deteccion de aves
path_to_model = str(os.getcwd()) + "/models/yolov8n.pt"
model = YOLO(path_to_model)

# esta funcion abre un archivo de texto para escribir sobre el.
archivo_detecciones = open(ruta_archivo, "w")

while True:
    success, img = cap.read()
    # img = cv2.imread("/home/felipe/Cuac/Proyecto_Humedales/dataset-humedal/valid/YECO/001.jpg")
    # Recortar 30% central de la imagen
    # img = img[int(img.shape[0]*0.3):int(img.shape[0]*0.7), int(img.shape[1]*0.3):int(img.shape[1]*0.7)]
    results = model(img, classes = 14, stream=True, verbose=False)
    for bird in results:
        boxes = bird.boxes
	#IMPORTANTE: espacio para tomar el recorte de la foto, pasarlo
	#al modelo entrenado del humedal, y que solo nos devuelva la
	#especie de pajaro identificada
        for box in boxes:
            n = n+1
            # se encierra la ave detectada en la webcam en un cuadrado
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # recorte de la imagen
            subimg = img[y1:y2, x1:x2]
            cv2.imshow('recorte', subimg)

            # reconocimiento de la especie
            label = reconocer_ave(subimg)
            print(label)
            # label = "LOICA"

            # se imprime el cuadrado en la camara
            cv2.rectangle(img, (x1, y1), (x2, y2), (12, 183, 242), 3)

            # se imprime el porcentaje de confianza en la deteccion
            grado_confianza = math.ceil((box.conf[0]*100))
            # print(f"Grado de confianza: {grado_confianza}%")

            # Detalle de los objetos, el color del texto, el tipo de letra
		#el grososr del txt
            org = [x1, y1 - 10]
            color = (12, 183, 242)
            thickness = 2
            font_type = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(img, str(label), org, font_type, 1, color, thickness)
            # cv2.putText(img, str(grado_confianza), org, font_type, 1, color, thickness)

            # Escribe la información de la detección en el archivo txt, 
            timestamp = int(time.time() - tiempo_inicio)
            info_deteccion = f"en el segundo {timestamp} - Ave detectada con {grado_confianza}% de confianza.\n"
            archivo_detecciones.write(info_deteccion)

#Al apretar la letra q, se rompe el ciclo, se libera la camara y se apaga
    cv2.namedWindow('Webcam', cv2.WINDOW_NORMAL)
    cv2.imshow('Webcam', img)
    if cv2.waitKey(30) == ord('q'):
        break

#se escribe la ultima linea del archivo de texto, con el numero de detecciones realizadas
archivo_detecciones.write(f"Se realizaron {n} detecciones duarante la grabación.")

#se cierra el archivo de texto junto con la camara
archivo_detecciones.close()
cap.release()
cv2.destroyAllWindows()


