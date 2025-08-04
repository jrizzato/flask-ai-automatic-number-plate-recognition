# Data preprocessing
from joblib import load, dump
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2

image_path = load('./data/image_path.joblib')
print('Image paths loaded from ./data/image_path.joblib ')
df = load('./data/labels.joblib')
print('Labels loaded from ./data/labels.joblib ')

# print(df.iloc[:,1:])
labels = df.iloc[:, 1:].values
print('labels shape: ', labels.shape) # labels shape:  (233, 4) significa que hay 233 imagenes y 4 coordenadas por imagen. Las coordenadas son xmin, xmax, ymin, ymax

data = []
output = []
for ind in range(len(image_path)):
    image = image_path[ind] # esta linea es la que se encarga de obtener el path de la imagen
    img_arr = cv2.imread(image) # esta linea lee la imagen
    h,w,d = img_arr.shape # esta linea obtiene las dimensiones de la imagen
    # print(f'Image shape: {h}x{w}x{d}')  # Display image dimensions
    # preprocesing
    load_image = load_img(image, target_size=(224, 224))
    load_image_arr = img_to_array(load_image) # esta linea convierte la imagen a un array
    norm_load_image_arr = load_image_arr / 255.0 # esta linea normaliza la imagen a valores entre 0 y 1
    # normalization of labels
    xmin, xmax, ymin, ymax = labels[ind] # esta linea obtiene las coordenadas de la caja delimitadora
    xmin = xmin / w # estas lineas normaliza las coordenadas de la caja delimitadora
    xmax = xmax / w
    ymin = ymin / h
    ymax = ymax / h
    norm_labels = (xmin, xmax, ymin, ymax) # esta linea crea una tupla con las coordenadas normalizadas

    data.append(norm_load_image_arr) # esta linea agrega la imagen normalizada a la lista de datos
    output.append(norm_labels) # esta linea agrega las etiquetas normalizadas a la lista de salida

X = np.array(data, dtype=np.float32) # X shape:  (233, 224, 224, 3) significa que hay 233 imagenes, cada una de 224x224 pixels y 3 canales de color (RGB)
y = np.array(output, dtype=np.float32) # Y shape:  (233, 4) significa que hay 233 imagenes y 4 coordenadas por imagen. Las coordenadas son xmin, xmax, ymin, ymax

print('X shape: ', X.shape)
print('Y shape: ', y.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print('x_train shape: ', x_train.shape)
print('x_test shape: ', x_test.shape)
print('y_train shape: ', y_train.shape)
print('y_test shape: ', y_test.shape)
print('Data preprocessing completed.')

dump((x_train, x_test, y_train, y_test), './data/preprocessed_data.joblib')
print('Preprocessed data saved to ./data/preprocessed_data.joblib') 