from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from joblib import load
from matplotlib import pyplot as plt
import cv2

print("******** ----------- Cargando modelo ---------------- *********")
model = load('./model/object_detection_model.joblib') 
print("******** ----------- Model loaded successfully. ---------------- *********")

# Cargar imagen
path = './data/validation_images/16.jpg'

# Cargar imagen original para visualización
image = load_img(path)
image = np.array(image, dtype=np.uint8)
image1 = load_img(path,target_size=(224,224))
image_arr_224 = img_to_array(image1)/255.0  # convert into array and get the normalized output

# size of the orginal image
h,w,d = image.shape
print('Height of the image =',h)
print('Width of the image =',w)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()

# Preprocesar la imagen
print("image_arr_224.shape")
print(image_arr_224.shape)  # (224, 224, 3)

test_arr = image_arr_224.reshape(1,224,224,3)
print("test_arr.shape")
print(test_arr.shape)  # (1, 224, 224, 3)

# make predictions
coords = model.predict(test_arr)
print("Predicted coordinates:")
print(coords)

# denormalize the values
denorm = np.array([w,w,h,h])
coords = coords * denorm
coords = coords.astype(int)
print("Denormalized coordinates:")
print(coords)

# draw bounding on top the image
xmin, xmax,ymin,ymax = coords[0]
pt1 =(xmin,ymin)
pt2 =(xmax,ymax)
print(pt1, pt2)
cv2.rectangle(image,pt1,pt2,(0,255,0),3)

plt.figure(figsize=(10,8))
plt.imshow(image)
plt.show()

