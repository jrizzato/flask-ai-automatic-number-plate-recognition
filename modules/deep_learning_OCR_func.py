from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from joblib import load
from matplotlib import pyplot as plt
import cv2
import easyocr
# from paddleocr import PaddleOCR

print("******** ----------- Cargando modelo ---------------- *********")
model = load('./model/object_detection_model.joblib') 
print("******** ----------- Model loaded successfully. ---------------- *********")

def object_detection(path, filename):
    # read image
    image = load_img(path) # PIL object
    image = np.array(image,dtype=np.uint8) # 8 bit array (0,255)
    image1 = load_img(path,target_size=(224,224))
    # data preprocessing
    image_arr_224 = img_to_array(image1)/255.0  # convert into array and get the normalized output
    h,w,d = image.shape
    test_arr = image_arr_224.reshape(1,224,224,3)
    # make predictions
    coords = model.predict(test_arr)
    # denormalize the values
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    # draw bounding on top the image
    xmin, xmax,ymin,ymax = coords[0]
    pt1 =(xmin,ymin)
    pt2 =(xmax,ymax)
    print(pt1, pt2)
    cv2.rectangle(image,pt1,pt2,(0,255,0),3)
    # convert to bgr
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./static/predict/{filename}', image_bgr) # es la imagen con el bounding box

    return coords

def OCR(path, filename):
    coords = object_detection(path, filename)

    img = np.array(load_img(path))
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]
    roi_bgr = cv2.cvtColor(roi, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'./static/roi/{filename}', roi_bgr) # es la ROI para OCR

    # OCR using EasyOCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(roi)
    print("******** ----------- OCR Result ---------------- *********")
    text = ' '.join([res[1] for res in result])
    print(text)
    
    return text

