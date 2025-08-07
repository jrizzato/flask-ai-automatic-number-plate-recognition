import pandas as pd
import os
import cv2
import xml.etree.ElementTree as xet
from joblib import dump

df = pd.read_csv('./data/labels.csv')
print()
print('df head: ')
print(df)

def get_file_name(filename):
    filename_image = xet.parse(filename).getroot().find('filename').text
    filepath_image = os.path.join('./data/images', filename_image)
    return filepath_image

image_path = list(df['filepath'].apply(get_file_name)) # 
print('Image paths extracted from DataFrame')
print('Number of images:', len(image_path))
# Display first 5 image paths
print('First 5 image paths:')
print(*image_path[:5], sep='\n')  # Display first 5 image paths

# verificar imagen y salida
file_path = image_path[0]
img = cv2.imread(file_path)
cv2.rectangle(img, (226, 173), (419, 125), (0, 255, 0), 3)
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

dump(image_path, './data/image_path.joblib')
dump(df, './data/labels.joblib')

print('Image paths saved to ./data/image_path.joblib')
print('Labels saved to ./data/labels.joblib')
