import pandas as pd
import xml.etree.ElementTree as xet
from glob import glob

path = glob('./data/images/*.xml')

labels_dict = dict(filepath=[], xmin=[], xmax=[], ymin=[], ymax=[])

for filename in path:
    info = xet.parse(filename)
    root = info.getroot()
    member_objects = root.find('object')
    labels_info = member_objects.find('bndbox')
    xmin = int(labels_info.find('xmin').text)
    xmax = int(labels_info.find('xmax').text)
    ymin = int(labels_info.find('ymin').text)
    ymax = int(labels_info.find('ymax').text)

    labels_dict['filepath'].append(filename)
    labels_dict['xmin'].append(xmin)
    labels_dict['xmax'].append(xmax)
    labels_dict['ymin'].append(ymin)
    labels_dict['ymax'].append(ymax)

df = pd.DataFrame(labels_dict)
print()
print('DataFrame created with shape:', df.shape) # (number of rows, number of columns)
print('Columns:', df.columns.tolist())
print('First 5 rows of the DataFrame:')
print(df.head())

df.to_csv('./data/labels.csv', index=False)
print('CSV file saved as ./data/labels.csv')

