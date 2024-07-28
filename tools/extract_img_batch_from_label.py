
import json
from collections import Counter
import cv2 
import numpy as np
import matplotlib as plt


with open('data/annotations/final_train.json', 'r') as f:
    data = json.load(f)

print('keys:',data.keys())
unique = []

## extract Runway bbox
for item in data['annotations']:
    file_name = item['file_name']
    for ele in item['segments_info']:
        # print('category_id:',ele['category_id'])
        if ele['category_id'] == 21:
            runway_bbox = ele['bbox']
            print('runway_bbox:', runway_bbox, file_name)
            file_list = [file_name, runway_bbox]
            unique.append(file_list)
        else:
            continue

print('unique:',unique)

img_path = "/data/JPEGImages/"
for item in unique:
    img = cv2.imread(img_path+item[0])
    bbox = item[1]
    x1,x2,y1,y2 = bbox[0]
    rect = img[y:y+h, x:x+w]

