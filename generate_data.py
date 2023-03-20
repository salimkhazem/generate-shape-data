import os
import sys
import cv2
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

output = "dataset"
num_classes = 5
line_width = 2
num_imgs_per_class = int(sys.argv[1]) if len(sys.argv)>1 else 1000
img_size = int(sys.argv[2]) if len(sys.argv)>2 else 64
shape_names = ["circle", "square", "rectangle", "triangle", "polygon"]
df = pd.DataFrame(columns=["path", "label"])
if not os.path.exists(output):
    os.makedirs(output)

def generate_(shape_name):
    img = np.zeros((img_size, img_size), dtype=np.uint8)
    center = (random.randint(img_size // 4, 3 * img_size // 4), random.randint(img_size // 4, 3 * img_size // 4))
    if shape_name == 'circle':
        radius = img_size // 4 - line_width // 2
        cv2.circle(img, center, radius, 255, line_width)
    elif shape_name == 'square':
        side = img_size // 2 - line_width // 2
        side = min(center[0], center[1], img_size - center[0], img_size - center[1])  - line_width // 2
        pt1 = (center[0] - side // 2, center[1] - side // 2)
        pt2 = (center[0] + side // 2, center[1] + side // 2)
        cv2.rectangle(img, pt1, pt2, 255, line_width)
    elif shape_name == 'rectangle':
        width = img_size // 4 - line_width // 2
        height = img_size // 4 - line_width // 2
        width = min(center[0], img_size - center[0]) - line_width // 2
        height = min(center[1], img_size - center[1]) - line_width // 2
        pt1 = (center[0] - width // 2, center[1] - height // 2)
        pt2 = (center[0] + width // 2, center[1] + height // 2)
        cv2.rectangle(img, pt1, pt2, 255, line_width)
    elif shape_name == 'triangle':
        side = img_size // 4 - line_width // 2
        side = min(center[0], center[1], img_size - center[0], img_size - center[1]) * 2 - line_width // 2
        pt1 = (center[0], center[1] - side // 2)
        pt2 = (center[0] - side // 2, center[1] + side // 2)
        pt3 = (center[0] + side // 2, center[1] + side // 2)
        pts = np.array([pt1, pt2, pt3], dtype=np.int32)
        cv2.polylines(img, [pts], True, 255, line_width)
    elif shape_name == 'polygon':
        pts = []
        poly_side = 5
        r = min(center[0], center[1], img_size - center[0], img_size - center[1]) - line_width // 2
        for i in range(poly_side):
            theta = i * (360 // poly_side)
            #r = random.randint(img_size // 4 - line_width, img_size // 2)
            x = center[0] + int(r * np.cos(np.deg2rad(theta)))
            y = center[1] + int(r * np.sin(np.deg2rad(theta)))
            pts.append([x, y])
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], True, 255, line_width)
    return img

count = 0
print("----- Generating Dataset -----")
for i in tqdm(range(num_classes), total=num_classes):
    shape_name = shape_names[i]
    for j in range(num_imgs_per_class):
        img = generate_(shape_name)
        label = i
        filename = f'{shape_name}_{j}.png'
        df.loc[count, 'path'] = filename
        df.loc[count, 'label'] = i
        cv2.imwrite(f"{output}/{filename}", img)
        count += 1
print(f'{num_classes*num_imgs_per_class} images generated')
print("Distribution of the data:")
print(df.label.value_counts())
df.to_csv("./dataset.csv", index=False)
