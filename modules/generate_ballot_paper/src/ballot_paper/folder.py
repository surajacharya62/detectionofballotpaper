import os
import random
import cv2

path = "../../../datasets/"
all_symbols = []
symbols = []
for folder in os.listdir(path):
    subfolder = os.path.join(path, folder) 
    for filename in os.listdir(subfolder):
        if filename.endswith(('.jepg','png','.jpg')):
            img = os.path.join(subfolder, filename)
            symbols.append(img)
    if symbols:
        print(len(symbols))
        image = random.choice(symbols)
        all_symbols.append(image)        
        symbols.clear()

print(len(all_symbols))