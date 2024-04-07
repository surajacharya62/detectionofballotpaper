import cv2
import os
import random
from PIL import Image, ImageDraw, ImageFont
import math

from sklearn.model_selection import train_test_split

def load_symbols(folder_path):
    """    
    """
    
    all_party_symbols = []
    symbols = []
    for folder in os.listdir(folder_path):
        sub_folder = os.path.join(folder_path, folder) 
       
        # sub_folder = folder_path + folder
        if os.path.isdir(sub_folder):
            
            for sub_folder2 in os.listdir(sub_folder):
                sub_dir = os.path.join(sub_folder, sub_folder2)
                
                if os.path.isdir(sub_dir):
                    # print(sub_dir)
                    for filename in os.listdir(sub_dir):
                        # print(filename)
                        if filename.lower().endswith(('.jepg','png','.jpg')):
                                img = cv2.imread(os.path.join(sub_dir, filename))
                                symbol_name = os.path.splitext(filename)[0] 
                                symbols.append((img,symbol_name))

                        if symbols:                            
                            image = random.choice(symbols)
                            all_party_symbols.append(image)        
                            symbols.clear() 
                else:
                    pass
        else:
            print("invalid directory")

    random.shuffle(all_party_symbols)    
    return all_party_symbols


symbols_test_path = '../../../datasets_symbol_train/'


symbols = load_symbols(symbols_test_path)
print(len(symbols))
# print(symbols[:5])

train_images, test_images = train_test_split(symbols, test_size=0.25, random_state=42)
main_train_images, val_images = train_test_split(train_images, test_size=0.20, random_state=42)
print(len(train_images))
print(len(main_train_images))
print(len(test_images))
print(len(val_images))
import shutilsp 
import os

def save_images(image_list, target_folder):
    # Create the target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    random.shuffle(image_list)
    for image_path in image_list:
        # Define the destination path for the image
        image, name = image_path
        # print(image)
        # print(name)
        cv2.imwrite(target_folder + f'{name}.jpg', image) 
        # destination_path = os.path.join(target_folder, os.path.basename(image_path))
        # Copy the image to the target folder
        # shutil.copy(image_path, destination_path)

# Example usage
save_images(main_train_images, '../../../train_folder/')
save_images(test_images, '../../../test_folder/')
save_images(test_images, '../../../validation_folder/')