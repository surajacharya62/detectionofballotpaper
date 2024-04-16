import cv2
import os
import random
from PIL import Image, ImageDraw, ImageFont
import math
from sklearn.model_selection import train_test_split

class SplitTrainTestValidation():

    def load_symbols(self, folder_path):

        for folder in os.listdir(folder_path):
            print("folder: " + folder)
            symbols = []
            file = ''
            sub_folder = os.path.join(folder_path, folder) 
        
            # sub_folder = folder_path + folder
            if os.path.isdir(sub_folder):
                
                for filename in os.listdir(sub_folder):      
                                        
                    if filename.lower().endswith(('.jepg','png','.jpg')):
                            img = cv2.imread(os.path.join(sub_folder, filename))
                            symbol_name = os.path.splitext(filename)[0]                             
                            symbols.append((img, symbol_name))

                    # if symbols:                            
                    #     image = random.choice(symbols)
                    #     all_party_symbols.append(image)        
                    #     symbols.clear() 
                # print(symbols)
                file = symbol_name.rpartition('_')[0]   
                print(file)             
                self.split_symbol(symbols, file)
                # break     
                
            else:
                print("invalid directory")
    

    def split_symbol(self,symbols, file):
        # print(symbols,file)
        train_images, test_images = train_test_split(symbols, test_size=0.2, random_state=42)
        main_train_images, val_images = train_test_split(train_images, test_size=0.2, random_state=42)
        self.save_images(
                            main_train_images, 
                            val_images, 
                            test_images,
                            file,
                            train_folder='train_folder1',
                            validation_folder='validation_folder1',
                            test_folder='test_folder1'
                         )
  
      
    def save_images(self,train_images, 
                            val_images, 
                            test_images,
                            folder_name,
                            train_folder,
                            validation_folder,
                            test_folder):
    
        train_path = '../../../train_folder/'  
        if not os.path.exists( '../../../train_folder/' + folder_name):
            os.makedirs( train_path + folder_name)
        random.shuffle(train_images)

        for image_path in train_images:
            # Define the destination path for the image
            image, name = image_path
            # print(image)
            # print(name)
            cv2.imwrite(train_path + folder_name + "/" + f'{name}.jpg', image) 
            # destination_path = os.path.join(target_folder, os.path.basename(image_path))
            # Copy the image to the target folder
            # shutil.copy(image_path, destination_path)
        
        validation_path = '../../../validation_folder/'  
        if not os.path.exists('../../../validation_folder/' + folder_name):
            os.makedirs( validation_path + folder_name)
        random.shuffle(val_images)

        for image_path in val_images:
            # Define the destination path for the image
            image, name = image_path
            # print(image)
            # print(name)
            cv2.imwrite(validation_path + folder_name + "/" + f'{name}.jpg', image) 
            # destination_path = os.path.join(target_folder, os.path.basename(image_path))
            # Copy the image to the target folder
            # shutil.copy(image_path, destination_path)
        
        test_path = '../../../test_folder/'  
        if not os.path.exists( '../../../test_folder/' + folder_name):
            os.makedirs( test_path + folder_name)
        random.shuffle(test_images)

        for image_path in test_images:
            # Define the destination path for the image
            image, name = image_path
            # print(image)
            # print(name)
            cv2.imwrite(test_path + folder_name + "/" + f'{name}.jpg', image) 
            # destination_path = os.path.join(target_folder, os.path.basename(image_path))
            # Copy the image to the target folder
            # shutil.copy(image_path, destination_path)
        
        
        
        
        
   


symbols_test_path = '../../../datasets_symbol_train/'

obj_split = SplitTrainTestValidation()
symbols = obj_split.load_symbols(symbols_test_path)



# print(len(train_images))
# print(len(main_train_images))
# print(len(test_images))
# print(len(val_images))
# import shutilsp 
# import os



# # Example usage
# save_images(main_train_images, '../../../train_folder/')
# save_images(test_images, '../../../test_folder/')
# save_images(test_images, '../../../validation_folder/')