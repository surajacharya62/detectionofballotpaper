import torchvision.transforms as transforms
from PIL import Image
import os
import cv2


def flip_image_horizontally_torchvision(image_path):
    # Define the horizontal flipping transform
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1)  # Set probability to 1 to always flip
    ])
    
    
    # Open the image
    image = Image.open(image_path)
    # flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
    
    # Apply the transform
    flipped_image = transform(image)
    # flipped_image.show()
    # Save the flipped image to a new file
    # flipped_image.save(save_path)
    return flipped_image

# Example usage:

def load_symbols1(folder_path):
    """    
    """
    
    all_party_symbols = []
    
    for folder in os.listdir(folder_path):
        sub_folder = os.path.join(folder_path, folder) 
        print('folder: ' + folder)
        symbols = []
       
        # sub_folder = folder_path + folder
        if os.path.isdir(sub_folder):   
            
            
            for i, filename in enumerate(os.listdir(sub_folder),150):
                # print(filename)
                if filename.lower().endswith(('.jepg','png','.jpg')):
                        image_path = os.path.join(sub_folder, filename)
                        symbol_name = filename.rpartition('_')[0]
                        flipped_image = flip_image_horizontally_torchvision(image_path)
                        save_path = os.path.join(sub_folder, f'{symbol_name}_{i:04}.png')
                        flipped_image.save(save_path)


         
                  
            
        else:
            print("invalid directory")

    # random.shuffle(all_party_symbols)    
   

image_path = '../../../datasets_symbol_train1/'
load_symbols1(image_path)





