import os
import csv
import random
from PIL import Image,ImageDraw
import cv2
# from PIL import Image, ImageDraw, ImageResampling

def load_symbols(folder_path):
    """
    
    """
    
    all_party_symbols = []
    symbols = {}
    for folder in os.listdir(folder_path):
        sub_folder1 = os.path.join(folder_path, folder) 
        # print(sub_folder1)
        # sub_folder = folder_path + folder 
        if os.path.isdir(sub_folder1): 
            
            for folder in os.listdir(sub_folder1):
                sub_dir = os.path.join(sub_folder1, folder)
                # print(sub_dir)
                
                if os.path.isdir(sub_dir):
                    for filename in os.listdir(sub_dir):
                        # print(filename)
                        if filename.endswith(('.jepg','png','.jpg')):
                                # img = cv2.imread(os.path.join(sub_dir, filename))
                                # symbols.append(img) 
                                file_name = os.path.splitext(filename)[0] 
                                # print(label)                                
                                split_list = file_name.split('_') # Use filename without extension as label
                                label = '_'.join(split_list[:-1])
                                img_path = os.path.join(sub_dir, filename)
                                symbols[label] = Image.open(img_path).convert("RGBA")
                                
                        # if symbols:
                            
                        #     image = random.choice(symbols)
                        #     all_party_symbols.append(image)        
                        #     symbols.clear() 
        else:
            pass
            # print("invalid directory path")
    
    return symbols

def generate_dataset(num_images, symbols, output_dir, image_size=(700, 700), max_symbols=10):
    annotations_path = os.path.join(output_dir, 'annotations.csv')
    with open(annotations_path, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'x1', 'y1', 'x2', 'y2', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for img_id in range(1, num_images + 1):
            img = Image.new('RGB', image_size, (255, 255, 255))
            draw = ImageDraw.Draw(img)
            num_symbols_in_img = random.randint(1, max_symbols)

            for _ in range(20):
                label, symbol_img = random.choice(list(symbols.items()))
                symbol_img = symbol_img.resize((random.randint(70, 200), random.randint(70, 200)),Image.LANCZOS)
                # symbol_img = cv2.resize(symbol_img,(random.randint(100, 300), random.randint(70, 200)),interpolation=cv2.INTER_LINEAR)
                x1, y1 = random.randint(0, image_size[0] - symbol_img.width), random.randint(0, image_size[1] - symbol_img.height)
                img.paste(symbol_img, (x1, y1), symbol_img)
                draw.rectangle(((x1, y1), (x1 + symbol_img.width, y1 + symbol_img.height)), outline="red")

                writer.writerow({
                    'image_id': f'image_{img_id}.png',
                    'x1': x1,
                    'y1': y1,
                    'x2': x1 + symbol_img.width,
                    'y2': y1 + symbol_img.height,
                    'label': label
                })

            img_path = os.path.join(output_dir, f'image_{img_id}.png')
            img.save(img_path)

# Example usage
folder_path = '../../../datasets/' 
symbols_dir = '/path/to/your/symbols'  # Update this path to your symbols directory
output_dir = '../../../datasets/annotatedataset'   # Update this to your desired output directory


# Load symbols
symbols = load_symbols(folder_path)
# print(symbols)

# Generate dataset
num_images = 50

generate_dataset(num_images, symbols, output_dir)
