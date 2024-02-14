import os
import csv
import random
from PIL import Image,ImageDraw
import cv2

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
                                label = os.path.splitext(filename)[0] 
                                # print(file_name)                                
                                # split_list = file_name.split('_') # Use filename without extension as label
                                # label = '_'.join(split_list[:-1])
                                # print(label)
                                img_path = os.path.join(sub_dir, filename)
                                symbols[label] = Image.open(img_path).convert("RGBA")
                                # print(symbols)
                              
                        # if symbols:
                            
                        #     image = random.choice(symbols)
                        #     all_party_symbols.append(image)        
                        #     symbols.clear() 
        else:
            pass
            # print("invalid directory path")
    # print(len(symbols))
    return symbols

def does_overlap(new_box, existing_boxes, threshold=0.1):
    """
    Check if the new_box overlaps with any of the existing_boxes more than the specified threshold.
    Threshold is a fraction of the new box's area.
    """
    new_x1, new_y1, new_x2, new_y2 = new_box
    new_area = (new_x2 - new_x1) * (new_y2 - new_y1)

    for box in existing_boxes:
        x1, y1, x2, y2 = box

        # Calculate intersection area
        inter_x1 = max(new_x1, x1)
        inter_y1 = max(new_y1, y1)
        inter_x2 = min(new_x2, x2)
        inter_y2 = min(new_y2, y2)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:  # Boxes overlap
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            if inter_area / new_area > threshold:
                return True  # Overlaps more than the threshold
    return False

def return_stamp(stamp_path):
    stamp = {}
    if os.path.isdir(stamp_path):
        for filename in os.listdir(stamp_path):
            # print(filename)
            if filename.endswith(('.jepg','png','.jpg')):
                    # img = cv2.imread(os.path.join(sub_dir, filename))
                    # symbols.append(img) 
                    label = os.path.splitext(filename)[0] 
                    # print(label)                                
                    # split_list = file_name.split('_') # Use filename without extension as label
                    # label = '_'.join(split_list[:-1])
                    img_path = os.path.join(stamp_path, filename)
                    stamp[label] = Image.open(img_path).convert("RGBA")
    else:
        print("Invalid directory")
    
    return stamp

def generate_dataset(num_images, symbols, output_dir,image_size=(700, 700), max_symbols=20):
    annotations_path = os.path.join(output_dir, 'annotations.csv')
    with open(annotations_path, 'w', newline='') as csvfile:
        fieldnames = ['image_id', 'x1', 'y1', 'x2', 'y2', 'label']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        # stamp = return_stamp(stamp_path) 
        
        for img_id in range(1, num_images + 1):
            img = Image.new('RGB', image_size, (255, 255, 255))
            draw = ImageDraw.Draw(img)
            existing_boxes = []  # Track bounding boxes of placed symbols
           
            for _ in range(max_symbols): 
                # print(label,image)
                
                label, symbol_img = random.choice(list(symbols.items()))
                attempts = 0
                placed = False
                while not placed and attempts < 50:  # Try to place the symbol, avoiding excessive overlap
                    symbol_img_resized = symbol_img.resize((random.randint(70, 200), random.randint(70, 200)), Image.LANCZOS)
                    x1, y1 = random.randint(0, image_size[0] - symbol_img_resized.width), random.randint(0, image_size[1] - symbol_img_resized.height)
                    new_box = (x1, y1, x1 + symbol_img_resized.width, y1 + symbol_img_resized.height)

                    if not does_overlap(new_box, existing_boxes): 
                        img.paste(symbol_img_resized, (x1, y1), symbol_img_resized)
                        # draw.rectangle(((x1, y1), (x1 + symbol_img_resized.width, y1 + symbol_img_resized.height)), outline="red")
                        existing_boxes.append(new_box)
                        writer.writerow({
                            'image_id': f'image_{img_id}.png',
                            'x1': x1,
                            'y1': y1,
                            'x2': x1 + symbol_img_resized.width,
                            'y2': y1 + symbol_img_resized.height,
                            'label': label
                        })
                        placed = True
                       
                    attempts += 1

            img_path = os.path.join(output_dir, f'image_{img_id}.png')
            img.save(img_path)


# Example usage
folder_path = '../../../datasets1/' 
# stamp_path = '../../../datasets/stamp/train'
output_dir = '../../../datasets1/annotatedataset' 


# Load symbols
symbols = load_symbols(folder_path)
print(len(symbols))
# print(symbols)
# print(symbols)

# Generate dataset
num_images = 500

generate_dataset(num_images, symbols, output_dir)
