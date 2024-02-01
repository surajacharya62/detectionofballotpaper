import os
import cv2


file_path = "../../../datasets/ballot_datasets/train1/"
file_to_save = "../../../datasets/ballot_datasets/train/"

list_of_images = []
# images = os.listdir(file_path)

if os.path.isdir(file_path):
    for file in os.listdir(file_path):
           
        if not file or file.endswith(('.jpeg','png','.jpg')):
            path = os.path.join(file_path, file)
            img = cv2.imread(path)
            list_of_images.append(img) 
        else:
            print("Do not have image file")

else:
    print("Invalid directory path")
# for i in range(1, 11):
#     filename = f"{i:03}.txt"  # Formats the number with 3 digits, padded with zeros
#     print(filename)
        
for i, image in enumerate(list_of_images, 1):
    cv2.imwrite(file_to_save + f'invalid_{i:04}.png',image)