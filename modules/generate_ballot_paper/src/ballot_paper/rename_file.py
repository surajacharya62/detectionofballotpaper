# import os
# import cv2


# file_path = '../../../datasets_stamp/invalid_newsets/'
# file_to_save = '../../../datasets_stamp/invalid_stamp/'

# list_of_images = []
# # images = os.listdir(file_path)

# if os.path.isdir(file_path):
#     for file in os.listdir(file_path):           
#         if not file or file.endswith(('.jpeg','png','.jpg')):
#             path = os.path.join(file_path, file)
#             img = cv2.imread(path)
#             list_of_images.append(img) 
#         else:
#             print("Do not have image file") 

# else:
#     print("Invalid directory path")

        
# for i, image in enumerate(list_of_images, 0):
#     cv2.imwrite(file_to_save + f'invalid_stamp_{i:04}.png',image) 




import cv2
import os
import numpy as np

file_path = '../../../datasets_stamp/invalid_stamp/'
file_to_save = '../../../datasets_stamp/valid_stamp/'

if os.path.isdir(file_path):
    list_of_images = []
    for file in os.listdir(file_path):
        if file.endswith(('.jpeg', '.png', '.jpg')):
            path = os.path.join(file_path, file)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Read with alpha channel if exists

            # Check if image has an alpha channel
            if img.shape[2] < 4:
                # Add alpha channel, setting all initial values to 255 (fully opaque)
                alpha_channel = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                img = np.concatenate((img, alpha_channel), axis=2)

            list_of_images.append(img)
        else:
            print("Do not have image file")
else:
    print("Invalid directory path")


# Save images with alpha channel
for i, image in enumerate(list_of_images, 0):
    save_path = os.path.join(file_to_save, f'valid_stamp_{i:04}.png')
    cv2.imwrite(save_path, image)



