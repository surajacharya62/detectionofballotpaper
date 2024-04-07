from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
import numpy as np
import ast

# Function to convert YOLO bounding box format to matplotlib rectangle format
def yolo_to_matplotlib(image_width, image_height, bbox):
    x_center, y_center, w, h = bbox
    x1 = (x_center - w / 2) * image_width
    y1 = (y_center - h / 2) * image_height
    rect_w = w * image_width
    rect_h = h * image_height 
    return x1, y1, rect_w, rect_h


image_path_test = '../../../training_set/set5/train/image_0002.jpg'
# image_path_test = '../../../testing_set/set4/test/image_0002.jpg'
image_path_ann = '../../../training_set/set5/annotations.csv'
# csv = '../../../ai_detection_model/src/model/yolo_pred_labels.csv'
# xlsx = '../../../ai_detection_model/src/model/yolo_truth_annotations.xlsx'
image = Image.open(image_path_test)
image_width, image_height = image.size
pred_labels = pd.read_csv(image_path_ann)
# training_labels = pd.read_csv(image_path_ann)
data_yolo = pd.DataFrame(pred_labels)
filtered_df = data_yolo[data_yolo['image_id'] == 'image_0002.jpg']
# print(filtered_df)


def yolo_to_corners(img_w, img_h, bbox):
    """
    Convert from YOLO format (normalized) to corners (xmin, ymin, xmax, ymax) in pixel coordinates.
    
    Args:
    - img_w: Image width in pixels.
    - img_h: Image height in pixels.
    - bbox: Bounding box in YOLO format [x_center, y_center, width, height], normalized.
    
    Returns:
    - A tuple (xmin, ymin, xmax, ymax) in pixel coordinates.
    """
    # De-normalize coordinates
    # x_center, y_center, w, h = bbox
    # x_center = x_center * img_w
    # y_center = y_center * img_h
    # w = w * img_w
    # h = h * img_h
    
    # Calculate corners
    x_center, y_center, w, h = bbox
    xmin = x_center - (w / 2)
    ymin = y_center - (h / 2)
    xmax = x_center + (w / 2)
    ymax = y_center + (h / 2)
    return [xmin, ymin, xmax, ymax]



fig, ax = plt.subplots(1)
ax.imshow(image)
# # # # # Iterate over the filtered DataFrame and draw each bounding box
for index, row in filtered_df.iterrows():
    # box = ast.literal_eval(row['box_coord'])[0] 
    xmin = row['x1']
    ymin = row['y1']
    xmax = row['x2']
    ymax = row['y2']
    box = xmin,ymin,xmax,ymax
    # print(row)
    # print(row[0])
    # Convert YOLO format to matplotlib format
    rect_params = yolo_to_matplotlib(image_width, image_height, box)
    # print(rect_params[0])
    
    # Create a rectangle patch
    # rect = patches.Rectangle((rect_params[0], rect_params[1]), rect_params[2], rect_params[3],
    #                          linewidth=1, edgecolor='r', facecolor='none')
    # # rect = patches.Rectangle((box[0], box[1]), box[2], box[3],
    #                          linewidth=1, edgecolor='r', facecolor='none')
    rect = patches.Rectangle((row['x1'], row['y1']), row['x2'], row['y2'],
                             linewidth=1, edgecolor='r', facecolor='none')
      
    # # Add the patch to the Axes
    # ax.add_patch(rect)

plt.show()

##---------------------------------------For predicted test images
# for index, row in filtered_df.iterrows():
#     # box = ast.literal_eval(row['box_coord'])[0] 
#     xmin = row['xmin']
#     ymin = row['ymin']
#     xmax = row['xmax']
#     ymax = row['ymax']
#     box = xmin,ymin,xmax,ymax 
#     # print(box)
#     # Assuming box is already a list of [x_center, y_center, width, height]
#     # box = row['box_coord']  # If it's a string, use ast.literal_eval(box)[0]
    
#     # Convert YOLO format to corners
#     xmin, ymin, xmax, ymax = yolo_to_corners(image_width, image_height, box)
#     print(xmin,ymin,xmax-xmin,ymax-ymin)
    
#     # Draw the bounding box
#     rect = patches.Rectangle((xmin, ymin), xmax, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

# plt.show()



###----------------------------------------for training set images
# image_path= '../../../training_set/set5/train/image_0001.jpg'
# image = Image.open(image_path)
# image_width, image_height = image.size

# # Load YOLO bounding box details from .txt file
# bbox_path = '../../../training_set/set5/yolov8annotations/image_0001.txt'
# with open(bbox_path, 'r') as f:
#     bboxes = [list(map(float, line.split())) for line in f.readlines()]

# class_id = 0
# # Create a matplotlib figure and axes
# fig, ax = plt.subplots(1)
# ax.imshow(image) 

# # Draw bounding boxes on the image
# for bbox in bboxes:
#     print(bbox)
    
#     # rect_params = yolo_to_matplotlib(image_width, image_height, bbox)
#     # print(rect_params)
#     # rect = patches.Rectangle((rect_params[0], rect_params[1]), rect_params[2], rect_params[3],
#     #                          linewidth=1, edgecolor='r', facecolor='none')
#     rect = patches.Rectangle((bbox[1], bbox[2]), bbox[3], bbox[4],
#                              linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)

# plt.show() 



# Assuming you want to visualize boxes of class_id_to_visualize only
# class_id_to_visualize = 20  # Example class ID

# for bbox in bboxes:
#     # print(bbox)
#     # class_id = int(bbox[0])
#     # print(class_id)
#     # if class_id == class_id_to_visualize:
#     rect_params = yolo_to_matplotlib(image_width, image_height, bbox[1:])
#     rect = patches.Rectangle((rect_params[0], rect_params[1]), rect_params[2], rect_params[3],
#                                 linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)


# plt.show()





