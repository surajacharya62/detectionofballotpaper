from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
import numpy as np
import ast
import os

 # To get class labels and sorting it in order  
df = pd.read_csv(os.path.join('../../../training_set/set5/', 'annotations.csv'))
label_to_id = {label: i for i, label in enumerate(df['label'].unique())}
sorted_labels = sorted(label_to_id)
label_to_id = {label: i for i, label in enumerate(sorted_labels)}  

class YoloVisualize():   

    # Function to convert YOLO bounding box format to matplotlib rectangle format
    def yolo_to_matplotlib(self, image_width, image_height, bbox):
        x_center, y_center, w, h = bbox
        x1 = (x_center - w / 2) * image_width
        y1 = (y_center - h / 2) * image_height
        rect_w = w * image_width
        rect_h = h * image_height 
        return x1, y1, rect_w, rect_h

    def yolo_to_corners(self, img_w, img_h, bbox):
        
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
    

    def visualize_test_images(self, image_path, pred_labels_path):          
        
        file_name = 'yolo_predicted_labels.csv' 
        pred_labels = pd.read_csv(os.path.join(pred_labels_path, file_name))
        labels_df = pd.DataFrame(pred_labels)  
        test_files = []
        for filename in os.listdir(image_path):
            test_files.append(filename)
        # print(test_files)

        for i, (image_file) in enumerate(test_files):
            image = os.path.join(image_path, image_file)            
            image = Image.open(image)
            image_width, image_height = image.size
            fig, ax = plt.subplots(1)
            ax.imshow(image)             
            filtered_df = labels_df[labels_df['image_name'] == image_file]

            for index, row in filtered_df.iterrows():
                # print(row)
                box = ast.literal_eval(row['box_coord'])[0]
                xmin = box[0]
                ymin = box[1]
                xmax = box[2]
                ymax = box[3]
                box = xmin, ymin, xmax, ymax 
                # print(box)
                # Assuming box is already a list of [x_center, y_center, width, height]
                # box = row['box_coord']  # If it's a string, use ast.literal_eval(box)[0]
                
                # Convert YOLO format to corners
                xmin, ymin, xmax, ymax = self.yolo_to_corners(image_width, image_height, box)
                # print(xmin, ymin, xmax-xmin, ymax-ymin) 
                
                # Draw the bounding box
                rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            
                labels = {value: key for key, value in label_to_id.items()}
                label_id = int(row['class_id'])
                class_name = labels.get(label_id)
                ax.text(xmin, ymin, class_name, color="red", fontsize=8)

            plt.axis('off')  # Optional: Remove axes for cleaner visualization
            plt.savefig(f'../../../output/visualization/yolo/{image_file}1.png', bbox_inches='tight', pad_inches=0, dpi=300)           
            plt.close()


    # def visualized_normalized_pred_labels():
        
    #     ##------------------------------------Visualized Normalized Predicted labels-----------------------------------------
    #     image_path= '../../../testing_set/set4/test/image_0002.jpg'
    #     image = Image.open(image_path)
    #     image_width, image_height = image.size

    #     # Load YOLO bounding box details from .txt file
    #     # bbox_path = '../../../training_set/set5/yolov8annotations/image_0001.txt'
    #     dff = pd.read_excel('../../../yolo_files/pred_labels_normalized.xlsx')
    #     # with open(bbox_path, 'r') as f:
    #     #     bboxes = [list(map(float, line.split())) for line in f.readlines()]
    #     filtered_df = dff[dff['image_name'] == 'image_0002.jpg']
    #     class_id = 0
    #     # Create a matplotlib figure and axes
    #     fig, ax = plt.subplots(1)
    #     ax.imshow(image) 
    #     print(filtered_df)

    #     # Draw bounding boxes on the image
    #     for bbox in filtered_df.iterrows():
    #         ls1 = ast.literal_eval(bbox[1]['box_coord'])
    #         ls = bbox[1]['box_coord'][0]
    #         bb1, bb2, bb3, bb4 = ls1

    #         print(bb1)
    #         rect_params = self.yolo_to_matplotlib(image_width, image_height, ls1)
    #         # print(rect_params)
    #         rect = patches.Rectangle((rect_params[0], rect_params[1]), rect_params[2], rect_params[3],
    #                                 linewidth=1, edgecolor='r', facecolor='none')
    #         # rect = patches.Rectangle((bbox[1], bbox[2]), bbox[3], bbox[4],
    #         #                          linewidth=1, edgecolor='r', facecolor='none')
            
    #         ax.add_patch(rect)
    #         labels = {value: key for key, value in label_to_id.items()}
    #         label_id = int(bbox[1]['class_id'])
    #         class_name = labels.get(label_id)
    #         ax.text(rect_params[0], rect_params[1], class_name, color="red", fontsize=8)

    #     plt.show() 




# #----------------------------------------------------------------------------------------------------------------------------------------
# ###----------------------------------------Visualized Normalized training set images-----------------------------------------------------
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
#     # bbox = bbox[1],bbox[2],bbox[3],bbox[]
    
#     rect_params = yolo_to_matplotlib(image_width, image_height, bbox)
#     # print(rect_params)
#     rect = patches.Rectangle((rect_params[0], rect_params[1]), rect_params[2], rect_params[3],
#                              linewidth=1, edgecolor='r', facecolor='none')
#     # rect = patches.Rectangle((bbox[1], bbox[2]), bbox[3], bbox[4],
#     #                          linewidth=1, edgecolor='r', facecolor='none')
    
#     ax.add_patch(rect)
#     labels = {value: key for key, value in label_to_id.items()}
#     label_id = int(bbox[0])
#     class_name = labels.get(label_id)
#     ax.text(rect_params[0], rect_params[1], class_name, color="red", fontsize=8)

# plt.show() 










