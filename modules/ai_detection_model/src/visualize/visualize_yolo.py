from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import cv2
import numpy as np
import ast
import os
import torch
import random

 # To get class labels and sorting it in order  
df = pd.read_csv(os.path.join('../../../../training_set/set7/', 'annotations.csv'))
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
    
    def nms(self,bboxes, iou_threshold, threshold=0.4, box_format="corners"):
        """
        Does Non Max Suppression given bboxes

        Parameters:
            bboxes (list): list of lists containing all bboxes with each bboxes
            specified as [class_pred, prob_score, x1, y1, x2, y2]
            iou_threshold (float): threshold where predicted bboxes is correct
            threshold (float): threshold to remove predicted bboxes (independent of IoU) 
            box_format (str): "midpoint" or "corners" used to specify bboxes

        Returns:
            list: bboxes after performing NMS given a specific IoU threshold
        """

        assert type(bboxes) == list

        bboxes = [box for box in bboxes if box[1] > threshold]
        bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
        bboxes_after_nms = []

        while bboxes:
            chosen_box = bboxes.pop(0)

            bboxes = [ 
                box
                for box in bboxes
                if box[0] != chosen_box[0]
                or self.intersection_over_union(
                    torch.tensor(chosen_box[2:]),
                    torch.tensor(box[2:]),
                    box_format=box_format,
                )
                < iou_threshold
            ]

            bboxes_after_nms.append(chosen_box)

        return bboxes_after_nms
    
    def intersection_over_union(self,boxes_preds, boxes_labels, box_format="corners"):
        """
        Calculates intersection over union

        Parameters:
            boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
            boxes_labels (tensor): Correct Labels of Boxes (BATCH_SIZE, 4)
            box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

        Returns:
            tensor: Intersection over union for all examples
        """

        # Slicing idx:idx+1 in order to keep tensor dimensionality
        # Doing ... in indexing if there would be additional dimensions
        # Like for Yolo algorithm which would have (N, S, S, 4) in shape
        if box_format == "midpoint":
            box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
            box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
            box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
            box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
            box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
            box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
            box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
            box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

        elif box_format == "corners":
            box1_x1 = boxes_preds[..., 0:1]
            box1_y1 = boxes_preds[..., 1:2]
            box1_x2 = boxes_preds[..., 2:3]
            box1_y2 = boxes_preds[..., 3:4]
            box2_x1 = boxes_labels[..., 0:1]
            box2_y1 = boxes_labels[..., 1:2]
            box2_x2 = boxes_labels[..., 2:3]
            box2_y2 = boxes_labels[..., 3:4]

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)

        # Need clamp(0) in case they do not intersect, then we want intersection to be 0
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)
    

    
    def apply_nms(self,original_boxes, original_scores, iou_threshold=0.55):
        # Apply NMS and return indices of kept boxes
        # original_scores = original_scores.float()
        keep = torch.ops.torchvision.nms(original_boxes, original_scores, iou_threshold)
        # print(keep)
        return keep

    
    
    def pred_normalize(self, row):

        box = ast.literal_eval(row['box_coord'])[0]
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]      
      
        normalized_boxes = xmin, ymin, xmax, ymax
        row['box_coord'] = list(normalized_boxes) 
        return row

    def random_color(self):
    # Generate random RGB components
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        
        # Convert RGB values to a hex color code
        return f"#{r:02x}{g:02x}{b:02x}" 

    def visualize_test_images(self, image_path, pred_labels_path):          
        
        file_name = 'yolo_predicted_labels.csv' 
        pred_labels = pd.read_csv(os.path.join(pred_labels_path, file_name))
        labels_df = pd.DataFrame(pred_labels)  
        test_files = []
        for filename in os.listdir(image_path):
            test_files.append(filename)

        # print(test_files)
        
        test_set_path = '../../../../testing_set/set6/'
        # yolo_files_path = '../../../yolo_files/'
        test = 'test'

        df = pd.read_csv(os.path.join(test_set_path, 'annotations.csv'))
        label_to_id = {label: i for i, label in enumerate(df['label'].unique())}

        sorted_labels = sorted(label_to_id)

        label_id = {label: i for i, label in enumerate(sorted_labels)}

        for i, (image_file) in enumerate(test_files):
            image = os.path.join(image_path, image_file)            
            image = Image.open(image)
            image_width, image_height = image.size
            fig, ax = plt.subplots(1)
            ax.imshow(image)             
            df = pred_labels[pred_labels['image_name'] == image_file]

            df = df.apply(self.pred_normalize, axis=1) 
            boxes = df['box_coord'].values.tolist()
            labels = df['class_id'].tolist()
            scores = df['score'].tolist()

            boxes_array = np.array(boxes, dtype='float32')
            boxes1 = torch.tensor(boxes_array)
            scores1 = torch.tensor(scores)
            # labels = [label_id[label] for label in labels]
            labels1 = torch.tensor(labels) 

            combined_data = [[labels[i], scores[i]] + boxes[i] for i in range(len(boxes))]
            # print(combined_data)
            indices = self.nms(combined_data, 0.5)   
            # print(indices)

            labels = [item[0] for item in indices]
            scores = [item[1] for item in indices]
            boxes = [item[2:] for item in indices]
            # final_boxes, final_scores, final_labels = self.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )      

            # final_boxes, final_scores, final_labels  =  boxes1_sorted[indices], scores1_sorted[indices], labels1_sorted[indices] 
            # print(len(final_boxes) , len(final_scores), len(final_labels))  

            for box, score, pred_label in zip(boxes, scores, labels):

            # for index, row in filtered_df.iterrows():
            #     # print(row)
            #     box = ast.literal_eval(row['box_coord'])[0]
            #     xmin = box[0]
            #     ymin = box[1]
            #     xmax = box[2]
            #     ymax = box[3]
            #     box = xmin, ymin, xmax, ymax 
                # print(box)
                # Assuming box is already a list of [x_center, y_center, width, height]
                # box = row['box_coord']  # If it's a string, use ast.literal_eval(box)[0]
                
                # Convert YOLO format to corners
                xmin, ymin, xmax, ymax = self.yolo_to_corners(image_width, image_height, box)
                # print(xmin, ymin, xmax-xmin, ymax-ymin) 
                
                # Draw the bounding box
                
                

               

            
                labels = {value: key for key, value in label_id.items()}
                # print(label_id)
                # print(labels)
                # pred_label = pred_label.cpu().numpy()
                pred_label = int(pred_label)

                class_name = labels.get(pred_label, "unkown")
                # print(class_name)
                if pred_label == 39 or pred_label == 14:

                    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='blue', facecolor='none')
                    ax.text(xmin, ymin, class_name, color="blue", fontsize=8)
                    ax.add_patch(rect) 
                else:
                    rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
                    ax.text(xmin, ymin, class_name, color=self.random_color(), fontsize=random.randint(5, 8))
                    ax.add_patch(rect) 
               

            plt.axis('off')  # Optional: Remove axes for cleaner visualization
            plt.savefig(f'../../../../output/visualization/yolo/{image_file}1.png', bbox_inches='tight', pad_inches=0, dpi=300)           
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










