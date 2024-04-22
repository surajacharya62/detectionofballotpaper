import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore', 'Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers)')


class VisualizePrediction():

    def __init__(self):
         pass
    
    def visualize_predicted_images(self, test_set, predicted_labels, label_to_id):
            
        for i, (test_data, label) in enumerate(zip(test_set, predicted_labels)):

            image = test_data[0]                       
            img_np = image.permute(1, 2, 0).numpy()         
            id_to_label = {value: key for key, value in label_to_id.items()}
            
            fig, ax = plt.subplots(1) 
            ax.imshow(img_np) 
            
            # actual_labels = test_data[1]['labels']  
            # actual_bounding_box = test_data[1]['boxes']
            # # image_name1 = test_data[2]

            boxes = label[0]['boxes'] 
            labels = label[0]['labels']
            scores = label[0]['scores']
            image_name = label[2]   

            indices = self.apply_nms(boxes, scores)   
            # final_boxes, final_scores, final_labels = self.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )      

            final_boxes, final_scores, final_labels  =  boxes[indices], scores[indices], labels[indices]             

            for box, score, pred_label in zip(final_boxes, final_scores, final_labels):                
                
                box1 = box.cpu().numpy() 
                label_id = int(f"{pred_label}")
                class_name = id_to_label.get(label_id, 'Unknown') 
                x1, y1, x2, y2 = box1 

                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect) 

                # Add label text
                # label_text = f"{label}"  # Replace `label` with a mapping to the actual class name if you have one
                ax.text(x1, y1, class_name, color='red', fontsize=4)             
         
            plt.axis('off')  # Optional: Remove axes for cleaner visualization
            plt.savefig(f'../../../output/visualization/faster_rcnn/{image_name}1.png', bbox_inches='tight', pad_inches=0, dpi=300)           
            plt.close()
            
                    
          
    def visualize_train_set(self, train_labels, label_to_id):
           
            image = train_labels[0]
            boxes = train_labels[3]['boxes'] 
            labels = train_labels[3]['labels']
            # scores = train_labels[1]['scores']
            image_name = train_labels[2]
            img_np = image.permute(1, 2, 0).numpy()         
            id_to_label = {value: key for key, value in label_to_id.items()}
            fig, ax = plt.subplots(1) 
            ax.imshow(img_np)                       
                         
                      

            # indices = self.apply_nms(boxes, scores)   
            # final_boxes, final_scores, final_labels = self.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )           

            for box, label in zip(boxes, labels):

                # score1 = score.cpu().numpy()                      
                box1 = box.cpu().numpy()
                label_id = int(f"{label}")
                class_name = id_to_label.get(label_id, 'Unknown')
                x1, y1, x2, y2 = box1

                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                # Add label text
                # label_text = f"{label}"  # Replace `label` with a mapping to the actual class name if you have one
                ax.text(x1, y1, class_name, color='blue', fontsize=7) 
            
            plt.axis('off')  # Optional: Remove axes for cleaner visualization
            # plt.savefig(f'../../../outputdir/{image_name}.png', bbox_inches='tight', pad_inches=0, dpi=300)  
            plt.show()         
            plt.close()
                
    
    def nms(self,bboxes, iou_threshold, threshold, box_format="corners"):
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
    
    def intersection_over_union(self,boxes_preds, boxes_labels, box_format="midpoint"):
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
    

    
    def apply_nms(self,original_boxes, original_scores, iou_threshold=0.5):
        # Apply NMS and return indices of kept boxes
        # original_scores = original_scores.float()
        keep = torch.ops.torchvision.nms(original_boxes, original_scores, iou_threshold)
        # print(keep)
        return keep


    def select_highest_confidence_per_class(self, boxes, scores, labels, keep_indices):
        # Filter boxes, labels, and scores based on NMS keep indices
        filtered_boxes = boxes[keep_indices]
        filtered_labels = labels[keep_indices]
        filtered_scores = scores[keep_indices]
        print(filtered_labels)

        unique_labels = filtered_labels.unique()
        # print('unique' + str(len(unique_labels)))
        

        final_indices = []
        for label in unique_labels:
            # Get indices of all occurrences of this label
            label_indices = (filtered_labels == label).nonzero().view(-1)

            # Find the index with the highest score among these
            highest_score_index = label_indices[filtered_scores[label_indices].argmax()]
            final_indices.append(highest_score_index)
        # print(boxes.device)
        # Convert to a tensor
        final_indices = torch.tensor(final_indices, device=boxes.device)

        # Select the final boxes, labels, and scores
        final_boxes = filtered_boxes[final_indices]
        final_labels = filtered_labels[final_indices]
        final_scores = filtered_scores[final_indices]
        
        return final_boxes, final_scores, final_labels
       