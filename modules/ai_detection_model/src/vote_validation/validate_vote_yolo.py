
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize.visualize_prediction import VisualizePrediction
import pandas as pd
from collections import Counter
import torch
import ast

visualize =  VisualizePrediction()

class ValidateVote():
    def __init__(self):
        pass

    def reconstruct_grid_cells(self, margins, ballot_size, symbol_size, rows, columns):
        mt, mb, ml, mr = margins
        ballot_width, ballot_height = ballot_size
        symbol_width, symbol_height = symbol_size

        # Calculate cell size based on the symbol size being half the width of the cell
        cell_width = symbol_width * 2  # Ensure each cell is double the width of the symbol
        cell_height = symbol_height    # Height of the cell matches the height of the symbol

        # Initialize grid cells list
        grid_cells = []

        # Calculate the starting y-coordinate of the grid
        header_box_bottom = mt
        grid_start_y = header_box_bottom  # Additional offset for the header box

        # Calculate the starting x-coordinate of the grid
        grid_start_x = ml

        # Generate grid cells based on calculated dimensions
        for row_idx in range(rows):
            for col_idx in range(columns):
                x1 = grid_start_x + col_idx * cell_width
                y1 = grid_start_y + row_idx * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height
                grid_cells.append((x1, y1, x2, y2))

        return grid_cells
 

    def is_stamp_valid(self, stamp_box, grid_cells, tolerance=10):
        # for cell_box in grid_cells:
        #     # Check if stamp box is entirely inside the grid cell box
        #     if (stamp_box[0] >= cell_box[0] and stamp_box[1] >= cell_box[1] and
        #         stamp_box[2] <= cell_box[2] and stamp_box[3] <= cell_box[3]):
        #         return True  # Stamp is valid
        # return False  # Stamp is not valid

        for cell_box in grid_cells:
            # Expand the cell box by the tolerance value
            adjusted_cell_box = (
                cell_box[0] - tolerance,  # left
                cell_box[1] - tolerance,  # top
                cell_box[2] + tolerance,  # right
                cell_box[3] + tolerance   # bottom
            )
            if (stamp_box[0] >= adjusted_cell_box[0] and
                stamp_box[1] >= adjusted_cell_box[1] and
                stamp_box[2] <= adjusted_cell_box[2] and
                stamp_box[3] <= adjusted_cell_box[3]):
                return True
        return False
    
    def nms(self,bboxes, iou_threshold, threshold=0.3, box_format="midpoint"):
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

    def predicted_images(self, test_images, pred_labels, label_to_id):   

        margins = (1560, 300, 200, 200)  # top, bottom, left, right margins
        ballot_size = (2668, 3413)  # width, height of the ballot paper
        symbol_size = (189, 189)  # width, height of the symbols
        rows = 7  # Number of symbol rows
        columns = 6  # Number of symbol columns  
        grid_cells = self.reconstruct_grid_cells(margins, ballot_size, symbol_size, rows, columns)  
        id_to_label = {value: key for key, value in label_to_id.items()}
       
        valid_stamp_id = 39
        invalid_stamp_id = 14
        symbols_class = []

       

        

        for i, (test_data) in enumerate(zip(test_images)):   
            # Convert tensor image to numpy array
            # img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            # print(test_data[0])
            img = test_data[0][0]  
            image_name = test_data[0][2] 
            actual_labels = test_data[0][3]['labels'].cpu().numpy()
            actual_bboxes = test_data[0][3]['boxes'].cpu().numpy()
            true_labels = zip(actual_labels,actual_bboxes)              
           
            img_np = img.permute(1, 2, 0).numpy() 
            # img_np = np.clip(img_np, 0, 1)  #Ensure the image array is between 0 and 1
            
            fig, ax = plt.subplots(1)
            ax.imshow(img_np) 
            transformed_data = [] 
           
            df = pred_labels[pred_labels['image_name'] == image_name]
            
            # boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            # labels = df['label'].tolist()
            # scores = df['score'].tolist()
            combined_data = []

            for _, row in df.iterrows():
                label = row['label']  # This should be a string.
                score = row['score']  # This should be a float.
                bbox = [row['xmin'], row['ymin'], row['xmax'], row['ymax']]  # These should be floats.

                # Append the combined information to the combined_data list
                combined_data.append([label, score] + bbox)

            # boxes = df['box_coord']
            # labels = df['label']
            # scores = df['score']
            
            # transformed_data.append({
            #     'boxes': boxes,
            #     'labels': labels,
            #     'scores': scores
            # }) 
            # print(img, image_name, df)

            # boxes = transformed_data['boxes'] 
            # labels = transformed_data['labels']
            # scores = transformed_data['scores']
            # # image_name = prediction[2] 

            # boxes_tolist = ast.literal_eval(boxes)[1]
            # boxes = torch.tensor(boxes, dtype=torch.float32)
            # scores = torch.tensor(scores)
            # labels = [label_to_id[label] for label in labels]
            # labels = torch.tensor(labels) 
            
            # combined_data = [[labels[i], scores[i]] + boxes[i] for i in range(len(boxes))]
            # print(combined_data)
            indices = self.nms(combined_data, 0.45)   
            # print(len(indices))

            labels = [item[0] for item in indices]
            scores = [item[1] for item in indices]
            boxes = [item[2:] for item in indices]
            # indices = self.apply_nms(boxes, scores)   
            # final_boxes, final_scores, final_labels  =  boxes[indices], scores[indices], labels[indices] 
            # final_boxes, final_scores, final_labels = visualize.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )  
                     
            # stamp_count = final_labels.count(stamp_id)                                      
            
            for box, score, pred_label in zip(boxes, scores, labels):
                               
                bounding_box = box                
                pred_label = label_to_id.get(pred_label, "unkown")
                label_id = int(pred_label)
                label_ids = [label_to_id.get(label, "unknown") for label in labels]
                counts = Counter(label_ids)
                numbers_to_check = [14, 39]       
                total_count = sum(counts[num] for num in numbers_to_check)

                if label_id == valid_stamp_id:    
                    # print("stamp")                    
                    if self.is_stamp_valid(bounding_box, grid_cells):
                          # print("valid_stamp")
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        
                        label_id = int(pred_label)
                        class_name = id_to_label.get(label_id, 'Unknown')  # Replace `label` with a mapping to the actual class name if you have one
                        ax.text(x1, y1, class_name, color='blue', fontsize=12)
                    
                        is_valid_symbol, symbol_label, symbol_box, closet_distance = self.is_stamp_for_symbol(bounding_box, zip(boxes, labels, scores), image_name)
                        # print(is_valid_symbol)
                        if is_valid_symbol:
                            # print("vaid symbol")
                            symbol_label = label_to_id.get(symbol_label, "unkown")
                            label_idx = int(symbol_label)
                            class_name1 = id_to_label.get(label_idx, 'Unknown')
                            x11, y11, x21, y21 = symbol_box
                            
                            stamp_box = [(label, bbox) for label, bbox in zip(actual_labels, actual_bboxes) if label == label_id]
                            # print(filtered_labels_and_bboxes, 'filterboxes')
                            # true_label, true_box = filtered_labels_and_bboxes[0],filtered_labels_and_bboxes[1]
                            bounding_boxes_stamp = [bbox for _, bbox in stamp_box]
                            # t_box = []
                            # Printing the bounding boxes
                            for bbox in bounding_boxes_stamp:
                                t_box_stamp = bbox
                                break
                            t_box_stamp = [int(coordinate) for coordinate in t_box_stamp]

                         
                            filtered_labels_and_bboxes = [(label, bbox) for label, bbox in zip(actual_labels, actual_bboxes) if int(label) == label_idx]
                            # print(filtered_labels_and_bboxes, 'filterboxes')
                            # true_label, true_box = filtered_labels_and_bboxes[0],filtered_labels_and_bboxes[1]
                            bounding_boxes = [bbox for _, bbox in filtered_labels_and_bboxes]
                            # t_box = []
                            # Printing the bounding boxes
                           
                            for bbox in bounding_boxes:
                                t_box1 = bbox
                                break
                         
                            t_box = [int(coordinate) for coordinate in t_box1]

                            if self.iou(symbol_box, t_box) > 0.5 and self.iou(bounding_box, t_box_stamp) > 0.5:
                                
                               

                               
                                rect = patches.Rectangle((x11, y11), x21-x11, y21-y11, linewidth=1, edgecolor='r', facecolor='none')                
                                ax.add_patch(rect)
                                # Add label text
                                # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                                ax.text(x11, y11, class_name1, color='Red', fontsize=8)  
                                # print('Symbols')                     
            
                                plt.axis('off')  # Optional: Remove axes for cleaner visualization
                                plt.savefig(f'../../../../output/vote_validation/yolo/valid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                                plt.close()
                                symbols_class.append((image_name, class_name1, 'valid', 'valid stamp and valid symbol', closet_distance))
                                print('Symbols: ' + str(id_to_label))
                                print('Vote: ' + class_name1 + "|" + "valid|" + "valid stamp and valid symbol")
                            else:
                                # x11, y11, x21, y21 = symbol_box
                                
                                rect = patches.Rectangle((x11, y11), x21-x11, y21-y11, linewidth=1, edgecolor='r', facecolor='none')                
                                ax.add_patch(rect)
                                # Add label text
                                # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                                ax.text(x11, y11, class_name1, color='Red', fontsize=8)                       
            
                                plt.axis('off')  # Optional: Remove axes for cleaner visualization
                                plt.savefig(f'../../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)  
                                plt.close()
                                symbols_class.append((image_name, class_name1, 'invalid', 'invalid symbol or stamp', closet_distance))
                                print('Symbols: ' + str(id_to_label))
                                print('Vote: ' + class_name1 + "|" + "invalid|" + "invalid symbol or stamp")                   


                        else:
                            x1, y1, x2, y2 = symbol_box

                            symbol_label = label_to_id.get(symbol_label,"unkown")
                            label_id = int(symbol_label)
                            class_name = id_to_label.get(label_id, 'Unknown')
                            
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                            ax.add_patch(rect)
                            # Add label text
                            # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                            ax.text(x1, y1, class_name, color='Red', fontsize=8)
                            plt.axis('off')  # Optional: Remove axes for cleaner visualization
                            plt.savefig(f'../../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                            plt.close()
                            symbols_class.append((image_name, class_name, 'invalid', 'valid vote and invalid symbol',closet_distance))
                            print('Symbols: ' + str(id_to_label))
                            print('Vote: ' + class_name + "|" + "invalid|" + "valid vote and invalid symbol")  
                    else:
                        # print(bounding_box, image_name, grid_cells)
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}" 
                        pred_label = label_to_id.get(pred_label,"unkown") 
                        label_id = int(pred_label)
                        class_name = id_to_label.get(label_id, 'Unknown')  # Replace `label` with a mapping to the actual class name if you have one
                        ax.text(x1, y1, class_name, color='blue', fontsize=12)
                        
                       
                        plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        plt.savefig(f'../../../output/vote_validation/yolo/valid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                        plt.close()
                        symbols_class.append((image_name, class_name, 'invalid', 'stamp not inside cell1','Nan'))
                        print('Symbols: ' + str(id_to_label))
                        print('Vote: ' + class_name + "|" + "invalid|" + "stamp not inside cell")  

                elif label_id == invalid_stamp_id:
                    # print(self.is_stamp_valid(bounding_box, grid_cells), 'gridcell', image_name, bounding_box, grid_cells)
                    
                    if self.is_stamp_valid(bounding_box, grid_cells):
                        # print("valid_stamp")
                        # print(self.is_stamp_valid(bounding_box, grid_cells), 'validgridcell', image_name)
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        # pred_label = label_to_id.get(pred_label, "unkown")
                        # label_id = int(pred_label)
                        class_name = id_to_label.get(label_id, 'Unknown')  # Replace `label` with a mapping to the actual class name if you have one
                        ax.text(x1, y1, class_name, color='blue', fontsize=12)
                        
                        plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        plt.savefig(f'../../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                        plt.close()
                        symbols_class.append((image_name, class_name, 'invalid', 'invalid stamp','nan'))
                        print('Symbols: ' + str(id_to_label))
                        print('Vote: ' + class_name + "|" + "invalid|" + "invalid stamp")  
                    
                        # is_valid_symbol, symbol_label, symbol_box, closet_distance = self.is_stamp_for_symbol(bounding_box, zip(boxes, labels, scores), image_name)
                        
                        # if is_valid_symbol: 

                        #     filtered_labels_and_bboxes = [(label, bbox) for label, bbox in zip(actual_labels, actual_bboxes) if label == symbol_label]
                        #     # print(filtered_labels_and_bboxes, 'filterboxes')
                        #     # true_label, true_box = filtered_labels_and_bboxes[0],filtered_labels_and_bboxes[1]
                        #     bounding_boxes = [bbox for _, bbox in filtered_labels_and_bboxes]
                        #     # t_box = []
                        #     # Printing the bounding boxes
                        #     for bbox in bounding_boxes:
                        #         t_box = bbox
                        #         break
                        #     t_box = [int(coordinate) for coordinate in t_box]

                        #     if self.iou(symbol_box, t_box) > 0.5:                          
                        #         x1, y1, x2, y2 = symbol_box
                        #         symbol_label = label_to_id.get(symbol_label,"unkown")
                        #         label_id = int(symbol_label)
                        #         class_name = id_to_label.get(label_id, 'Unknown')
                        #         symbols_class.append((image_name, class_name, 'invalid', 'invalid stamp and valid symbol'))
                        #         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                        #         ax.add_patch(rect)
                        #         # Add label text
                        #         # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                        #         ax.text(x1, y1, class_name, color='Red', fontsize=8)                       
            
                        #         plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        #         plt.savefig(f'../../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                        #         # plt.close()

                        #     else:
                                  
                        #         symbols_class.append((image_name, class_name, 'invalid', 'invalid stamp and invalid symbol', closet_distance))
                        #         rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                        #         ax.add_patch(rect)
                        #         # Add label text
                        #         # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                        #         ax.text(x1, y1, class_name, color='Red', fontsize=8)                       
            
                        #         plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        #         plt.savefig(f'../../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                      
                            

                        # else:
                        #     x1, y1, x2, y2 = symbol_box
                        #     symbol_label = label_to_id.get(symbol_label,"unkown")
                        #     label_id = int(symbol_label)
                        #     class_name = id_to_label.get(label_id, 'Unknown')
                            
                        #     rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                        #     ax.add_patch(rect)
                        #     # Add label text
                        #     # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                        #     ax.text(x1, y1, class_name, color='Red', fontsize=8)  

                        #     symbols_class.append((image_name, class_name, 'invalid', 'invalid stamp and invalid symbol',closet_distance))
                        #     plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        #     plt.savefig(f'../../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                        #     # plt.close()

                    else:
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='blue', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        # pred_label = label_to_id.get(pred_label,"unkown")
                        # label_id = int(pred_label)
                        class_name = id_to_label.get(label_id, 'Unknown')  # Replace `label` with a mapping to the actual class name if you have one
                        ax.text(x1, y1, class_name, color='blue', fontsize=12)                       
                        
                        plt.axis('off')  # Optional: Remove axes for cleaner visualization2
                        plt.savefig(f'../../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                        plt.close()
                        symbols_class.append((image_name, pred_label, 'invalid', 'Stamp not inside cell2','nan'))
                        print('Symbols: ' + str(id_to_label))
                        print('Vote: ' + class_name + "|" + "invalid|" + "stamp not inside cell")  

                elif all(x not in label_ids for x in [14, 39]):
                   
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization2
                    plt.savefig(f'../../../../output/vote_validation/yolo/no_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                    plt.close()
                    symbols_class.append((image_name, 'stamp not detected', 'invalid', 'no stamp','nan'))
                    print('Symbols: ' + str(id_to_label))
                    print('Vote: ' + 'stamp not detected' + "|" + "invalid|" + "no stamp")  
                    break

                elif all(x in label_ids for x in [14, 39]):
                   
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization2
                    plt.savefig(f'../../../../output/vote_validation/yolo/multi_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                    plt.close()
                    symbols_class.append((image_name, 'multiple stamp detected', 'invalid', 'multi stamp','nan'))
                    print('Symbols: ' + str(id_to_label))
                    print('Vote: ' + 'multiple stamp detected' + "|" + "invalid|" + "mulit stamp")  
                    break
                
                elif total_count > 2:
                    ax.axis('off')  # Optional: Remove axes for cleaner visualization2
                    plt.savefig(f'../../../../output/vote_validation/yolo/multi_stamp_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                    plt.close()
                    symbols_class.append((image_name, 'multiple stamp detected', 'invalid', 'multi stamp','nan'))
                    print('Symbols: ' + str(id_to_label))
                    print('Vote: ' + 'multiple stamp detected' + "|" + "invalid|" + "mulit stamp")  
                    break


                    

        #   Symbols: 1-Tree 2-Sun …. and Vote - Tree|Invalid|No stamp.                    
        df = pd.DataFrame(symbols_class, columns=['Image Id','Vote Symbol','Valid','Remarks','closet_distance'])
        df.to_excel('../../../../output/vote_results/vote_results_yolo.xlsx')

    def iou(self, boxA, boxB):
        """Compute the Intersection Over Union (IoU) of two bounding boxes."""
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3]- boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3]- boxB[1])


        # Compute the IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
    
    def center(self, box):
        """Calculate the center point of a bounding box."""
        x_center = (box[0] + box[2]) / 2
        y_center = (box[1] + box[3]) / 2
        return (x_center, y_center)

    def proximity(self, boxA, boxB, threshold):
        """Check if two boxes are within a certain threshold distance."""
        centerA = self.center(boxA)
        centerB = self.center(boxB)
        distance = ((centerB[0] - centerA[0]) ** 2 + (centerB[1] - centerA[1]) ** 2) ** 0.5
        return distance < threshold

    def calculate_edge_distance(self, stamp_box, symbol_box):
        """Calculate the horizontal and vertical distances between the stamp and symbol."""
        horizontal_distance = max(0, stamp_box[0] - symbol_box[2]) 
         # Stamp to the right of the symbol
        # print(stamp_box[0],symbol_box[2])
        vertical_distance = 0  # Default to 0 if vertically aligned or overlapping
        if stamp_box[3] < symbol_box[1]: 
             # Stamp above symbol
      
            vertical_distance = symbol_box[1] - stamp_box[3]
        elif symbol_box[3] < stamp_box[1]:  # Stamp below symbol
            vertical_distance = stamp_box[1] - symbol_box[3]
        
        return horizontal_distance, vertical_distance
    
    # def is_stamp_for_symbol(self, stamp_box, symbol_boxes):
    #     """Determine if a stamp is for a symbol based on calculated edge distances."""
    #     closest_distance = float('inf')
    #     closest_symbol_label = None
    #     closest_symbol_box = None        

    #     for symbol_box, symbol_label, _ in symbol_boxes:  # Assume '_' is a placeholder for another value, like 'score'
    #         # print(symbol_box)
    #         symbol_box = symbol_box.cpu().numpy()
    #         if symbol_label != 30:  # Assuming '30' is the label for 'stamp'
    #             if self.iou(stamp_box, symbol_box) > 0.0:  # There is an overlap
    #                 return True, symbol_label, symbol_box
    #             else:
                    
    #                 horizontal_distance, vertical_distance = self.calculate_edge_distance(stamp_box, symbol_box)
    #                 # print(horizontal_distance)
                    
    #                 # For simplicity, let's focus on horizontal distance
    #                 if horizontal_distance < closest_distance:
    #                     closest_distance = horizontal_distance
    #                     closest_symbol_label = symbol_label
    #                     closest_symbol_box = symbol_box

    #     adjusted_proximity_threshold = 378 - 189  # Example adjustment
        
    #     if closest_distance <= adjusted_proximity_threshold:
    #         print("True")
    #         return True, closest_symbol_label, closest_symbol_box
    #     else:
    #         print("False")
    #         return False, None, None

    

    def is_stamp_for_symbol(self, stamp_box, symbol_boxes,image_name):
        """Determine if a stamp is for a symbol based on overlap or proximity."""
        closest_distance = float('inf') 
        closest_symbol_label = None
        for symbol_box, symbol_label, score in symbol_boxes:
            # print(symbol_label)
            # symbol_label = symbol_label
            # symbol_box = symbol_box.cpu().numpy()
            # print(symbol_box)
            # symbol_box = [round(coordinate) for coordinate in symbol_box]
            
            if symbol_label not in ['invalid_stamp' , 'valid_stamp']:          
               
                if self.iou(stamp_box, symbol_box) > 0.0:  # There is an overlap
                    # print(self.iou(stamp_box, symbol_box), symbol_box, stamp_box, image_name, symbol_label)
                    return True, symbol_label, symbol_box, closest_distance
                else:  # Check for proximity
                    dist = ((self.center(symbol_box)[0] - self.center(stamp_box)[0]) ** 2 + 
                            (self.center(symbol_box)[1] - self.center(stamp_box)[1]) ** 2) ** 0.5
                    if dist <= closest_distance:
                        print(image_name, symbol_label, symbol_box, stamp_box)
                        closest_distance = dist
                        # print(closest_distance)
                        closest_symbol_label = symbol_label
                        closest_symbol_box = symbol_box
        # Check if the closest symbol is within the acceptable threshold distance
        # if closest_distance < proximity_threshold:
        #     return True, closest_symbol_label, symbol_box
        adjusted_proximity_threshold = 378 - 189  # Example adjustment
        
        if closest_distance <= adjusted_proximity_threshold:
            # print("True",closest_distance, image_name)
            return True, closest_symbol_label, closest_symbol_box, closest_distance
        else:
            # print("False",closest_distance, image_name)
            
            return False, closest_symbol_label, closest_symbol_box, closest_distance

