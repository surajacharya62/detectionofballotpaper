
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize.visualize_prediction import VisualizePrediction
import pandas as pd
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

        # Calculate cell size
        cell_width = symbol_width * 2
        cell_height = max(symbol_height, ballot_height // rows)

        # Initialize grid cells list
        grid_cells = []

        # Calculate the starting y-coordinate of the grid
        header_box_bottom = mt  # Assuming the top margin includes the header
        grid_start_y = header_box_bottom + 100  # Additional offset for the header box

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

    def is_stamp_valid(self, stamp_box, grid_cells):
        for cell_box in grid_cells:
            # Check if stamp box is entirely inside the grid cell box
            if (stamp_box[0] >= cell_box[0] and stamp_box[1] >= cell_box[1] and
                stamp_box[2] <= cell_box[2] and stamp_box[3] <= cell_box[3]):
                return True  # Stamp is valid
        return False  # Stamp is not valid
    

    def predicted_images(self, test_images, pred_labels, label_to_id):   

        margins = (1560, 300, 200, 200)  # top, bottom, left, right margins
        ballot_size = (2668, 3413)  # width, height of the ballot paper
        symbol_size = (189, 189)  # width, height of the symbols
        rows = 7  # Number of symbol rows
        columns = 6  # Number of symbol columns  
        grid_cells = self.reconstruct_grid_cells(margins, ballot_size, symbol_size, rows, columns)  
        id_to_label = {value: key for key, value in label_to_id.items()}
        stamp_id = 30
        symbols_class = []

        for i, (test_data) in enumerate(zip(test_images)):   
            # Convert tensor image to numpy array
            # img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            # print(test_data[0])
            img = test_data[0][0]  
            image_name = test_data[0][2]               
           
            img_np = img.permute(1, 2, 0).numpy() 
            # img_np = np.clip(img_np, 0, 1)  #Ensure the image array is between 0 and 1
            
            fig, ax = plt.subplots(1)
            ax.imshow(img_np) 
            transformed_data = [] 
           
            df = pred_labels[pred_labels['image_name'] == image_name]
            
            boxes = df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
            labels = df['label'].tolist()
            scores = df['score'].tolist()

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
            boxes = torch.tensor(boxes)
            scores = torch.tensor(scores)
            labels = [label_to_id[label] for label in labels]
            labels = torch.tensor(labels) 
            
            indices = visualize.apply_nms(boxes, scores)   
            final_boxes, final_scores, final_labels  =  boxes[indices], scores[indices], labels[indices] 
            # final_boxes, final_scores, final_labels = visualize.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )  
                     
            # stamp_count = final_labels.count(stamp_id)                                      
            
            for box, score, pred_label in zip(final_boxes, final_scores, final_labels):
                               
                # if score2 > 0.5:
                bounding_box = box.cpu().numpy() 
                # print('box1')
                # print(label)
                label_id = int(pred_label.cpu().numpy())

                if label_id == stamp_id:    
                    print("stamp")                    
                    if self.is_stamp_valid(bounding_box, grid_cells):
                        print("valid_stamp")
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        # label_text = f"{pred_label}"  
                        label_id = int(pred_label.cpu().numpy())
                        class_name = id_to_label.get(label_id, 'Unknown')  # Replace `label` with a mapping to the actual class name if you have one
                        ax.text(x1, y1, class_name, color='blue', fontsize=12)
                    
                        is_valid_symbol, symbol_label, symbol_box = self.is_stamp_for_symbol(bounding_box, zip(final_boxes, final_labels, final_scores))
                        # print(is_valid_symbol)
                        if is_valid_symbol:
                            print("vaid symbol")
                            x1, y1, x2, y2 = symbol_box
                            label_id = int(symbol_label.cpu().numpy())
                            class_name = id_to_label.get(label_id, 'Unknown')
                            symbols_class.append((image_name, class_name, class_name, 'Valid', 'Valid stamp'))
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                            ax.add_patch(rect)
                            # Add label text
                            # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                            ax.text(x1, y1, class_name, color='Red', fontsize=8)                       
        
                            plt.axis('off')  # Optional: Remove axes for cleaner visualization
                            plt.savefig(f'../../../output/vote_validation/yolo/valid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                            # plt.close()

                        else:
                            x1, y1, x2, y2 = symbol_box
                            label_id = int(symbol_label.cpu().numpy())
                            class_name = id_to_label.get(label_id, 'Unknown')
                            
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                            ax.add_patch(rect)
                            # Add label text
                            # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                            ax.text(x1, y1, class_name, color='Red', fontsize=8)  

                            symbols_class.append((image_name, class_name, class_name, 'Invalid', 'Invalid symbol'))
                            plt.axis('off')  # Optional: Remove axes for cleaner visualization
                            plt.savefig(f'../../../output/vote_validation/yolo/invalid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                            # plt.close()

                    # else:
                    #     pass
                        # symbols_class.append((image_name, pred_label, pred_label, 'Invalid', 'Invalid stamp'))
                        # plt.axis('off')  # Optional: Remove axes for cleaner visualization
                        # plt.savefig(f'../../../output/vote_validation/valid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                        # plt.close()

                # else:
                #     pass
                    # symbols_class.append((image_name, pred_label, pred_label, 'Invalid', 'No stamp'))
                    # plt.axis('off')  # Optional: Remove axes for cleaner visualization
                    # plt.savefig(f'../../../output/vote_validation/valid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                    # plt.close()

        #   Symbols: 1-Tree 2-Sun …. and Vote - Tree|Invalid|No stamp.                    
        df = pd.DataFrame(symbols_class, columns=['Image Id','Symbol','Vote','Valid','Remarks'])
        df.to_excel('../../../output/vote_results/vote_results_yolo.xlsx')

        # if df.empty:
        #     print("Vote Not Detected")
        # else:
        #     print(df)

    def iou(self, boxA, boxB):
        """Compute the Intersection Over Union (IoU) of two bounding boxes."""
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0].item(), boxB[0].item())
        yA = max(boxA[1].item(), boxB[1].item())
        xB = min(boxA[2].item(), boxB[2].item())
        yB = min(boxA[3].item(), boxB[3].item())

        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2].item() - boxA[0].item()) * (boxA[3].item() - boxA[1].item())
        boxBArea = (boxB[2].item() - boxB[0].item()) * (boxB[3].item() - boxB[1].item())


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
    
    def is_stamp_for_symbol(self, stamp_box, symbol_boxes):
        """Determine if a stamp is for a symbol based on calculated edge distances."""
        closest_distance = float('inf')
        closest_symbol_label = None
        closest_symbol_box = None        

        for symbol_box, symbol_label, _ in symbol_boxes:  # Assume '_' is a placeholder for another value, like 'score'
            print(symbol_box)
            if symbol_label != 30:  # Assuming '30' is the label for 'stamp'
                if self.iou(stamp_box, symbol_box) > 0.0:  # There is an overlap
                    return True, symbol_label, symbol_box
                else:
                    symbol_box = symbol_box.cpu().numpy()
                    horizontal_distance, vertical_distance = self.calculate_edge_distance(stamp_box, symbol_box)
                    # print(horizontal_distance)
                    
                    # For simplicity, let's focus on horizontal distance
                    if horizontal_distance < closest_distance:
                        closest_distance = horizontal_distance
                        closest_symbol_label = symbol_label
                        closest_symbol_box = symbol_box

        adjusted_proximity_threshold = 378 - 189  # Example adjustment
        
        if closest_distance <= adjusted_proximity_threshold:
            print("True")
            return True, closest_symbol_label, closest_symbol_box
        else:
            print("False")
            return False, None, None



    # def is_stamp_for_symbol(self, stamp_box, symbol_boxes, proximity_threshold):
    #     """Determine if a stamp is for a symbol based on overlap or proximity."""
    #     closest_distance = float('inf') 
    #     closest_symbol_label = None
    #     for symbol_box, symbol_label, score in symbol_boxes:
    #         # print(symbol_label)
    #         if symbol_label != 30:
    #             if self.iou(stamp_box, symbol_box) > 0.0:  # There is an overlap
    #                 return True, symbol_label, symbol_box
    #             else:  # Check for proximity
    #                 dist = ((self.center(symbol_box)[0] - self.center(stamp_box)[0]) ** 2 + 
    #                         (self.center(symbol_box)[1] - self.center(stamp_box)[1]) ** 2) ** 0.5
    #                 if dist < closest_distance:
    #                     closest_distance = dist
    #                     print(closest_distance)
    #                     closest_symbol_label = symbol_label
    #     # Check if the closest symbol is within the acceptable threshold distance
    #     # if closest_distance < proximity_threshold:
    #     #     return True, closest_symbol_label, symbol_box
    #     return True, closest_symbol_label, symbol_box

