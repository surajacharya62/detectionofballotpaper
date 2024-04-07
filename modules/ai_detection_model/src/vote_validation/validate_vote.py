
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from visualize.visualize_prediction import VisualizePrediction
import pandas as pd
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
        ballot_size = (2668, 3791)  # width, height of the ballot paper
        symbol_size = (189, 189)  # width, height of the symbols
        rows = 7  # Number of symbol rows
        columns = 6  # Number of symbol columns  
        grid_cells = self.reconstruct_grid_cells(margins, ballot_size, symbol_size, rows, columns)  
        id_to_label = {value: key for key, value in label_to_id.items()}
        stamp_id = 30
        symbols_class = []

        for i, (test_data, prediction) in enumerate(zip(test_images, pred_labels)):   
            # Convert tensor image to numpy array
            # img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img = test_data[0]           
            img_np = img.permute(1, 2, 0).numpy() 
            # img_np = np.clip(img_np, 0, 1)  #Ensure the image array is between 0 and 1
            
            fig, ax = plt.subplots(1)
            ax.imshow(img_np)
          

            
            boxes = prediction[0]['boxes'] 
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']
            image_name = prediction[2]
            

            indices = visualize.apply_nms(boxes, scores)   
            final_boxes, final_scores, final_labels  =  boxes[indices], scores[indices], labels[indices] 
            # final_boxes, final_scores, final_labels = visualize.select_highest_confidence_per_class( 
            #                                                 boxes, scores, labels, indices )  
                     
                                                    
            
            for box, score, pred_label in zip(final_boxes, final_scores, final_labels):
                               
                # if score2 > 0.5:
                bounding_box = box.cpu().numpy() 
                # print('box1')
                # print(label)
                label_id = int(pred_label.cpu().numpy())


                if label_id == stamp_id:    
                    # print("stamp")                    
                    if self.is_stamp_valid(bounding_box, grid_cells):
                        print("valid_stamp")
                        x1, y1, x2, y2 = bounding_box
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                        ax.add_patch(rect)
                        # Add label text
                        label_text = f"{pred_label}"  # Replace `label` with a mapping to the actual class name if you have one
                        ax.text(x1, y1, label_text, color='blue', fontsize=12)

                        is_valid_symbol, symbol_label, symbol_box = self.is_stamp_for_symbol(bounding_box, zip(final_boxes, final_labels, final_scores), 30)
                        if is_valid_symbol:
                            x1, y1, x2, y2 = symbol_box.numpy()
                            label_id = int(symbol_label.cpu().numpy())
                            class_name = id_to_label.get(label_id, 'Unknown')
                            symbols_class.append((class_name))
                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                            ax.add_patch(rect)
                            # Add label text
                            # label_text = f"{class_name}"  # Replace `label` with a mapping to the actual class name if you have one
                            ax.text(x1, y1, class_name, color='Red', fontsize=8)                       
        
                            plt.axis('off')  # Optional: Remove axes for cleaner visualization
                            plt.savefig(f'../../../output/vote_validation/valid_{image_name}.jpg', bbox_inches='tight', pad_inches=0,dpi=600)
                            plt.close()
                            
        df = pd.DataFrame(symbols_class, columns=['pred_symbol'])

        if df.empty:
            print("Vote Not Detected")
        else:
            print(df)

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

    def is_stamp_for_symbol(self, stamp_box, symbol_boxes, proximity_threshold):
        """Determine if a stamp is for a symbol based on overlap or proximity."""
        closest_distance = float('inf') 
        closest_symbol_label = None
        for symbol_box, symbol_label, score in symbol_boxes:
            # print(symbol_label)
            if symbol_label != 30:
                if self.iou(stamp_box, symbol_box) > 0.0:  # There is an overlap
                    return True, symbol_label, symbol_box
                else:  # Check for proximity
                    dist = ((self.center(symbol_box)[0] - self.center(stamp_box)[0]) ** 2 + 
                            (self.center(symbol_box)[1] - self.center(stamp_box)[1]) ** 2) ** 0.5
                    if dist < closest_distance:
                        closest_distance = dist
                        print(closest_distance)
                        closest_symbol_label = symbol_label
        # Check if the closest symbol is within the acceptable threshold distance
        if closest_distance < proximity_threshold:
            return True, closest_symbol_label, symbol_box
        return False, closest_symbol_label, symbol_box

