
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class CheckVote():
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
    
    def predicted_images(self, predicted_images, predicted_bounding_box):   
        margins = (1560, 300, 200, 200)  # top, bottom, left, right margins
        ballot_size = (2668, 3791)  # width, height of the ballot paper
        symbol_size = (189, 189)  # width, height of the symbols
        rows = 9  # Number of symbol rows
        columns = 6  # Number of symbol columns  
        grid_cells = self.reconstruct_grid_cells(margins, ballot_size, symbol_size, rows, columns)       
        for i, (img, prediction) in enumerate(zip(predicted_images, predicted_bounding_box)):   
            # Convert tensor image to numpy array
            # img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = img.permute(1, 2, 0).numpy() 
            # img_np = np.clip(img_np, 0, 1)  #Ensure the image array is between 0 and 1
            
            fig, ax = plt.subplots(1)
            ax.imshow(img_np)

            # Prediction boxes, labels, and scores
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']

            for box, score, label in zip(boxes, scores, labels):
                if score > 0.5:
                    box1 = box.cpu() 
                    if label == 24:
                        if self.is_stamp_valid(box1.numpy(), grid_cells):
                            x1, y1, x2, y2 = box1.numpy()

                            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                            ax.add_patch(rect)

                            # Add label text
                            label_text = f"{label}"  # Replace `label` with a mapping to the actual class name if you have one
                            ax.text(x1, y1, label_text, color='blue', fontsize=12)

                            plt.axis('off')  # Optional: Remove axes for cleaner visualization
                            plt.savefig(f'../../../outputdir/valid_stamp_{i}.png', bbox_inches='tight', pad_inches=0)
                            plt.close()  

# Define your ballot and grid parameters (use the same values as in the creation script)


# Reconstruct the grid cells


# Assuming you have a list of detected stamps during testing
detected_stamps = [...]  # List of detected stamps as bounding boxes

# Check each detected stamp for validity
# for stamp_box in detected_stamps:
#     if is_stamp_valid(stamp_box, grid_cells):
#         print("Stamp is valid.")
#     else:
#         print("Stamp is invalid.")