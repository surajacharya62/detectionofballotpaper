import pandas as pd
import os

# Specify the directory where your annotation text files are stored
annotation_dir = '../../../testing_set/set4/yolov8'

annotation_files = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]  # Adjust if necessary

# Prepare a list to hold all annotation data
annotations = []

# Define a function to parse a single annotation file
def parse_annotation(file_path):
    with open(file_path, 'r') as file:
        # Assuming each line of the file is an annotation in the format "label xmin ymin xmax ymax"
        # Adjust the splitting logic based on the actual delimiter in your files (e.g., ',', ' ', etc.)
        annotations = [line.strip().split() for line in file.readlines()]  # Splitting by space as an example
    return annotations

# # Loop through each annotation file
for annotation_file in annotation_files:
    file_path = os.path.join(annotation_dir, annotation_file)
    # Parse the current file
    file_annotations = parse_annotation(file_path)
    # Process each annotation for the current file
    for annotation in file_annotations:
        # Extract label, xmin, ymin, xmax, ymax from the parsed data
        # Ensure the correct number of elements are present to avoid errors
        if len(annotation) == 5:
            label, xmin, ymin, xmax, ymax = annotation
            annotations.append({
                'image_name': annotation_file.replace('.txt', '.jpg'),
                'label': label,
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            })

# Convert the annotations list to a DataFrame
df_annotations = pd.DataFrame(annotations)

# Save the DataFrame to an Excel file
excel_path = 'annotations.xlsx'
df_annotations.to_excel(excel_path, index=False)

print(f'Annotations have been saved to {excel_path}')

def normalize_bbox(img_w, img_h, bbox):
    # Unpack the bounding box
    x_center, y_center, w, h = bbox
    
    # Calculate top-left and bottom-right corners
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    
    # Normalize the coordinates
    xmin /= img_w
    ymin /= img_h
    xmax /= img_w
    ymax /= img_h
    
    return xmin, ymin, xmax, ymax

# def yolo_to_corners(img_w, img_h, bbox):
#     """
#     Convert from YOLO format (normalized) to corners (xmin, ymin, xmax, ymax) in pixel coordinates.
    
#     Args:
#     - img_w: Image width in pixels.
#     - img_h: Image height in pixels.
#     - bbox: Bounding box in YOLO format [x_center, y_center, width, height], normalized.
    
#     Returns:
#     - A tuple (xmin, ymin, xmax, ymax) in pixel coordinates.
#     """
#     # De-normalize coordinates
#     x_center, y_center, w, h = bbox
#     x_center = x_center * img_w
#     y_center = y_center * img_h
#     w = w * img_w
#     h = h * img_h
    
#     # Calculate corners
#     x_center, y_center, w, h = bbox
#     xmin = x_center - (w / 2)
#     ymin = y_center - (h / 2)
#     xmax = x_center + (w / 2)
#     ymax = y_center + (h / 2)
#     return [int(xmin), int(ymin), int(xmax), int(ymax)]

def normalize_yolo( box, img_width, img_height):
    """
    Normalize YOLO format bounding box coordinates.

    Args:
    - x_center, y_center, width, height: YOLO format bounding box dimensions in pixels.
    - img_width, img_height: Dimensions of the image.

    Returns:
    - Tuple of normalized (x_center, y_center, width, height).
    """
    x_center, y_center, width, height = box
    x_center /= img_width
    y_center /= img_height
    width /= img_width
    height /= img_height
    
    return (x_center, y_center, width, height)

def yolo_to_corners_normalized(x_center, y_center, width, height):
    """
    Convert normalized YOLO format to normalized corners format.

    Args:
    - x_center, y_center, width, height: Normalized YOLO bounding box dimensions.

    Returns:
    - Tuple of normalized corners (xmin, ymin, xmax, ymax).
    """
    xmin = x_center - width / 2
    ymin = y_center - height / 2
    xmax = x_center + width / 2
    ymax = y_center + height / 2
    
    return (xmin, ymin, xmax, ymax)



import pandas as pd

# Load the annotations from the Excel file
df_annotations = pd.read_excel('yolo_pred_labels.xlsx')

# Example function for obtaining image dimensions
# Replace or modify this with your actual method for getting dimensions
def get_image_dimensions(image_name):
    # Placeholder: return fixed dimensions or look up actual dimensions
    return 2668, 3413  # Example dimensions, replace with actual values if available

# Apply the denormalization for each row in the DataFrame
def apply_denormalization(row):
    img_w, img_h = get_image_dimensions(row['image_name'])
    # Assuming 'xmin', 'ymin', 'xmax', 'ymax' are normalized and stored as 'label', 'xmin', 'ymin', 'xmax', 'ymax' in the DataFrame
    bbox = (row['xmin'], row['ymin'], row['xmax'], row['ymax'])  # This assumes bbox is already in a normalized form
    denormalized_bbox = normalize_yolo(bbox, img_w, img_h)
    x_center, y_center, width, height = denormalized_bbox 
    bb = yolo_to_corners_normalized(x_center, y_center, width, height)
    # Update row with denormalized values
    row['xmin'], row['ymin'], row['xmax'], row['ymax'] = bb
    return row

# Apply the function to each row of the DataFrame
df_annotations = df_annotations.apply(apply_denormalization, axis=1)

# Save the updated DataFrame back to Excel
df_annotations.to_excel('yolo_pred_labels_normalized.xlsx', index=False)

print('The bounding boxes have been denormalized and saved to denormalized_annotations.xlsx')
