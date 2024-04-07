import pandas as pd

# Load your CSV annotations
df = pd.read_csv('../../../testing_set/set4/annotations.csv')


label_to_id = {label: i for i, label in enumerate(df['label'].unique())}

all_labels = sorted(label_to_id)

# Create a consistent mapping from label names to label IDs, starting from 1
label_to_id1 = {label: i for i, label in enumerate(all_labels)}
# label_to_id1['stamp'] = 1

for _, row in df.iterrows():

    class_id = label_to_id1[row['label']]  
    x_center = (row['x1'] + row['x2']) / 2 / 2668
    y_center = (row['y1'] + row['y2']) / 2 / 3413
    width = (row['x2'] - row['x1']) / 2668 
    height = (row['y2'] - row['y1']) / 3413
    
    # Write to a new .txt file for each image
    with open(f'../../../testing_set/set4/yolov8/{row["image_id"].replace(".jpg", ".txt")}', 'a') as file:
        file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# import os

# def check_dataset_annotations(root_dir):
#     for subdir, dirs, files in os.walk(root_dir):
#         for file in files:
#             if file.endswith(".txt"):  # Assuming annotation files are .txt
#                 with open(os.path.join(subdir, file), 'r') as f:
#                     for line in f:
#                         class_id = int(line.split()[0])  # Assuming the class ID is the first entry in each line
#                         if class_id > 42:
#                             print(f"Invalid class ID {class_id} found in {file}")

# # Example usage
# check_dataset_annotations('E:/Oslo/OsloMet/Fourth semester/DetectionOfBallotPaper/modules/yolodata/set2/train_set/labels/train')
