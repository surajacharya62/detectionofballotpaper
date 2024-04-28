# import pandas as pd
# import json

# # Load data from Excel
# true_labels_path = '../../../faster_rcnn_files/true_labels_df.xlsx'
# pred_labels_path = '../../../faster_rcnn_files/preds_df.xlsx'

# true_labels_df = pd.read_excel(true_labels_path)
# pred_labels_df = pd.read_excel(pred_labels_path)

# # Convert bounding box from [xmin, ymin, xmax, ymax] to [x, y, width, height]
# def convert_bbox(coco_bbox):
#     xmin, ymin, xmax, ymax = coco_bbox
#     return [xmin, ymin, xmax - xmin, ymax - ymin]

# # Convert to COCO format
# def convert_to_coco(df, is_prediction=False):
#     images = []
#     annotations = []
#     categories = set()
    
#     for i, row in df.iterrows():
#         image_id = row['image_name']
#         category_id = row['category_id']
#         bbox = row['xmin'],row['ymin'],row['xmax'],row['ymax']
#         bbox = convert_bbox(bbox)  # Convert the bbox format
        
#         images.append({"id": image_id, "file_name": f"image_{image_id}.jpg"})
        
#         annotation = {
#             "id": i,
#             "image_id": image_id,
#             "category_id": category_id,
#             "bbox": bbox,
#             "area": bbox[2] * bbox[3],  # width * height
#             "iscrowd": 0
#         }
#         if is_prediction:
#             annotation["score"] = row['score']
            
#         annotations.append(annotation)
#         categories.add((category_id, f"category_{category_id}"))  # Example category naming
    
#     categories = [{"id": cat_id, "name": name} for cat_id, name in categories]
#     return {"images": images, "annotations": annotations, "categories": categories}

# # Convert and save the JSON files
# coco_true = convert_to_coco(true_labels_df)
# coco_pred = convert_to_coco(pred_labels_df, is_prediction=True)

# with open('coco_true.json', 'w') as f:
#     json.dump(coco_true, f)

# with open('coco_pred.json', 'w') as f:
#     json.dump(coco_pred, f)

# import pandas as pd
# import json

# # Assuming pred_labels_df is your DataFrame with predictions
# true_labels_path = '../../../faster_rcnn_files/true_labels_df.xlsx'
# # pred_labels_path = '../../../faster_rcnn_files/preds_df.xlsx'
# pred_labels_df = pd.read_excel(true_labels_path)

# def convert_bbox_to_coco(xmin, ymin, xmax, ymax):
#     width = xmax - xmin
#     height = ymax - ymin
#     return [xmin, ymin, width, height]

# def convert_predictions_to_coco(df):
#     results = []
#     for _, row in df.iterrows():
#         image_id = row['id']
#         category_id = row['category_id']
#         xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
#         bbox = convert_bbox_to_coco(xmin, ymin, xmax, ymax)
#         score = row['score']  # Assuming a 'score' column exists in your predictions
        
#         result = {
#             "image_id": image_id,
#             "category_id": category_id,
#             "bbox": bbox,
#             "score": score
#         }
#         results.append(result)
#     return results

# # Convert and save the predictions JSON
# coco_pred = convert_predictions_to_coco(pred_labels_df)
# with open('true_pred.json', 'w') as f:
#     json.dump(coco_pred, f)




import pandas as pd
import json

# Load the Excel file into a DataFrame
true_labels_path = '../../../faster_rcnn_files/true_labels_df.xlsx'
# pred_labels_path = '../../../faster_rcnn_files/preds_df.xlsx'
# pred_labels_df = pd.read_excel(true_labels_path)# Adjust this to the path of your Excel file
df_true_labels = pd.read_excel(true_labels_path)  # Adjust sheet name as necessary
# print(df_true_labels.head(5))

# A function to convert bounding boxes
def convert_bbox_to_coco(xmin, ymin, xmax, ymax):
    width = xmax - xmin
    height = ymax - ymin
    return [xmin, ymin, width, height]

# Process images and annotations
def process_true_labels(df):
    images = []
    annotations = []
    category_ids = set()

    # Create 'image_id' from 'image_name' if missing
    if 'image_id' not in df.columns:
        df['image_id'] = df['image_name'].apply(lambda x: int(x.split('_')[1].split('.')[0]) - 1)

    
    for idx, row in df.iterrows():
        image_id = int(row['image_id'])
        category_id = row['category_id']
        # score = row['score']
        
        # Add to images list if not already added
        if not any(img['id'] == image_id for img in images):
            images.append({
                "id": image_id,
                "file_name": row['image_name'],
                # Include other fields as necessary
            })
        
        # Convert bbox and add to annotations list
        bbox_coco = convert_bbox_to_coco(row['xmin'], row['ymin'], row['xmax'], row['ymax'])
        annotations.append({
            "id": idx,
            "image_id": int(image_id),
            "category_id": category_id,
            "bbox": bbox_coco,
            "area": bbox_coco[2] * bbox_coco[3], 
            #  "score": score,# width * height
            "iscrowd": 0,
        })
        
        # Keep track of category IDs
        category_ids.add(category_id)
    
    # Create categories (assuming 'label' corresponds to the category name)
    categories = [{"id": cid, "name": df[df['category_id'] == cid]['label'].iloc[0]} for cid in category_ids]
    
    return images, annotations, categories

# Process the DataFrame
images, annotations, categories = process_true_labels(df_true_labels)

# Compile into COCO format
coco_format = {
    "images": images,
    "annotations": annotations,
    "categories": categories,
}

coco_format1 = [annotations]
    


# Write to JSON file
with open('coco_true.json', 'w') as f:
    json.dump(coco_format, f, indent=4)





