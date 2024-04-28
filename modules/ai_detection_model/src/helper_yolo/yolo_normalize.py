import pandas as pd
import os
import ast
from PIL import Image
import pandas as pd



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

class YoloNormalize():

    # def objeval_format_true_labels(self, true_labels_path, output_save_path):
        
    #     annotations = []      
    #     annotation_files = [f for f in os.listdir(true_labels_path) if f.endswith('.txt')] 

    #     def parse_annotation(file_path):
    #         with open(file_path, 'r') as file:                
    #             annotations = [line.strip().split() for line in file.readlines()]
    #         return annotations
       
    #     for annotation_file in annotation_files:
    #         file_path = os.path.join(true_labels_path, annotation_file)            
    #         file_annotations = parse_annotation(file_path)           
    #         for annotation in file_annotations:                
    #             if len(annotation) == 5:
    #                 label, xmin, ymin, xmax, ymax = annotation
    #                 annotations.append({
    #                     'image_name': annotation_file.replace('.txt', '.jpg'),
    #                     'label': label,
    #                     'xmin': xmin,
    #                     'ymin': ymin,
    #                     'xmax': xmax,
    #                     'ymax': ymax
    #                 })
     
    #     df_annotations = pd.DataFrame(annotations)    

    #     excel_path = 'true_labels_objeval_format.xlsx'
    #     df_annotations.to_excel(os.path.join(output_save_path, excel_path), index=False)


    # def objeval_format_pred_labels(self, preds_path):    
        
    #     objeval_notations = []
    #     file_name = 'pred_labels_normalized.xlsx'
    #     labels_df = pd.read_excel(os.path.join(preds_path, file_name ))       
            
    #     for annotation in labels_df.iterrows():  
                         
    #         box = ast.literal_eval(annotation[1]['box_coord'])
    #         xmin = box[0]
    #         ymin = box[1]
    #         xmax = box[2]
    #         ymax = box[3]  
    #         xmin, ymin, xmax, ymax =  xmin, ymin, xmax, ymax
    #         objeval_notations.append({
    #             'image_name': annotation[1]['image_name'],
    #             'label': annotation[1]['label'],
    #             'score': annotation[1]['score'],
    #             'xmin': xmin,
    #             'ymin': ymin,
    #             'xmax': xmax,
    #             'ymax': ymax
    #         })
    
    #     df_annotations = pd.DataFrame(objeval_notations)
    #     excel_path = 'pred_labels_objeval_format.xlsx'
    #     df_annotations.to_excel(os.path.join(preds_path, excel_path), index=False)

    def get_image_id(self, filename):
   
        return int(filename.split('_')[1].split('.')[0]) - 1
    
    def pred_labels_conerized_normalized(self, preds_path):    
        
        objeval_notations = []
        file_name = 'pred_labels_cornerized.xlsx'
        labels_df = pd.read_excel(os.path.join(preds_path, file_name ))       
            
        for annotation in labels_df.iterrows():                           
            box = ast.literal_eval(annotation[1]['box_coord'])
            xmin = box[0]
            ymin = box[1]
            xmax = box[2]
            ymax = box[3]  
            xmin, ymin, xmax, ymax =  xmin, ymin, xmax, ymax
            objeval_notations.append({
                'image_name': annotation[1]['image_name'],
                'image_id': self.get_image_id(annotation[1]['image_name']),
                'label': annotation[1]['label'],
                'category_id':annotation[1]['class_id'],
                'score': annotation[1]['score'],
                'xmin': xmin,
                'ymin': ymin,
                'xmax': xmax,
                'ymax': ymax
            }) 

        df_annotations = pd.DataFrame(objeval_notations)
        excel_path = 'pred_labels_conerized_normalized.xlsx'
        df_annotations.to_excel(os.path.join(preds_path, excel_path), index=False)

    def normalize_yolo(self, box, img_width, img_height):      
        x_center, y_center, width, height = box
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return (x_center, y_center, width, height)

    def yolo_to_corners_normalized(slef, x_center, y_center, width, height):
      
        xmin = x_center - width / 2
        ymin = y_center - height / 2
        xmax = x_center + width / 2
        ymax = y_center + height / 2
        
        return (xmin, ymin, xmax, ymax)

    def yolo_to_corners( self, img_w, img_h, bbox):
            
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
            

    # def apply_normalization1(self, row):
    #     box = ast.literal_eval(row['box_coord'])[0]
    #     img_w, img_w = get_image_dimensions(row['image_name'])
    #     box = ast.literal_eval(row['box_coord'])[0]
    #     xmin = box[0]
    #     ymin = box[1]
    #     xmax = box[2]
    #     ymax = box[3]
    #     box = xmin, ymin, xmax, ymax 

    #     boxes = self.yolo_to_corners(img_w, img_w, box)
    
    #     row['box_coord'] = list(boxes)
    #     return row 


    def set_corner(self, row):
        # df_annotations = pd.read_csv()
        image_path= '../../../../testing_set/set6/test/image_0001.jpg'
        image = Image.open(image_path)
        image_width, image_height = image.size   
        # print(row)
        box = ast.literal_eval(row['box_coord'])[0]
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        box = xmin, ymin, xmax, ymax         
        xmin, ymin, xmax, ymax = self.yolo_to_corners(image_width, image_height, box)
        box = xmin,ymin,xmax,ymax
        row['box_coord'] = list(box)

        return row


    def pred_normalize(self, row):

        box = ast.literal_eval(row['box_coord'])
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]      
        x_center = (xmin + xmax) / 2 / 2668
        y_center = (ymin + ymax) / 2 / 3413
        width = (xmax - xmin) / 2668 
        height = (ymax - ymin) / 3413
        normalized_boxes = x_center, y_center, width, height
        row['box_coord'] = list(normalized_boxes) 
        return row
    

    def normalize(self, pred_labels_path, ouput_save_path):
        # try:
        file_name = 'pred_labels_cornerized.xlsx'
        pred_labels = 'yolo_predicted_labels.csv' 
        df_annotations = pd.read_csv(os.path.join(pred_labels_path, pred_labels))
        df_annotations = df_annotations.apply(self.set_corner, axis=1) 
        df_annotations.to_excel(os.path.join(ouput_save_path, file_name))
        df_annotations = pd.read_excel(os.path.join(ouput_save_path, file_name))
        df_annotations = df_annotations.apply(self.pred_normalize, axis=1) 
        df_annotations.to_excel(os.path.join(ouput_save_path,
                                                'pred_labels_normalized.xlsx'))
    
        # except:
        #     print(" Error in file path")
        


# # Example function for obtaining image dimensions
# # Replace or modify this with your actual method for getting dimensions
# def get_image_dimensions(image_name):
#     # Placeholder: return fixed dimensions or look up actual dimensions
#     return 2668, 3413  # Example dimensions, replace with actual values if available


# # Apply the denormalization for each row in the DataFrame
# def apply_normalization(row):
#     img_w, img_h = get_image_dimensions(row['image_name'])
#     box = ast.literal_eval(row['box_coord'])[0]
#     xmin = box[0]
#     ymin = box[1]
#     xmax = box[2]
#     ymax = box[3]
#     box = xmin, ymin, xmax, ymax 
#     # Assuming 'xmin', 'ymin', 'xmax', 'ymax' are normalized and stored as 'label', 'xmin', 'ymin', 'xmax', 'ymax' in the DataFrame
#     # bbox = (row['box_coord'], row['ymin'], row['xmax'], row['ymax'])  # This assumes bbox is already in a normalized form

#     denormalized_bbox = normalize_yolo(box, img_w, img_h)
#     x_center, y_center, width, height = denormalized_bbox 
#     bb = yolo_to_corners_normalized(x_center, y_center, width, height)
#     # Update row with denormalized values
#     # row['xmin'], row['ymin'], row['xmax'], row['ymax'] = bb
#     row['box_coord'] = list(bb)
#     return row 




