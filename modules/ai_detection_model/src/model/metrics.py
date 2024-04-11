
from objectdetect.objdetecteval.metrics import image_metrics as im, coco_metrics as cm
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

class Metrics():
    
    def metrics(self, prediction_labels, true_labels_path, label_to_id):
        pred_label = []
        true_labels = pd.read_csv(true_labels_path)
        true_labels = true_labels.rename(columns={'image_id':'image_name','x1':'xmin','y1':'ymin','x2':'xmax','y2':'ymax'})
        
        for data in prediction_labels:
            id = data['imageid'] # Use .item() to get the value as a standard Python scalar
            boxes = data['boxes'].cpu().numpy()
            labels = data['labels'].cpu().numpy()
            scores = data['scores'].cpu().numpy()
            image_name = data['imagename']

            for box, label, score in zip(boxes, labels, scores):
                xmin, ymin, xmax, ymax = box  # Unpack each box's coordinates
                pred_label.append({
                    'id': id,
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax,
                    'label': label,  # Convert to Python scalar if necessary
                    'score': score,  # Convert to Python scalar if necessary
                    'image_name': image_name
                })

        inv_label = {v:k for k,v in label_to_id.items()}            
        preds_df = pd.DataFrame(pred_label)
        preds_df['label']= preds_df['label'].replace(inv_label)  
        preds_df.to_excel('preds_df.xlsx')

        true_labels_df = pd.DataFrame(true_labels)
        true_labels_df.to_excel('true_labels_df.xlsx')        
        
        infer_df = im.get_inference_metrics_from_df(preds_df, true_labels_df)
        infer_df.to_excel('infer_df.xlsx')

        class_summary_df = im.summarise_inference_metrics(infer_df)
        class_summary_df.to_excel('class_summary_df.xlsx')
   
    
    def generate_precision_recall_curve(self,df, class_name):
        # Filter DataFrame for the current class
        class_df = df[df['class'] == class_name].copy()
        
        # Sort by Confidence in descending order
        class_df.sort_values('Confidence', ascending=False, inplace=True)
        
        # Initialize lists to store precision and recall values
        precisions = []
        recalls = []
        
        # Cumulatively calculate TP, FP, FN to determine precision and recall at each threshold
        cumulative_tp = 0
        cumulative_fp = 0
        for i in range(len(class_df)):
            cumulative_tp += class_df.iloc[i]['TP']
            cumulative_fp += class_df.iloc[i]['FP']
            cumulative_fn = class_df['FN'].sum()  # FN remains constant for a given class
            
            precision = cumulative_tp / (cumulative_tp + cumulative_fp) if (cumulative_tp + cumulative_fp) > 0 else 0
            recall = cumulative_tp / (cumulative_tp + cumulative_fn) if (cumulative_tp + cumulative_fn) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, marker='.', label=f'Precision-Recall Curve for {class_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {class_name}')
        plt.legend()
        plt.show()


    def generate_separate_precision_recall_curves(self, df):
        unique_classes = df['class'].unique()
        n_classes = 43
        
        # Calculate the number of rows/columns for the subplot grid
        n_cols = 5  # For example, 3 columns
        n_rows = (n_classes + n_cols - 1) // n_cols  # Calculate rows needed
        
        plt.figure(figsize=(2 * n_cols, 3 * n_rows))  # Adjust figure size as needed
        
        # Loop through each class to plot its precision-recall curve
        for index, class_name in enumerate(sorted(unique_classes), start=1):
            plt.subplot(n_rows, n_cols, index)
            
            class_df = df[df['class'] == class_name].copy()
            class_df.sort_values('Confidence', ascending=False, inplace=True)
            
            # Initialize lists to store precision and recall values
            precisions = []
            recalls = []

            # Cumulatively calculate TP, FP, FN to determine precision and recall at each threshold
            cumulative_tp = 0
            cumulative_fp = 0
            for i in range(len(class_df)):
                cumulative_tp += class_df.iloc[i]['TP']
                cumulative_fp += class_df.iloc[i]['FP']
                cumulative_fn = class_df['FN'].sum()  # FN remains constant for a given class

                precision = cumulative_tp / (cumulative_tp + cumulative_fp) if (cumulative_tp + cumulative_fp) > 0 else 0
                recall = cumulative_tp / (cumulative_tp + cumulative_fn) if (cumulative_tp + cumulative_fn) > 0 else 0

                precisions.append(precision)
                recalls.append(recall)

            # Plotting for the current class
            # plt.plot(recalls, precisions, marker='.', label=f'{class_name}')
            # plt.xlabel('Recall')
            # plt.ylabel('Precision')
            # plt.title(f'Class: {class_name}')
            # plt.legend()

            plt.plot(recalls, precisions, marker='.',color='pink', markerfacecolor='blue', label=f'{class_name}')
            plt.xlabel('Recall', fontsize=6)  # Adjust font size here
            plt.ylabel('Precision', fontsize=6)  # Adjust font size here
            # plt.title(f'Class: {class_name}', fontsize=8)  # Adjust font size here
            plt.legend(prop={'size': 5})  # Adjust legend font size here
            plt.tick_params(axis='both', which='major', labelsize=5) 
        # plt.xticks(rotation=45) 
        plt.tight_layout()
        plt.savefig(f'../../../outputdir/FasterRCNNMetrics.png', bbox_inches='tight', pad_inches=0, dpi=300)  
        # plt.subplots_adjust(bottom=0.3) # Adjust subplots to fit in the figure area
        plt.show()


    def call_metrics(self):
        
        f1_micro_data = pd.read_excel('./infer_df.xlsx')   
        df = pd.DataFrame(f1_micro_data)     

        ##-----------------Precision Recall Curve Visualization
        # # obje.generate_precision_recall_curve(df, 'heart')
        self.generate_separate_precision_recall_curves(df)

        # =CONCAT("\hline ",TEXT(B2,"0.00")," & ",TEXT(G2,"0.00")," & ", TEXT(H2,"0.00")," & ",TEXT(I2,"0.00")," & ",TEXT(J2,"0.00")," \\")


        # # -------------------------------Calculate Macro F1 Score
        data_summary = pd.read_excel('./class_summary_df.xlsx')   
        df_summary = pd.DataFrame(data_summary)
        df_summary['F1'] = 2 * (df_summary['Precision'] * df_summary['Recall']) / (df_summary['Precision'] + df_summary['Recall'])

        macro_f1_score = df_summary['F1'].mean()
        print("Macro F1 Score:", macro_f1_score)


        # #-------------------------------Calculating micro F1 Score
        total_TP = df['TP'].sum()
        total_FP = df['FP'].sum()
        total_FN = df['FN'].sum()

        precision_micro = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
        recall_micro = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
        f1_micro = 2 * (precision_micro * recall_micro) / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0

        print(f"Micro-average Precision: {precision_micro}")
        print(f"Micro-average Recall: {recall_micro}")
        print(f"Micro F1 Score: {f1_micro}")


# import ast
# data_yolo = pd.read_csv('./yolo_predicted_labels.csv')
# df_yolo = pd.DataFrame(data_yolo)
# # print(df_yolo)
# df_yolo['xmin'] = None
# df_yolo['ymin'] = None
# df_yolo['xmax'] = None
# df_yolo['ymax'] = None
# for index,row in data_yolo.iterrows():
#     box = ast.literal_eval(row['box_coord'])[0] 
#     # print(box)
#     xmin,ymin,xmax,ymax = box
#     # df_yolo['xmin','ymin','xmax','ymax'] =  xmin,ymin,xmax,ymax
#     # df_yolo['xmin'] = xmin
#     # df_yolo['ymin'] = ymin
#     # df_yolo['xmax'] = xmax
#     # df_yolo['ymax'] = ymax
    
#     df_yolo.at[index, 'xmin'] = box[0]
#     df_yolo.at[index, 'ymin'] = box[1]
#     df_yolo.at[index, 'xmax'] = box[2]
#     df_yolo.at[index, 'ymax'] = box[3]


# # print(df_yolo)
# df_yolo.to_excel('yolo_pred_labels.xlsx')

