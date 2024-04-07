import pandas as pd
import os

from objectdetect.objdetecteval.metrics import image_metrics as im, coco_metrics as cm



df = pd.read_csv(os.path.join('../../../testing_set/set4/', 'annotations.csv'))
label_to_id = {label: i for i, label in enumerate(df['label'].unique())}

all_labels = sorted(label_to_id)

# Create a consistent mapping from label names to label IDs, starting from 1
label_to_id = {label: i for i, label in enumerate(all_labels)}
inv_label = {v:k for k,v in label_to_id.items()} 

true_label = pd.read_excel('annotations.xlsx')
true_labels_df = pd.DataFrame(true_label)
true_labels_df['label']= true_labels_df['label'].replace(inv_label)  
true_labels_df.to_excel('true_labels_df_yolo.xlsx') 


# pred_label = pd.read_excel('yolo_pred_labels_normalized.xlsx')
# preds_df = pd.DataFrame(pred_label)
# preds_df.to_excel('pred_df_yolo.xlsx')

pred =  pd.read_excel('yolo_pred_labels_normalized.xlsx')
preds = pd.DataFrame(pred)
   
true =  pd.read_excel('true_labels_df_yolo.xlsx')
true_labels_df = pd.DataFrame(true)

infer_df = im.get_inference_metrics_from_df(preds, true_labels_df)
infer_df.to_excel('infer_df_yolo.xlsx')

class_summary_df = im.summarise_inference_metrics(infer_df)
class_summary_df.to_excel('class_summary_df_yolo.xlsx')