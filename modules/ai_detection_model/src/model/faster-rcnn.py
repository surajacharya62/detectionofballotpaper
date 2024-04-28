import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import ast
import time
import argparse

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from visualize.visualize_prediction import VisualizePrediction
from vote_validation.validate_vote import ValidateVote
from model.metrics import Metrics
from model.compare_bounding_boxes_faster import CompareBoundingBox
from model.test1 import ReshapeData

import warnings
warnings.filterwarnings("ignore")


visualize = VisualizePrediction()
object_compare = CompareBoundingBox()
obj_reshape = ReshapeData()

start_time = time.time()
# class ElectoralSymbolTrainDataset(Dataset):
#     def __init__(self, image_path, train_or_test_path, label, transforms):
#         self.image_path = image_path
#         self.train_or_test_path = train_or_test_path
#         self.transforms = transforms
#         self.label_to_id = label
#         # if len(os.listdir(os.path.join(image_path, train_or_test_path))
#         self.df = pd.read_csv(os.path.join(image_path, 'annotations.csv'))
#         self.imgs_list = sorted(os.listdir(os.path.join(image_path, train_or_test_path)))

#     def __getitem__(self, idx):
#         img_name = self.imgs_list[idx]
#         img_path = os.path.join(self.image_path, self.train_or_test_path, img_name)
#         img = Image.open(img_path).convert('RGB')      
#         filtered_rows = self.df[self.df['image_id'] == img_name]
#         boxes = filtered_rows[['x1', 'y1', 'x2', 'y2']].values.astype('float32')
#         labels = filtered_rows['label'].apply(lambda x: self.label_to_id[x]).values       
#         boxes = torch.as_tensor(boxes, dtype=torch.float32)
#         labels = torch.as_tensor(labels, dtype=torch.int64)
#         image_id = torch.tensor(idx)
#         area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])      
#         iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
#         target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd':iscrowd}

       
#         if self.transforms is not None:           
#             img = self.transforms(img)          

#         return img, image_id, img_name, target

#     def __len__(self):
#         return len(self.imgs_list)


class ElectoralSymbolDataset(Dataset):
    def __init__(self, image_path, image_name, annotation_path, label, transforms, is_single_image=False):
        self.image_path = image_path
        # self.train_or_test_path = train_or_test_path
        self.transforms = transforms
        self.label_to_id = label
        self.is_single_image = is_single_image
        self.annotation_path = annotation_path        

        if is_single_image:
            # Handle single image case
            self.imgs_list = [image_name]
            print("Length of dataset:", len(self.imgs_list))  # Single image in the list
            annotations_path = os.path.join(self.annotation_path, 'annotations.csv')
            if os.path.exists(annotations_path):
                self.df = pd.read_csv(annotations_path)
            else:
                # If annotations for a single image are to be handled differently
                raise FileNotFoundError("Annotations file not found for the single image mode.")
        else:
            
            # Handle directory case
            self.df = pd.read_csv(os.path.join(self.annotation_path, 'annotations.csv'))
            self.imgs_list = sorted(os.listdir(image_path))
            print("Length of dataset:", len(self.imgs_list)) 

    def __getitem__(self, idx):
        img_name = self.imgs_list[idx]
        
        if self.is_single_image:
            img_name = img_name  
            img_path = os.path.join(self.image_path, img_name)
            
        else:
            img_path = os.path.join(self.image_path, img_name)

        img = Image.open(img_path).convert('RGB')
        filtered_rows = self.df[self.df['image_id'] == img_name]
        boxes = filtered_rows[['x1', 'y1', 'x2', 'y2']].values.astype('float32')
        labels = filtered_rows['label'].apply(lambda x: self.label_to_id[x]).values
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd': iscrowd}

        if self.transforms is not None:
            img = self.transforms(img)

        return img, image_id, img_name, target

    def __len__(self):
        return len(self.imgs_list)


def get_transform(train):
    transforms = []
    if train:
        transforms = T.Compose([
                # T.Resize((224, 224)),  # Example resize, adjust as needed
                T.ToTensor(),
                # T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization, adjust as needed
            ])
    else:
        transforms = T.Compose([
                # T.Resize((224, 224)),  # Example resize, adjust as needed
                T.ToTensor(),
                # T.RandomHorizontalFlip(p=0.5),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization, adjust as needed
            ])

    return transforms


# class UnlabeledTestDataset(Dataset):
#     def __init__(self, image_dir, transform=None):
#         self.image_dir = image_dir
#         self.transform = transform
#         self.images = os.listdir(image_dir)

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image_path = os.path.join(self.image_dir, self.images[idx])
#         image = Image.open(image_path).convert("RGB")
#         if self.transform:
#             image = self.transform(image)
#         return image 
    

def collate_fn(batch):
    return tuple(zip(*batch))



parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--image_name', type=str, help='image name')
parser.add_argument('--singleORmulti_images', type=str, help='singe multi images')
parser.add_argument('--image_path', type=str, help='image path')
args = parser.parse_args()


# image_path = '../../../training_set/set7/'
# train_path = 'train' 

# test_image_path = '../../../testing/set6/'
# test_path = 'test'

annotations_path = '../../../testing_set/set6/'
if args.singleORmulti_images == 'single':
    test_image_path = args.image_path
    is_single_image = True
    image_name = args.image_name
    

else:
    is_single_image = False
    image_name = None    
    test_image_path = args.image_path

df = pd.read_csv(os.path.join('../../../testing_set/set6/', 'annotations.csv'))
label_to_id = {label: i for i, label in enumerate(df['label'].unique())}
sorted_labels = sorted(label_to_id)


# Create a consistent mapping from label names to label IDs, starting from 1
label_to_id = {label: i for i, label in enumerate(sorted_labels)}

# train_set = ElectoralSymbolTrainDataset(image_path, train_path,label_to_id, get_transform(True))

# train_loader = DataLoader(train_set, batch_size=4,
#                                          shuffle= True, 
#                                          pin_memory= True if torch.cuda.is_available() else False, collate_fn=collate_fn )


test_set = ElectoralSymbolDataset(test_image_path, image_name, annotations_path, label_to_id, get_transform(True), is_single_image)
# test_set = ElectoralSymbolTrainDataset(test_image_path, test_path, label_to_id, get_transform(True))

test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


######---------------------------------------------- Model Preparation
def get_object_detection_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


num_classes = 44
model1 = get_object_detection_model(num_classes)
optimizer = torch.optim.SGD(model1.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0005)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')



# ###--------------------------------------------Testing the model

# torch.cuda.empty_cache()

# model2 = get_object_detection_model(num_classes)
# model2.load_state_dict(torch.load('../../../trained_models/trained_model_fasterrcnn_16_2.pth', map_location= torch.device('cpu')))
# model2.eval()
# predictions = []
# for images, imageids, imagenames, target in test_loader:  # No labels if your test set is unlabeled
#     image = images.to(device)  # Move images to the device where your model is
#     with torch.no_grad():  # No gradients needed
#         output = model2(image)

#         for output, imageid, imagename in zip(output, imageids, imagenames):
#             # Structure each prediction as a tuple
#             prediction = (output, imageid, imagename)
#             predictions.append(prediction) 



# ----------------------------------------------For Visualizing The Predictions

obj_visualize = VisualizePrediction()
# obj_visualize.visualize_predicted_images(test_set, predictions, label_to_id)
print(test_set[0])
obj_visualize.visualize_train_set(test_set, label_to_id)


# #-----------------------------------------------For Vote Validation Visualization
# obj_vote_val = ValidateVote()
# obj_vote_val.predicted_images(test_set, predictions, label_to_id)


#----------------------------------For Bounding Box comparision(Predicted Labels with Ground Truth labels)
# object_compare.labels(test_set, predictions, label_to_id)
# # Specify the path to your Excel file
# file_path = '../../../faster_rcnn_files/df_total_comparisions.xlsx'
# obj_reshape.process_and_reshape_data_v2(file_path)


# #------------------------------------------------For Metrics Calculation
# dataframe_predicted_data = []
# for data in predictions:  # Loop through each batch
    
#     prediction_dict = data[0]  
#     imageid = data[1] # The second item is the imageid
#     imagename = data[2] # The third item is the imagename
    
#     boxes = data[0]['boxes']
#     scores = data[0]['scores']
#     labels = data[0]['labels']

#     indices = visualize.apply_nms(boxes, scores)   
#     final_boxes, final_scores, final_labels = boxes[indices], scores[indices], labels[indices]
                                                    
#     #                                                 
#     # If you want to collect these for all images
#     dataframe_predicted_data.append({
#         'imageid': imageid,
#         'imagename': imagename,
#         'boxes': final_boxes,
#         'scores': final_scores,
#         'labels': final_labels
#     })

# metrics_file_path = '../../../faster_rcnn_files/'
# true_labels_path = '../../../testing_set/set6/annotations.csv'
# obj = Metrics()
# obj.metrics(dataframe_predicted_data, true_labels_path, label_to_id, metrics_file_path)
# obj.call_metrics(metrics_file_path)

      




end_time = time.time()
elapsed_time = end_time - start_time
print(f"Prediction time: {elapsed_time} seconds.")










