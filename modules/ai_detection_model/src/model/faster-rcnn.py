import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import ast
import time

from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch.cuda.amp import GradScaler, autocast

from visualize.visualize_prediction import VisualizePrediction
from vote_validation.validate_vote import ValidateVote
from model.metrics import Metrics
from model.test import CompareBoundingBox


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# check_vote_obj = CheckVote()
# torch.cuda.empty_cache()
visualize = VisualizePrediction()
object_compare = CompareBoundingBox()


class ElectoralSymbolTrainDataset(Dataset):
    def __init__(self, image_path, train_or_test_path, label, transforms):
        self.image_path = image_path
        self.train_or_test_path = train_or_test_path
        self.transforms = transforms
        self.label_to_id = label
        self.df = pd.read_csv(os.path.join(image_path, 'annotations.csv'))
        self.imgs_list = sorted(os.listdir(os.path.join(image_path, train_or_test_path)))

    def __getitem__(self, idx):
        img_name = self.imgs_list[idx]
        img_path = os.path.join(self.image_path, self.train_or_test_path, img_name)
        img = Image.open(img_path).convert('RGB')
        # image = np.array(img)
        # img = read_image(img_path)

        # Filter rows for the current image and convert boxes to tensor
        filtered_rows = self.df[self.df['image_id'] == img_name]
        boxes = filtered_rows[['x1', 'y1', 'x2', 'y2']].values.astype('float32')

        labels = filtered_rows['label'].apply(lambda x: self.label_to_id[x]).values
        # labels = labels.astype(int)

        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        image_id = torch.tensor(idx)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        # Wrap sample and targets into torchvision tv_tensors:
        # img = tv_tensors.Image(img)

        target = {'boxes': boxes, 'labels': labels, 'image_id': image_id, 'area': area, 'iscrowd':iscrowd}

        # Optionally, apply transformations
        if self.transforms is not None:
            # tensor_img = self.transforms(img)
            img = self.transforms(img)          

        return img, image_id, img_name

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


class UnlabeledTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image 
    

def collate_fn(batch):
    return tuple(zip(*batch))


image_path = '../../../training_set/set6/'
train_path = 'train' 
test_image_path = '../../../testing_set/set4/'
test_path = 'test'
df = pd.read_csv(os.path.join('../../../training_set/set6/', 'annotations.csv'))
label_to_id = {label: i for i, label in enumerate(df['label'].unique())}

sorted_labels = sorted(label_to_id)
# print(all_labels)

# Create a consistent mapping from label names to label IDs, starting from 1
label_to_id = {label: i for i, label in enumerate(sorted_labels)}

# train_set = ElectoralSymbolTrainDataset(image_path, train_path,label_to_id, get_transform(True))

# train_loader = DataLoader(train_set, batch_size=4,
#                                          shuffle= True, 
#                                          pin_memory= True if torch.cuda.is_available() else False, collate_fn=collate_fn )


test_set = ElectoralSymbolTrainDataset(test_image_path, test_path, label_to_id, get_transform(True))
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)


######---------------------------------------------- Model Preparation
def get_object_detection_model(num_classes):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model


num_classes = 43
model1 = get_object_detection_model(num_classes)
optimizer = torch.optim.SGD(model1.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0005)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# ##--------------------------------------------Testing the model

torch.cuda.empty_cache()
start_time = time.time()
model2 = get_object_detection_model(num_classes)
model2.load_state_dict(torch.load('../../../trained_models/trained_model_fasterrcnn_new.pth', map_location= torch.device('cpu')))
model2.eval()
predictions = []
for images, imageids, imagenames in test_loader:  # No labels if your test set is unlabeled
    image = images.to(device)  # Move images to the device where your model is
    with torch.no_grad():  # No gradients needed
        output = model2(image)

        for output, imageid, imagename in zip(output, imageids, imagenames):
            # Structure each prediction as a tuple
            prediction = (output, imageid, imagename)
            predictions.append(prediction) 



#------------------------------------------------For Metrics Calculation
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


# true_labels_path = '../../../testing_set/set4/annotations.csv'
# obj = Metrics()
# obj.metrics(dataframe_predicted_data, true_labels_path, label_to_id)
# obj.call_metrics()

      

#-----------------------------------------------For Vote Validation Visualization
obj_vote_val = ValidateVote()
obj_vote_val.predicted_images(test_set, predictions, label_to_id)


#----------------------------------------------For Visualizing The Predictions
# for test in test_set:   
#     print(test[0].shape)
    # break

# obj_visualize = VisualizePrediction()
# obj_visualize.visualize_predicted_images(test_set, predictions, label_to_id)
# # obj_visualize.visualize_train_set(train_set[1], label_to_id)


#----------------------------------For Bounding Box comparision(Predicted Labels with Ground Truth labels)
# object_compare.labels(test_set, predictions)


# end_time = time.time()
# elapsed_time = end_time - start_time

# print(f"Prediction time: {elapsed_time} seconds.")
# # visualize.visualize_predicted_images(test_image[0], predicted_data[0])









