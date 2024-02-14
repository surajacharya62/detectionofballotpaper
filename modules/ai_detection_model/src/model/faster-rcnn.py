import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
import ast


from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import warnings
warnings.filterwarnings('ignore')


# class ElectoralSymbolDataset(Dataset):
#     def __init__(self, image_path, train_or_test_path, transforms): 
#         self.image_path = image_path 
#         self.train_or_test_path = train_or_test_path
#         self.transforms = transforms 
#         self.df = pd.read_csv(image_path + 'annotation1.csv')
#         self.ids = {v:k for k, v in enumerate(np.unique(self.df.image_id.values))}
#         self.imgs_list = list(sorted(os.listdir(os.path.join(image_path, train_or_test_path))))

#         self.classes = [_,""] 
        

#     def __getitem__(self, idx):
#         id_from_path_image = self.imgs_list[idx]       
#         filtered_rows = self.df[self.df.image_id == id_from_path_image]
#         boxes = filtered_rows.iloc[:,1:].values .astype("float")        
#         img = Image.open(image_path + train_path + id_from_path_image).convert('RGB')         
#         labels = torch.ones((boxes.shape[0]),dtype=torch.int64)
#         target = {} 
#         target['boxes'] = torch.tensor(boxes)
#         target['labels'] = labels 
        
#         return T.ToTensor()(img), target
        
#     def __len__(self):
#         return len(self.imgs_list)



class ElectoralSymbolDataset(Dataset):
    def __init__(self, image_path, train_or_test_path, use_tranforms=False):
        self.image_path = image_path
        self.train_or_test_path = train_or_test_path
        # self.transforms = transforms
        self.df = pd.read_csv(os.path.join(image_path, 'annotations1.csv'))
        self.imgs_list = sorted(os.listdir(os.path.join(image_path, train_or_test_path)))
        
        # Extract unique labels and create a mapping to integers
        self.label_to_id = {label: i for i, label in enumerate(self.df['label'].unique())}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}

        if use_tranforms:
            self.transforms = T.Compose([
                # T.Resize((224, 224)),  # Example resize, adjust as needed
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization, adjust as needed
            ])
        else:
            self.transforms = None

    def __getitem__(self, idx):
        img_name = self.imgs_list[idx]
        img_path = os.path.join(self.image_path, self.train_or_test_path, img_name)
        img = Image.open(img_path).convert('RGB')
        
        # Filter rows for the current image and convert boxes to tensor
        filtered_rows = self.df[self.df['image_id'] == img_name]
        boxes = filtered_rows[['x1', 'y1', 'x2', 'y2']].values.astype('float32')
        labels = filtered_rows['label'].apply(lambda x: self.label_to_id[x]).values
        
        # Convert boxes and labels to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {'boxes': boxes, 'labels': labels, 'image_id': torch.tensor([idx])}
        
        # Optionally, apply transformations
        if self.transforms is not None:
            tensor_img = self.transforms(img)
        
        return tensor_img, target 

    def __len__(self):
        return len(self.imgs_list)


def collate_fn(batch):
    """
    Custom collate function for handling batches of images with different
    numbers of bounding boxes (targets).

    Args:
    - batch: A list of tuples (image, target), where 'image' is the tensor
             representing the image, and 'target' is a dictionary containing
             the bounding boxes and labels for the image.

    Returns:
    - A tuple of two lists: (images, targets), where 'images' is a list of
      image tensors, and 'targets' is a list of target dictionaries.
    """
    images = [item[0] for item in batch]  # Extract images
    targets = [item[1] for item in batch]  # Extract targets

    # Images can be stacked because they should have the same size
    # (if your transformations ensure this). If not, you need to handle varying image sizes.
    images = torch.stack(images, 0)

    # Targets are returned as a list of dictionaries because their sizes may vary
    return images, targets


image_path = '../../../datasets1/annotateddataset/'
train_path = 'train/' 
test_path = 'test'


dataset = ElectoralSymbolDataset(image_path, train_path, use_tranforms=True)

train_set = torch.utils.data.DataLoader(dataset, batch_size=4,
                                         shuffle= True, 
                                         pin_memory= True if torch.cuda.is_available() else False, collate_fn=collate_fn )





#########-------------------------------------------------- For visualization

# def plot_img_bbox(img, target):
#     # plot the image and bboxes
#     # Bounding boxes are defined as follows: x-min y-min width height
#     img = img.permute(1, 2, 0).numpy() 
#     fig, a = plt.subplots(1,1)
#     fig.set_size_inches(5,5)
#     a.imshow(img)
#     for box in (target['boxes']):

#         x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
#         rect = patches.Rectangle((x, y),
#                                  width, height,
#                                  linewidth = 2,
#                                  edgecolor = 'r',
#                                  facecolor = 'none')

#         # Draw the bounding box on top of the image
#         a.add_patch(rect)
#     plt.show()
    
# # plotting the image with bboxes. Feel free to change the index

# img, target = dataset[220]
# plot_img_bbox(img, target)


######---------------------------------------------- Model preparation
def get_object_detection_model(num_classes):

    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    return model



num_classes = 57

# get the model using our helper function
model = get_object_detection_model(num_classes)

optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0005)
num_epochs = 5

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model.to(device)
total_losses = []
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0

    for images, targets in train_set:
        images = [image.to(device) for image in images]        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]       

        optimizer.zero_grad()

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    # Average loss for the epoch
    epoch_loss /= len(train_set)
    total_losses.append(epoch_loss)

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}")

    # for data in train_set:
    #     imgs = []
    #     targets = []
    #     for d in data:
    #         # print(d)
    #         imgs.append(d[0].to(device))
    #         targ = {}
    #         targ['boxes'] = d[1]['boxes'].to(device)
    #         targ['labels'] = d[1]['label'].to(device)
    #         targets.append(targ)
    #     loss_dict = model(imgs,targets)
    #     loss = sum(v for v in loss_dict.values())
    #     epochs_loss += loss.cpu().detach.numpy()
    #     optimizer.zero_grad()
    #     loss.backward()
    # print(epochs_loss)






