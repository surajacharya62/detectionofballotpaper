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
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image
from torch.cuda.amp import GradScaler, autocast
from vote_checker import CheckVote

import warnings

warnings.filterwarnings('ignore')

check_vote_obj = CheckVote()

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
        self.df = pd.read_csv(os.path.join(image_path, 'annotations.csv'))
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

df = pd.read_csv(os.path.join('../../../datasets1/annotateddataset/', 'annotations.csv'))
label_to_id = {label: i for i, label in enumerate(df['label'].unique())}
print(label_to_id)

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
    

def transform_image(use_tranforms):
    if use_tranforms:
        transforms = T.Compose([
                # T.Resize((224, 224)),  # Example resize, adjust as needed
                T.ToTensor(),               
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization, adjust as needed
            ])

    else:
        transforms = None
    return transforms

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
train_path = 'train' 
test_path = '../../../datasets/ballot/valid'


dataset = ElectoralSymbolDataset(image_path, train_path, use_tranforms=True)

train_set = DataLoader(dataset, batch_size=4,
                                         shuffle= True, 
                                         pin_memory= True if torch.cuda.is_available() else False, collate_fn=collate_fn )


test_set = UnlabeledTestDataset(test_path, transform=transform_image(True))
test_loader = DataLoader(test_set, batch_size=4, shuffle=False)
# print(next(iter(test_loader)))
# print(test_set[500])/
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


torch.cuda.empty_cache()
num_classes = 67

# get the model using our helper function
model = get_object_detection_model(num_classes)

optimizer = torch.optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=0.0005)
num_epochs = 2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


model.to(device)
total_losses = []
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    epoch_loss = 0
    scaler = GradScaler()
    for images, targets in train_set:
        images = [image.to(device) for image in images]        
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]       

        optimizer.zero_grad()
        with autocast():
            loss_dict = model(images, targets) 
            losses = sum(loss for loss in loss_dict.values())

        # Backpropagation
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += losses.item()

    # Average loss for the epoch
    epoch_loss /= len(train_set)
    total_losses.append(epoch_loss)

    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {epoch_loss:.3f}")

##--------------------------------------------Testing the model
torch.cuda.empty_cache()
model.eval()
predictions = []
for images in test_loader:  # No labels if your test set is unlabeled
    images = images.to(device)  # Move images to the device where your model is
    with torch.no_grad():  # No gradients needed
        output = model(images)
        # predictions.append(output)
        predictions.extend(output)  # Use extend to flatten the list if processing batch by batch


def inverse_normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Apply inverse normalization to a tensor."""
    mean = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device)
    std = torch.tensor(std, dtype=tensor.dtype, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor


def visualize_prediction(test_images, predictions, threshold=0.5):
    """
    Visualize the prediction on the image.
    
    Parameters:
    - image: the PIL image
    - prediction: the prediction output from the model
    - threshold: threshold for prediction score
    """
    
    for i, (img, prediction) in enumerate(zip(test_images, predictions)):      

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
            print('label:-----')
            print(label)
            if score > threshold:
                box1 = box.cpu() 
                x1, y1, x2, y2 = box1.numpy()
                label_text = f"{label}"  # Replace `label` with a mapping to the actual class name if you have one
                print(label_text)
                print(label)
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')                
                ax.add_patch(rect)

                # Add label text
               
                ax.text(x1, y1, label_text, color='blue', fontsize=12)

        plt.axis('off')  # Optional: Remove axes for cleaner visualization
        plt.savefig(f'../../../outputdir/output_image_{i}.png', bbox_inches='tight', pad_inches=0)
        plt.close()  
      


if len(predictions) > 0:
    image_tensor, pred = test_set, predictions # Assuming test_set[0] returns a tuple (image, target)
    visualize_prediction(image_tensor, pred, threshold=0.5)
else:
    print("No predictions to visualize.") # 



# check_vote_obj.predicted_images(test_set, predictions)
    




# def check_stamp_validity(stamp_box, grid_cells):
#     """
#     Check if the stamp is entirely within any grid cell and return the valid cell or None.
#     """
#     for cell_box in grid_cells:
#         if (stamp_box[0] >= cell_box[0] and stamp_box[1] >= cell_box[1] and
#                 stamp_box[2] <= cell_box[2] and stamp_box[3] <= cell_box[3]):
#             return cell_box  # Stamp is valid in this cell
#     return None

# def get_voted_symbol(stamp_valid_cell, symbol_boxes):
#     """
#     Determine which symbol the stamp was placed on based on cell overlap.
#     """
#     for symbol_box in symbol_boxes:
#         # Check if the symbol is within the valid cell, this can be adjusted based on requirements
#         if (symbol_box[0] >= stamp_valid_cell[0] and symbol_box[1] >= stamp_valid_cell[1] and
#                 symbol_box[2] <= stamp_valid_cell[2] and symbol_box[3] <= stamp_valid_cell[3]):
#             return symbol_box  # This is the voted symbol
#     return None

# # Assuming detected stamps, defined grid cells, and symbols are in these lists
# detected_stamps = [...]  # List of detected stamps as bounding boxes
# grid_cells = [...]  # List of grid cells as bounding boxes
# symbol_boxes = [...]  # List of symbols as bounding boxes with associated party names

# # Process each detected stamp to check its validity and determine the voted symbol
# for stamp_box in detected_stamps:
#     valid_cell = check_stamp_validity(stamp_box, grid_cells)
#     if valid_cell:
#         voted_symbol = get_voted_symbol(valid_cell, symbol_boxes)
#         if voted_symbol:
#             print("Vote is valid for symbol:", voted_symbol)
#         else:
#             print("Stamp is in a valid position, but no symbol matched. Check for errors.")
#     else:
#         print("Vote is invalid due to stamp placement.")





