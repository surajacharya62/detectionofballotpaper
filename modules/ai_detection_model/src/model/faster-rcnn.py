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


class GlobalWheatDataset(Dataset):
    def __init__(self, path, train_or_test_path, transforms):
        self.path = path
        self.train_or_test_path = train_or_test_path
        self.transforms = transforms

        self.df = pd.read_csv(path + 'train.csv')

        self.ids = {v:k for k, v in enumerate(np.unique(self.df.image_id.values))}
        self.imgs_list = list(sorted(os.listdir(os.path.join(path, train_or_test_path))))

    def get_rectangles(self, idx):
        id = self.imgs_list[idx].split('/')[-1].split('.jpg')[0]
        rectangles = []

        for box in self.df[self.df.image_id == id]['bbox'].values:
            bbox = ast.literal_eval(box)
            x = bbox[0]
            y = bbox[1]
            w = bbox[2]
            h = bbox[3]
            rectangles.append(patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='r',facecolor='none'))

        return rectangles

    def format_boxes(self, boxes):
        # replace width, height with xmax, ymax
        try:
            boxes[:, 2] =  boxes[:, 2] + boxes[:, 0]
            boxes[:, 3] =  boxes[:, 3] + boxes[:, 1]
        except:
            pass
        return boxes

    def get_image(self, idx):
        img_path = os.path.join(self.path, self.train_or_test_path, self.imgs_list[idx])
        return np.array(Image.open(img_path).convert("RGB"))

    def draw(self, idx):
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(dataset.get_image(idx))
        for rectangle in dataset.get_rectangles(idx):
            ax.add_patch(rectangle)
        plt.show

    def __getitem__(self, idx):
        id = self.df.iloc[idx].image_id
        boxes = np.int64(np.array([ast.literal_eval(box) for box in self.df[self.df.image_id == id]['bbox'].values]))

        # format boxes width, height
        boxes = self.format_boxes(boxes)
        
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.int64)
        target["labels"] = torch.ones((len(boxes),), dtype=torch.int64)
        target["image_id"] = torch.tensor([self.ids[id]])
        target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        img_path = os.path.join(self.path, self.train_or_test_path, self.imgs_list[idx])
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            img = self.transforms(img)

        return T.ToTensor()(img), target

    def __len__(self):
        return len(self.imgs_list)


def get_transform(train):
    transforms = []
    if train:
        # random horizontal flip with 50% probability
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

path = '../../../datasets/adhunik_nepal_samajwadi_party/'
train_path = 'train'
test_path = 'test'


dataset = GlobalWheatDataset(path, train_path, get_transform(train=True))
dataset.draw(8)

