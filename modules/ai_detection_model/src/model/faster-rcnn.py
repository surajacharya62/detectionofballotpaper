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