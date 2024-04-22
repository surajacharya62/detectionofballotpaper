from ultralytics import YOLO
import os
import pandas as pd
import csv
import json
from PIL import Image
from torch.utils.data import DataLoader
import torch


root_path = '../../../../'


class ModelYolo():

    def train_model(self, root_path):
        config_path = os.path.join(root_path, "config.yaml")
        if not os.path.exists(config_path):
            raise ValueError(f"The specified config path does not exist: {config_path}")

        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')
        # Training the model
        results = model.train(data=os.path.join(root_path,"config.yaml"), epochs=10)  # train the model

        #validation
        metrics = model.val()  # evaluate model performance on the validation set
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category

        # = CONCAT("\hline ",B2, " & ",TEXT(G2,"0.00")," & ",TEXT(H2,"0.00")," & \\")
        # Exporting the model
        model.export(format='onnx')


obj_yolo_train = ModelYolo()
obj_yolo_train.train_model(root_path)
   