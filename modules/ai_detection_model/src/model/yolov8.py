from ultralytics import YOLO
import os
import pandas as pd
import csv
import json
from PIL import Image
from torch.utils.data import DataLoader
import torch

from visualize.visualize_yolo import YoloVisualize
from helper_yolo.yolo_normalize import YoloNormalize
from model.metrics_yolo import YoloMetrics
from vote_validation.validate_vote import ValidateVote
from helper_yolo.data_for_vote import CreateDataForVote

obj_visualize = YoloVisualize()
obj_normalize = YoloNormalize()
obj_metrics = YoloMetrics()
obj_vote_validate = ValidateVote()
obj_data_for_vote = CreateDataForVote()

root_path = '../../../'


class ModelYolo():

    # def train_model():
    #     # config_path = os.path.join(root_path, "config.yaml")
    #     # if not os.path.exists(config_path):
    #     #     raise ValueError(f"The specified config path does not exist: {config_path}")

    #     # # Load a model
    #     # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    #     # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    #     # model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    #     # # Training the model
    #     # results = model.train(data=os.path.join(root_path,"config.yaml"), epochs=10)  # train the model

    #     # #validation
    #     # metrics = model.val()  # evaluate model performance on the validation set
    #     # metrics.box.map    # map50-95
    #     # metrics.box.map50  # map50
    #     # metrics.box.map75  # map75
    #     # metrics.box.maps   # a list contains map50-95 of each category

    #     # # = CONCAT("\hline ",B2, " & ",TEXT(G2,"0.00")," & ",TEXT(H2,"0.00")," & \\")
    #     # # Exporting the model
    #     # model.export(format='onnx')

    def predict(self, model_path, test_set, pred_labels_save_path):
        
        model = YOLO(model_path, test_set)
        results = model(test_set, save_txt=None)
        # Process results list
        # for result in results:
        #     boxes = result.boxes
        #     masks = result.masks  # Masks object for segmentation masks outputs
        #     keypoints = result.keypoints  # Keypoints object for pose outputs
        #     scores = result.probs  # Probs object for classification outputs   
        #     labels = result.names    
        #     result.show()  # display to screen    
        #     scores = result.probs  # Assuming this gives you a list of probabilities
        #     result.save(filename='image_0001.jpg')
        #     Find the index of the max probability to get the predicted class index
        #         print(dir(result))
        # result = results.tojson()

        detection_list = []  # Changed variable name from 'list' to 'detection_list'
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                cls = int(box.cls[0])
                path = result.path
                class_name = model.names[cls]
                conf = int(box.conf[0] * 100)
                bx = box.xywh.tolist()
                df = pd.DataFrame({'image_name': path.split("\\")[15],  # Changed index 15 to -1 for general case
                                'label': class_name,
                                'class_id': cls,
                                'score': conf / 100,
                                'box_coord': [bx]})  # Ensuring bx is inside a list
                detection_list.append(df)
        # Concatenate all DataFrames in the list into a single DataFrame
        df = pd.concat(detection_list)
        file_name = 'yolo_predicted_labels.csv' 
        # Save the concatenated DataFrame to a CSV file        
        df.to_csv(os.path.join(pred_labels_save_path, file_name), index=False)


model_path = './runs/detect/train3/weights/best.pt'
test_set = '../../../testing_set/set4/test/' 
pred_labels_path = '../../../yolo_files/'


#------------------------------For prediction-------------------------------------
obj_model = ModelYolo()
obj_model.predict(model_path, test_set, pred_labels_path)


#----------For visualizing the predictions-----------------------------------------
obj_visualize.visualize_test_images(test_set, pred_labels_path)


#--------For normalizing the predicted labels-----------------------
output_save_path = '../../../yolo_files/'
true_labels_path = '../../../testing_set/set4/yolov8'
obj_normalize.normalize(pred_labels_path, output_save_path)
obj_normalize.objeval_format_true_labels(true_labels_path, output_save_path)
obj_normalize.objeval_format_pred_labels(pred_labels_path)
obj_normalize.pred_labels_conerized_normalized(output_save_path)


#---------------For metrics----------------------------------------------------

obj_metrics.metrics(pred_labels_path)
obj_metrics.generate_separate_precision_recall_curves(pred_labels_path)
obj_metrics.call_metrics(pred_labels_path)


#----------------For vote validation------------------------------------------------
test_path = '../../../testing_set/set4/'
obj_data_for_vote.data(output_save_path, test_path)





# with open('detection_results.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['id', 'xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_name'])

#     # Assuming each entry in 'results' corresponds to a single detection in JSON format
#     for image_id, result_json in enumerate(results, start=1):
#         result = result_json.tojson()
#         # detection = json.loads(result)  # Parsing the JSON string to a Python dictionary
#         print(len(result))
        
    # # Access the nested 'box' dictionary for coordinates
        # xmin = detection['box']['x1']
        # ymin = detection['box']['y1']
        # xmax = detection['box']['x2']
        # ymax = detection['box']['y2']
        # print(detection)
        
        # label = detection['name']  # Based on your structure, the label is under 'name'
        # score = detection['confidence']  # The confidence score is under 'confidence'
            
            # # Assuming 'image_name' is available somewhere in your context
            # image_name = "YourImageNameHere"  # Replace or adjust this to get the actual image name

        # Write the extracted information into the CSV file
        # writer.writerow([image_id, xmin, ymin, xmax, ymax, label, score, image_name])

# for result in results[0:1]:
#     data = result.tojson()
#     print(data[0])
# print(len(results))



# with open("predicted_labels.xlsx", 'w', newline='') as file:
#     writer = csv.writer(file)
#     for i in len(results):
#         for idx, prediction in enumerate(results[i].boxes.xywhn): # change final attribute to desired box format
#             cls = int(results[i].boxes.cls[idx].item())
#             # Write line to file in YOLO label format : cls x y w h
#             writer.writerow(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")


# labels = []
# for result in results:
#     boxes = result.boxes.cpu().numpy()
#     for box in boxes:
#         cls = int(box.cls[0])
#         path = result.path
#         class_name = model.names[cls]
#         conf = int(box.conf[0]*100)
#         bx = box.xywh.tolist()
#         df = pd.DataFrame({'image_name': path.split("\\")[15],'label': class_name, 'class_id': cls, 'score': conf/100, 'box_coord': bx})
#         labels.append(df)

# df = pd.concat(labels)
# df.to_csv('predicted_labels.csv', index=False)


