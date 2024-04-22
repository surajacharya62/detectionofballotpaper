import torch
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from vote_validation.validate_vote_yolo import ValidateVote
from helper_yolo.compare_bounding_boxes import CompareBoundingBox
from helper_yolo.reshape_compared_boxes import ReshapeData
import torchvision.transforms as T



class ElectoralSymbolDataset(Dataset):
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

        return img, image_id, img_name , target

    def __len__(self):
        return len(self.imgs_list)
    
 


class CreateDataForVote():

    def get_transform(self, train):
        transforms = []
        if train:
            transforms = T.Compose([
                    # T.Resize((224, 224)),  # Example resize, adjust as needed
                    T.ToTensor(),
                    # T.RandomHorizontalFlip(p=0.5),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example normalization, adjust as needed
                ])  

        return transforms   


    def data(self, yolo_files, test_files):

        test_set_path = '../../../../testing_set/set6/'
        # yolo_files_path = '../../../yolo_files/'
        test = 'test'

        df = pd.read_csv(os.path.join(test_set_path, 'annotations.csv'))
        label_to_id = {label: i for i, label in enumerate(df['label'].unique())}

        sorted_labels = sorted(label_to_id)

        labe_id = {label: i for i, label in enumerate(sorted_labels)}

        test_set = ElectoralSymbolDataset(test_set_path, test, labe_id , self.get_transform(True))
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
        
        transformed_data = []

        # Assuming all rows correspond to a single image or you want to group by 'image_name'
        # If you need to group by 'image_name', you can adjust the code accordingly
        pred_df = pd.read_excel(os.path.join(yolo_files, 'pred_labels_conerized_normalized.xlsx'))

        # boxes = pred_df[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
        # labels = pred_df['label'].tolist()
        # scores = pred_df['score'].tolist()

        # transformed_data.append({
        #     'boxes': boxes,
        #     'labels': labels,
        #     'scores': scores
        # })

        df = pd.read_csv(os.path.join(test_files, 'annotations.csv'))
        label_to_id = {label: i for i, label in enumerate(df['label'].unique())}
        sorted_labels = sorted(label_to_id)

        obj_val = ValidateVote()
        obj_val.predicted_images(test_set, pred_df, labe_id)

        obj_com = CompareBoundingBox()
        obj_com.labels(test_set, pred_df, labe_id)

        obj_reshape = ReshapeData()
        file_path = '../../../../yolo_files/df_total_comparisions.xlsx'
        obj_reshape.process_and_reshape_data_v2(file_path)


    
