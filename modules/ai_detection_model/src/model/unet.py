import torch
import torch.nn as nn
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision
import torch.optim as optim

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt




def double_conv(in_channel, out_channel):
    conv = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channel, out_channel, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv


def crop_img(tensor, target_tensor):
    # target_size = target_tensor.size()[2]
    # tensor_size = tensor.size()[2]
    # delta = tensor_size - target_size 
    # delta = delta // 2
    # return tensor[:,:,delta:tensor_size-delta, delta:tensor_size-delta]
    target_size = target_tensor.size()[2:4]  # Get the height and width of the target tensor
    tensor_size = tensor.size()[2:4]  # Get the height and width of the tensor to be cropped

    delta_height = tensor_size[0] - target_size[0]
    delta_width = tensor_size[1] - target_size[1]

    delta_height = delta_height // 2
    delta_width = delta_width // 2

    return tensor[:,:,delta_height:delta_height+target_size[0], delta_width:delta_width+ target_size[1]]

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        self.final_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(1024, 1)

        # self.up_trans_1 = nn.ConvTranspose2d(
        #     in_channels= 1024,
        #     out_channels= 512,
        #     kernel_size=2,
        #     stride=2
        #     )        
        # self.up_conv_1 = double_conv(1024, 512)

        # self.up_trans_2 = nn.ConvTranspose2d(
        #     in_channels= 512,
        #     out_channels= 256,
        #     kernel_size=2,
        #     stride=2
        #     )        
        # self.up_conv_2 = double_conv(512, 256)

        # self.up_trans_3 = nn.ConvTranspose2d(
        #     in_channels= 256,
        #     out_channels= 128,
        #     kernel_size=2,
        #     stride=2
        #     )        
        # self.up_conv_3 = double_conv(256, 128)

        # self.up_trans_4 = nn.ConvTranspose2d(
        #     in_channels= 128, 
        #     out_channels= 64,
        #     kernel_size=2,
        #     stride=2
        #     )        
        # self.up_conv_4 = double_conv(128, 64)

        # self.out = nn.Conv2d(
        #     in_channels=64,
        #     out_channels=2,
        #     kernel_size=1
        # )
    
    #encoder
    def forward(self, image):
        # print(image.size())
        x1 = self.down_conv_1(image)
        # print(x1.size())
        x2 = self.max_pool_2x2(x1)
        # print(x2.size())
        x3 = self.down_conv_2(x2)
        # print(x3.size())
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)

        # print(x9.size())
        
        #decoder
        # x = self.up_trans_1(x9)
        # y = crop_img(x7, x)
        # x = self.up_conv_1(torch.cat([x, y], 1))

        # x = self.up_trans_2(x)
        # y = crop_img(x5, x)
        # x = self.up_conv_2(torch.cat([x, y], 1))

        # x = self.up_trans_3(x)
        # y = crop_img(x3, x)
        # x = self.up_conv_3(torch.cat([x, y], 1))

        # x = self.up_trans_4(x)
        # y = crop_img(x1, x)
        # x = self.up_conv_4(torch.cat([x, y], 1))

        # x = self.out(x)


        x = self.final_pool(x9)
        x = x.view(x.size(0), -1)
    
        # x = torch.sigmoid(self.classifier(x))
        x = self.classifier(x)
        # print(x.size())
        return x
        
        # print(x7.size())
        # print(y.size())


class BallotPaperDataset(Dataset):
    def __init__(self, valid_dir, invalid_dir, transform=None):
        self.transform = transform

        # Get paths and labels for valid images
        valid_paths = [os.path.join(valid_dir, fname) for fname in os.listdir(valid_dir) if fname.endswith(('.jpg','.jpeg','.png'))]
        valid_labels = [1] * len(valid_paths)  # 1 for valid
        # print(valid_labels)

        # Get paths and labels for invalid images
        invalid_paths = [os.path.join(invalid_dir, fname) for fname in os.listdir(invalid_dir) if fname.endswith(('.jpg','.jpeg','.png'))]
        invalid_labels = [0] * len(invalid_paths)  # 0 for invalid
        # print(invalid_labels)

        # Combine the paths and labels
        self.image_paths = valid_paths + invalid_paths
        self.labels = valid_labels + invalid_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image_file = cv2.imread(img_path)
        image_file = cv2.cvtColor(image_file, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        if self.transform:
            image = self.transform(image_file)

        label = self.labels[idx]
        return image, label


transforms = transforms.Compose([
    # transforms.Resize(572, 572),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


valid_file_path = "E://Oslo//OsloMet//Fourth semester//ballot//valid samples/train"
invalid_file_path = "E://Oslo//OsloMet//Fourth semester//ballot//Invalid samples/train"
valid_file_path_test = "E://Oslo//OsloMet//Fourth semester//ballot//valid samples/test"
invalid_file_path_test = "E://Oslo//OsloMet//Fourth semester//ballot//Invalid samples/test"

# Instantiate dataset
train_dataset = BallotPaperDataset(
    valid_file_path, invalid_file_path, transform=transforms
    )
# test_dataset = BallotPaperDataset('path/to/test/dataset', transform=your_transforms)


test_dataset = BallotPaperDataset(
    valid_file_path_test, invalid_file_path_test, transform=transforms
    )

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
model = UNet() 

# Loss function and optimer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        print(i)
        inputs, labels = data 
        # print('image:'+ str(inputs), 'Label:' + str(labels))
        inputs, labels = inputs.float(), labels.float()

        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = torch.squeeze(outputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        print(running_loss)
        # if i % 1 == 1:
        #     print(f'[{epoch + 1}, {i+1:5d}] loss: { running_loss / 10:.3f}') 
        #     running_loss = 0.0


print("Finished Training")





# dataiter = iter(train_loader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# print(' '.join(f'{labels[j]}' for j in range(batch_size)))
# print(labels)
# for lable in labels:
#     print(lable)



# showing the image by reversing the tensor to numpy format
# for batch in train_loader:
#     images = batch
#     for image in images:
#         # Reverse the normalization and convert to numpy
#         image = image.numpy().transpose((1, 2, 0))  # Convert from CHW to HWC format
#         image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
#         image = np.clip(image, 0, 1)  # Ensure the image is in [0, 1] range
#         image = (image * 255).astype(np.uint8)  # Convert to uint8

#         # Convert from RGB (PyTorch) to BGR (OpenCV)
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

#         # Display the image
#         cv2.imshow("Image", image)
#         cv2.waitKey(0)

# cv2.destroyAllWindows()

# if __name__ == "__main__":
#     image = torch.rand((1, 1, 572, 572)) # batch_size, channel, height, width
#     # print(image[0][2][2])
#     # print(len(image[0][0]))
#     model = UNet() 
#     # print(model(image))   



PATH = './u_net.pth'
torch.save(model.state_dict(), PATH)


dataiter_test = iter(test_loader)
images, labels = next(dataiter_test)



# print('GroundTruth: ', ' '.join(f'{labels[j]}' for j in range(4)))


model_test = UNet()
model_test.load_state_dict(torch.load(PATH))

outputs = model_test(images)
print('test-outputs:'+ str(outputs))

_, predicted = torch.max(outputs, 1)

image_predicted = []

print('Predicted: ', ' '.join(f'{predicted[j]}'
                              for j in range(4)))


def imshow(img):
    
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # plt.title()
    plt.show()


import torchvision.transforms.functional as TF
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = model_test(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()

        for i in range(len(images)):
            img = images[i]
            pred = predicted[i]
            print(pred)
            imshow(img)
            
            
            # img = TF.to_pil_image(img)
            # plt.imshow(img)
            # plt.title(f'Predicted:{pred.item()}')

        break   

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')




# print images
# imshow(torchvision.utils.make_grid(image_predicted))




