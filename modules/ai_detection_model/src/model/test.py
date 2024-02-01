
# import torch
# import torchvision
# from torchvision import datasets, transforms

# traindir = "../../../datasets/ballot_datasets/training"
# testdir = "../../../datasets/ballot_datasets/testing"

# train_transforms = transforms.Compose([transforms.Resize((500,500)),
#                                        transforms.ToTensor(),                                
#                                        torchvision.transforms.Normalize(
#                                            mean=[0.485, 0.456, 0.406],
#                                            std=[0.229, 0.224, 0.225],
#     ),
#                                        ])

# test_transforms = transforms.Compose([transforms.Resize((500,500)),
#                                       transforms.ToTensor(),
#                                       torchvision.transforms.Normalize(
#                                           mean=[0.485, 0.456, 0.406],
#                                           std=[0.229, 0.224, 0.225],
#     ),
#                                       ])

# train_data = datasets.ImageFolder(traindir,transform=train_transforms)
# test_data = datasets.ImageFolder(testdir,transform=test_transforms)

# trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
# testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)

# import cv2
# import numpy as np
# images, label = next(iter(trainloader))
# print(images.shape[0])
# print(label)

# # for i,data  in enumerate(datas):
# #    print(data.size[0])
# def tensor_to_image(tensor):
#     # Undo normalization
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     tensor = tensor * torch.tensor(std[:, None, None]) + torch.tensor(mean[:, None, None])
#     tensor = torch.clamp(tensor, 0, 1)
    
#     # Convert to numpy and transpose
#     numpy_image = tensor.numpy().transpose(1, 2, 0)
#     numpy_image = (numpy_image * 255).astype(np.uint8)
#     return numpy_image


# for i in range(images.shape[0]):
#     numpy_image = tensor_to_image(images[i])
#     cv2.imshow('Image', numpy_image)
#     cv2.waitKey(0) 

# cv2.destroyAllWindows()


import matplotlib.pyplot as plt

trainloss = [0.7047932147979736,0.7074649333953857,0.7029479146003723,0.6927050352096558,0.6797118782997131]
testloss = [0.7801194787025452,0.7157936692237854,0.7342474460601807,0.692432701587677,0.6031248569488525]


plt.plot(trainloss, label="Training loss")
plt.plot(testloss, label="Val loss")
plt.title('Training and Test Loss for Resnet18')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


