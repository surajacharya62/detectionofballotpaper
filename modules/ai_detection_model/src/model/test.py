
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
import numpy as np
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

# train = [np.array(0.7005201, dtype=float32), array(0.69502616, dtype=float32), array(0.6918939, dtype=float32), array(0.6907588, dtype=float32), array(0.68251705, dtype=float32), array(0.7193494, dtype=float32), array(0.71271116, dtype=float32), array(0.6991049, dtype=float32), array(0.6822881, dtype=float32), array(0.6719694, dtype=float32), array(0.67873454, dtype=float32), array(0.67182237, dtype=float32), array(0.66507864, dtype=float32), array(0.6702996, dtype=float32), array(0.666229, dtype=float32), array(0.6793914, dtype=float32), array(0.66706353, dtype=float32), array(0.6706848, dtype=float32), array(0.672568, dtype=float32), array(0.6658086, dtype=float32), array(0.663919, dtype=float32), array(0.6664936, dtype=float32), array(0.67700815, dtype=float32), array(0.65485543, dtype=float32), array(0.6726405, dtype=float32), array(0.65289664, dtype=float32), array(0.6508676, dtype=float32), array(0.6627098, dtype=float32), array(0.6654881, dtype=float32), array(0.65696156, dtype=float32), array(0.6702744, dtype=float32), array(0.6419332, dtype=float32), array(0.6440389, dtype=float32), array(0.6483857, dtype=float32), array(0.6455904, dtype=float32), array(0.63974196, dtype=float32), array(0.64787865, dtype=float32), array(0.65279025, dtype=float32), array(0.64690757, dtype=float32), array(0.6484465, dtype=float32), array(0.64069057, dtype=float32), array(0.6344263, dtype=float32), array(0.6450904, dtype=float32), array(0.6436385, dtype=float32), array(0.6413799, dtype=float32), array(0.6381174, dtype=float32), array(0.6372319, dtype=float32), array(0.6305471, dtype=float32), array(0.629254, dtype=float32), array(0.6324579, dtype=float32)]
# loss = [array(1.1081887, dtype=float32), array(0.5994436, dtype=float32), array(0.7301602, dtype=float32), array(1.0985765, dtype=float32), array(0.65765667, dtype=float32), array(0.58383393, dtype=float32), array(0.7071645, dtype=float32), array(0.68641216, dtype=float32), array(0.61261445, dtype=float32), array(0.69518703, dtype=float32), array(0.6842506, dtype=float32), array(0.57812047, dtype=float32), array(0.6435528, dtype=float32), array(0.6701226, dtype=float32), array(0.5586138, dtype=float32), array(0.6717968, dtype=float32), array(0.61406386, dtype=float32), array(0.6595597, dtype=float32), array(0.71634126, dtype=float32), array(0.43451324, dtype=float32), array(0.73738647, dtype=float32), array(0.7565312, dtype=float32), array(0.8755924, dtype=float32), array(0.8178384, dtype=float32), array(0.6090582, dtype=float32), array(0.5485822, dtype=float32), array(0.7065742, dtype=float32), array(0.7771247, dtype=float32), array(1.3289089, dtype=float32), array(0.5684599, dtype=float32), array(0.7417024, dtype=float32), array(0.64558446, dtype=float32), array(0.6608634, dtype=float32), array(0.62802124, dtype=float32), array(0.6918328, dtype=float32), array(0.6330901, dtype=float32), array(0.6004498, dtype=float32), array(0.64431673, dtype=float32), array(0.5459239, dtype=float32), array(0.5713643, dtype=float32), array(0.60349417, dtype=float32), array(0.6884322, dtype=float32), array(0.7491126, dtype=float32), array(0.7616717, dtype=float32), array(0.63557833, dtype=float32), array(0.638938, dtype=float32), array(0.70609385, dtype=float32), array(0.71591455, dtype=float32), array(0.5678423, dtype=float32), array(0.6491304, dtype=float32)]



import matplotlib.pyplot as plt

# trainloss = [0.7047932147979736,0.7074649333953857,0.7029479146003723,0.6927050352096558,0.6797118782997131]
# testloss = [0.7801194787025452,0.7157936692237854,0.7342474460601807,0.692432701587677,0.6031248569488525]
train = [np.array(0.7005201, dtype=np.float32), np.array(0.69502616, dtype=np.float32), np.array(0.6918939, dtype=np.float32), np.array(0.6907588, dtype=np.float32), np.array(0.68251705, dtype=np.float32), np.array(0.7193494, dtype=np.float32), np.array(0.71271116, dtype=np.float32), np.array(0.6991049, dtype=np.float32), np.array(0.6822881, dtype=np.float32), np.array(0.6719694, dtype=np.float32), np.array(0.67873454, dtype=np.float32), np.array(0.67182237, dtype=np.float32), np.array(0.66507864, dtype=np.float32), np.array(0.6702996, dtype=np.float32), np.array(0.666229, dtype=np.float32), np.array(0.6793914, dtype=np.float32), np.array(0.66706353, dtype=np.float32), np.array(0.6706848, dtype=np.float32), np.array(0.672568, dtype=np.float32), np.array(0.6658086, dtype=np.float32), np.array(0.663919, dtype=np.float32), np.array(0.6664936, dtype=np.float32), np.array(0.67700815, dtype=np.float32), np.array(0.65485543, dtype=np.float32), np.array(0.6726405, dtype=np.float32), np.array(0.65289664, dtype=np.float32), np.array(0.6508676, dtype=np.float32), np.array(0.6627098, dtype=np.float32), np.array(0.6654881, dtype=np.float32), np.array(0.65696156, dtype=np.float32), np.array(0.6702744, dtype=np.float32), np.array(0.6419332, dtype=np.float32), np.array(0.6440389, dtype=np.float32), np.array(0.6483857, dtype=np.float32), np.array(0.6455904, dtype=np.float32), np.array(0.63974196, dtype=np.float32), np.array(0.64787865, dtype=np.float32), np.array(0.65279025, dtype=np.float32), np.array(0.64690757, dtype=np.float32), np.array(0.6484465, dtype=np.float32), np.array(0.64069057, dtype=np.float32), np.array(0.6344263, dtype=np.float32), np.array(0.6450904, dtype=np.float32), np.array(0.6436385, dtype=np.float32), np.array(0.6413799, dtype=np.float32), np.array(0.6381174, dtype=np.float32), np.array(0.6372319, dtype=np.float32), np.array(0.6305471, dtype=np.float32), np.array(0.629254, dtype=np.float32), np.array(0.6324579, dtype=np.float32)]

# Convert train list to a NumPy array for easier handling
train_values = np.array(train)


loss = [np.array(1.1081887, dtype=np.float32), np.array(0.5994436, dtype=np.float32), np.array(0.7301602, dtype=np.float32), np.array(1.0985765, dtype=np.float32), np.array(0.65765667, dtype=np.float32), np.array(0.58383393, dtype=np.float32), np.array(0.7071645, dtype=np.float32), np.array(0.68641216, dtype=np.float32), np.array(0.61261445, dtype=np.float32), np.array(0.69518703, dtype=np.float32), np.array(0.6842506, dtype=np.float32), np.array(0.57812047, dtype=np.float32), np.array(0.6435528, dtype=np.float32), np.array(0.6701226, dtype=np.float32), np.array(0.5586138, dtype=np.float32), np.array(0.6717968, dtype=np.float32), np.array(0.61406386, dtype=np.float32), np.array(0.6595597, dtype=np.float32), np.array(0.71634126, dtype=np.float32), np.array(0.43451324, dtype=np.float32), np.array(0.73738647, dtype=np.float32), np.array(0.7565312, dtype=np.float32), np.array(0.8755924, dtype=np.float32), np.array(0.8178384, dtype=np.float32), np.array(0.6090582, dtype=np.float32), np.array(0.5485822, dtype=np.float32), np.array(0.7065742, dtype=np.float32), np.array(0.7771247, dtype=np.float32), np.array(1.3289089, dtype=np.float32), np.array(0.5684599, dtype=np.float32), np.array(0.7417024, dtype=np.float32), np.array(0.64558446, dtype=np.float32), np.array(0.6608634, dtype=np.float32), np.array(0.62802124, dtype=np.float32), np.array(0.6918328, dtype=np.float32), np.array(0.6330901, dtype=np.float32), np.array(0.6004498, dtype=np.float32), np.array(0.64431673, dtype=np.float32), np.array(0.5459239, dtype=np.float32), np.array(0.5713643, dtype=np.float32), np.array(0.60349417, dtype=np.float32), np.array(0.6884322, dtype=np.float32), np.array(0.7491126, dtype=np.float32), np.array(0.7616717, dtype=np.float32), np.array(0.63557833, dtype=np.float32), np.array(0.638938, dtype=np.float32), np.array(0.70609385, dtype=np.float32), np.array(0.71591455, dtype=np.float32), np.array(0.5678423, dtype=np.float32), np.array(0.6491304, dtype=np.float32)]

# Convert the list of np.array values to a single np.array for easier handling and plotting
loss_values = np.array(loss)

plt.plot(train_values, label="Training loss")
plt.plot(loss_values, label="Val loss")
plt.title('Training and Test Loss for Resnet18')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


