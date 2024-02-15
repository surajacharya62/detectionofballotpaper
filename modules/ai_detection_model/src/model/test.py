
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
# train = [np.array(0.7005201, dtype=np.float32), np.array(0.69502616, dtype=np.float32), np.array(0.6918939, dtype=np.float32), np.array(0.6907588, dtype=np.float32), np.array(0.68251705, dtype=np.float32), np.array(0.7193494, dtype=np.float32), np.array(0.71271116, dtype=np.float32), np.array(0.6991049, dtype=np.float32), np.array(0.6822881, dtype=np.float32), np.array(0.6719694, dtype=np.float32), np.array(0.67873454, dtype=np.float32), np.array(0.67182237, dtype=np.float32), np.array(0.66507864, dtype=np.float32), np.array(0.6702996, dtype=np.float32), np.array(0.666229, dtype=np.float32), np.array(0.6793914, dtype=np.float32), np.array(0.66706353, dtype=np.float32), np.array(0.6706848, dtype=np.float32), np.array(0.672568, dtype=np.float32), np.array(0.6658086, dtype=np.float32), np.array(0.663919, dtype=np.float32), np.array(0.6664936, dtype=np.float32), np.array(0.67700815, dtype=np.float32), np.array(0.65485543, dtype=np.float32), np.array(0.6726405, dtype=np.float32), np.array(0.65289664, dtype=np.float32), np.array(0.6508676, dtype=np.float32), np.array(0.6627098, dtype=np.float32), np.array(0.6654881, dtype=np.float32), np.array(0.65696156, dtype=np.float32), np.array(0.6702744, dtype=np.float32), np.array(0.6419332, dtype=np.float32), np.array(0.6440389, dtype=np.float32), np.array(0.6483857, dtype=np.float32), np.array(0.6455904, dtype=np.float32), np.array(0.63974196, dtype=np.float32), np.array(0.64787865, dtype=np.float32), np.array(0.65279025, dtype=np.float32), np.array(0.64690757, dtype=np.float32), np.array(0.6484465, dtype=np.float32), np.array(0.64069057, dtype=np.float32), np.array(0.6344263, dtype=np.float32), np.array(0.6450904, dtype=np.float32), np.array(0.6436385, dtype=np.float32), np.array(0.6413799, dtype=np.float32), np.array(0.6381174, dtype=np.float32), np.array(0.6372319, dtype=np.float32), np.array(0.6305471, dtype=np.float32), np.array(0.629254, dtype=np.float32), np.array(0.6324579, dtype=np.float32)]

# # Convert train list to a NumPy array for easier handling
# train_values = np.array(train)


# loss = [np.array(1.1081887, dtype=np.float32), np.array(0.5994436, dtype=np.float32), np.array(0.7301602, dtype=np.float32), np.array(1.0985765, dtype=np.float32), np.array(0.65765667, dtype=np.float32), np.array(0.58383393, dtype=np.float32), np.array(0.7071645, dtype=np.float32), np.array(0.68641216, dtype=np.float32), np.array(0.61261445, dtype=np.float32), np.array(0.69518703, dtype=np.float32), np.array(0.6842506, dtype=np.float32), np.array(0.57812047, dtype=np.float32), np.array(0.6435528, dtype=np.float32), np.array(0.6701226, dtype=np.float32), np.array(0.5586138, dtype=np.float32), np.array(0.6717968, dtype=np.float32), np.array(0.61406386, dtype=np.float32), np.array(0.6595597, dtype=np.float32), np.array(0.71634126, dtype=np.float32), np.array(0.43451324, dtype=np.float32), np.array(0.73738647, dtype=np.float32), np.array(0.7565312, dtype=np.float32), np.array(0.8755924, dtype=np.float32), np.array(0.8178384, dtype=np.float32), np.array(0.6090582, dtype=np.float32), np.array(0.5485822, dtype=np.float32), np.array(0.7065742, dtype=np.float32), np.array(0.7771247, dtype=np.float32), np.array(1.3289089, dtype=np.float32), np.array(0.5684599, dtype=np.float32), np.array(0.7417024, dtype=np.float32), np.array(0.64558446, dtype=np.float32), np.array(0.6608634, dtype=np.float32), np.array(0.62802124, dtype=np.float32), np.array(0.6918328, dtype=np.float32), np.array(0.6330901, dtype=np.float32), np.array(0.6004498, dtype=np.float32), np.array(0.64431673, dtype=np.float32), np.array(0.5459239, dtype=np.float32), np.array(0.5713643, dtype=np.float32), np.array(0.60349417, dtype=np.float32), np.array(0.6884322, dtype=np.float32), np.array(0.7491126, dtype=np.float32), np.array(0.7616717, dtype=np.float32), np.array(0.63557833, dtype=np.float32), np.array(0.638938, dtype=np.float32), np.array(0.70609385, dtype=np.float32), np.array(0.71591455, dtype=np.float32), np.array(0.5678423, dtype=np.float32), np.array(0.6491304, dtype=np.float32)]

# # Convert the list of np.array values to a single np.array for easier handling and plotting
# loss_values = np.array(loss)

# plt.plot(train_values, label="Training loss")
# plt.plot(loss_values, label="Val loss")
# plt.title('Training and Test Loss for Resnet18')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()

import torch
predictions = {'boxes': torch.tensor([[ 43.7332, 213.5269, 111.2414, 372.6928],
        [144.2942, 143.8887, 360.3079, 183.2080],
        [140.0604, 146.5337, 384.2130, 184.0802],
        [187.1325, 212.2351, 252.5611, 297.4113],
        [189.3467, 317.2808, 251.1578, 447.3797],
        [327.7851, 197.9452, 394.5064, 430.7094],
        [179.9222, 216.4984, 255.0771, 326.7257],
        [ 36.9537,  52.4798, 162.3382, 135.8132],
        [ 27.5536, 224.6903, 111.1299, 449.6593],
        [400.1011, 209.0044, 469.4872, 318.3181],
        [108.4521, 210.8337, 179.2451, 289.4620],
        [ 44.2276, 350.2479, 109.8677, 440.1537],
        [256.3594, 201.5649, 324.1126, 322.2958],
        [111.0046, 214.4788, 183.1559, 264.7868],
        [395.4152, 201.5765, 473.2253, 403.9206],
        [ 44.2392, 360.4931, 105.9727, 441.3641],
        [185.8333, 304.4647, 250.1510, 438.9623],
        [256.7484, 315.1017, 326.6691, 433.4090],
        [ 39.7129, 221.7265, 105.5536, 425.4415],
        [ 31.0928, 359.9680, 105.8520, 444.7391],
        [327.2137, 208.6843, 395.3152, 318.6118],
        [ 41.9462, 214.9508, 103.1827, 316.7636],
        [330.9976, 328.9052, 395.7101, 436.6003],
        [112.5121, 308.5482, 182.3063, 442.4894],
        [ 39.6219,  49.8180, 155.1566, 133.5434],
        [180.9735, 304.1204, 255.2358, 437.1417],
        [ 32.3822, 220.3020, 115.3819, 319.4560],
        [403.3217, 217.2181, 470.8431, 411.5781],
        [108.3221, 202.4852, 179.7579, 383.3499],
        [106.6256, 205.0252, 183.3906, 350.2986],
        [393.4212, 206.8400, 475.0756, 414.0876],
        [330.4275, 200.3359, 396.0984, 427.1608],
        [331.2699, 208.0384, 396.2911, 317.1698],
        [325.2518, 335.1362, 394.4467, 438.0014],
        [188.0617, 202.1207, 256.6792, 374.2356],
        [188.0323, 211.3035, 253.1574, 430.1757],
        [176.5048, 314.5686, 249.1274, 452.0683],
        [182.9795, 207.7845, 254.0110, 308.3762],
        [181.3702, 212.6850, 255.1862, 296.9006],
        [114.6198, 220.7239, 181.7154, 348.8929],
        [398.0839, 212.8182, 474.4420, 317.7780],
        [109.6282, 215.9647, 185.4577, 268.5195],
        [326.7979, 205.9993, 394.6747, 316.8679],
        [110.3082, 308.6912, 183.2275, 432.4457],
        [332.3318, 213.0410, 392.8253, 324.2051],
        [103.4397, 306.3077, 185.7821, 436.7786],
        [ 34.1547,  71.2756, 157.5084, 137.0765],
        [394.4037, 209.0485, 470.2132, 325.3848],
        [ 44.2555,  58.9437, 257.2470, 108.2195],
        [392.8712, 185.7965, 472.4409, 430.6940],
        [178.5426, 268.7610, 252.6142, 442.1004],
        [181.5426, 212.4068, 251.4391, 430.6089],
        [134.3986, 142.7555, 367.3970, 179.2872],
        [399.7759, 199.8803, 476.8888, 368.7017],
        [182.5879, 203.7275, 250.8727, 331.0030],
        [328.7737, 347.1626, 391.9911, 444.6764],
        [183.0701, 276.8831, 253.3894, 437.3199],
        [ 22.1461,  48.7611, 169.0864, 133.5777],
        [262.3084, 336.2265, 327.7475, 440.2613],
        [333.7722, 221.1070, 392.4583, 438.5511],
        [188.0934, 222.5310, 250.5706, 326.8428],
        [134.9425, 140.6096, 376.7438, 182.6516],
        [396.3344, 341.4020, 469.3115, 420.6736]], device='cuda:0'), 'labels': torch.tensor([10, 25, 14, 11, 10, 12,  2, 25, 14, 12, 12, 10, 12, 20, 12, 25, 11, 12,
        25, 14, 12, 25, 11, 10, 10, 12, 14, 10, 12, 20, 28, 11, 11, 12, 11, 10,
        14, 28, 12, 10, 28,  4, 28, 11, 10, 12,  2, 20, 25, 20, 20,  2, 20, 11,
        20, 10, 28, 14, 10, 10, 10, 10, 25], device='cuda:0'), 'scores': torch.tensor([0.4960, 0.4687, 0.3923, 0.3652, 0.3295, 0.2598, 0.2555, 0.2408, 0.2322,
        0.2234, 0.2140, 0.2116, 0.1990, 0.1882, 0.1701, 0.1556, 0.1522, 0.1453,
        0.1443, 0.1338, 0.1289, 0.1265, 0.1195, 0.1187, 0.1129, 0.1099, 0.1054,
        0.1009, 0.0948, 0.0926, 0.0904, 0.0849, 0.0830, 0.0828, 0.0826, 0.0820,
        0.0793, 0.0776, 0.0765, 0.0756, 0.0756, 0.0706, 0.0683, 0.0661, 0.0648,
        0.0639, 0.0629, 0.0620, 0.0619, 0.0591, 0.0587, 0.0582, 0.0568, 0.0565,
        0.0559, 0.0531, 0.0529, 0.0528, 0.0527, 0.0520, 0.0516, 0.0510, 0.0504],
       device='cuda:0')}



image = torch.tensor([[2.2489, 2.2489, 2.2489,  ..., 2.2489, 2.2489, 2.2489],
        [2.2489, 2.2489, 2.2489,  ..., 2.2489, 2.2489, 2.2489],
        [2.2489, 2.2489, 2.2489,  ..., 2.2489, 2.2489, 2.2489],
        ...,
        [2.2489, 2.2489, 2.2489,  ..., 2.2489, 2.2489, 2.2489],
        [2.2489, 2.2489, 2.2489,  ..., 2.2489, 2.2489, 2.2489],
        [2.2489, 2.2489, 2.2489,  ..., 2.2489, 2.2489, 2.2489]])


import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Example normalization parameters, adjust as necessary
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(tensor, means, stds):
    """Denormalizes image tensors using mean and std"""
    denormalized = torch.clone(tensor)
    for t, m, s in zip(denormalized, means, stds):
        t.mul_(s).add_(m)
    return denormalized

def visualize_prediction(image_tensor, prediction, threshold=0.5):
    """
    Visualize the prediction on the image tensor.
    
    Parameters:
    - image_tensor: the image tensor in CxHxW format
    - prediction: the prediction output from the model
    - threshold: threshold for prediction score
    """
    # Denormalize the image
    image_tensor = denormalize(image_tensor, mean, std)
    
    # Convert tensor to PIL Image
    image = to_pil_image(image_tensor.cpu())
    
    # Convert image to numpy array for plotting
    image_np = np.array(image)
    fig, ax = plt.subplots(1)
    ax.imshow(image_np)

    # Draw the bounding boxes
    boxes = prediction['boxes'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()
    
    for box, score in zip(boxes, scores):
        if score > threshold:
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    plt.axis('off')  # Optionally remove axes for cleaner visualization
    plt.show()

# Assuming 'image' and 'predictions' are defined as in your provided code snippet
# Convert the list representation of the image to a tensor
image_tensor = torch.tensor(image)  # Make sure to add `.to(device)` if your model and data are on CUDA

# Now you can call the visualization function
visualize_prediction(image_tensor, predictions, threshold=0.5)
