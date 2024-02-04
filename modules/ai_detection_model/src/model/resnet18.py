import zipfile
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

# # #root directory
# # data_zip = "../archive.zip"
# # data_dir = "../data"
# # data_zip_ref = zipfile.ZipFile(data_zip,"r")
# # data_zip_ref.extractall(data_dir)



# import os

# #creating training dir
# data_dir = "../data"
# training_dir = os.path.join(data_dir,"training")

# if not os.path.isdir(training_dir):
#     os.mkdir(training_dir)

# #creating dog in training
# dog_training_dir = os.path.join(training_dir,"dog")
# if not os.path.isdir(dog_training_dir):
#     os.mkdir(dog_training_dir)


# #creating cat in training
# cat_training_dir = os.path.join(training_dir,"cat")
# if not os.path.isdir(cat_training_dir):
#     os.mkdir(cat_training_dir)


# #create validation dir
# validation_dir = os.path.join(data_dir,"validation")
# if not os.path.isdir(validation_dir):
#     os.mkdir(validation_dir)


# #create dog in validation
# dog_validation_dir = os.path.join(validation_dir,"dog")
# if not os.path.isdir(dog_validation_dir):
#     os.mkdir(dog_validation_dir)


# #create cat in validation
# cat_validation_dir = os.path.join(validation_dir,"cat")
# if not os.path.isdir(cat_validation_dir):
#     os.mkdir(cat_validation_dir) 

# import os
# import shutil
# import cv2

# base_source_path = '../data/train/train/'
base_dog_destination = '../data/training/dog'
base_cat_destination = '../data/training/cat'
# dog_images = os.listdir(base_source_path)
# # print(type(dog_images))

# for img in dog_images:
#     source_path = os.path.join(base_source_path, img)
#     print(type(source_path))
#     # if img.lower().startswith("dog"):
#     #     destination_path = os.path.join("../data/training/dog")
#     #     # image = cv2.imread(destination_path)
       
#     # else:
#     #     destination_path = os.path.join("../data/training/cat")
      
#     if img.lower().startswith("dog"):
#         # destination_directory = "../data/training/dog"
#         destination_path = os.path.join(base_dog_destination, img)
#     elif img.lower().startswith("cat"):
#         # destination_directory = "../data/training/cat"
#         destination_path = os.path.join(base_cat_destination, img)

#     # destination_path = os.path.join(destination_directory, img)

#     shutil.move(source_path, destination_path)

# from IPython.core.pylabtools import figsize

# import os
# samples_dog = [os.path.join(base_dog_destination,np.random.choice(os.listdir(base_dog_destination),1)[0]) for _ in range(8)]
# samples_cat = [os.path.join(base_cat_destination,np.random.choice(os.listdir(base_cat_destination),1)[0]) for _ in range(8)]
# nrows = 4
# ncols = 4
# fig, ax = plt.subplots(nrows,ncols,figsize = (10,10))
# ax = ax.flatten()

# for i in range(nrows*ncols):
#   if i < 8:
#     pic = plt.imread(samples_dog[i%8])
#     ax[i].imshow(pic)
#     ax[i].set_axis_off()
#   else:
#     pic = plt.imread(samples_cat[i%8])
#     ax[i].imshow(pic)
#     ax[i].set_axis_off()
# plt.show()


import torch
import torchvision
from torchvision import datasets, transforms

traindir = "../../../datasets/ballot_datasets/training"
testdir = "../../../datasets/ballot_datasets/testing"

train_transforms = transforms.Compose([transforms.Resize((500,500)),
                                       transforms.ToTensor(),                                
                                       torchvision.transforms.Normalize(
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225],
    ),
                                       ])

test_transforms = transforms.Compose([transforms.Resize((500,500)),
                                      transforms.ToTensor(),
                                      torchvision.transforms.Normalize(
                                          mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225],
    ),
                                      ])

train_data = datasets.ImageFolder(traindir,transform=train_transforms)
test_data = datasets.ImageFolder(testdir,transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, shuffle = True, batch_size=16)
testloader = torch.utils.data.DataLoader(test_data, shuffle = True, batch_size=16)


def make_train_step(model, optimizer, loss_fn):
  def train_step(x,y):
    #make prediction
    yhat = model(x)
    #enter train mode
    model.train()
    #compute loss
    loss = loss_fn(yhat,y)

    loss.backward() 
    optimizer.step()
    optimizer.zero_grad()
    #optimizer.cleargrads()

    return loss
  return train_step

from torchvision import datasets, models, transforms
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.resnet18(pretrained=True)


for params in model.parameters():
  params.requires_grad_ = False


nr_filters = model.fc.in_features  #number of input features of last layer
model.fc = nn.Linear(nr_filters, 1)
model = model.to(device)

from torch.nn.modules.loss import BCEWithLogitsLoss
from torch.optim import lr_scheduler

#loss
loss_fn = BCEWithLogitsLoss() #binary cross entropy with sigmoid, so no need to use sigmoid in the model

#optimizer
optimizer = torch.optim.Adam(model.fc.parameters()) 

#train step
train_step = make_train_step(model, optimizer, loss_fn)

losses = []
val_losses = []

epoch_train_losses = []
epoch_test_losses = []


n_epochs = 50
early_stopping_tolerance = 3
early_stopping_threshold = 0.03

for epoch in range(n_epochs):  
  
  epoch_loss = 0 

  for i ,data in tqdm(enumerate(trainloader), total = len(trainloader)): #iterate ove batches
    x_batch , y_batch = data
    # print(x_batch,y_batch)
    x_batch = x_batch.to(device) #move to gpu
    y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
    y_batch = y_batch.to(device) #move to gpu 

    loss = train_step(x_batch, y_batch)
    print(len(trainloader))
    epoch_loss += loss/len(trainloader)


  epoch_train_losses.append(epoch_loss)
  print('\nEpoch : {}, train loss : {}'.format(epoch+1, epoch_loss))


    #validation does not requires gradient 
  with torch.no_grad():
      cum_loss = 0
      for x_batch, y_batch in testloader:
          x_batch = x_batch.to(device)
          y_batch = y_batch.unsqueeze(1).float() #convert target to same nn output shape
          y_batch = y_batch.to(device) 

          #model to eval mode
          model.eval()

          yhat = model(x_batch)
          val_loss = loss_fn(yhat,y_batch)
          cum_loss += loss/len(testloader)
          val_losses.append(val_loss.item())

          
      epoch_test_losses.append(cum_loss)
      print('Epoch : {}, val loss : {}'.format(epoch+1,cum_loss))  

      best_loss = min(epoch_test_losses)


      #save best model
      if cum_loss <= best_loss:
          best_model_wts = model.state_dict()

      
      #early stopping
      early_stopping_counter = 0
      if cum_loss > best_loss:
          early_stopping_counter +=1
      
      
      if (early_stopping_counter == early_stopping_tolerance) or (best_loss <= early_stopping_threshold):
          print("/nTerminating: early stopping")
          break #terminate training


#load best model
model.load_state_dict(best_model_wts)


import matplotlib.pyplot as plt
print(epoch_train_losses)
print(epoch_test_losses)
epoch_train_losses_np = [loss.cpu().numpy() for loss in epoch_train_losses]
epoch_test_losses_np = [loss.cpu().numpy() for loss in epoch_test_losses]

plt.plot(epoch_train_losses_np, label="Training loss")
plt.plot(epoch_test_losses_np, label="Val loss")
plt.title('Training and Test Loss for Resnet18')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()





# trainloss = [0.7047932147979736,0.7074649333953857,0.7029479146003723,0.6927050352096558,0.6797118782997131]
# testloss = [0.7801194787025452,0.7157936692237854,0.7342474460601807,0.692432701587677,0.6031248569488525]
