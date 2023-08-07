# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:22:58 2023

@author: bmkea
"""
from audio_classification import AudioClassifier
from Sound_Dataset import SoundDS
from torch.utils.data import random_split, DataLoader
from torch import nn
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_csv("C:\\Users\\bmkea\\Documents\\Denso_Test_cell\\Python Scripts\\Audio_Prediction\\Audio_Files\\Denso_4_Seconds\\directory_labels.csv")
data_path = "C:\\Users\\bmkea\\Documents\\Denso_Test_cell\\Python Scripts\\Audio_Prediction\\Audio_Files\\Denso_4_Seconds\\"
myds = SoundDS(df, data_path)

#X, y = myds.__getitem__(1)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

X, y = train_ds.__getitem__(1)
#print(X[0])


#X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)

# Create training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)


train_features, train_labels = next(iter(train_dl))
print(train_features)
print(train_labels)


# Create the model and put it on the GPU if available
myModel = AudioClassifier()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)
# Check that it is on Cuda
next(myModel.parameters()).device

# ----------------------------
# Training Loop
# ----------------------------
def training(model, train_dl, num_epochs):
  # Loss Function, Optimizer and Scheduler
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
  scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

  # Repeat for each epoch
  for epoch in range(num_epochs):
    running_loss = 0.0
    correct_prediction = 0
    total_prediction = 0

    # Repeat for each batch in the training set
    for inputs, labels in next(iter(train_dl)):
        print(inputs)
        #print(data[0])
        # Get the input features and target labels, and put them on the GPU
        inputs, labels = inputs[0], labels
        #print(inputs)
        # Normalize the inputs
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s
        print(inputs.size)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        #print(labels)
        loss = criterion(outputs, labels.view(len(labels), 1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        running_loss += loss.item()

        # Get the predicted class with the highest score
        _, prediction = torch.max(outputs,1)
        # Count of predictions that matched the target label
        correct_prediction += (prediction == labels).sum().item()
        total_prediction += prediction.shape[0]

        #if i % 10 == 0:    # print every 10 mini-batches
            #print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 10))
    
    # Print stats at the end of the epoch
    num_batches = len(train_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

  print('Finished Training')
  
num_epochs=10   # Just for demo, adjust this higher.
training(myModel, train_dl, num_epochs)