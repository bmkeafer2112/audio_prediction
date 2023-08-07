# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 11:19:54 2023

@author: bmkea
"""

from torch.utils.data import random_split, DataLoader
from Sound_Dataset import SoundDS
import pandas as pd


df = pd.read_csv("C:\\Users\\bmkea\\Documents\\Denso_Test_cell\\Python Scripts\\Audio_Prediction\\Audio_Files\\Denso_4_Seconds\\directory_labels.csv")
data_path = "C:\\Users\\bmkea\\Documents\\Denso_Test_cell\\Python Scripts\\Audio_Prediction\\Audio_Files\\Denso_4_Seconds\\"
myds = SoundDS(df, data_path)

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])

# Create training and validation data loaders
train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)