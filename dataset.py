import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torchvision.datasets

import numpy as np
from torch.utils.data import Dataset
import os
from skimage import io
from skimage.color import rgb2gray
from random import seed,randint
from augmentations_3D import *
import albumentations as A
from utils import *
class BeeDataset_2class(Dataset):

    def __init__(self, source_dir:str, transform, channel=0, labelclass=0):
        self.source_dir = source_dir
        self.data = []
        self.img = []
        self.transform = transform
        self.channel = channel
        # path to folder
        self.path = source_dir
        # takes all names in source folder and opens each image
        self.files = os.listdir(self.path+ "medium_imagesTr/")
        #self.labels = os.listdir(self.path+ "medium_labelsTr/")
        self.labelclass = labelclass

    def __len__(self):
        # returns the length of the dataset/folder
        return len(self.files)

    def __getitem__(self,idx):
        seed()
        InnerSeed = randint(1,100000)
        ## better for biiiig datasets : loads the indexes data as soon as you ask for it (memory for speed)


        img = io.imread(self.path+"medium_imagesTr/"+self.files[idx])
        label = io.imread(self.path+"medium_labelsTr/"+self.files[idx].replace("medium_", "medium_label_processedgray_"))
        #label = np.int16(rgb2gray(label))
        #img = np.uint8(img) zu viel informationsverlust
        # Some Random Transforms
        #seed(InnerSeed)
        #img = random_rotate3D(img, InnerSeed = InnerSeed)
        #img = random_rotate3D_transpose(img, InnerSeed= InnerSeed)
        #img = random_rotate3D_flip(img, InnerSeed= InnerSeed)
        #img = np.flip(img,axis=-1)
        #seed(InnerSeed)
        #label = random_rotate3D(label, InnerSeed = InnerSeed)
        #label = random_rotate3D_transpose(label, InnerSeed= InnerSeed)
        #label = random_rotate3D_flip(label, InnerSeed= InnerSeed)
        #label = np.flip(label, axis = -1)
        label = multi_dim_normalize(label)
        img = multi_dim_normalize(img)
        for i in range(len(img)):
            if i == 0:
                data = self.transform(image = img[i], mask = label[i])
                img[i], label[i] = data["image"] , data["mask"]
            else:
                replay = A.ReplayCompose.replay(data["replay"], image = img[i], mask = label[i])
                img[i], label[i] = replay["image"], replay["mask"]
        #img = self.transform(img)
        # adds 1 empty dimension
        img = random_noise(img, p=0.5)
        
        #shift image randomly up/down
        if randint(1,1000) >= 500:
            img, label = random_ShiftDown(img,InnerSeed, p=0.3), random_ShiftDown(label,InnerSeed, p=0.3)
        else: 
            img, label = random_ShiftUp(img,InnerSeed, p=0.3), random_ShiftUp(label,InnerSeed, p=0.3)

        #shift image randomly left/right
        if randint(1,1000) >= 500:
            img, label = random_ShiftRight(img,InnerSeed, p=0.3), random_ShiftRight(label,InnerSeed, p=0.3)
        else: 
            img, label = random_ShiftLeft(img,InnerSeed, p=0.3), random_ShiftLeft(label,InnerSeed, p=0.3)

        #shift image randomly in Z up/down
        if randint(1,1000) >= 500:
            img, label = random_ShiftZDown(img,InnerSeed, p=0.3), random_ShiftZDown(label,InnerSeed, p=0.3)
        else: 
            img, label = random_ShiftZUp(img,InnerSeed, p=0.3), random_ShiftZUp(label,InnerSeed, p=0.3)

        #img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
        img = np.expand_dims(img,0)


        # normalisation in transform pipeline not on labels
        label = torch.tensor(label, dtype= torch.float)
        img = torch.tensor(img, dtype=torch.float)
        #label = self.transform(label)



        ## In 2class variant: find out which class is background -> always use this as 0 and 1 as the class
        ## after converting to grayscale, takes unique values as classes

        ## convertion to 0-9 maybe not needed
        #unique = label.unique()
        #for i in range(0,len(unique)):
        #    label[label==unique[self.labelclass]] = 0
        #    label[label!=self.labelclass] = 1


        return img, label
