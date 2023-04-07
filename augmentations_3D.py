## Transformation Functions -> outsource in Classfile later
from skimage.transform import rotate
from skimage import io
import matplotlib.pyplot as plt
from numpy import transpose
from random import randint, sample,seed, random
import numpy as np
from PIL import Image
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
import albumentations as A
#torchvision.transforms.RandomRotation(degrees, resample=False, expand=False, center=None, fill=None)
#torchvision.transforms.RandomHorizontalFlip(p=0.5)

# transformation with np.flip

def rotate3D(image, degreeX=0, degreeY=0, resize = True):
    stack = list()
    stack2 = list()
    if degreeX != 0:
        for i in image:
            i = Image.fromarray(np.uint16(i))
            i = i.rotate(degreeX)
            i = np.array(i)
            stack.append(i)
    else: stack = image

    if degreeY != 0:
        stack = transpose(stack)
        for i in stack:
            i = Image.fromarray(np.uint16(i))
            i = i.rotate(degreeY)
            i = np.array(i)
            stack2.append(i)
        stack = transpose(stack2)

    else: stack2 = stack
    return stack2

def random_rotate3D_90(image,InnerSeed, p=0.5, resize = True):
    #seed(InnerSeed/2)
    if randint(1,100) < p*100:

        stack = list()
        stack2 = list()

        degreeX = sample((90,0,-90),1)[0]
        degreeY = sample((90,0,-90),1)[0]

        if degreeX != 0:
            for i in image:
                i = Image.fromarray(np.uint16(i))
                i = i.rotate(degreeX)
                i = np.array(i)
                stack.append(i)
        else: stack = image

        if degreeY != 0:
            stack = transpose(stack)
            for i in stack:
                i = Image.fromarray(np.uint16(i))
                i = i.rotate(degreeY)
                i = np.array(i)
                stack2.append(i)
            stack = transpose(stack2)

        else: stack2 = stack
        return np.array(stack2)
    else:
        return(image)

def random_rotate3D_1830(image,InnerSeed, p=0.5, resize = True):
    seed(InnerSeed/3)
    print("seed: ", InnerSeed)
    print("random: ", randint(1,100))
    if randint(1,100) < p*100:

        stack = list()
        stack2 = list()

        degreeX = sample((180,0),1)[0]
        degreeY = sample((180,0),1)[0]

        if degreeX != 0:
            for i in image:
                i = Image.fromarray(np.uint16(i))
                i = i.rotate(degreeX)
                i = np.array(i)
                stack.append(i)
        else: stack = image

        if degreeY != 0:
            stack = transpose(stack)
            for i in stack:
                i = Image.fromarray(np.uint16(i))
                i = i.rotate(degreeY)
                i = np.array(i)
                stack2.append(i)
            stack = transpose(stack2)

        else: stack2 = stack
        #print(np.array(stack2).shape)
        return np.array(stack2)
    else:
        #print(image.shape)
        return(image)


def random_rotate3D_flip(image,InnerSeed, p=0.5, resize = True):
    seed(InnerSeed/3)
    if randint(1,100) < p*100:


        degreeX = sample((180,0),1)[0]
        #print("X", degreeX)
        degreeY = sample((180,0),1)[0]
        #print("Y", degreeY)
        #degreeZ = sample((180,0),1)[0]
        degreeZ=0
        #print("Z", degreeZ)

        if degreeX != 0:
            image = np.flip(image, axis=-1)

        if degreeY != 0:
            image = np.flip(image, axis=-2)

        if degreeZ != 0:
            image = np.flip(image, axis=-3)
        return image
    else:
        return(image)

def random_rotate3D_transpose(image,InnerSeed, p=0.5, resize = True):
    seed(InnerSeed/2)
    if randint(1,100) < p*100:


        degreeX = sample((180,0),1)[0]
        #print("Xt", degreeX)
        degreeY = sample((180,0),1)[0]
        #print("Yt", degreeY)
        degreeZ = sample((180,0),1)[0]
        #print("Zt", degreeZ)

        if degreeX != 0:
            image = np.transpose(image, (0,2,1))

        if degreeY != 0:
            image = np.transpose(image, (1,0,2))

        if degreeZ != 0:
            image = np.transpose(image, (2,1,0))
        return image
    else:
        return(image)

def random_noise_torch(image,  p=0.5):
    noise_intensity = random()/2
    if randint(1,100) < p*100:

        image = image+ torch.rand(image.shape)*noise_intensity
        return image
    else:
        return(image)
        
def random_noise(image, **kwargs):
    return (image + np.random.uniform(0,0.4)*np.random.rand(image.shape[0], image.shape[1], image.shape[2]))
    
def random_ShiftUp(image,InnerSeed,p=0.5, max_frames =20, **kwargs):
    seed(InnerSeed/2)
    if randint(1,100) < p*100:
            
        distance = np.random.randint(0,max_frames)
        for i in range(len(image)):
            image[i,:image.shape[1]-distance,:] = image[i,distance:,:]
            image[i,image.shape[1]-distance:,:]= 0
        return image
    else: return image
def random_ShiftDown(image,InnerSeed,p=0.5,max_frames =20, **kwargs):
    seed(InnerSeed/2.1)
    if randint(1,100) < p*100:
            
        distance = np.random.randint(0,max_frames)
        for i in range(len(image)):
            image[i,distance:,:] = image[i,:image.shape[1]-distance,:]
            image[i,:distance,:]= 0
        return image
    else: return image
def random_ShiftRight(image,InnerSeed,p=0.5,max_frames =20, **kwargs):
    seed(InnerSeed/2.2)
    if randint(1,100) < p*100:
        
        distance = np.random.randint(0,max_frames)
        for i in range(len(image)):
            image[i,:,distance:] = image[i,:,:image.shape[1]-distance]
            image[i,:,:distance]= 0

        return image
    else: return image
def random_ShiftLeft(image,InnerSeed,p=0.5,max_frames =20, **kwargs):
    seed(InnerSeed/2.4)
    if randint(1,100) < p*100:
            
        distance = np.random.randint(0,max_frames)
        for i in range(len(image)):
            image[i,:,:image.shape[1]-distance] = image[i,:,distance:]
            image[i,:,image.shape[1]-distance:]= 0

        return image
    else: return image
def random_ShiftZUp(image,InnerSeed,p=0.5,max_frames =5, **kwargs):
    seed(InnerSeed/2.5)
    if randint(1,100) < p*100:
            
        distance = np.random.randint(0,max_frames)
        for i in range(image.shape[1]):
            image[:image.shape[0]-distance,i,:] = image[distance:,i,:]
            image[image.shape[0]-distance:,i,:]= 0

        return image
    else: return image
def random_ShiftZDown(image,InnerSeed,p=0.5,max_frames =5, **kwargs):
    seed(InnerSeed/2.6)
    if randint(1,100) < p*100:
            
        distance = np.random.randint(0,max_frames)
        for i in range(image.shape[1]):
            image[distance:,i,:] = image[:image.shape[0]-distance,i,:]
            image[:distance,i,:]= 0

        return image
    else: return image