import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
#import itertools
import matplotlib.pyplot as plt
import time
import argparse
import os
import pickle
import subprocess
from numpy.fft import ifft2
from PIL import Image
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
from skimage import io
from skimage import *
from skimage.metrics import normalized_root_mse as nrm
import numpy as np
from skimage.metrics import normalized_root_mse as nrm
from skfuzzy.image import nmse
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import numpy.matlib 

class grappaDataset(Dataset):
    """Standard GRAPPA dataset.
    TRAINING DATA: Aliased multi-channel complex in -> single slice, single coil
    The GRAPPA method does not include any augmentation"""

    def __init__(self,calK,targetPacket,targetSlice,targetCoil):
        #self.aliasedCplx = muxK
        self.calibrationCplx = calK
        
        self.makeRealData()
        
        self.cols = calK.shape[4]
        self.rows = calK.shape[3]
        self.coils = calK.shape[2]
        self.packets = calK.shape[1]
        self.smsR = calK.shape[0]
        
        self.targetPacket = targetPacket
        self.targetSlice = targetSlice
        self.targetCoil = targetCoil
        
        self.setupXY()
        
    def makeRealData(self):
        #self.realAliased = np.concatenate( [np.real(self.aliasedCplx), \
                                           # np.imag(self.aliasedCplx)], axis=1).astype(np.float32)
        self.realCal = np.concatenate( [np.real(self.calibrationCplx), \
                                        np.imag(self.calibrationCplx)], axis=2).astype(np.float32)
        
        #Below for normalization--disabled (mean=0, variance=1)
        self.aliasedMean = 0#np.mean(self.realAliased)
        self.aliasedVar = 1#np.var(self.realAliased)
        
        #self.realAliased = (self.realAliased-self.aliasedMean)/self.aliasedVar
        
        self.calMean = 0#np.mean(self.realCal)
        self.calVar = 1#np.var(self.realCal)
        
        self.realCal = (self.realCal-self.calMean)/self.calVar
        
        
    def __len__(self):
        return 1
    
    def setupXY(self):
        #X is the aliased data--summed across list of slice idcs: 2*coils x rows x cols
        self.calX = np.zeros((self.realCal.shape[2],self.realCal.shape[3],self.realCal.shape[4]), \
                             dtype=np.float32)
        #Y is the ideal, single-slice, single-coil data: (2 (R/I) x rows x cols
        self.calY = np.zeros((2,self.realCal.shape[3],\
                              self.realCal.shape[4]),dtype=np.float32)

        #Fill X by summing across all excited slices for this packet
        self.calX[::,::,::] = np.sum(self.realCal[::,self.targetPacket,::,::,::], \
                                     axis=0, dtype=np.float32)
        #Grab the ideal slice, packet, coil real and imaginary data
        self.calY[0,::,::] = self.realCal[self.targetSlice,self.targetPacket,self.targetCoil,::,::]
        self.calY[1,::,::] = self.realCal[self.targetSlice,self.targetPacket,self.targetCoil+self.coils,::,::]
        
    def __getitem__(self, idx):            
        return(self.calX[::,::,::].astype(np.float32), self.calY[::,::,::].astype(np.float32))


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def makeRakiNetwork(inChannels,kernSize,midChannels,networkLayers,dropOut,batchNorm,groups,biasFL,numFilters):
    #Works well with 
    # L1: inChannels->inChannels; kernel 7; no bias; BatchNorm; ReLU; Dropout (.5)
    # L2: inChannels->512; kernel 1; no bias; BatchNorm; ReLU; Dropout (.5)
    # L3: 512->inChannels*numSlices; kernel 7; no bias
    #kernSize = 7
    #midChannels = 1024

    netList = []
    outputChan=inChannels
    for idx in range(networkLayers):
        #Set the input channel count equal to the output channel count of last layer
        inChan = outputChan
        #Final output has 2 channels for real/imag
        if idx+1 == networkLayers:
            outputChan = 2
            thisKern = kernSize
        #Penultimate output has midChannels
        elif idx+2 == networkLayers:
            outputChan = midChannels
            thisKern = 1
        #Input the number of in channels, output number of filters if first layer
        elif idx == 0:
            outputChan = numFilters
            thisKern = kernSize
        #All other layers keep number of channels constant
        else:
            outputChan = inChannels
            thisKern = kernSize

        #Make the 2D convolution layer
        netList.append(nn.Conv2d(in_channels = inChan, \
                                 out_channels = outputChan, \
                                 kernel_size = thisKern, \
                                 stride = 1, \
                                 padding = int(np.floor(thisKern/2.)), \
                                 dilation = 1, \
                                 groups = groups, \
                                 bias = biasFL, \
                                 padding_mode='zeros'))
        #Make the batch normalizaiton layer (if desired)
        if ((batchNorm) and (idx<networkLayers-1)):
            netList.append(nn.BatchNorm2d(outputChan))
        #Make the activation layer (only if not last layer of network)
            #Add the dropout layer
            netList.append(nn.Dropout2d(dropOut))

    rakiCNN = nn.Sequential(*netList)
    rakiCNN.apply(weights_init)

    return rakiCNN


def train_model(d ,eps,name,path):
  learningRate=0.001
  grappaEpochs=eps
  network=makeRakiNetwork(64,5,1024,7,0,True,1,True,5)
  costFN = nn.L1Loss()
  optimizer = torch.optim.Adam(network.parameters(),lr=learningRate)
  


  errors = []
  ep = 0
  myContinue=True
  #while myContinue:
  for ep in range(grappaEpochs):
          ep+=1
          network.train(True)
      
          dataLoader = DataLoader(d)
          for i_batch, sample_batched in enumerate(dataLoader):
              #Run the prediction
              #thisCNN.half()
              sampleX = sample_batched[0]
              yPred = network(sampleX) 
              #yPred = yPred.float()s 
      
              #Compute the loss
              sampleY = sample_batched[1]
              #sampleY = sampleY.float()
              loss = costFN(yPred, sampleY)
      
              #Zero the gradients
              optimizer.zero_grad()
      
              #Do the back propogation
              loss.backward()
      
              #Take the next step
              #thisCNN.float()
              optimizer.step()
          errors.append(loss.item())
          #duration = time.time()-startTime
          #if ep%100 == 0:
              #print('Duration %f s'%(duration))
          print('    %i %f'%(ep, loss.item(),))
          stdEpRun = ep
        
  torch.save(network.state_dict(),"{p}/{n}.pth".format(n=name,p=path))



def transform_kspace_to_image(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = fftshift(ifftn(ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img
#here
def get_image(pj,i):
 image2=(abs((transform_kspace_to_image(pj))/np.max(transform_kspace_to_image(pj)))*255)
#sos = np.sqrt(np.sum((image2**2), 2))
#sos= sos/np.max(sos)*255
 new_p2=Image.fromarray(image2)
 new_p2 = new_p2.convert("L")
 new_p2.save("{i}.png".format(i=i))
 #new_p2

def get_sos(check2,name):

  image1=(abs((transform_kspace_to_image(check2))/np.max(transform_kspace_to_image(check2)))*255)
  sos = np.sqrt(np.sum((image1**2), 2))
  sos= sos/np.max(sos)*255
  new_p1=Image.fromarray(sos)
  new_p1 = new_p1.convert("L")
  new_p1.save("{n}.png".format(n=name))
  new_p1


def prepare_data(data):
  coils=np.zeros((32,68,68),dtype=complex)
  for i in range(32):
    d=data[:,:,i]
    coils[i,:,:] =d
  return coils

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [1,dilate_rate])
