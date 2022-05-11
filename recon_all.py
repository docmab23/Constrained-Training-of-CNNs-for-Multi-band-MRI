#!~/.conda/envs/slice/bin/python
#reconstruct the test data 


# input would be the test_matrix 
# slice extraction
# Transform shapes
# run the s-raki models 3 times
# run st-raki models on the 3 rexonstructe slices
# calcualte the metrics

import numpy  as np
from scipy import io
from numpy.fft import fftshift, ifftshift, fftn, ifftn
from PIL import Image
import sys
from utils import transform_kspace_to_image , grappaDataset , makeRakiNetwork , prepare_data , get_image, get_sos,conv2d ,conv2d_dilate
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
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import scipy.io as sio
import numpy as np
#from tensorflow.examples.tutorials.mnist import input_data
import time
import os
import numpy.matlib 


mat_file = sys.argv[1]
slice_no = int(sys.argv[2])
dir_name = sys.argv[3]
results_dir = sys.argv[4]
calib_dir = sys.argv[5]

os.mkdir(results_dir)
pwd = os.getcwd()
sess = tf.InteractiveSession()
def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,acc_rate):                
    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate)) 
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3,acc_rate) 
    return sess.run(h_conv3)     
mat=io.loadmat(mat_file)
matr = mat["measPc1Slice"]
matr=np.squeeze(matr)
image = matr[:,:,slice_no,0,:]
#68,68,33,3,32
np.save("{r}/mb_ipat".format(r=results_dir),image)
image1=(abs((transform_kspace_to_image(image))/np.max(transform_kspace_to_image(image)))*255)
sos = np.sqrt(np.sum((image1**2), 2))
sos= sos/np.max(sos)*255
new_p1=Image.fromarray(sos)
new_p1 = new_p1.convert("L")
new_p1.save("{c}/mb_ipat.png".format(c=calib_dir))


mb_ipat_data = np.load("{r}/mb_ipat.npy".format(r=results_dir))

mb_ipat_data=prepare_data(mb_ipat_data)

for m in range(3):
    real = np.real(mb_ipat_data)
    comp = np.imag(mb_ipat_data)
    total_ = np.concatenate((real,comp),axis=0)
    total_ = np.expand_dims(total_,axis=0)
    model_path = "./{c}".format(c=calib_dir)
    whole_kspace= np.zeros((68,68,32),dtype=complex)
    for i in range(32):
     print("Reconstructing Coil {i}".format(i=i+1))
     j = torch.Tensor(total_)
     inferred=[]
     with torch.no_grad(): #Needed to avoid memory errors!!!
        model=makeRakiNetwork(64,5,1024,7,0,True,1,True,5)
        pk = os.path.join(model_path,"{m}_{i}.pth".format(i=i,m=m))
        model.load_state_dict(torch.load(pk))
        model.eval()
        inferred=[]
        yPred = model(j)    
        inferred.append(yPred.cpu().detach().numpy())
        pred = np.squeeze(inferred)
        pj = np.array(pred[0,:,:],dtype=complex)
        pj.imag=pred[1,:,:]
        whole_kspace[:,:,i]=pj
    os.chdir(results_dir)
    get_sos(whole_kspace,str(m))
    os.chdir(pwd)

    np.save("{r}/whole{m}".format(r=results_dir,m=m), whole_kspace)
    
    try_data=np.load("{r}/whole{m}.npy".format(r=results_dir,m=m))
    data=np.load("{c}/{d}_under.npy".format(c=calib_dir,d=dir_name))
    data= data[m,0,:,:,:]
    coils2=[]
    for i in range(data.shape[0]):
        coils2.append(data[i,:,:])

    check2 =np.stack((c for c in coils2),axis=-1)
    kspace=check2

    weights = sio.loadmat("{c}/RAKI_recon{m}_weight_52,11,32_256,256.mat".format(c=calib_dir,m=m))
    [m1,n1,no_ch] = np.shape(kspace)
    no_inds = 1

    kspace_all = kspace
    kx = np.transpose(np.int32([(range(1,m1+1))]))                  
    ky = np.int32([(range(1,n1+1))])

    kspace = np.copy(kspace_all)
    mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0; 
    picks = np.where(mask == 1);                                  
    kspace = kspace[:,np.int32(picks[0][0]):n1+1,:]
    kspace_all = kspace_all[:,np.int32(picks[0][0]):n1+1,:]  

    kspace_NEVER_TOUCH = np.copy(kspace_all)

    mask = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),0),1))>0;  
    picks = np.where(mask == 1);                                  
    d_picks = np.diff(picks,1)  
    indic = np.where(d_picks == 1);

    mask_x = np.squeeze(np.matlib.sum(np.matlib.sum(np.abs(kspace),2),1))>0;
    picks_x = np.where(mask_x == 1);
    x_start = picks_x[0][0]
    x_end = picks_x[0][-1]
    indic = indic[1][:]
    center_start = picks[0][indic[0]];
    center_end = picks[0][indic[-1]+1];
    ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
    ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
    ACS_re[:,:,0:no_ch] = np.real(ACS)
    ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)
    acc_rate = d_picks[0][0]
    kspace_recon_all = np.copy(kspace_all)
    kspace_recon_all_nocenter = np.copy(kspace_all)
    kspace = np.copy(kspace_all)
    over_samp = np.setdiff1d(picks,np.int32([range(0, n1,2)]))
    kspace_und = try_data
    kspace_und[:,over_samp,:] = 0
    [dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

    kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
    kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
    kspace_und_re = np.float32(kspace_und_re)
    kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_recon = kspace_und_re
    kspace_NEVER_TOUCH = np.copy(kspace_all)
    w1_all=weights["w1"]
    w2_all=weights["w2"]
    w3_all=weights["w3"]
    over_samp = np.setdiff1d(picks,np.int32([range(0, n1,2)]))
    kernel_x_1 = 5
    kernel_y_1 = 2

    kernel_x_2 = 1
    kernel_y_2 = 1

    kernel_last_x = 3
    kernel_last_y = 2

    layer1_channels = 128
    layer2_channels = 128
    b1_flag = 0
    b2_flag = 0                       
    b3_flag = 0

    if (b1_flag == 1):
        b1_all = np.zeros([1,1, layer1_channels,64])
    else:
        b1 = []

    if (b2_flag == 1):
        b2_all = np.zeros([1,1, layer2_channels,64])
    else:
        b2 = []

    if (b3_flag == 1):
        b3_all = np.zeros([1,1, 64, 64])
    else:
        b3 = []
    target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1)
    target_x_end = np.int32(ACS_dim_X - target_x_start -1)

    for ind_c in range(0,64):
        print('Reconstruting Channel #',ind_c+1)
        

        if (b1_flag == 1):
            b1 = b1_all[:,:,:,ind_c];
        if (b2_flag == 1):
            b2 = b2_all[:,:,:,ind_c];
        if (b3_flag == 1):
            b3 = b3_all[:,:,:,ind_c];
        w1 = np.float32(w1_all[:,:,:,:,ind_c])
        w2 = np.float32(w2_all[:,:,:,:,ind_c])     
        w3 = np.float32(w3_all[:,:,:,:,ind_c])
                    
            
        res = cnn_3layer(kspace_und_re,w1,b1,w2,b2,w3,b3,2) 
        target_x_end_kspace = dim_kspaceUnd_X - target_x_start;
        
        for ind_acc in range(0,2-1):

            target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * 2 + ind_acc + 1;             
            target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * 2 + ind_acc;
            kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:2,ind_c] = res[0,:,::2,ind_acc]

    kspace_recon = np.squeeze(kspace_recon)
    kspace_recon_complex = (kspace_recon[:,:,0:np.int32(64/2)] + np.multiply(kspace_recon[:,:,np.int32(64/2):64],1j))
    kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex)
#kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]

#kspace_recon_all_nocenter = np.copy(kspace_all)
#kspace_recon_complex = (kspace_recon[:,:,0:np.int32(64/2)] + np.multiply(kspace_recon[:,:,np.int32(64/2):64],1j))
#kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex); 


#kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]


    kspace_recon_all[:,:,:] = kspace_recon_complex; 

    for sli in range(0,no_ch):
       kspace_recon_all[:,:,sli] =  np.fft.ifft2(kspace_recon_all[:,:,sli])
    os.chdir(results_dir)   
    np.save("recon_slice_{m}".format(m=m),kspace_recon_complex)
    get_sos(kspace_recon_complex,"recon_slice_{m}".format(m=m))
    os.chdir(pwd)





