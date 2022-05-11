from numpy.fft import fftshift, ifftshift, fftn, ifftn
#from skimage import data, img_as_float
#from skimage.metrics import structural_similarity as ssim
#from skimage import data, img_as_float
#rom skimage import io
#from skimage import *
#from skimage.metrics import normalized_root_mse as nrm
#from skfuzzy.image import nmse
import os
from numpy.fft import ifft2
from PIL import Image
import torch.nn as nn
import numpy as np
from numpy.fft import fftshift, ifftshift, fftn, ifftn
import numpy as np
from numpy import matlib
import time
import argparse
import sys 
import scipy.io as sio
import torch 
from torch.utils.data import Dataset, DataLoader
import tensorflow.compat.v1 as tf
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage import data, img_as_float
from skimage import io
import mat73
tf.disable_v2_behavior()

from utils import grappaDataset , get_sos,transform_kspace_to_image


#parser = argparse.ArgumentParser(description='Train models')
#parser.add_argument("f", dest="filename/path", required=True,
    #help="your calibration data", metavar="FILE")
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def makeRakiNetwork(inChannels,kernSize,midChannels,networkLayers,dropOut,batchNorm,groups,biasFL,numFilters):
  
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
      
          dataLoader = DataLoader(d, 
                                  batch_size=8,
                                  shuffle=True, 
                                  num_workers=4)
          for i_batch, sample_batched in enumerate(dataLoader):
              #Run the prediction
              #thisCNN.half()
              sampleX = sample_batched[0]
              yPred = network(sampleX) 
              #yPred = yPred.float()
      
              #Compute the loss
              sampleY = sample_batched[1]
              #sampleY = sampleY.float()
              loss = costFN(yPred, sampleY)
              #print(yPred[0][1])
              #print(sampleY[0][1])
      
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

#args = parser.parse_args()

file_path = sys.argv[1]

slice_no = int(sys.argv[2])

dir_name = sys.argv[3]
calib_dir = sys.argv[4]

print(calib_dir)
if not os.path.exists("./{c}".format(c=calib_dir)):
  os.mkdir(calib_dir)
  print("making directory")
else:
    print("not making directory")

data_dict = mat73.loadmat(file_path, only_include='MB_IPAT_calibration_ksp') 
struct = data_dict['MB_IPAT_calibration_ksp'] # now only structure is loaded and nothing else
print(struct.shape)
np.save("{c}/all_calib".format(c=calib_dir),struct)

matr = np.load("{c}/all_calib.npy".format(c=calib_dir))
print(slice_no , slice_no+11, slice_no+22)
matr1 = matr[:,:,slice_no,:,:]
matr2= matr[:,:,slice_no+11,:,:]
matr3= matr[:,:,slice_no+22,:,:]
#matr4= matr[:,:,18,:,:]
matr5 = np.stack((matr1,matr2,matr3),axis=2)

#print(np.shape(matr3))

#matr3= matr3.reshape(2,32,100,100,order="C")
    # 100,100,2,3,32
#matr4 = matr3[:,:,:,0,:]
    # 100,100,3,323
kk= np.stack((matr5[:,:,0,:,:],matr5[:,:,1,:,:],matr5[:,:,2,:,:]),axis=0)
#print(kk.shape)
epis=[]
for i in range(kk.shape[-2]):
    epis.append(kk[:,:,:,i,:])
kj = np.stack((e for e in epis),axis=1)
#print(kj.shape)
coils=[]
for i in range(kj.shape[-1]):
    coils.append(kj[:,:,:,:,i])

kl =np.stack((c for c in coils),axis=2)

#kj=np.expand_dims(kj, axis=1)
print(kl.shape)
np.save("{c}/{d}".format(c=calib_dir,d=dir_name),kl)
data_new = np.load("{c}/{d}.npy".format(c=calib_dir,d=dir_name))
#matr3=np.load("calib_sms3_try.npy")
print(data_new)
all=np.zeros((3,3,32,68,68),dtype=complex)
for k in range(data_new.shape[0]):
 check=data_new[k,0,:,:,:]
 coils=[]
 for i in range(check.shape[0]):
    coils.append(check[i,:,:])

 putt =np.stack((c for c in coils),axis=-1)
 #RAKI-cart
 for i in range(1,18,2):
    putt[:,i,:]=0
    #kspace[:,i+1,:]=0
    #kspace[:,i+2,:]=0
    #kspace[:,i+3,:]=0
    #kspace[:,i+4,:]=0
  #  unaq.append(slice_[:,1,:])
    #data.append(slice_[:,i+1:i+4,:])

 for i in range(50,68,2):
    putt[:,i,:]=0
    #kspace[:,i+1,:]=0
    #kspace[:,i+2,:]=0
    #kspace[:,i+3,:]=0
    #kspace[:,i+4,:]=0
 for b in range(32): 
   all[k,0,b,:,:]=putt[:,:,b]
   
   
np.save("{c}/{d}_under".format(c=calib_dir,d=dir_name),all)

s_raki_in= np.load("{c}/{d}_under.npy".format(c=calib_dir,d=dir_name))
print(s_raki_in.shape)

for slice_ in range(3):
    print(slice_)
    im=s_raki_in[slice_,0,:,:,:]
    coils2=[]
    for i in range(im.shape[0]):
        coils2.append(im[i,:,:])
    im=np.stack((c for c in coils2),axis=-1)
    get_sos(im,"{sl_no}/gos_im_{k}".format(sl_no=calib_dir,k=slice_))
     
    whole2= np.zeros((68,68,32),dtype=complex)
    for i in range(32):

        
        print("Coil {i}".format(i=i+1))
        
        j = grappaDataset(s_raki_in,0,slice_,i)
        #print(j.__getitem__(0)[0].shape,j.__getitem__(0)[1].shape)
        train_model(j,150,"{k}_{m}".format(m=i,k=slice_),calib_dir)
        """inferred=[]
        with torch.no_grad(): #Needed to avoid memory errors!!!
         dataLoader = DataLoader(j)
         model=makeRakiNetwork(64,5,1024,7,0,True,1,True,5)
         model.load_state_dict(torch.load("{c}/{k}_{m}.pth".format(k=k,c=calib_dir,m=i)))
         model.eval()
        for i_batch, s_b in enumerate(dataLoader):
            #Run the prediction
            sampleX = s_b[0]
            yPred2 = model(sampleX)
    
            inferred.append(yPred2.cpu().detach().numpy())
            pred = np.squeeze(inferred)
            pj = np.array(pred[0,:,:],dtype=complex)
            pj.imag=pred[1,:,:]
            whole2[:,:,i]=pj
    np.save("{c}/whole{k}".format(k=k,c=calib_dir),whole2)"""

def weight_variable(shape,vari_name):                   
  initial = tf.truncated_normal(shape, stddev=0.1,dtype=tf.float32)
  return tf.Variable(initial,name = vari_name)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def conv2d_dilate(x, W,dilate_rate):
    return tf.nn.convolution(x, W,padding='VALID',dilation_rate = [1,dilate_rate])

#### LEANING FUNCTION ####
def learning(ACS_input,target_input,accrate_input,sess):

    input_ACS = tf.placeholder(tf.float32, [1, ACS_dim_X,ACS_dim_Y,ACS_dim_Z])                                  
    input_Target = tf.placeholder(tf.float32, [1, target_dim_X,target_dim_Y,target_dim_Z])         

    Input = tf.reshape(input_ACS, [1, ACS_dim_X, ACS_dim_Y, ACS_dim_Z])         

    [target_dim0,target_dim1,target_dim2,target_dim3] = np.shape(target)

    W_conv1 = weight_variable([kernel_x_1, kernel_y_1, ACS_dim_Z, layer1_channels],'W1') 
    h_conv1 = tf.nn.relu(conv2d_dilate(Input, W_conv1,accrate_input)) 

    W_conv2 = weight_variable([kernel_x_2, kernel_y_2, layer1_channels, layer2_channels],'W2')
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, W_conv2,accrate_input))

    W_conv3 = weight_variable([kernel_last_x, kernel_last_y, layer2_channels, target_dim3],'W3')
    h_conv3 = conv2d_dilate(h_conv2, W_conv3,accrate_input)

    error_norm = tf.norm(input_Target - h_conv3)       
    train_step = tf.train.AdamOptimizer(LearningRate).minimize(error_norm)

    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)

    error_prev = 1 
    error_tot=[]
    for i in range(MaxIteration):
        
        sess.run(train_step, feed_dict={input_ACS: ACS, input_Target: target})
        if i % 100 == 0:                                                                      
            error_now=sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})    
            print('The',i,'th iteration gives an error',error_now)  
            error_tot.append(error_now)                           
            
            
        
    error = sess.run(error_norm,feed_dict={input_ACS: ACS, input_Target: target})
    return [sess.run(W_conv1),sess.run(W_conv2),sess.run(W_conv3),error,error_tot]  



def cnn_3layer(input_kspace,w1,b1,w2,b2,w3,b3,acc_rate,sess):                
    h_conv1 = tf.nn.relu(conv2d_dilate(input_kspace, w1,acc_rate)) 
    h_conv2 = tf.nn.relu(conv2d_dilate(h_conv1, w2,acc_rate))
    h_conv3 = conv2d_dilate(h_conv2, w3,acc_rate) 
    return sess.run(h_conv3)                                 

# Standard RAKI training 

for s in range(3):
    data=np.load("{c}/{d}_under.npy".format(c=calib_dir,d=dir_name))
    data= data[s,0,:,:,:]
    coils2=[]
    for i in range(data.shape[0]):
        coils2.append(data[i,:,:])

    check2 =np.stack((c for c in coils2),axis=-1)

    kspace= check2
    #print(check2.shape)

     

    
    kernel_x_1 = 5
    kernel_y_1 = 2

    kernel_x_2 = 1
    kernel_y_2 = 1

    kernel_last_x = 3
    kernel_last_y = 2

    layer1_channels = 256
    layer2_channels = 256

    MaxIteration = 2000
    LearningRate = 1e-2

    #### Input/Output Data ####
    #loaded = np.load('sliced_data6.npz')
    #actual_data=loaded['a']
    #actual_data = np.load(data_)
    #actual_data=actual_data.real
    #image= ifft2(datas)
    #max_ =np.max(image[:,:,1])
    #image/(max_)
    #real=actual_data.real
    #imag= actual_data.imag
    #kspace=np.concatenate((real , imag),2)
    #kspace=actual_data
    """for i in range(1,18,2):
    kspace[:,i,:]=0
    #kspace[:,i+4,:]=0
    #  unaq.append(slice_[:,1,:])
    #data.append(slice_[:,i+1:i+4,:])

    for i in range(50,68,2):
    kspace[:,i,:]=0
    #kspace[:,i+1,:]=0
    #kspace[:,i+2,:]=0
    #kspace[:,i+3,:]=0
    #kspace[:,i+4,:]=0"""
    resultName = 'RAKI_recon'
    recon_variable_name = 'kspace_recon'
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

    if np.size(indic)==0:    
        no_ACS_flag=1;       
        print('No ACS signal in input data, using individual ACS file.')
        matfn = 'ACS.mat'   
        ACS = sio.loadmat(matfn)
        ACS = ACS['ACS']     
        [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
        ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
        ACS_re[:,:,0:no_ch] = np.real(ACS)
        ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)
    else:
        no_ACS_flag=0;
        print('ACS signal found in the input data')
        indic = indic[1][:]
        center_start = picks[0][indic[0]];
        center_end = picks[0][indic[-1]+1];
        ACS = kspace[x_start:x_end+1,center_start:center_end+1,:]
        [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS)
        ACS_re = np.zeros([ACS_dim_X,ACS_dim_Y,ACS_dim_Z*2])
        ACS_re[:,:,0:no_ch] = np.real(ACS)
        ACS_re[:,:,no_ch:no_ch*2] = np.imag(ACS)

    acc_rate = d_picks[0][0]
    no_channels = ACS_dim_Z*2
    print(acc_rate)
    name_weight = resultName + ('{s}_weight_%d%d,%d%d,%d%d_%d,%d.mat'.format(s=s) % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))
    name_image = resultName + ('_image_%d%d,%d%d,%d%d_%d,%d.mat' % (kernel_x_1,kernel_y_1,kernel_x_2,kernel_y_2,kernel_last_x,kernel_last_y,layer1_channels,layer2_channels))

    existFlag = os.path.isfile(name_image)
    print(acc_rate)
    w1_all = np.zeros([kernel_x_1, kernel_y_1, no_channels, layer1_channels, no_channels],dtype=np.float32)
    w2_all = np.zeros([kernel_x_2, kernel_y_2, layer1_channels,layer2_channels,no_channels],dtype=np.float32)
    w3_all = np.zeros([kernel_last_x, kernel_last_y, layer2_channels,acc_rate - 1, no_channels],dtype=np.float32)    

    b1_flag = 0;
    b2_flag = 0;                       
    b3_flag = 0;

    if (b1_flag == 1):
     b1_all = np.zeros([1,1, layer1_channels,no_channels]);
    else:
     b1 = []

    if (b2_flag == 1):
     b2_all = np.zeros([1,1, layer2_channels,no_channels])
    else:
     b2 = []

    if (b3_flag == 1):
     b3_all = np.zeros([1,1, layer3_channels, no_channels])
    else:
     b3 = []

    target_x_start = np.int32(np.ceil(kernel_x_1/2) + np.floor(kernel_x_2/2) + np.floor(kernel_last_x/2) -1); 
    target_x_end = np.int32(ACS_dim_X - target_x_start -1); 

    time_ALL_start = time.time()

    [ACS_dim_X, ACS_dim_Y, ACS_dim_Z] = np.shape(ACS_re)
    ACS = np.reshape(ACS_re, [1,ACS_dim_X, ACS_dim_Y, ACS_dim_Z]) 
    ACS = np.float32(ACS)  

    target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate;     
    target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + np.floor(kernel_y_2/2) + np.floor(kernel_last_y/2))) * acc_rate -1;

    target_dim_X = target_x_end - target_x_start + 1
    target_dim_Y = target_y_end - target_y_start + 1
    target_dim_Z = acc_rate - 1

    print('go!')
    time_Learn_start = time.time() 

    errorSum = 0;
    config = tf.ConfigProto()

    error_total=[]
    for ind_c in range(ACS_dim_Z):

     sess = tf.Session(config=config)
    # set target lines
     target = np.zeros([1,target_dim_X,target_dim_Y,target_dim_Z])
     print('learning channel #',ind_c+1)
     time_channel_start = time.time()
    
     for ind_acc in range(acc_rate-1):
        target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + (np.ceil(kernel_y_2/2)-1) + (np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1 
        target_y_end = ACS_dim_Y  - np.int32((np.floor(kernel_y_1/2) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2))) * acc_rate + ind_acc
        target[0,:,:,ind_acc] = ACS[0,target_x_start:target_x_end + 1, target_y_start:target_y_end +1,ind_c];

    # learning
    
     [w1,w2,w3,error,error_tot_]=learning(ACS,target,acc_rate,sess) 
     w1_all[:,:,:,:,ind_c] = w1
     w2_all[:,:,:,:,ind_c] = w2
     w3_all[:,:,:,:,ind_c] = w3                               
     time_channel_end = time.time()
     print('Time Cost:',time_channel_end-time_channel_start,'s')
     print('Norm of Error = ',error)
     errorSum = errorSum + error
     error_total.append(error_tot_)

    sess.close()
    tf.reset_default_graph()
    
    time_Learn_end = time.time();
    print('lerning step costs:',(time_Learn_end - time_Learn_start)/60,'min')
    sio.savemat("{c}/{nw}".format(nw=name_weight,c=calib_dir), {'w1': w1_all,'w2': w2_all,'w3': w3_all})
    np.save("errors.npy",error_total)  


    kspace_recon_all = np.copy(kspace_all)
    kspace_recon_all_nocenter = np.copy(kspace_all)

    kspace = np.copy(kspace_all)

    over_samp = np.setdiff1d(picks,np.int32([range(0, n1,acc_rate)]))
    kspace_und = kspace
    kspace_und[:,over_samp,:] = 0;
    [dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z] = np.shape(kspace_und)

    kspace_und_re = np.zeros([dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_und_re[:,:,0:dim_kspaceUnd_Z] = np.real(kspace_und)
    kspace_und_re[:,:,dim_kspaceUnd_Z:dim_kspaceUnd_Z*2] = np.imag(kspace_und)
    kspace_und_re = np.float32(kspace_und_re)
    kspace_und_re = np.reshape(kspace_und_re,[1,dim_kspaceUnd_X,dim_kspaceUnd_Y,dim_kspaceUnd_Z*2])
    kspace_recon = kspace_und_re

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1/3 ; 

    for ind_c in range(0,no_channels):
        print('Reconstruting Channel #',ind_c+1)
        
        sess = tf.Session(config=config) 
        if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        sess.run(init)
        
        # grab w and b
        w1 = np.float32(w1_all[:,:,:,:,ind_c])
        w2 = np.float32(w2_all[:,:,:,:,ind_c])     
        w3 = np.float32(w3_all[:,:,:,:,ind_c])

        if (b1_flag == 1):
            b1 = b1_all[:,:,:,ind_c];
        if (b2_flag == 1):
            b2 = b2_all[:,:,:,ind_c];
        if (b3_flag == 1):
            b3 = b3_all[:,:,:,ind_c];                
            
        res = cnn_3layer(kspace_und_re,w1,b1,w2,b2,w3,b3,acc_rate,sess) 
        target_x_end_kspace = dim_kspaceUnd_X - target_x_start;
        
        for ind_acc in range(0,acc_rate-1):

            target_y_start = np.int32((np.ceil(kernel_y_1/2)-1) + np.int32((np.ceil(kernel_y_2/2)-1)) + np.int32(np.ceil(kernel_last_y/2)-1)) * acc_rate + ind_acc + 1;             
            target_y_end_kspace = dim_kspaceUnd_Y - np.int32((np.floor(kernel_y_1/2)) + (np.floor(kernel_y_2/2)) + np.floor(kernel_last_y/2)) * acc_rate + ind_acc;
            kspace_recon[0,target_x_start:target_x_end_kspace,target_y_start:target_y_end_kspace+1:acc_rate,ind_c] = res[0,:,::acc_rate,ind_acc]

    sess.close()
    tf.reset_default_graph()
    
    kspace_recon = np.squeeze(kspace_recon)

    kspace_recon_complex = (kspace_recon[:,:,0:np.int32(no_channels/2)] + np.multiply(kspace_recon[:,:,np.int32(no_channels/2):no_channels],1j))
    kspace_recon_all_nocenter[:,:,:] = np.copy(kspace_recon_complex); 


    if no_ACS_flag == 0: 
     kspace_recon_complex[:,center_start:center_end,:] = kspace_NEVER_TOUCH[:,center_start:center_end,:]
     print('ACS signal has been putted back')
    else:
     print('No ACS signal is putted into k-space')

    kspace_recon_all[:,:,:] = kspace_recon_complex; 

    for sli in range(0,no_ch):
     kspace_recon_all[:,:,sli] = np.fft.ifft2(kspace_recon_all[:,:,sli])

    rssq = (np.sum(np.abs(kspace_recon_all)**2,2)**(0.5))
    sio.savemat(name_image,{recon_variable_name:kspace_recon_complex})

    time_ALL_end = time.time()
    print('All process costs ',(time_ALL_end-time_ALL_start)/60,'mins')
    print('Error Average in Training is ',errorSum/no_channels)
            
            
            


