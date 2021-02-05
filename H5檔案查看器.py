# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 21:27:35 2021

@author: user
"""

import h5py    #HDF5的读取： 
import numpy as np 
import cv2

def h5_print(file):
    
    f = h5py.File(file,'r')   #打开h5文件  # 可以查看所有的主键  
    for key in f.keys():      
        print(f[key].name)      
        print(f[key].shape) 
        
        print(f[key][1][0])   
        img=f[key][1][0]
        cv2.imshow(f[key].name, img)
        #cv2.destroyAllWindows()
        
        #print(f[key].value)

# f = h5py.File('./data/train.h5','r')   #打开h5文件  # 可以查看所有的主键  
# for key in f.keys():      
#     print(f[key].name)      
#     print(f[key].shape)      
#     print(f[key].value)




# imgData = np.zeros((30,3,128,256))  
# f = h5py.File('HDF5_FILE.h5','w')   #創建一個h5文件，文件指針是f  
# f['data'] = imgData                 #將數據寫入文件的主鍵data下面  
# f['labels'] = range(100)            #將數據寫入文件的主鍵labels下面  
# f.close()                           #關閉文件  


# file='mytestfile.hdf5'
file='./data/train.h5'
file='D:/mytestfile_41x41_all_small_x2.h5'
file='D:/mytestfile_40x40_all_small_x4.h5'
data_count=0
label_count=0







frame=np.zeros((2,1,41,41))

a=np.zeros((1,4))
b=np.zeros((1,4))
b=((1,2,3,4))
# initial=> create h5 
def create_h5_file(file):
    with h5py.File(file, "w") as f:
        dset = f.create_dataset('data', data=frame, maxshape=(None, 1,41,41), chunks=True)
        dset = f.create_dataset('label', data=frame,maxshape=(None, 1,41,41), chunks=True)
        # dset = f.create_dataset('test', data=a, maxshape=(None, 4), chunks=True)
        # print(f['test'].value)
    f.close() 
    


# extend dataset
def extend_dataset(file,num,key):  #擴展資料
    with h5py.File(file, "a") as f:
        f[key].resize((f[key].shape[0] + num), axis=0)
        # f['test'].resize((f['test'].shape[0] + num), axis=0)
        # f['test'] [2]= b
        # print(f['test'].value)
        # print("====",f['test'].shape[0] )
        # hf[key][-1:] = frame
    f.close() 
    
def label_conut(file,key): #暫時不寫
    with h5py.File(file, "a") as f:
        print(key,"=" , f[key].shape[0])

def add_image_data(file,data,label): #放入1筆資料
    with h5py.File(file, "a") as f:
        f['data'] [data_count]= data
        data_count+=1
        f['label'] [label_count]= label
        label_count+=1
    f.close() 
    
    
# create_h5_file(file)

# h5_print(file)

# extend_dataset(file,2649,'data')
# extend_dataset(file,2549,'label')

# h5_print(file)
    
    
h5_print(file)
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    