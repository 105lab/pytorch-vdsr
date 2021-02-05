# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 20:16:56 2021

@author: user
"""

"""
自動數據集製作
包含  插值 
     轉y值
     讀圖
     寫入h5檔案

"""

import numpy as np
import cv2
import h5py    #HDF5的读取： 
import time

def image_segmentation(file,pix):
    global first,data_count,label_count,img_file,h5_file
    big_img = cv2.imread(file)
    #cv2.imshow('big Image', big_img)
    big_img=cv2.resize(big_img, (800, 800), interpolation=cv2.INTER_CUBIC) #妥協方法 都縮到1000 避免存h5檔案報錯
    # cv2.imshow('Image_gt', big_img)#---------------------------------show
    #1000 40
    # h=big_img.shape[0]
    # w=big_img.shape[1]
    # 裁切圖片
    print("+++++++++++"+file+"+++++++++++")
    for x in range(1,700,200):
        for y in range(1,700,200):
            img = big_img[y:y+41, x:x+41]
            #img=cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_CUBIC) #妥協方法 都縮到1000 避免存h5檔案報錯
            h=img.shape[0]
            w=img.shape[1]
            img_gt_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #轉 YCR_CB
            img_gt_y = img_gt_y[:,:,0].astype(float) #得到y通道
            # print(img_gt_y)
            img_gt_y=img_gt_y/255      #  /255在儲存 原圖
            # print(img_gt_y)
            # cv2.imshow('My Image_gt', img_gt_y)#---------------------------------show
            # print("=======",img_gt_y.shape)
            img_gt_y=img_gt_y[np.newaxis,np.newaxis,:,:] #增加維度
            # print("=============",img_gt_y.shape)
            img = cv2.resize(img, (int(w/2), int(h/2)), interpolation=cv2.INTER_CUBIC) #雙三次降採樣
            img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC) #resize(img, 寬, 高) #雙三次升採樣
            #cv2.imshow('My Image_bic', img)
            # time.sleep(2000)
            img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #轉 YCR_CB
            img_y = img_y[:,:,0].astype(float) #得到y通道
            img_x4_y=img_y/255      #  /255在儲存
            # print(img_x4_y)
            # cv2.imshow('My Image_bic', img_x4_y)#---------------------------------show
            # print("=======",img_x4_y.shape)
            img_x4_y=img_x4_y[np.newaxis,np.newaxis,:,:] #增加維度
            # print("=============",img_x4_y.shape)
          #  return (img_gt_y,img_x4_y) #回傳圖 原圖_y值圖 , 降採樣x4_y值圖
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # time.sleep(2000)
            
            if first==True:
                first=False
                create_h5_file(h5_file,img_x4_y,img_gt_y)
                data_count+=1
                label_count+=1
                continue
            else:
                extend_dataset(h5_file,1,'data')
                extend_dataset(h5_file,1,'label')
                add_image_data(h5_file,img_x4_y,img_gt_y)
                data_count+=1
                label_count+=1



def y_img_make(file): #讀取圖片並轉呈訓練資料
    img = cv2.imread(file)
    #print(img.shape[0],img.shape[1],img.shape[2])#高 寬 通道
    img=cv2.resize(img, (1000, 1000), interpolation=cv2.INTER_CUBIC) #妥協方法 都縮到1000 避免存h5檔案報錯
    h=img.shape[0]
    w=img.shape[1]
    
    img_gt_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #轉 YCR_CB
    img_gt_y = img[:,:,0].astype(float) #得到y通道
    img_gt_y=img_gt_y/255      #  /255在儲存 原圖
    print("=======",img_gt_y.shape)
    img_gt_y=img_gt_y[np.newaxis,np.newaxis,:,:] #增加維度
    print("=============",img_gt_y.shape)

    img = cv2.resize(img, (int(w/4), int(h/4)), interpolation=cv2.INTER_CUBIC) #雙三次降採樣
    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC) #resize(img, 寬, 高) #雙三次升採樣
    
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #轉 YCR_CB
    img_y = img_y[:,:,0].astype(float) #得到y通道
    img_x4_y=img_y/255      #  /255在儲存
    print("=======",img_x4_y.shape)
    img_x4_y=img_x4_y[np.newaxis,np.newaxis,:,:] #增加維度
    print("=============",img_x4_y.shape)
    return (img_gt_y,img_x4_y) #回傳圖 原圖_y值圖 , 降採樣x4_y值圖










def h5_print(file):
    
    f = h5py.File(file,'r')   #打开h5文件  # 可以查看所有的主键  
    for key in f.keys():      
        print(f[key].name)      
        print(f[key].shape)      
        #print(f[key].value)







# frame=np.zeros((2,1,41,41))

# a=np.zeros((1,4))
# b=np.zeros((1,4))
# b=((1,2,3,4))
# initial=> create h5 
def create_h5_file(file,data,label):
    with h5py.File(file, "w") as f:
        dset = f.create_dataset('data', data=data, maxshape=(None, 1,None,None), chunks=True)
        dset = f.create_dataset('label', data=label,maxshape=(None, 1,None,None), chunks=True)
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
    global data_count
    global label_count
    print("data_count=",data_count,"label_count=",label_count)
    with h5py.File(file, "a") as f:
        f['data'] [data_count]= data
        
        f['label'] [label_count]= label
        
    f.close() 
    
    







#=====================================

img_file="C:/Users/user/Desktop/Flickr2K_vdsr_train_data/Flickr2K/Flickr2K_HR/"
h5_file='D:/mytestfile_41x41_all_small_x2.h5' #mytestfile.hdf5
data_count=0
label_count=0
first=True
segment=True #s需要分割

# h5_print('D:/mytestfile.h5')


if __name__ == "__main__":
    
    for i in range(1,2649):
        print("===第",i,"張圖片")
        if (i//10  ==0   ):
            img_file_no="00000"+str(i)+".png"
        elif (i//100 ==0    ):
            img_file_no="0000"+str(i)+".png"
        elif (i//1000 ==0   ):
            img_file_no="000"+str(i)+".png"
        elif (i//10000==0   ):
            img_file_no="00"+str(i)+".png"
        print(img_file_no+"\n")
        
        if segment==False:
            img_gt_y,img_x4_y=y_img_make(img_file+img_file_no)
            if first==True:
                first=False
                create_h5_file(h5_file,img_x4_y,img_gt_y)
                data_count+=1
                label_count+=1
                continue
            else:
                extend_dataset(h5_file,1,'data')
                extend_dataset(h5_file,1,'label')
                add_image_data(h5_file,img_x4_y,img_gt_y)
                data_count+=1
                label_count+=1
        elif segment==True:
            image_segmentation(img_file+img_file_no,40)
            
            
            
            















