# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 19:22:01 2021

@author: user
"""

import numpy as np
import cv2




# 讀取圖檔

def y_img_make(file):
    img = cv2.imread(file)
    #print(img.shape[0],img.shape[1],img.shape[2])#高 寬 通道
    h=img.shape[0]
    w=img.shape[1]
    img_gt_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #轉 YCR_CB
    img_gt_y = img[:,:,0].astype(float) #得到y通道
    img_gt_y=img_gt_y/255      #  /255在儲存 原圖



    img = cv2.resize(img, (int(w/4), int(h/4)), interpolation=cv2.INTER_CUBIC) #雙三次降採樣
    img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC) #resize(img, 寬, 高) #雙三次升採樣
    
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #轉 YCR_CB
    img_y = img_y[:,:,0].astype(float) #得到y通道
    img_x4_y=img_y/255      #  /255在儲存
    return (img_gt_y,img_x4_y) #回傳圖 原圖_y值圖 , 降採樣x4_y值圖
    

img = cv2.imread('123456.bmp')

print(img.shape[0],img.shape[1],img.shape[2])#高 寬 通道
h=img.shape[0]
w=img.shape[1]


img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB) #轉 YCR_CB
img_y = img[:,:,0].astype(float) #得到y通道
# print(img_y.shape)
# print(img_y)
img_y=img_y/255      #  /255在儲存
# print(img_y.shape)
# print(img_y)

# 顯示圖片
cv2.imshow('My Image', img_y)

img = cv2.resize(img, (int(w/9), int(h/9)), interpolation=cv2.INTER_CUBIC) #雙三次降採樣
img = cv2.resize(img, (w,h), interpolation=cv2.INTER_CUBIC) #resize(img, 寬, 高) #雙三次升採樣

img_x4_y = img[:,:,0].astype(float) #得到y通道
img_x4_y=img_x4_y/255      #  /255在儲存
# 顯示圖片
cv2.imshow('My Image2', img_x4_y)

# 按下任意鍵則關閉所有視窗
cv2.waitKey(0)
cv2.destroyAllWindows()





