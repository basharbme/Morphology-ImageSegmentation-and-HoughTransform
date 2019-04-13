import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import math

img = cv2.imread("original_imgs/noise.jpg",0)

mask = np.zeros([5,3])
mask[:,:] = 255
mask.size
print(mask)

def dilation(img,mask):
    new_img = np.zeros(img.shape)
    
    h= len(mask)//2
    w= len(mask[0])//2
    
    for i in range(h,len(img)-h):
        for j in range(w,len(img[0])-w):
            val_region = img[i - h: i + h + 1, j - w: j + w + 1]
            val = 0
            for k in range(len(mask)):
                for l in range(len(mask[0])):
                    if(val_region[k,l] == mask[k,l]):
                        val += 1
            if(val != 0):
                new_img[i][j] = 255
    return new_img

def erosion(img,mask):
    new_img = np.zeros(img.shape)
    
    h= len(mask)//2
    w= len(mask[0])//2
    
    for i in range(h,len(img)-h):
        for j in range(w,len(img[0])-w):
            val_region = img[i - h: i + h + 1, j - w: j + w + 1]
            val = 0
            for k in range(len(mask)):
                for l in range(len(mask[0])):
                    if(val_region[k,l] == mask[k,l]):
                        val += 1                       
            if(val == mask.size):
                new_img[i][j] = 255
    return new_img


img_1=erosion(dilation(dilation(erosion(img,mask),mask),mask),mask)
img_2=dilation(erosion(erosion(dilation(img,mask),mask),mask),mask)  
cv2.imwrite('res_noise1.jpg',img_1) #opening followed by closing
cv2.imwrite('res_noise2.jpg',img_2) #closing followed by opening


img_3=dilation(img_1,mask)
img_bound_1=img_3-img_1 #Image boundary using dilation
cv2.imwrite('res_bound1.jpg',img_bound_1)

img_4=erosion(img_2,mask)
img_bound_2=img_2-img_4 #Image boundary using erosion
cv2.imwrite('res_bound2.jpg',img_bound_2)