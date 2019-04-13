# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 18:22:57 2018

@author: disha
"""

import numpy as np
import cv2

img1 = cv2.imread('original_imgs/turbine-blade.jpg')

gray_scale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype = np.int8)

image_h, image_w = gray_scale.shape

kernel_h, kernel_w=kernel.shape

new_img = np.zeros(gray_scale.shape) 

image_padded = np.zeros((gray_scale.shape[0] + 2, gray_scale.shape[1] + 2))   
image_padded[1:-1, 1:-1] = gray_scale

img_1 = gray_scale.copy() 

for i in range(image_h):
    for j in range(image_w):
        val=np.sum(kernel*image_padded[i:i+3,j:j+3])        

        if val>600:
            new_img[i,j]=255
        else:
            new_img[i,j]=0
            
cv2.imwrite('point_threshold.jpg',new_img)

largest_num = new_img[0][0]
for row_idx, row in enumerate(new_img):
    for col_idx, num in enumerate(row):
        if num > largest_num:
            largest_num = num
            id_x=col_idx
            id_y=row_idx

large_val = largest_num
id_f_x=id_x
id_f_y=id_y
print(id_f_x,id_f_y)

cv2.circle(img1,(id_f_x,id_f_y), 10, (0,0,255), 2)
font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1, "[445,249]", (410,230), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
cv2.imwrite('porous_img.jpg',img1)


from matplotlib import pyplot as plt

img_segment = cv2.imread('original_imgs/segment.jpg')

segment = cv2.cvtColor(img_segment, cv2.COLOR_BGR2GRAY)
print(len(segment))
print(len(segment[0]))


arr = []

for i in range(len(segment)):
    for j in range (len(segment[0])):
        if segment[i][j]!=0:
            arr.append(segment[i][j]) 
            
count_arr=np.zeros([256,1])

for i in range(0,len(arr)):
    count_arr[arr[i]]=count_arr[arr[i]]+1
    
    
plt.plot(count_arr,color='green')
plt.show()

new_img=np.zeros(segment.shape)

for i in range(len(segment)):
    for j in range (len(segment[0])):
        if segment[i][j]<200:
            new_img[i][j]=0
        else:
            new_img[i][j]=255
            
cv2.imwrite('threshold-image.jpg',new_img)

i, j = np.where(new_img == 255)
print(i,j)

cv2.rectangle(img_segment,(160,125),(210,168),(0,0,255),2)
cv2.rectangle(img_segment,(250,210),(305,76),(0,0,255),2)
cv2.rectangle(img_segment,(330,285),(363,20),(0,0,255),2)
cv2.rectangle(img_segment,(388,255),(425,38),(0,0,255),2)
cv2.imwrite('final_img.jpg',img_segment)