# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 04:03:49 2018

@author: disha
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

def edge_detection(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow('Original image', cv2.WINDOW_NORMAL)
    cv2.imshow('Original image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    img_b = img.copy()

    img_b[:, :, 1] = 0
    img_b[:, :, 2] = 0
    cv2.imshow('Blue Image', img_b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#    image = cv2.GaussianBlur(img_b, (5, 5), 1.5)
    
    image = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Blue Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    sizey, sizex = image.shape[0:2]
    
    h=3//2
    w=3//2
    
    sobelimagey = image.copy() 
    img_1 = image.copy() 
    
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.int8) #Sobel kernel along y-axis
    
    
    for i in range(h,sizey-h):
        for j in range(w,sizex-w):
            val=0
            
            for m in range(3):
                for n in range(3):
                    val=val+ kernel_y[m][n]*img_1[i-h+m][j-w+n]
                               
            if val > 255:
               sobelimagey[i][j]=255
            elif val < 0:
                sobelimagey[i][j]=0
        
    cv2.namedWindow('New image y', cv2.WINDOW_NORMAL)
    cv2.imshow('New image y',sobelimagey)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    sobelimagex = image.copy() 
    img_2=image.copy()
    img_final=image.copy()
    
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.int8)
    
    for i in range(h,sizey-h):
        for j in range(w,sizex-w):
            val=0
            
            for m in range(3):
                for n in range(3):
                    val=val+ kernel_x[m][n]*img_2[i-h+m][j-w+n]
                   
            
            if val > 255:
                val = 255
            elif val < 0:
                val = 0
            
            sobelimagex[i][j]=val
        
    cv2.namedWindow('New image x', cv2.WINDOW_NORMAL)
    cv2.imshow('New image x',sobelimagex)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    for i in range(sizey):
        for j in range(sizex):
            q=np.sqrt(sobelimagex[i][j] ** 2 + sobelimagey[i][j] ** 2)      
            if(q>255):
                img_final[i][j]=255
            elif(q<0):
                img_final[i][j]=0
    
            img_final[i][j]=q
    
    
    cv2.namedWindow('Final image', cv2.WINDOW_NORMAL)
    cv2.imshow('Final image',img_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.subplot(2,2,1),plt.imshow(image,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(sobelimagex,cmap = 'gray')
    plt.title('SobelX'), plt.xticks([]), plt.yticks([]) 
    plt.subplot(2,2,3),plt.imshow(sobelimagey,cmap = 'gray')
    plt.title('SobelY'), plt.xticks([]), plt.yticks([])
    
    replicate = cv2.copyMakeBorder(img_final,10,10,10,10,cv2.BORDER_REPLICATE)
    plt.subplot(2,2,4),plt.imshow(replicate,cmap = 'gray')
    plt.title('Final Image'), plt.xticks([]), plt.yticks([])
    
    plt.show()