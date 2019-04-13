# -*- coding: utf-8 -*-
"""
@author: disha
"""


import numpy as np
import cv2

def hough_lines_acc(img, theta_resolution=1):
    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height*height + width*width))
    rhos = np.linspace(-img_diagonal, img_diagonal, img_diagonal*2)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # create the empty Hough Accumulator with dimensions equal to the size of
    # rhos and thetas
    hough_acc = np.zeros((len(rhos), len(thetas)), dtype=np.uint8)
    y_id, x_id = np.nonzero(img) # find all edge (nonzero) pixel indexes

    for i in range(len(x_id)): # cycle through edge points
        x = x_id[i]
        y = y_id[i]

        for i in range(len(thetas)): 
            rho = int((x * np.cos(thetas[i]) + y * np.sin(thetas[i])) + img_diagonal)
            hough_acc[rho, i] += 1

    return hough_acc, rhos, thetas

def hough_peaks(hough_acc, num_peaks, nhood_size):
    # loop through number of peaks to identify
    indicies = []
    hough_acc_1 = np.copy(hough_acc)
    for i in range(num_peaks):
        idmax = np.argmax(hough_acc_1) 
        hough_acc_idmax = np.unravel_index(idmax, hough_acc_1.shape) 
        indicies.append(hough_acc_idmax)


        idmax_y, idmax_x = hough_acc_idmax 
        
        if (idmax_x - (nhood_size/2)) < 0:
            min_x = 0
        else: 
            min_x = idmax_x - (nhood_size/2)
            
        if ((idmax_x + (nhood_size/2) + 1) > hough_acc.shape[1]): 
            max_x = hough_acc.shape[1]
        else: 
            max_x = idmax_x + (nhood_size/2) + 1


        if (idmax_y - (nhood_size/2)) < 0: 
            min_y = 0
        else: 
            min_y = idmax_y - (nhood_size/2)
        if ((idmax_y + (nhood_size/2) + 1) > hough_acc.shape[0]): 
            max_y = hough_acc.shape[0]
        else: 
            max_y = idmax_y + (nhood_size/2) + 1

        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                
                hough_acc_1[y, x] = 0

                
                if (x == min_x or x == (max_x - 1)):
                    hough_acc[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    hough_acc[y, x] = 255

    return indicies, hough_acc

def hough_lines_draw(img, indicies, rhos, thetas):
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
# read in shapes image and convert to grayscale
coin_img = cv2.imread('original_imgs/hough.jpg')

img_b = coin_img.copy()

img_b[:, :, 1] = 0
img_b[:, :, 2] = 0

blur_img_b = cv2.GaussianBlur(img_b, (5, 5), 1.5)

coin_edges = cv2.Canny(blur_img_b, 100, 200)


# run hough_lines_accumulator on the shapes canny_edges image
hough_acc, rhos, thetas = hough_lines_acc(coin_edges)
indicies, hough_acc = hough_peaks(hough_acc, 6, 20) # find peaks
hough_lines_draw(coin_img, indicies, rhos, thetas)


cv2.imwrite('red_lines.jpg',coin_img)

# read in shapes image and convert to grayscale
coin_img_b = cv2.imread('original_imgs/hough.jpg')

img_b = coin_img_b.copy()

img_b[:, :, 0] = 0
img_b[:, :, 1] = 0


blur_img_b = cv2.GaussianBlur(img_b, (5, 5), 0)

coin_edges_b = cv2.Canny(blur_img_b, 100, 200)


hough_acc, rhos, thetas = hough_lines_acc(coin_edges_b)
indicies, hough_acc = hough_peaks(hough_acc, 8, 40) # find peaks
hough_lines_draw(coin_img_b, indicies, rhos, thetas)

cv2.imwrite('blue_lines.jpg',coin_img_b)