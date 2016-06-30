# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 16:16:00 2016

@author: ckirst
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np



import experiment as exp;

img = exp.load_img(t = 529204)


#img = cv2.medianBlur(img,5);
img = cv2.GaussianBlur(img, (5,5), 1);

dst = cv2.cornerHarris(img,6,2,0.1)

#result is dilated for marking the corners, not important
#dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
#img[dst>0.01*dst.max()]=[0,0,255]

#cv2.imshow('dst',img)
plt.figure(1); plt.clf();
plt.subplot(1,2,1)
plt.imshow(img);
plt.subplot(1,2,2)
plt.imshow(dst)



#if cv2.waitKey(0) & 0xff == 27:
#    cv2.destroyAllWindows()