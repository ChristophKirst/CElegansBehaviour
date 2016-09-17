# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 17:07:02 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters


import analysis.experiment as exp
import worm.model as aw;


t = 0;
# load image near an intersection
t = t+1;
print(t);
img = exp.load_img(wid = 80, t= 524700+914+t);
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);
#imgs = img;
th = 90;
imgs[imgs  > th] = th;
#imgs =- th -imgs;


plt.figure(1); plt.clf();
plt.imshow(imgs);

# find worm model

ws = aw.WormModel();
ws.from_image(imgs, verbose = False, sigma = None, threshold_factor = 0.95);

plt.figure(1); plt.clf();
ws.plot(image = imgs)


# represent worm by spline coefficients for angle of center line + fixed width profile









