# -*- coding: utf-8 -*-
"""
Deep Network to map images to worm shapes

Note:
  Network structure img -> conv -> conv -> hidden -> worm parameter
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'











import numpy as np
import matplotlib.pyplot as plt

import analysis.experiment as exp
import analysis.plot as aplt

import worm.model as wm;


# load image
img = exp.load_img(wid = 80, t= 500000, smooth = 1.0);

aplt.plot_array(img);


from skimage.filters import threshold_otsu
threshold_factor = 0.95;
level = threshold_factor * threshold_otsu(img);

from imageprocessing.masking import mask_to_phi 
phi_img = mask_to_phi(img < level);

### worm  -> Phi

w = wm.WormModel(npoints = 20);

w.from_image(img, sigma = None)
w.plot(image = img);

phi =  w.phi()


plt.figure(1);
plt.subplot(1,3,1);
plt.imshow(phi);
plt.subplot(1,3,2);
plt.imshow(phi_img);
plt.subplot(1,3,3);
plt.imshow(phi-phi_img);

dist = np.sum((phi-phi_img)**2);


### Set up Network

# structure img -> conv -> conv -> hidden -> worm parameter


import tensorflow as tf;





