# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:21:43 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt

import scripts.analyse_wormshape as aws



import experiment as exp

import scipy.ndimage.filters as filters

from skimage.filters import threshold_otsu


t = 529202;

# load image

t = t + 1;
img = exp.load_img(wid = 80, t= t);
#img = exp.load_img(wid = 80, t= 529204);
#img = exp.load_img(wid = 80, t= 529218);

imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);

reload(aws)
sh = aws.analyse_shape(img, verbose = True);