# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:43:48 2016

@author: ckirst
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.filters as filters

import experiment as exp;

img = exp.load_img(t = 529204)
imgs = filters.gaussian_filter(np.asarray(img, float), 1.0);
  
imgdata = imgs.flatten();
imgdata.shape = (imgdata.shape[0], 1);

plt.figure(1); plt.clf();
plt.hist(imgdata[:,0], 256)


### fit mixture of tw gaussians to it

from sklearn import mixture

clf = mixture.GMM(n_components=3, covariance_type='full',  n_init = 15, verbose = 2)
clf.fit(imgdata)


npts = 100;
x = np.linspace(imgdata[:,0].min(), imgdata[:,0].max(), npts);
x.shape = (x.shape[0], 1);
z = np.exp(clf.score_samples(x)[0]);

pred = clf.predict(imgdata);
pred.shape = img.shape;

plt.figure(100); plt.clf();
plt.subplot(2,2,1);
plt.hist(imgdata[:,0], 256, normed = True)
plt.plot(x,z)
plt.subplot(2,2,2);
plt.imshow(img);
plt.subplot(2,2,3);
plt.imshow(pred);
plt.subplot(2,2,4);
plt.imshow(imgs < 80)

plt.title('GMM fit')
plt.axis('tight')
plt.show()

print clf.means_


from skimage.filters import threshold_otsu

threshold_otsu(imgs)