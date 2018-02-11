# -*- coding: utf-8 -*-
"""
Created on Sat Feb 10 15:28:03 2018

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt


def arrange_plots(n):
  m = int(np.ceil(np.sqrt(n)));
  k = int(np.ceil(1.0*n/m));
  return m,k;


def plot(image, title = None, link = True, fig = 19):
  plt.figure(fig); 
  plt.clf(); 
  if not isinstance(image, list):
    image = [image];
  if not isinstance(title, list):
    title = [title]; 
  title.extend([None] * len(image));
  m,k = arrange_plots(len(image));
  fig, axs = plt.subplots(k,m, sharex = link, sharey = link, num = fig);
  if not isinstance(axs, np.ndarray):
    axs = np.array([axs]);
  axs = axs.flatten();
  for j,i in enumerate(image):
    axs[j].imshow(i, interpolation = 'none' )
    if title[j] is not None:    
      axs[j].set_title(title[j]);
  plt.tight_layout();
