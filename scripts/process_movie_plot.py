# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 15:03:07 2018

@author: ckirst
"""
import numpy as np
import matplotlib.pyplot as plt


def arrange_plots(n):
  m = int(np.ceil(np.sqrt(n)));
  k = int(np.ceil(1.0*n/m));
  return m,k;


def plot(image, link = True, fig = 19):
  plt.figure(fig); 
  plt.clf(); 
  if not isinstance(image, list):
    image = [image];
  m,k = arrange_plots(len(image));
  fig, axs = plt.subplots(k,m, sharex = link, sharey = link, num = fig);
  if not isinstance(axs, np.ndarray):
    axs = np.array([axs]);
  axs = axs.flatten();
  for j,i in enumerate(image):
    axs[j].imshow(i, interpolation = 'none' )
  plt.tight_layout();


def plot_plate(img, circles = None, region  = None, mask = None, fig = 2):
  fig = plt.figure(fig); plt.clf();  
  if mask is not None:
    plt.subplot(1,2,1);
  
  plt.imshow(img, cmap = plt.cm.gray, interpolation = 'none')
  ax = plt.gcf().gca();
  
  if circles is not None:
    i = 0; 
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow'];
    for (x, y, r) in circles:
      # draw the outer circle
      ax.add_artist(plt.Circle((x,y),r, color = colors[i%6],fill = False, linewidth = 2))
      # draw the center of the circle
      ax.add_artist(plt.Circle((x,y),20, color = colors[i%6], fill = True))
      ax.text(x,y, '%d' % i, horizontalalignment='center',  verticalalignment='center', color = 'white')
      i += 1;
  
  if region is not None:
      ax.add_artist(plt.Rectangle(region[:2], region[2], region[3], color = 'white', fill = False));
      
  if mask is not None:
    plt.subplot(1,2,2);
    plt.imshow(mask, cmap = plt.cm.gray, interpolation = 'none');
    ax = plt.gcf().gca();
    if circles is not None:
      i = 0; 
      for (x, y, r) in circles:
        ax.add_artist(plt.Circle((x,y),r, color = colors[i%6],fill = False, linewidth = 2))

  
    
    
  plt.tight_layout();
  plt.draw();
  
  
def plot_worm_detection(frame  = None, norm = None, detect = None, worm = None, fig = 3):
  plt.figure(fig); 
  plt.clf();  
  fig, axs = plt.subplots(2,2, num = fig);  
  axs = axs.flatten();
  
  if frame is not None:
    axs[0].imshow(frame, interpolation = 'none');
  if norm is not None:
    axs[1].imshow(norm, interpolation = 'none');
  if detect is not None:
    axs[2].imshow(detect, interpolation = 'none');
  if worm is not None:
    axs[3].imshow(worm, interpolation = 'none');
  
  plt.tight_layout()
