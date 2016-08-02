# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 08:59:50 2016

@author: ckirst
"""

import os

import numpy as np
import matplotlib.pyplot as plt;

import copy
import time

import imageprocessing.active_worm as aw;



import experiment as exp;

wid = 80;

xydata = exp.load(wid = wid)
ntime = xydata.shape[0];

#create results data array for traces, points

basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/Data/2016_07_23_Wormshapes/'
shapefile = os.path.join(basedir, "shape_wid=%d_shapes.npy" % wid);
measurefile  = os.path.join(basedir, "shape_wid=%d_measures.npy" % wid);
figfile =  os.path.join(basedir, "shape_wid=%d_t=%s.png" % (wid, "%d"));

# create the files

npoints = 21;

#shapedata = np.zeros((ntime,3*npoints));
#np.save(shapefile, shapedata);
#shapedata = [];

#measuredata = np.zeros((ntime, 12));
#np.save(measurefile, measuredata)
#measuredata = [];#

reload(aw)

def analyse_shape_at_time(t):
  print t;
  img = exp.load_img(wid = wid, t = t);
  
  ws = aw.WormModel(npoints = npoints);
  ws.from_image(img, verbose = False, save = figfile % t);
    
  sdat = np.load(shapefile, mmap_mode = 'r+');
  sdat[t,:] = ws.to_array();
  
  mdat = np.load(measurefile, mmap_mode = 'r+');
  mdat[t,:] = ws.measure();



analyse_shape_at_time(105820)

analyse_shape_at_time(105816)

analyse_shape_at_time(151870)


analyse_shape_at_time(151873)

analyse_shape_at_time(164166)


plt.figure(100); plt.clf();
plt.imshow(exp.load_img(wid = wid, t=167169))


from joblib import Parallel, delayed, cpu_count

Parallel(n_jobs = cpu_count())( delayed(analyse_shape_at_time)(t) for t in xrange(167870, ntime))











import matplotlib.cm as cm;

def plotTrajectory(xydata, colordata = 'time', cmap = cm.jet, size = 20, ax = None, line = True):
    """Plot trajectories with color using time or specified values"""
    
    if isinstance(colordata, basestring) and colordata in ['time']:
      c = np.linspace(0, len(xydata[:,0]), len(xydata[:,0]));
    else:
      c = colordata;
    
    if ax is None:
      ax = plt.gca();
    s = ax.scatter(xydata[:,0], xydata[:,1], c = c, cmap = cmap, s = size, marker = 'o', lw = 0);
    if isinstance(line, basestring):
      ax.plot(xydata[:,0], xydata[:,1], color = line);    
    
    ax.get_figure().colorbar(s, ax = ax);
    return ax



def plot_shape_trajectory(tstart = 500000, tend = ntime, nfig = 100):
  xy = exp.load(wid = wid);
  
  # head tail positions
  xycht = np.load(posfile);
  print xycht.shape  
  
  # reduce to time window of interest
  xy = xy[tstart:tend,:];
  xycht = xycht[tstart:tend,:];
  
  plt.figure(nfig); plt.clf();
  plotTrajectory(xy, cmap = cm.Greys, line = 'gray');
  off= [75,75]
  plotTrajectory(xycht[:,0,:] + xy - off, cmap = cm.Reds, line = 'red');
  plotTrajectory(xycht[:,1,:] + xy - off, cmap = cm.Blues, line = None);
  plotTrajectory(xycht[:,2,:] + xy - off, cmap = cm.Greens, line = None);
  

plot_shape_trajectory(tstart = 0, tend = 1, nfig = 105);
  

plot_shape_trajectory(tend = 502000, nfig = 100);

plot_shape_trajectory(tstart = 505000, tend = 508000, nfig = 101);

plot_shape_trajectory(tstart = 510000, tend = 510500, nfig = 102);



















