# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 02:35:04 2016

@author: ckirst
"""

import os
import numpy as np
import matplotlib.pyplot as plt


from scripts.analyse_wormshape import analyse_shape


import experiment as exp;

xydata = exp.load(wid= 80)
ntime = xydata.shape[0];

#create results data array for traces, points

wid = 80;

basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/Data/2016_06_21_Wormshapes/'
linefile = os.path.join(basedir, "shape_wid=%d_lines.npy" % wid);
posfile  = os.path.join(basedir, "shape_wid=%d_positions.npy" % wid);
curfile =  os.path.join(basedir, "shape_wid=%d_curvature.npy" % wid);
figfile =  os.path.join(basedir, "shape_wid=%d_t=%s.png" % (wid, "%d"));

# create the files

linedata = np.zeros((ntime, 3, 2, 50));
np.save(linefile, linedata)

posdata = np.zeros((ntime, 3, 2));
np.save(posfile, posdata)

curdata = np.zeros((ntime, 2));
np.save(curfile, curdata)

linedata = []; posdata = []; curdata = [];


def analyse_shape_at_time(t):
  img = exp.load_img(t = t);
  pos_center, pos_head, pos_tail, curvature_mean, curvature_variation, line_center, line_left, line_right = analyse_shape(img, verbose = True, save = figfile % t);
  
  ldat = np.load(linefile, mmap_mode = 'r+');
  ldat[t,0,:,:] = line_center;
  ldat[t,1,:,:] = line_left;
  ldat[t,2,:,:] = line_right;
  #ldat.close();
  
  pdat = np.load(posfile, mmap_mode = 'r+');
  pdat[t,0,:] = pos_center;
  pdat[t,1,:] = pos_head;
  pdat[t,2,:] = pos_tail;
  #pdat.close();
  
  cdat = np.load(curfile, mmap_mode = 'r+');
  cdat[t,0] = curvature_mean;
  cdat[t,1] = curvature_variation;
  #cdat.close();



from joblib import Parallel, delayed, cpu_count

Parallel(n_jobs = cpu_count())( delayed(analyse_shape_at_time)(t) for t in xrange(ntime))

Parallel(n_jobs = cpu_count())( delayed(analyse_shape_at_time)(t) for t in xrange(3212, 3300))

Parallel(n_jobs = cpu_count())( delayed(analyse_shape_at_time)(t) for t in xrange(3300, ntime))

Parallel(n_jobs = cpu_count())( delayed(analyse_shape_at_time)(t) for t in xrange(500000, ntime))

analyse_shape_at_time(500000)

analyse_shape_at_time(0)

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




def plot_body_line(tstart = 500000, tend = ntime, dt = 1, nfig = 200, lineids = [0]):
  xy = exp.load(wid = wid);
  
  # head tail positions
  lines = np.load(linefile, mmap_mode = 'r');
  #print lines.shape  
  pos = np.load(posfile);
  
  # reduce to time window of interest
  plt.figure(nfig); plt.clf();
  off = [75,75];
  for i in lineids:
    for t in range(tstart, tend, dt):
      xyt = xy[t,:];
      cl = lines[t,i,:,:];
      print xyt
      cc = cm.Blues;
      plt.plot(cl[0,:] + np.round(xyt[0]) - off[0], -cl[1,:] + np.round(xyt[1]) - off[1], color = cc(1.0 * t/(tend-tstart)));  
      


plot_body_line(tstart = 500000, tend = 500004, dt = 1)

plt.figure(300); plt.clf();
nts = 4;
cc = cm.Blues;
lines = np.load(linefile, mmap_mode = 'r');
poss = np.load(posfile, mmap_mode = 'r');
xy = exp.load(wid = wid);

for ti in range(nts):
  plt.subplot(2,nts,ti+1);
  tt = 500000 + ti;
  img = exp.load_img(t = tt);
  plt.imshow(img, cmap = 'gray', interpolation = 'nearest')
  
  cl = lines[tt,0,:,:];
  label = "%s\n%s\n%s" % (str(poss[tt, 0,:]), str(xy[tt,:]), str(np.round(xy[tt,:])));
  plt.plot(cl[0,:], cl[1,:], color = 'red');
  
  if ti > 0:  
    plt.subplot(2,nts,ti);
    plt.plot(cl[0,:], cl[1,:], color = 'blue');  
  
  plt.subplot(2,nts,ti+nts+1);
  img2 =  exp.load_img(t = tt-1);
  plt.imshow(np.asarray(img, dtype = float)-img2, cmap = 'jet', interpolation = 'nearest')
  plt.title(label)
