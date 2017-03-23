# -*- coding: utf-8 -*-
"""
CElegans Behaviour


Video Generator
"""

import os
import time

import numpy as np;

import matplotlib.pyplot as plt;
import matplotlib.patches as patches;

import analysis.experiment as exp;

import matplotlib.animation as manimation


###############################################################################
### Frame generation
###############################################################################

def round_xy(xy):
  """Pixel position of image"""
  return np.array(np.round(np.array(xy, dtype = float)), dtype = int);
  #return np.array(np.floor(xy), dtype = int);
  #return np.array(np.ceil(xy), dtype = int);


def generate_image_extent(xy, xylim, worm_size, plot_range):
  """Generate extent of the worm image within the plotting area"""
  x0 = max(xy[0] - xylim[0][0] - worm_size[0]/2, 0);
  x1 = min(xy[0] - xylim[0][0] + worm_size[0]/2 + 1, plot_range[0]);
  y0 = max(xy[1] - xylim[1][0] - worm_size[1]/2, 0);
  y1 = min(xy[1] - xylim[1][0] + worm_size[1]/2 + 1, plot_range[1]);
  return (x0,x1,y0,y1);


def generate_image_data(strain = 'n2', wid = 0, t = 0, size = None, xy = None, worm = None, roi = None, background = None, sigma = 1.0, border = 75):
  """Generate data and parameter to plot worm image data"""
  # load worm image
  if worm is None:
    worm = exp.load_img(strain = 'n2', wid = wid, t = t, sigma = sigma);
  else:
    worm = exp.smooth_image(worm.copy(), sigma);
    
  worm_size = worm.shape;
  if background is not None:
    worm[worm > background] = background; 
  #print worm.shape
  
  # load position
  if xy is None:
    xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
    xy = xy[t,:];
  xy = round_xy(xy); 
  
  if roi is None:
    roi = exp.load_roi(strain = strain,  wid = wid);
    
  # plot limits
  xlim = [int(np.floor(roi[0]-roi[2]-border)), int(np.ceil(roi[0]+roi[2]+border))];
  ylim = [int(np.floor(roi[1]-roi[2]-border)), int(np.ceil(roi[1]+roi[2]+border))];
  #print xmm,ymm
  
  if size is None:
    plot_range = (xlim[1]-xlim[0], ylim[1]-ylim[0]);
  else:
    plot_range = size;
    
  extent = generate_image_extent(xy, [xlim, ylim], worm_size, plot_range)
    
  # place worm image
  return (worm, xy, roi, [xlim,ylim], extent, plot_range)


def worm_image(strain = 'n2', wid = 0, t = 0, size = None, xy = None, worm = None, roi = None, background = None, sigma = 1.0, border = 75):
  """Generate image of the worm at spatial location in full plate"""
  worm, xy, roi, xylim, extent, plot_range = generate_image_data(strain = strain, wid = wid, t = t, size = size, 
                                                        xy = xy, worm = worm, roi = roi, 
                                                        background = background, sigma = sigma, border = border)
  
  # create image
  img = np.ones(size, dtype = int) * background;
  
  if np.any(np.isnan(extent)):
    return img;
  
  x0, x1, y0, y1 = extent;
  xr = x1 - x0;
  yr = y1 - y0;  
  #img[x0:x1, y0:y1] = worm[:xr, :yr];
  if xr > 0 and yr > 0:
    img[y0:y1, x0:x1] = worm[:yr,:xr];
  
  return img;


def plot_worm(strain = 'n2', wid = 0, t = 0, size = None, xy = None, worm = None, roi = None, background = None, sigma = 1.0, border = 75, vmin = 60, vmax = 90, cmap = plt.cm.gray):
  """Plot worm at position in full plate"""
  worm, xy, roi, xylim, extent, plot_range = generate_image_data(strain = strain, wid = wid, t = t, size = size, 
                                                        xy = xy, worm = worm, roi = roi, 
                                                        background = background, sigma = sigma, border = border);
  
  center = np.array(np.array(plot_range) /2, dtype = int);
  ax = plt.gca();

  # draw plate outline  
  #r =roi[3];  
  r = center[0] - border;
  ax.add_artist(plt.Circle((center[0], center[1]), r, color = 'black',fill = False, linewidth = 1))
  ax.set_xlim(0, plot_range[0]);
  ax.set_ylim(0, plot_range[1]);
  
  # place worm image
  if np.any(np.isnan(extent)):
    return ax;
  
  ax.imshow(worm, extent = extent, vmin = vmin, vmax = vmax, cmap = cmap);
  return ax;
  

def generate_feature_data(strain = 'n2', wid = 0, features = [], feature_filters = None):
  """Generate feature data"""
  if feature_filters is None:
    feature_filters = [None for f in features];
  
  feature_data = [];
  feature_label = [];
  for f,ff in zip(features, feature_filters):
    if isinstance(f, str):
      feature_label.append(f);
      dat = exp.load(strain = strain, wid = wid, dtype = f);
      if ff is not None:
        dat = ff(dat);
      feature_data.append(dat);
    elif isinstance(f, tuple):
      feature_label.append(f[0]);
      feature_data.append(f[1]);
    else:
      feature_label.append('feature');
      feature_data.append(f);
  
  return (feature_label, feature_data);
  
def generate_feature_indicator(strain = 'n2', wid = 0, feature_indicator = None):
  """Generate data for the feature indicator"""
  if isinstance(feature_indicator, basestring):
    feature_indicator = exp.load(strain = strain, wid = wid, dtype = feature_indicator);
    feature_indicator[np.isnan(feature_indicator)] = 0.0;
    feature_indicator = np.array(feature_indicator, dtype = bool);
  return feature_indicator;

def generate_time_str(sec):
  m, s = divmod(sec, 60);
  h, m = divmod(m,   60);
  return '%s:%02d:%02d' % (str(h).rjust(3), m, s)

def generate_stage_str(stage):
  if stage <= 4:
    return 'L%d' % stage;
  else:
    return 'A';
    
    
def arange_plots(n, prefer_rows = True):
  r = int(np.sqrt(n));
  c = int(np.ceil(n * 1.0 / r));
  if prefer_rows:
    return r,c
  else:
    return c,r

def animate_frames(strain = 'n2', wid = 0,  
                size = None, xy = None, roi = None, 
                background = None, sigma = 1.0, border = 75, vmin = 60, vmax = 90, cmap = plt.cm.gray,
                times = all,
                features = ['speed', 'rotation'], feature_filters = None, history = 10, history_delta = 1, linecolor = 'gray', 
                feature_indicator = 'roam',
                time_data = None, time_cmap = plt.cm.rainbow, time_size = 30,
                time_stamp = True, sample_rate = 3, tref = None, stage_stamp = True, font_size = 16,
                pause = 0.01, legend = None,
                save = None, fps = 20, dpi = 300,
                verbose = True):
  """Animate frames and generate video"""         
  if times is all:
    t = 0;
  else:
    t = times[0];
  
  xynans = np.nan * np.ones(history);
  
  # memmap to images
  worm = exp.load_img(strain = strain, wid = wid);
  
  # xy positions
  if xy is None:
    xy = exp.load(strain = strain, wid = wid, dtype = 'xy');

  if times is all:    
    times = (0, len(xy));
  
  if isinstance(time_data, basestring):
    time_data = exp.load(strain = strain, wid = wid, dtype = time_data);
  
  # initialize image data
  wormt, xyt, roi, xylim, extent, plot_range = generate_image_data(strain = strain, wid = wid, t = times[0], size = size, 
                                                        xy = xy[t], worm = worm[t], roi = roi, 
                                                        background = background, sigma = sigma, border = border);
                                                        
  # initialize feature data                                                      
  feature_label, feature_data = generate_feature_data(strain, wid, features, feature_filters);
  nplt = len(feature_data)
  ncols = 1 + int(nplt > 0);
  
  # create plot
  fig = plt.gcf(); plt.clf();
  ax = plt.subplot(1,ncols,1);
  ax.set_xticks([])
  ax.set_yticks([])
  ax.set_axis_off();
  
  if legend is not None:
    for li,l in enumerate(legend):
      font = {'size': font_size }
      plt.text(10, plot_range[1]-100 + li * 50, l[1], color = l[0], fontdict = font)
  
  # draw plate outline
  center = np.array(np.array(plot_range) /2, dtype = int);
  r = center[0] - border;
  ax.add_artist(plt.Circle((center[0], center[1]), r, color = 'black', fill = False, linewidth = 0.5))
  ax.set_xlim(0, plot_range[0]);
  ax.set_ylim(0, plot_range[1]);
  
  # place worm image
  wimg = ax.imshow(wormt, extent = extent, vmin = vmin, vmax = vmax, cmap = cmap);
    
  # plot history of trace
  def generate_xy(t):
    hsteps = int(t / history_delta) + 1; # number of available steps in the past
    if hsteps - history > 0:
      t0 = t - history_delta * (history - 1);
      t1 = 0;
    else:
      t0 = t - history_delta * (hsteps - 1);
      t1 = -hsteps;
    
    xdat = xynans.copy();
    ydat = xynans.copy();
    xdat[t1:] = xy[t0:t+1:history_delta,0] - xylim[0][0]
    ydat[t1:] = xy[t0:t+1:history_delta,1] - xylim[1][0]
    return [xdat, ydat], t0, t1;
  
  if history > 0:
    xydat, t0, t1 = generate_xy(t);
    if time_data is None:
      cdat = np.linspace(0,1,history);
      vmin = 0;
      vmax = 1;
    else:
      cdat = xynans.copy();
      cdat[t1:] = time_data[t0:t+1:history_delta];
      vmin = np.nanmin(time_data);
      vmax = np.nanmax(time_data);
    
    print time_cmap.name
    scatter = plt.scatter(xydat[0], xydat[1], c = cdat, cmap = time_cmap, edgecolor = 'none', vmin = vmin, vmax = vmax, s = time_size)  ;
    
    # feature plots
    iplt = 0;
    pl = [];
    for fl,fd in zip(feature_label, feature_data):
      iplt += 1;
      plt.subplot(nplt, ncols, iplt * 2);
      pl.append(plt.plot(fd[t0:t+1:history_delta], c = linecolor));
      #plt.scatter(np.arange(nt), fd[t0:t+1:history_delta], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
      plt.title(fl);      
  
  else:
      scatter = None;
  
  # plot feature indicator
  feature_indicator = generate_feature_indicator(strain, wid, feature_indicator);
  if feature_indicator is not None:
    if feature_indicator[t]:
      fc = 'r';
    else:
      fc = 'w';
    posfac = 0.95; sizefac = 0.03;
    feature_rect = patches.Rectangle((posfac * plot_range[0], posfac * plot_range[1]),  sizefac * plot_range[0], sizefac * plot_range[1], edgecolor = None, facecolor = fc);
    ax.add_patch(feature_rect);
  
  # time stamp text  
  if time_stamp:
    font = {'size': font_size }
    tt = generate_time_str(0);
    time_stamp = plt.text(10, 10, tt, fontdict = font)
    if tref is None:
      tref = times[0];
    
  # stage stamp text
  if stage_stamp:
    stage_times = exp.load_stage_times(strain = strain, wid = wid);
    font = {'size': font_size }
    tt = generate_stage_str(1);
    stage_stamp = plt.text(plot_range[0] - 10, 10, tt, fontdict = font, horizontalalignment='right',)
  
  # movie saver
  if save is not None:
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Strain %s Worm %d', artist='Chirstoph Kirst',
                    comment='C Elegans Dataset, Bargmann Lab')
    writer = FFMpegWriter(fps=15, metadata=metadata);
    writer.setup(fig, save, dpi = dpi)
    
  
  
  # loop over times
  for t in times:
    if verbose:
      print '%d / %d' % (t, times[-1]);
    xyt = round_xy(xy[t]);
    
    # update worm image and position 
    wormt = exp.smooth_image(worm[t], sigma = sigma);
    wormt = wormt[::-1, :];
    wimg.set_data(wormt);
    extent = generate_image_extent(xyt, xylim, wormt.shape, plot_range);
    #extent = np.array(extent); extent = extent[[2,3,0,1]];
    wimg.set_extent(extent)
    
    # update scatter plot data
    if scatter is not None:
      #plt.scatter(xy[t0:t+1,1] - ymm[0], xy[t0:t+1,0] - xmm[0], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
      xydat, t0, t1 = generate_xy(t);
      scatter.set_offsets(np.array(xydat).T);
      if time_data is not None:
        cdat = xynans.copy();
        cdat[t1:] = time_data[t0:t+1:history_delta];
      scatter.set_array(cdat);
        
      for p,fd in zip(pl, feature_data):
        p[0].set_ydata(fd[t0:t+1:history_delta]);
        
    if feature_indicator is not None:
      if feature_indicator[t]:
        feature_rect.set_color('r');
      else:
        feature_rect.set_color('w');
    
    if time_stamp:
      time_stamp.set_text(generate_time_str((t-tref)/sample_rate));
    
    if stage_stamp:
      stage_stamp.set_text(generate_stage_str(np.sum(t >= stage_times)));
    
    fig.canvas.draw();
    fig.canvas.flush_events()   
    
    if pause is not None:
      #plt.show();
      plt.pause(pause);
    
    if save is not None:
      writer.grab_frame();
      #fig.savefig(save % t, dpi = 'figure', pad_inches = 0);
  
  if save is not None:
    writer.cleanup();
    





def animate_worms(strain = 'n2', wid = 0, times = all,  
                size = None, xy = None, roi = None, 
                background = None, sigma = 1.0, border = 75, vmin = 60, vmax = 90, cmap = plt.cm.gray,
                time_data = None, time_cmap = plt.cm.rainbow, time_size = 30, history = 10, history_delta = 1, 
                time_stamp = True, stage_stamp = True, sample_rate = 3, font_size = 16,
                pause = 0.01, save = None, fps = 20, dpi = 300,
                verbose = True):
  """Animate multiple worms and generate video"""
  
  # worms to plot
  wid = np.array([wid]).flatten();  
  nworms = len(wid);
  nrows,ncols = arange_plots(nworms)#

  # draw worms       
  xynans = np.nan * np.ones(history);
  
  # memmap to images
  worms = [exp.load_img(strain = strain, wid = w) for w in wid];
  #print worms  
  
  # xy positions
  if xy is None:
    xys = [exp.load(strain = strain, wid = w, dtype = 'xy') for w in wid];
  else:
    xys = np.array(xy);
  #print xys

  if times is all:
    tm = np.min([len(x) for x in xys]);
    times = np.array([[0, tm] for w in wid]);
  elif isinstance(times, np.ndarray) and times.ndim == 1:
    times = np.array([times for w in wid]);
  else:
    times = np.array(times);
  t0s = times[:,0];
  #print t0s
  
  if isinstance(time_data, basestring):
    time_data = [exp.load(strain = strain, wid = w, dtype = time_data) for w in wid];
  elif time_data is None:
    time_data = [None for w in wid];
  #print time_data
  
  #if time_stamp is True:
  #  time_stamp = [None for w in wid];
  #  time_stamp[0] = True;
  
  # initialize image data
  worm0s = []; xy0s = []; xylims = []; rois = []; extents = []; plot_ranges = [];
  for w,t,x,ws in zip(wid, t0s, xys, worms):
    wormt, xyt, roiw, xylim, extent, plot_range = generate_image_data(strain = strain, wid = w, t = t, size = size, 
                                                                     xy = x[t], worm = ws[t], roi = roi, 
                                                                     background = background, sigma = sigma, border = border);
    worm0s.append(wormt); xy0s.append(xyt); xylims.append(xylim); rois.append(roiw); extents.append(extent); plot_ranges.append(plot_range);
  
  # helper to create xy data
  def generate_xy(wi, t):
    hsteps = int(t / history_delta) + 1; # number of available steps in the past
    if hsteps - history > 0:
      t0 = t - history_delta * (history - 1);
      t1 = 0;
    else:
      t0 = t - history_delta * (hsteps - 1);
      t1 = -hsteps;
          
    xdat = xynans.copy();
    ydat = xynans.copy();
    xdat[t1:] = xys[wi][t0:t+1:history_delta,0] - xylims[wi][0][0]
    ydat[t1:] = xys[wi][t0:t+1:history_delta,1] - xylims[wi][1][0]
    return [xdat, ydat], t0, t1;  
  
  # create figure
  fig = plt.gcf(); plt.clf();
  axs = [];
  wimgs = []; scatters = []; time_stamps = []; stage_times = []; stage_stamps = [];
  for wi, w in enumerate(wid):
    ax = plt.subplot(nrows,ncols,wi+1);
    axs.append(ax);
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off();
    #ax.set_title('%d' % w, fontsize = font_size);
  
    # draw plate outline
    center = np.array(np.array(plot_ranges[wi]) /2, dtype = int);
    r = center[0] - border;
    ax.add_artist(plt.Circle((center[0], center[1]), r, color = 'black', fill = False, linewidth = 0.5))
    ax.set_xlim(0, plot_ranges[wi][0]);
    ax.set_ylim(0, plot_ranges[wi][1]);
  
    # place worm image
    wimgs.append(ax.imshow(worm0s[wi], extent = extents[wi], vmin = vmin, vmax = vmax, cmap = cmap));
    #print wimgs    
    
    if history > 0:
      xydat, t0, t1 = generate_xy(wi, t);
      if time_data[wi] is None:
        cdat = np.linspace(0,1,history);
        vmin = 0;
        vmax = 1;
      else:
        cdat = xynans.copy();
        cdat[t1:] = time_data[wi][t0:t+1:history_delta];
        vmin = np.nanmin(time_data[wi]);
        vmax = np.nanmax(time_data[wi]);
      
      scatters.append(plt.scatter(xydat[0], xydat[1], c = cdat, cmap = time_cmap, edgecolor = 'none', vmin = vmin, vmax = vmax, s = time_size));
    else:
      scatters.append(None);
    #print scatters
  
    # plot feature indicator
    #feature_indicator = generate_feature_indicator(strain, wid, feature_indicator);
    #if feature_indicator is not None:
    #if feature_indicator[t]:
    #  fc = 'r';
    #else:
    #  fc = 'w';
    #posfac = 0.95; sizefac = 0.03;
    #feature_rect = patches.Rectangle((posfac * plot_range[0], posfac * plot_range[1]),  sizefac * plot_range[0], sizefac * plot_range[1], edgecolor = None, facecolor = fc);
    #ax.add_patch(feature_rect);
  
    # time stamp text

    #if time_stamp[wi] is not None:
    #  font = {'size': font_size }
    #  tt = generate_time_str(0);
    #  time_stamps.append(plt.text(10, 10, tt, fontdict = font));
    
    # stage stamp text
    if stage_stamp:
      stage_times.append(exp.load_stage_times(strain = strain, wid = w));
      font = {'size': font_size, 'weight' : 'bold' }
      tt = generate_stage_str(1);
      stage_stamps.append(plt.text(plot_range[0] - 10, 10, tt, fontdict = font, horizontalalignment='right'));
      # stage_colors = plt.cm.Paired(np.linspace(0,1,12))[::2]
      #stage_colors = plt.cm.nipy_spectral(np.linspace(0,1,12))[[10,8,5,2,1,0]]
      stage_colors = plt.cm.nipy_spectral(np.linspace(0,1,20))[[18, 18, 15, 9,3,1,0]];
  
  if time_stamp:
    font = {'size': font_size + 1 }
    tt = 'N2 %s' % generate_time_str(0);
    time_stamp = fig.suptitle(tt, fontdict = font);
  
  # movie saver
  if save is not None:
    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Strain %s Worm %d', artist='Chirstoph Kirst',
                    comment='C Elegans Dataset, Bargmann Lab')
    writer = FFMpegWriter(fps=15, metadata=metadata);
    writer.setup(fig, save, dpi = dpi)
  
  # loop over times
  times_len = [len(ts) for ts in times];
  nsteps = np.min(times_len);
  #print plot_ranges
  #print xylims
  #print xys
  for s in range(nsteps):
    for wi,w in enumerate(wid):
      t = times[wi][s];
      if verbose:
        print '%d / %d' % (t, times[wi][-1]);        
      
      xyt = round_xy(xys[wi][t]);
      #print xyt
    
      # update worm image and position 
      wormt = exp.smooth_image(worms[wi][t], sigma = sigma);
      wormt = wormt[::-1, :];
      if wormt.min() == 0:
        wormt[:,:] = vmax;
      wimgs[wi].set_data(wormt);
      extent = generate_image_extent(xyt, xylims[wi], wormt.shape, plot_ranges[wi]);
      #print wi, xylims[wi], plot_ranges[wi]
      #print extent;
      wimgs[wi].set_extent(extent)
    
      # update scatter plot data
      if scatters[wi] is not None:
        #plt.scatter(xy[t0:t+1,1] - ymm[0], xy[t0:t+1,0] - xmm[0], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
        xydat, t0, t1 = generate_xy(wi,t);
        scatters[wi].set_offsets(np.array(xydat).T);
        if time_data[wi] is not None:
          cdat = xynans.copy();
          cdat[t1:] = time_data[wi][t0:t+1:history_delta];
        scatters[wi].set_array(cdat);
        #print scatters[wi];
      
      #if time_stamp[wi]:
      #  time_stamps[wi].set_text(generate_time_str((t-times[wi][0])/sample_rate));
    
      if stage_stamps[wi]:
        stage = np.sum(t >= stage_times[wi]);
        stage_stamps[wi].set_text(generate_stage_str(stage));
        stage_stamps[wi].set_color(stage_colors[stage]);
    
    if time_stamp:
      time_stamp.set_text('N2 %s' % generate_time_str((t-times[wi][0])/sample_rate));    
    
    fig.canvas.draw();
    fig.canvas.flush_events()   
    
    if pause is not None:
      #plt.show();
      plt.pause(pause);
    
    if save is not None:
      writer.grab_frame();
      #fig.savefig(save % t, dpi = 'figure', pad_inches = 0);
  
  if save is not None:
    writer.cleanup();
    


###############################################################################
### Gui
###############################################################################


import pyqtgraph as pg
from functools import partial

def worm_feature_gui(strain = 'n2', wid = 0, times = all, tstart = None, xy = None, roi = None,
                     size = None, cmap = 'rainbow', linecolor = 'gray', border = 75, levels = None, background = 90, sigma = 1.0,
                     features = ['speed', 'rotation', 'roam'], feature_filters = None, history = 10, history_delta = 1, 
                     feature_indicator = None, stage_times = None):

  # load feature data
  feature_label, feature_data = generate_feature_data(strain, wid, features, feature_filters);
  feature_indicator = generate_feature_indicator(strain, wid, feature_indicator = feature_indicator);
  
  if xy is None:
    xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
  
  if stage_times is None:
    stage_times = exp.load_stage_times(strain = strain, wid = wid)[:-1];  
  
  if times is all:
    times = (0, len(xy));
  t0 = times[0]; t1 = times[1];
  
  if tstart is None:
    tstart = stage_times[0];
  tstart = min(max(tstart, t0), t1);

  # memmap to images
  worm = exp.load_img(strain = strain, wid = wid);
  
  # generate image data
  wormt, xyt, roi, xylim, extent, plot_range = generate_image_data(strain = strain, wid = wid, t = tstart, size = size, 
                                                                   xy = xy[tstart], worm = worm[tstart], roi = roi, 
                                                                   background = background, sigma = sigma, border = border);
  
  img_size = wormt.shape;
  img_size2 = (img_size[0]/2, img_size[1]/2);
  
  
  # create the gui
  pg.mkQApp()  
  
  widget = pg.QtGui.QWidget();
  widget.setWindowTitle('Feature analysis: Strain: %s, Worm %d' % (strain, wid));
  widget.resize(1000,800)  
  
  layout = pg.QtGui.QVBoxLayout();
  layout.setContentsMargins(0,0,0,0)        
   
  splitter0 = pg.QtGui.QSplitter();
  splitter0.setOrientation(pg.QtCore.Qt.Vertical)
  splitter0.setSizes([int(widget.height()*0.99), int(widget.height()*0.01)]);
  layout.addWidget(splitter0);
   
   
  splitter = pg.QtGui.QSplitter()
  splitter.setOrientation(pg.QtCore.Qt.Horizontal)
  splitter.setSizes([int(widget.width()*0.5), int(widget.width()*0.5)]);
  splitter0.addWidget(splitter);
  
  
  #  Image plot
  img = pg.ImageView();
  img.view.setXRange(0, plot_range[0]);
  img.view.setYRange(0, plot_range[1]);
  
  # xy history
  x = np.zeros(history);
  fade = np.array(np.linspace(5, 255, history), dtype = int)[::-1];
  #brushes = np.array([pg.QtGui.QBrush(pg.QtGui.QColor(255, i, i)) for i in fade]);
  brushes = np.array([pg.QtGui.QBrush(pg.QtGui.QColor(255, 0, 0)) for i in fade]);
  pxy = pg.ScatterPlotItem(x, x, size = 2, pen=pg.mkPen(None), brush = brushes)  # brush=pg.mkBrush(255, 255, 255, 120))
  img.addItem(pxy)
  
  # circle
  k = 100;
  center = np.array(plot_range)/2.0;
  x = roi[2] * np.cos(np.linspace(0, 2*np.pi, k)) + center[0];  
  y = roi[2] * np.sin(np.linspace(0, 2*np.pi, k)) + center[1];
  circle = pg.PlotCurveItem(x,y, pen = pg.mkPen(pg.QtGui.QColor(0,0,0)));
  img.addItem(circle);
  
  # feature indicator
  if feature_indicator is not None:
    posfac = 0.9; sizefac = 0.03;
    indicator = pg.QtGui.QGraphicsRectItem(int(posfac * plot_range[0]), int(posfac * plot_range[1]), int(sizefac * plot_range[0])+1, int(sizefac * plot_range[1])+1);
    indicator.setPen(pg.mkPen(None))
    img.addItem(indicator);
  
  # Contrast/color control
  #hist = pg.HistogramLUTItem()
  #hist.setImageItem(img)
  #win.addItem(hist)
  #splitter.addWidget(gl1);
  splitter.addWidget(img);
  
  # Feture data plots
  gl2 = pg.GraphicsLayoutWidget(border=(50,50,50))
  pf = [];
  for f in feature_label:
    pf.append(gl2.addPlot(title = f));
    gl2.nextRow();
    
  splitter.addWidget(gl2);
  
  
  # counter and Scroll bar
  widget_tools = pg.QtGui.QWidget();
  layout_tools = pg.QtGui.QGridLayout()

  spin = pg.SpinBox(value=tstart, int = True, bounds=[t0,t1], decimals = 10);
  spin.setMaximumWidth(200);
  layout_tools.addWidget(spin,0,0);
  
  sb = pg.QtGui.QScrollBar(pg.QtCore.Qt.Horizontal);
  sb.setMinimum(t0); sb.setMaximum(t1);
  sb.setValue(tstart);
  layout_tools.addWidget(sb,0,1);
  
  # add stage times
  iitem = 1;
  stb = [];
  for st in stage_times:
    b = pg.QtGui.QPushButton('L%d - %d' % (iitem, st));
    b.setMaximumWidth(100);
    iitem += 1;
    layout_tools.addWidget(b,0,iitem);
    stb.append(b);
  
  cb = pg.QtGui.QCheckBox('>'); 
  cb.setCheckState(False);
  cb.setMaximumWidth(50);
  layout_tools.addWidget(cb,0,iitem+1);
  
  spin2 = pg.SpinBox(value=1, int = True, bounds=[1,1000], decimals = 3, step = 1);
  spin2.setMaximumWidth(100);
  layout_tools.addWidget(spin2,0,iitem+2);  
  
  spin3 = pg.SpinBox(value=1, int = True, bounds=[1,10000], decimals = 3, step = 1);
  spin3.setMaximumWidth(100);
  layout_tools.addWidget(spin3,0,iitem+3);  
  
  widget_tools.setLayout(layout_tools);
  splitter0.addWidget(widget_tools);
  
  widget.setLayout(layout)
  widget.show();
  
  # Callbacks for handling user interaction
  def updatePlot():
    #global strain, wid, img_data, xy, roi, border, sb, feature_data, pf, img, history, history_delta
    t0 = sb.value();    
    spin.setValue(t0);
    
    # Generate image data
    #wimg = wormimage(strain = strain, wid = wid, t = t0, xy = xy[t0], roi = roi, border = border, worm = img_data[t0]); 
    #img.setImage(wimg, levels = levels);
    wimg = worm[t0].copy();
    if background is not None:
      wimg[wimg > background] = background;
    
    x0 = max(xy[t0,0] - xylim[0][0] - img_size2[0], 0);
    y0 = max(xy[t0,1] - xylim[1][0] - img_size2[1], 0);
    img.setImage(wimg.T, levels = levels, pos = round_xy([x0,y0]), autoRange = False)

    #history
    hsteps = int(t0 / history_delta) + 1; # number of available steps in the past
    if hsteps - history > 0:
      ts = t0 - history_delta * (history - 1);
    else:
      ts = t0 - history_delta * (hsteps - 1);
    te = t0 + 1;
    
    # update xy positions
    x = xy[ts:te:history_delta,0] - roi[0] + center[0];
    y = xy[ts:te:history_delta,1] - roi[1] + center[1];
    pxy.setData(x,y);
    pxy.setBrush(brushes[-len(x):]);

    # feature traces
    for fd, pl in zip(feature_data, pf):
      pl.plot(fd[ts:te:history_delta],clear=True);
    
    if feature_indicator is not None:
      if feature_indicator[t0]:
        indicator.setBrush(pg.mkBrush('r'))
      else:
        indicator.setBrush(pg.mkBrush('w'))
      
  def updateScaleBar():
    t0 = int(spin.val);
    sb.setValue(t0);
    updatePlot();

  def animate():
    ta = int(spin.val);
    ta += int(spin3.val);
    if ta > t1:
      ta = t0;
    sb.setValue(ta);
    spin.setValue(ta);
    updatePlot();
   
  timer = pg.QtCore.QTimer();
  timer.timeout.connect(animate);
  
  def toggleTimer():
    if cb.checkState():
      timer.start(int(spin2.val));
    else:
      timer.stop();

  def updateTimer():
    timer.setInterval(int(spin2.val));
  
  def updateStage(i):
    spin.setValue(stage_times[i]);
    updatePlot();
  
  
  sb.valueChanged.connect(updatePlot);
  spin.sigValueChanged.connect(updateScaleBar);
  
  cb.stateChanged.connect(toggleTimer);
  spin2.sigValueChanged.connect(updateTimer)
  
  for i,s in enumerate(stb):
    s.clicked.connect(partial(updateStage, i));
  
  updatePlot();
  
  return widget;



## Start Qt event loop unless running in interactive mode or using pyside.

if __name__ == '__main__':
  pass
 