# -*- coding: utf-8 -*-
"""
CElegans Behaviour


Video Generator
"""

import os
import numpy as np;

import matplotlib.pyplot as plt

#dir_behaviour = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Code';
#os.chdir(dir_behaviour)

import analysis.experiment as exp;
#import scripts.preprocessing.file_order as fo;

#import analysis.plot as fplt

# remove backgroung noise

def wormimage(strain = 'n2', wid = 0, t = 0, size = None, xy = None, worm = None, roi = None, background = 90, sigma = 1.0, border = 75):
  """Plot the worm in full position frame"""
  
  # load position
  if xy is None:
    xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
    xy = xy[t,:];
  xy = np.array(np.round(xy), dtype = int);
  
  
  # load worm image
  if worm is None:
    worm = exp.load_img(strain = 'n2', wid = wid, t = t, sigma = sigma);
  else:
    worm = worm.copy();
  worm_size = worm.shape;
  if background is not None:
    worm[worm > background] = background; 
  #print worm.shape
    
  if roi is None:
    roi = exp.load_roi(strain = strain,  wid = wid);
  xmm = [int(np.floor(roi[0]-roi[2]-border)), int(np.ceil(roi[0]+roi[2]+border))];
  ymm = [int(np.floor(roi[1]-roi[2]-border)), int(np.ceil(roi[1]+roi[2]+border))];
  #print xmm,ymm
  
  if size is None:
    size = (xmm[1]-xmm[0], ymm[1]-ymm[0]);
    #size = [int(np.ceil(s)) for s in size];
    
  # create image
  img = np.ones(size, dtype = int) * background;
  
  # place worm image
  #print xy, xmm, worm_size
  x0 = max(xy[0] - xmm[0] - worm_size[0]/2, 0);
  x1 = min(xy[0] - xmm[0] + worm_size[0]/2 + 1, size[0]);
  y0 = max(xy[1] - ymm[0] - worm_size[1]/2, 0);
  y1 = min(xy[1] - ymm[0] + worm_size[1]/2 + 1, size[1]);
  xr = x1 - x0;
  yr = y1 - y0;  
  #img[x0:x1, y0:y1] = worm[:xr, :yr];
  img[y0:y1, x0:x1] = worm[:yr,:xr];
  
  return img;


def compose_frames(strain = 'n2', wid = 0, times = range(0,10), size = (1500, 1500), xy = None, 
                   features = ['speed', 'rotation', 'roam'], history = 10, history_delta = 1,
                   cmap = plt.cm.rainbow, linecolor = 'gray', border = 75, save = None):
  plt.clf();
  fig = plt.gcf();
  
  if xy is all:
    xy  = exp.load(strain = strain, wid = wid, dtype = 'xy');
    
  feature_data = [];
  feature_label = [];
  for f in features:
    if isinstance(f, str):
      feature_label.append(f);
      feature_data.append(exp.load(strain = strain, wid = wid, dtype = f));
    elif isinstance(f, tuple):
      feature_label.append(f[0]);
      feature_data.append(f[1]);
    else:
      feature_label.append('feature');
      feature_data.append(f);
  
  roi = exp.load_roi(strain = strain, wid = wid);
  xmm = [roi[0]-roi[2]-border, roi[0]+roi[2]+border];
  ymm = [roi[1]-roi[2]-border, roi[1]+roi[2]+border];  
  
  nplt = len(feature_data);  
  
  for t in times:
    plt.clf();
    
    # plot image + trajectory
    ax = plt.subplot(1,2,1);
    img = wormimage(strain = strain, wid = wid, t= t, xy = xy[t], roi = roi, border = border);
    #print img.dtype, img.max(), img.min(), img.shape
    sh = img.shape;
    plt.imshow(img, cmap = plt.cm.gray);

    ax.add_artist(plt.Circle((sh[0]/2,sh[1]/2),sh[0]/2-border, color = 'black',fill = False, linewidth = 1))

    t0 = max(0, t - history * history_delta); 
    nt = len(range(t0, t+1, history_delta));
    
    if xy is not None:
      #plt.scatter(xy[t0:t+1,1] - ymm[0], xy[t0:t+1,0] - xmm[0], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
      plt.scatter(xy[t0:t+1:history_delta,0] - xmm[0], xy[t0:t+1:history_delta,1] - ymm[0], c = np.arange(nt), cmap = cmap, edgecolor = 'face')  
  
    iplt = 0;
    for fl,fd in zip(feature_label, feature_data):
      iplt+=1
      plt.subplot(nplt,2, iplt * 2);
      plt.plot(fd[t0:t+1:history_delta], c = linecolor);
      plt.scatter(np.arange(nt), fd[t0:t+1:history_delta], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
      plt.title(fl);
      
    plt.show();
    plt.pause(0.01);
    
    if save is not None:
      fig.savefig(save % t);
      


import pyqtgraph as pg

def image_feature_gui(strain = 'n2', wid = 0, times = all, tstart = 200000, xy = None, 
                      features = ['speed', 'rotation', 'roam'], history = 10, history_delta = 1, 
                      cmap = 'rainbow', linecolor = 'gray', border = 75):

  # load feature data
  feature_data = [];
  feature_label = [];
  for f in features:
    if isinstance(f, str):
      feature_label.append(f);
      feature_data.append(exp.load(strain = strain, wid = wid, dtype = f));
    elif isinstance(f, tuple):
      feature_label.append(f[0]);
      feature_data.append(f[1]);
    else:
      feature_label.append('feature');
      feature_data.append(f);
  #nfeat = len(feature_data);
  
  if xy is None:
    xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
  
  if times is all:
    times = (0, len(xy));
  t0 = times[0];
  t1 = times[1];
  
  if tstart is None:
    tstart = t0;
  tstart = max(t0, tstart);
  tstart = min(t1, tstart);

  # memmap to images
  fn = exp.filename(strain = strain, wid = wid, dtype = 'img');
  img_data = np.load(fn, mmap_mode = 'r');
  
  #roi
  roi = exp.load_roi(strain = strain, wid = wid);
  xmm = [roi[0]-roi[2]-border, roi[0]+roi[2]+border];
  ymm = [roi[1]-roi[2]-border, roi[1]+roi[2]+border];  
  img_size = (xmm[1] - xmm[0], ymm[1] - ymm[0]);
  img_size2 = (img_size[0]/2, img_size[1]/2);
  print img_size, img_size2;
  
  # create the gui
  pg.mkQApp()  
  
  widget = pg.QtGui.QWidget();
  widget.setWindowTitle('Feature analysis: Strain: %s, Worm %d' % (strain, wid));
  widget.resize(1000,800)  
  
  layout = pg.QtGui.QVBoxLayout();
  layout.setContentsMargins(0,0,0,0)        
   
  splitter0 =  pg.QtGui.QSplitter();
  splitter0.setOrientation(pg.QtCore.Qt.Vertical)
  splitter0.setSizes([int(widget.height()*0.99), int(widget.height()*0.01)]);
  layout.addWidget(splitter0);
   
   
  splitter = pg.QtGui.QSplitter()
  splitter.setOrientation(pg.QtCore.Qt.Horizontal)
  splitter.setSizes([int(widget.width()*0.5), int(widget.width()*0.5)]);
  splitter0.addWidget(splitter);
  
  
  #  Image plot
  gl1 = pg.GraphicsLayoutWidget(border=(50,50,50))

  pimg = gl1.addPlot();
  img = pg.ImageItem()
  #img.translate(-, -ymm[0])
  pimg.addItem(img)
  
  # xy history
  x = np.zeros(history);
  fade = np.array(np.linspace(5, 255, history+1), dtype = int)[::-1];
  brushes = np.array([pg.QtGui.QBrush(pg.QtGui.QColor(255, i, i)) for i in fade]);
  pxy = pg.ScatterPlotItem(x, x, size = 5, pen=pg.mkPen(None), brush = brushes[:len(x)])  # brush=pg.mkBrush(255, 255, 255, 120))
  pimg.addItem(pxy)
  
  # circle
  k = 100;
  x = roi[2] * np.cos(np.linspace(0, 2*np.pi, k)) + img_size2[0];  
  y = roi[2] * np.sin(np.linspace(0, 2*np.pi, k)) + img_size2[1];
  circle = pg.PlotCurveItem(x,y, pen = pg.mkPen(pg.QtGui.QColor(0,0,0)));
  pimg.addItem(circle);
  
  # Contrast/color control
  #hist = pg.HistogramLUTItem()
  #hist.setImageItem(img)
  #win.addItem(hist)
  splitter.addWidget(gl1);
  
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
  
  cb = pg.QtGui.QCheckBox('>'); 
  cb.setCheckState(False);
  cb.setMaximumWidth(50);
  layout_tools.addWidget(cb,0,2);
  
  spin2 = pg.SpinBox(value=1, int = True, bounds=[1,1000], decimals = 3, step = 1);
  spin2.setMaximumWidth(100);
  layout_tools.addWidget(spin2,0,3);  
  
  spin3 = pg.SpinBox(value=1, int = True, bounds=[1,10000], decimals = 3, step = 1);
  spin3.setMaximumWidth(100);
  layout_tools.addWidget(spin3,0,4);  
  
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
    wimg = wormimage(strain = strain, wid = wid, t = t0, xy = xy[t0], roi = roi, border = border, worm = img_data[t0]); 
    img.setImage(wimg);

    #history    
    ts = max(0, t0 - history * history_delta); 
    te = t0+1;   
    
    # update xy positions
    x = xy[ts:te:history_delta,0] - roi[0] + img_size2[0];
    y = xy[ts:te:history_delta,1] - roi[1] + img_size2[1];
    pxy.setData(y,x);
    pxy.setBrush(brushes[:len(x)]);

    # feature traces
    for fd, pl in zip(feature_data, pf):
      pl.plot(fd[ts:te:history_delta],clear=True);
      
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
  
  
  sb.valueChanged.connect(updatePlot);
  spin.sigValueChanged.connect(updateScaleBar);
  
  cb.stateChanged.connect(toggleTimer);
  spin2.sigValueChanged.connect(updateTimer)
  
  updatePlot();
  
  return widget;



## Start Qt event loop unless running in interactive mode or using pyside.

if __name__ == '__main__':
  
  import numpy as np
  import os
  import analysis.experiment as exp
  import analysis.plot as fplt
  import matplotlib.pyplot as plt  
  
  import analysis.video as vd
  reload(vd)
  
  w =vd.image_feature_gui(strain = 'n2', wid = 0, history = 100)

  strain = 'n2'; wid = 0;
  
  trans = np.load(os.path.join(exp.data_directory, 'transitions_times.npy'))
  trans = trans[wid];
  t0 = trans[0];

  reload(vd)

  
  speed = exp.load(strain = strain, wid = wid, dtype = 'speed')
  plt.figure(6); plt.clf();
  plt.plot(speed)
  plt.ylim(0,20)
  
  t0= 350600
  vid = [vd.wormimage(t = t0+ti) for ti in range(0,1000,5)]
  vid = np.array(vid)

  fplt.pg.image(vid)  
  plt.figure(7); plt.clf();
  plt.imshow(vid[0], cmap = plt.cm.gray)
  
  t0= 350600
  img = vd.wormimage(t = t0, strain = strain, wid = wid, border = 75)
  plt.figure(7); plt.clf(); 
  plt.subplot(1,2,1)
  plt.imshow(img, cmap = plt.cm.gray)
  sh = img.shape;
  plt.gca().add_artist(plt.Circle((sh[0]/2,sh[1]/2),sh[0]/2-75, color = 'black',fill = False, linewidth = 1))

  reload(vd)
  plt.figure(10); plt.clf();
  vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+500, 5),
                    history= 50, history_delta=5)
  
  
  file_format = 'movie_%d.png';
  plt.figure(11);
  t0 = 350600;
  vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+5000, 1),
                    history= 150, history_delta=1, save = file_format);



    
  os.system("mencoder 'mf://movie_*.png' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.mpg")
  
  import glob
  files = np.sort(np.unique(glob.glob('movie_*.png')))
  for f in files:
    os.remove(f)


  #os.system("convert _tmp*.png animation.mng")

  # cleanup
  #for fname in files:
  #  os.remove(fname)



#  import matplotlib
#  matplotlib.use("Agg")
#  import matplotlib.animation as manimation
#
#  FFMpegWriter = manimation.writers['ffmpeg']
#  metadata = dict(title='Worm %d t0 = %d' % (wid, t0), artist='Christoph Kirst', comment='CElegans Behaviour Project')
#  writer = FFMpegWriter(fps=15, metadata=metadata)
#
#  fig = plt.figure(11);
#  writer.saving(fig, "writer_test.mp4", 100);
#  vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+500, 5),
#                    history= 50, history_delta=5, writer= writer)