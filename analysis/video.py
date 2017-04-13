# -*- coding: utf-8 -*-
"""
CElegans Behaviour


Video Generator
"""


import numpy as np;

import matplotlib.pyplot as plt;
import matplotlib.patches as patches;

import analysis.experiment as exp;

import matplotlib.animation as manimation



###############################################################################
### Default Settings
###############################################################################

def time_str(sec, sample_rate = 3.0):
  sec /= sample_rate;
  m, s = divmod(sec, 60);
  h, m = divmod(m,   60);
  return '%s:%02d:%02d' % (str(int(h)).rjust(3), m, s)

def stage_str(stage):
  if stage <= 4:
    return 'L%d' % stage;
  else:
    return 'A';

def layout_plots(n, prefer_rows = True):
  r = int(np.sqrt(n));
  c = int(np.ceil(n * 1.0 / r));
  if prefer_rows:
    return r,c
  else:
    return c,r

def round_xy(xy):
  """Pixel position of image from floating point numbers"""
  return np.array(np.round(np.array(xy, dtype = float)), dtype = int);
  #return np.array(np.floor(xy), dtype = int);
  #return np.array(np.ceil(xy), dtype = int);

default_stage_colors = ['#ed1c24', '#231f20', '#2e3192', '#27963c', '#918f8f', '#918f8f'];

def default_time_stamp_position(xlim, ylim):
  return [xlim[0] + 10, ylim[0] + 50];

def default_stage_stamp_position(xlim, ylim):
  return [xlim[1] - 10, ylim[0] + 50];

def default_stage_indicator_poistion(time, time_max, time_min, xlim, ylim, time_end = None):
  sizefac= 0.9;
  plot_range = xlim[1] - xlim[0];
  off = xlim[0] + (1 - sizefac) * plot_range / 2.0;
  x = sizefac * (time - time_min) / (time_max - time_min) * plot_range + off;
  y = 20 + ylim[0];
  if time_end is None:
    return [x,y];
  else:
    h = 10;
    w = sizefac * (time_end - time) / (time_max - time_min) * plot_range;
    return x,y,w,h;

def default_legend_position(index, xlim, ylim):
  return [10, ylim[1]-100 + index * 50];

###############################################################################
### Worm trajecotry / plate plotting
###############################################################################


class WormPlot(object):
  """Class to plot behaviour of a single worm"""
  
  def __init__(self, axis, strain = 'n2', wid = 0, 
               time = 0, xy = None,
               
               image = True, 
               image_source = None, image_data = None,
               image_vmin = 84, image_vmax = 92, image_sigma = 1.0, image_border = 75, image_cmap = plt.cm.gray,
               
               plate = True, 
               plate_color = 'k', plate_fill = False, plate_linewidth = 1,
               
               trajectory = True, 
               trajectory_length = 10, trajectory_delta = 1, trajectory_data = None,
               trajectory_size = 5, trajectory_vmin = None, trajectory_vmax = None, trajectory_cmap = plt.cm.Reds,
               
               time_stamp = True, 
               time_stamp_offset = 0, time_stamp_font_size = 12, time_stamp_color = 'k', time_stamp_position = default_time_stamp_position,
               time_stamp_text = time_str,
          
               stage_stamp = True,
               stage_stamp_text = stage_str, stage_stamp_colors = default_stage_colors,
               stage_stamp_font_size = 12,  stage_stamp_position = default_stage_stamp_position,
               
               stage_indicator = True, 
               stage_indicator_colors = default_stage_colors, stage_indicator_size = 7.5,
               stage_indicator_color  = 'black',
               stage_indicator_position = default_stage_indicator_poistion, stage_indicator_label = stage_str, stage_indicator_label_offset = 50, stage_indicator_label_font_size = 12,
               
               legend = False, 
               legend_text = None,
               legend_position = default_legend_position, legend_font_size = 1,
               

               
               border = [[0,0],[10,0]], title = None, 
               ):
    
    ### geometry initialization     
    
    self.axis = axis;
    self.strain = strain;
    self.wid = wid;
    self.time = time;
    if xy is None:
      self.xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
    else:
      self.xy = xy.copy();
    xy = round_xy(self.xy[time]);
    
    
    # generate roi / plot ranges
    plate_data = exp.load_roi(strain = strain, wid = wid);
    self.center = plate_data[:2];
    self.radius = plate_data[2];
    
    self.xlim = np.array([np.floor(self.center[0] - self.radius - border[0][0]), np.ceil(self.center[0] + self.radius + border[0][1])], dtype = int);
    self.ylim = np.array([np.floor(self.center[1] - self.radius - border[1][0]), np.ceil(self.center[1] + self.radius + border[1][1])], dtype = int);
    self.plot_range = np.array([self.xlim[1]- self.xlim[0], self.ylim[1] - self.ylim[0]]);    
    
    if image:
      self.image_sigma = image_sigma;
      
      if image_source is None:
        self.image_source = exp.load_img(strain = strain , wid = wid);
      else:
        self.image_source = image_source;
       
      if image_data is None:
        image_data = exp.load_img(strain = strain, wid = wid, t = time, sigma = image_sigma);
      else:
        image_data = image.copy();
      self.image_size = image_data.shape;
      self.image_vmin = image_vmin;
    
    if trajectory:
      self.trajectory_length = trajectory_length;
      self.trajectory_delta  = trajectory_delta;
      self.trajectory_data   = trajectory_data;
      if isinstance(trajectory_data, basestring):
        self.trajectory_data = exp.load(strain = strain, wid = wid, dtype = trajectory_data);      
      self.trajectory_empty  = np.full(trajectory_length, np.nan);
    
    
    ### graphics initgializaton 
    
    # prepare axis
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_axis_off();
    axis.set_xlim(self.xlim[0], self.xlim[1]);
    axis.set_ylim(self.ylim[0], self.ylim[1]);
    axis.set_aspect('equal');
    if title is not None:
      axis.set_title(title);
    
    #legend
    if legend:
      self.legend = [];
      for li,l in enumerate(legend_text):
        font = {'size': legend_font_size }
        lpos = legend_position(li, self.xlim, self.ylim);
        self.legend.append(plt.text(lpos[0], lpos[1], l[1], color = l[0], fontdict = font));
    else:
      self.legend = None;
  
    # draw plate outline
    if plate:
      self.plate = plt.Circle(self.center, self.radius, color = plate_color, fill = plate_fill, linewidth = plate_linewidth);
      axis.add_artist(self.plate);
    else:
      self.plate = None;

    # place worm image
    if image:
      self.image = axis.imshow(image_data, extent = self.generate_image_extent(xy), vmin = image_vmin, vmax = image_vmax, cmap = image_cmap);
    else:
      self.image = None;
   
    #draw trajectory
    if trajectory:
      x, y, c = self.generate_trajectory(time);
      if c is None:
        c = np.linspace(0,1,self.trajectory_length);
        trajectory_vmin = 0;
        trajectory_vmax = 1;  
    
      self.trajectory = axis.scatter(x, y, c = c, cmap = trajectory_cmap, edgecolor = 'none', vmin = trajectory_vmin, vmax = trajectory_vmax, s = trajectory_size);
    else:
      self.trajectory = None;
    
    # time stamp text  
    if time_stamp:
      self.time_stamp_offset = time_stamp_offset;
      self.time_stamp_text = time_stamp_text;
      font = {'size': time_stamp_font_size }
      tt = time_stamp_text(time - time_stamp_offset);
      tpos = time_stamp_position(self.xlim, self.ylim);
      self.time_stamp = axis.text(tpos[0], tpos[1], tt, fontdict = font);
    else:
      self.time_stamp = None;
    
    # stage stamp text
    if stage_stamp:
      self.stage_stamp_colors = stage_stamp_colors;
      self.stage_times = exp.load_stage_times(strain = strain, wid = wid);
      self.stage_stamp_text = stage_stamp_text;
      font = {'size': stage_stamp_font_size }
      stage = np.sum(time >= self.stage_times);
      tt = stage_stamp_text(stage);
      tpos = stage_stamp_position(self.xlim, self.ylim)
      self.stage_stamp = axis.text(tpos[0], tpos[1], tt, fontdict = font, horizontalalignment='right', color = stage_stamp_colors[stage-1]);
    else:
      self.stage_stamp= None;
    
    # stage indicator
    if stage_indicator:
      self.stage_times = exp.load_stage_times(strain = strain, wid = wid);
      self.stage_time_min = self.stage_times[0]
      self.stage_time_max = self.stage_times[-2];
      self.stage_indicator_colors = stage_indicator_colors;
      self.stage_indicator_position = stage_indicator_position;
    
      self.stage_indicator_rect = [];
      self.stage_indicator_label = [];
      for s in range(5):
        x,y,w,h = stage_indicator_position(self.stage_times[s], self.stage_time_max, self.stage_time_min, self.xlim, self.ylim, time_end = self.stage_times[s+1]);
        self.stage_indicator_rect.append(patches.Rectangle((x,y), w, h, edgecolor = None, facecolor = stage_indicator_colors[s]));
        axis.add_patch(self.stage_indicator_rect[s]);
        if stage_indicator_label is not None:
          font = {'size': stage_indicator_label_font_size }
          self.stage_indicator_label.append(axis.text(x + w / 2.0, y - stage_indicator_label_offset, stage_indicator_label(s+1), 
                                                     fontdict = font, horizontalalignment='center', color = stage_indicator_colors[s]));
      
      x, y = stage_indicator_position(time, self.stage_time_max, self.stage_time_min, self.xlim, self.ylim);
      self.stage_indicator = plt.Circle((x, y + h/2.0), stage_indicator_size, color = stage_indicator_color, fill = True, linewidth = 0.5);
      axis.add_artist(self.stage_indicator);
    else:
      self.stage_indicator = None;
      
  
  def generate_trajectory(self, time):
    steps = int(time / self.trajectory_delta) + 1; # number of available steps in the past
    if steps - self.trajectory_length > 0:
      t0 = time - self.trajectory_delta * (self.trajectory_length - 1);
      t1 = 0;
    else:
      t0 = time - self.trajectory_delta * (steps - 1);
      t1 = -steps;
    
    x = self.trajectory_empty.copy();
    y = self.trajectory_empty.copy();
    x[t1:] = self.xy[t0:time+1:self.trajectory_delta,0];
    y[t1:] = self.xy[t0:time+1:self.trajectory_delta,1];
    
    if self.trajectory_data is not None:
      c = self.trajectory_empty.copy();
      c[t1:] = self.trajectory_data[t0:time+1:self.trajectory_delta];
    else:
      c = None;
    return x,y,c;
    
  def generate_image_extent(self, xy):
    return np.array([xy[0] - self.image_size[0]/2, xy[0] + self.image_size[0]/2 + 1,
                     xy[1] - self.image_size[1]/2, xy[1] + self.image_size[1]/2 + 1]);
  
  def update(self, time):
    self.time = time;
    xy = round_xy(self.xy[time]);
    
    # update worm image and position
    if self.image is not None:
      image_data = exp.smooth_image(self.image_source[time], sigma = self.image_sigma);
      image_data = image_data[::-1, :];
      if image_data[0,0] > 10e-10 and image_data[0,0] != np.nan: # exclude invalid images
        self.image.set_data(image_data);
        self.image.set_extent(self.generate_image_extent(xy))
    
    # update scatter plot data
    if self.trajectory is not None:
      x,y,c = self.generate_trajectory(time);
      self.trajectory.set_offsets(np.vstack([x,y]).T);
      if c is not None:
        self.trajectory.set_array(c);
      else:
        self.trajectory.set_array(np.linspace(0,1,self.trajectory_length));
    
    if self.time_stamp is not None:
      self.time_stamp.set_text(self.time_stamp_text(time-self.time_stamp_offset));
    
    if self.stage_stamp is not None:    
      stage = np.sum(time >= self.stage_times);
      self.stage_stamp.set_text(self.stage_stamp_text(stage));
      self.stage_stamp.set_color(self.stage_stamp_colors[stage-1]);
      
    if self.stage_indicator is not None:
      x,y = self.stage_indicator_position(time, self.stage_time_max, self.stage_time_min, self.xlim, self.ylim);
      c = self.stage_indicator.center; 
      self.stage_indicator.center = (x, c[1]);
      self.stage_indicator.set_fill(True);
    

def time_stamp_text(time, index):
  return time_str(time);
    
class WormAnimation(object):
  def __init__(self, plots, times,
                     time_stamp = True, 
                     time_stamp_text = time_stamp_text, time_stamp_font_size = 1, time_stamp_color = 'k', time_stamp_position = None,
                     figure = None, 
                     save = None, dpi = 300, fps = 30, 
                     pause = None, verbose = True):
    self.plots = plots;
    self.times = times;
    if figure is None:
      self.figure = plt.gcf();
    else:
      self.figure = figure;
    self.pause = pause;
    
    # time stamp text  
    if time_stamp:
      self.time_stamp_text = time_stamp_text;
      tt = time_stamp_text(0,0);
      if time_stamp_position is None:
        self.time_stamp = figure.suptitle(tt, fontsize = time_stamp_font_size, color = time_stamp_color);
      else:
        self.time_stamp = plt.annotate(tt, time_stamp_position, xycoords = 'figure fraction',
                                       verticalalignment = 'center', horizontalalignment = 'left',
                                       fontsize = time_stamp_font_size, color = time_stamp_color);
    else:
      self.time_stamp = None;      
    
    # movie saver
    if save is not None:
      FFMpegWriter = manimation.writers['ffmpeg']
      #FFMpegWriter = manimation.writers['mencoder']
      metadata = dict(title='Strain %s Worm %d', artist='Chirstoph Kirst',
                      comment='C Elegans Dataset, Bargmann Lab')
      self.writer = FFMpegWriter(fps=fps, metadata=metadata);
      self.writer.setup(self.figure, save, dpi = dpi);
    else:
      self.writer = None;
    
  
  def animate(self):
    for ti in range(len(self.times[0])):
      for pi,p in enumerate(self.plots):
        p.update(self.times[pi][ti]);
    
      if self.time_stamp is not None:
        self.time_stamp.set_text(self.time_stamp_text(self.times[0][ti]-self.times[0][0], ti));
      
      self.figure.canvas.draw();
      self.figure.canvas.flush_events()   
      self.figure.show();
    
      if self.pause is not None:
        #plt.show();
        plt.pause(self.pause);
    
      if self.writer is not None:
        self.writer.grab_frame();
        
    if self.writer is not None:
      self.writer.cleanup();   
   
   



###############################################################################
#### Gui
###############################################################################


import pyqtgraph as pg
from functools import partial

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class WormGui(object):
  
  def __init__(self, strain = 'n2', wid = 0, times = all, stage_times = None,
               time = 0, xy = None,
               
               image = True, 
               image_source = None, image_data = None,
               image_vmin = 84, image_vmax = 92, image_sigma = 1.0, image_border = 75, image_cmap = plt.cm.gray,
               image_levels = None, image_background = 90,
               
               plate = True, 
               plate_color = 'k', plate_fill = False, plate_linewidth = 1,
               
               trajectory = True, 
               trajectory_length = 10, trajectory_delta = 1, trajectory_data = None,
               trajectory_size = 5, trajectory_vmin = None, trajectory_vmax = None, trajectory_cmap = plt.cm.Reds,
               
               border = [[0,0],[10,0]],
               title = None):
                 
    self.strain = strain;
    self.wid = wid;                 
                 
    if xy is None:
      self.xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
    else:
      self.xy = xy.copy();
    self.xy = round_xy(self.xy);
    
    if stage_times is None:
      self.stage_times = exp.load_stage_times(strain = strain, wid = wid)[:-1];  
    else:
      self.stage_times = stage_times;
  
    if times is all:
      self.times = (0, len(self.xy));
    else:
      self.times = times;
    self.t0 = self.times[0]; self.t1 = self.times[1];
  
    if time is None:
      self.time = self.stage_times[0];
    else:
      self.time = time;
    self.time = min(max(self.time, self.t0), self.t1);
    
    
    # generate roi / plot ranges
    plate_data = exp.load_roi(strain = strain, wid = wid);
    self.center = plate_data[:2];
    self.radius = plate_data[2];
    
    self.xlim = np.array([np.floor(self.center[0] - self.radius - border[0][0]), np.ceil(self.center[0] + self.radius + border[0][1])], dtype = int);
    self.ylim = np.array([np.floor(self.center[1] - self.radius - border[1][0]), np.ceil(self.center[1] + self.radius + border[1][1])], dtype = int);
    self.plot_range = np.array([self.xlim[1]- self.xlim[0], self.ylim[1] - self.ylim[0]]);    
    
    if image:
      self.image_sigma = image_sigma;
      
      if image_source is None:
        self.image_source = exp.load_img(strain = strain , wid = wid);
      else:
        self.image_source = image_source;
       
      if image_data is None:
        image_data = exp.load_img(strain = strain, wid = wid, t = time, sigma = image_sigma);
      else:
        image_data = image.copy();
      self.image_size = np.array(image_data.shape);
      self.image_size2= np.array(0.5 * self.image_size, dtype = int);
      self.image_vmin = image_vmin;
      self.image_data = None;
      self.image_background = image_background;
      self.image_levels = image_levels;
    
    if trajectory:
      self.trajectory_length = trajectory_length;
      self.trajectory_delta  = trajectory_delta;
      self.trajectory_data   = trajectory_data;
      if isinstance(trajectory_data, basestring):
        self.trajectory_data = exp.load(strain = strain, wid = wid, dtype = trajectory_data);      
      self.trajectory_empty  = np.full(trajectory_length, np.nan);
      
    # create the gui
    pg.mkQApp()  
  
    self.widget = pg.QtGui.QWidget();
    if title is None:
      title = 'Feature analysis: Strain: %s, Worm %d' % (strain, wid);
    self.widget.setWindowTitle(title);
    self.widget.resize(1000,800)  
  
    self.layout = pg.QtGui.QVBoxLayout();
    self.layout.setContentsMargins(0,0,0,0)        
   
    self.splitter0 = pg.QtGui.QSplitter();
    self.splitter0.setOrientation(pg.QtCore.Qt.Vertical)
    self.splitter0.setSizes([int(self.widget.height()*0.99), int(self.widget.height()*0.01)]);
    self.layout.addWidget(self.splitter0);
    
    self.splitter = pg.QtGui.QSplitter()
    self.splitter.setOrientation(pg.QtCore.Qt.Horizontal)
    self.splitter.setSizes([int(self.widget.width()*0.5), int(self.widget.width()*0.5)]);
    self.splitter0.addWidget(self.splitter);
    
    #  Image plot
    self.img = pg.ImageView();
    self.img.view.setXRange(self.xlim[0], self.xlim[1]);
    self.img.view.setYRange(self.ylim[0], self.ylim[1]);
  
    # xy history
    x = np.zeros(self.trajectory_length);
    fade = np.array(np.linspace(5, 255, self.trajectory_length), dtype = int)[::-1];
    self.brushes = np.array([pg.QtGui.QBrush(pg.QtGui.QColor(255, i, i)) for i in fade]);
    #self.brushes = np.array([pg.QtGui.QBrush(pg.QtGui.QColor(255, 0, 0)) for i in fade]);
    self.pxy = pg.ScatterPlotItem(x, x, size = 2, pen=pg.mkPen(None), brush = self.brushes)  # brush=pg.mkBrush(255, 255, 255, 120))
    self.img.addItem(self.pxy)
  
    # circle
    k = 100;
    x = self.radius * np.cos(np.linspace(0, 2*np.pi, k)) + self.center[0];  
    y = self.radius * np.sin(np.linspace(0, 2*np.pi, k)) + self.center[1];
    self.plate = pg.PlotCurveItem(x,y, pen = pg.mkPen(pg.QtGui.QColor(0,0,0)));
    self.img.addItem(self.plate);
    
    self.splitter.addWidget(self.img);
  
    # feature indicator
    #if feature_indicator is not None:
    #posfac = 0.9; sizefac = 0.03;
    #indicator = pg.QtGui.QGraphicsRectItem(int(posfac * plot_range[0]), int(posfac * plot_range[1]), int(sizefac * plot_range[0])+1, int(sizefac * plot_range[1])+1);
    #indicator.setPen(pg.mkPen(None))
    #img.addItem(indicator);
  
    # Contrast/color control
    #hist = pg.HistogramLUTItem()
    #hist.setImageItem(img)
    #win.addItem(hist)
    #splitter.addWidget(gl1);

  
    # Feture data plots
    #gl2 = pg.GraphicsLayoutWidget(border=(50,50,50))
    #pf = [];
    #for f in feature_label:
    #  pf.append(gl2.addPlot(title = f));
    #  gl2.nextRow();
    
    #splitter.addWidget(gl2);
  
  
    # counter and Scroll bar
    self.widget_tools = pg.QtGui.QWidget();
    self.layout_tools = pg.QtGui.QGridLayout()

    self.spin = pg.SpinBox(value=self.time, int = True, bounds=[self.t0,self.t1], decimals = 10);
    self.spin.setMaximumWidth(200);
    self.layout_tools.addWidget(self.spin,0,0);
  
    self.sb = pg.QtGui.QScrollBar(pg.QtCore.Qt.Horizontal);
    print self.t0, self.t1
    self.sb.setMinimum(self.t0); self.sb.setMaximum(self.t1);
    self.sb.setValue(self.time);
    self.layout_tools.addWidget(self.sb,0,1);
    
    # add stage times
    iitem = 1;
    self.stb = [];
    for st in self.stage_times:
      b = pg.QtGui.QPushButton('L%d - %d' % (iitem, st));
      b.setMaximumWidth(100);
      iitem += 1;
      self.layout_tools.addWidget(b,0,iitem);
      self.stb.append(b);
  
    self.cb = pg.QtGui.QCheckBox('>'); 
    self.cb.setCheckState(False);
    self.cb.setMaximumWidth(50);
    self.layout_tools.addWidget(self.cb,0,iitem+1);
    
    self.spin2 = pg.SpinBox(value=1, int = True, bounds=[1,1000], decimals = 3, step = 1);
    self.spin2.setMaximumWidth(100);
    self.layout_tools.addWidget(self.spin2,0,iitem+2);  
  
    self.spin3 = pg.SpinBox(value=1, int = True, bounds=[1,10000], decimals = 3, step = 1);
    self.spin3.setMaximumWidth(100);
    self.layout_tools.addWidget(self.spin3,0,iitem+3);  
    
    self.widget_tools.setLayout(self.layout_tools);
    self.splitter0.addWidget(self.widget_tools);
    
    self.widget.setLayout(self.layout)
    self.widget.show();
    
    #self.img.setImage(image_data.T, levels = image_levels, pos = self.xy[self.time], autoRange = False)
    # update xy positions
    #x = xy[ts:te:history_delta,0] - roi[0] + center[0];
    #y = xy[ts:te:history_delta,1] - roi[1] + center[1];
    #self.pxy.setData(x,y);
    #self.pxy.setBrush(brushes[-len(x):]);

    # feature traces
    #for fd, pl in zip(feature_data, pf):
    #  pl.plot(fd[ts:te:history_delta],clear=True);
    #
    #if feature_indicator is not None:
    #  if feature_indicator[t0]:
    #    indicator.setBrush(pg.mkBrush('r'))
    #  else:
    #    indicator.setBrush(pg.mkBrush('w'))

    self.timer = pg.QtCore.QTimer();
    self.timer.timeout.connect(self.animate);
    
    
    self.sb.valueChanged.connect(self.updatePlot);
    self.spin.sigValueChanged.connect(self.updateScaleBar);
  
    self.cb.stateChanged.connect(self.toggleTimer);
    self.spin2.sigValueChanged.connect(self.updateTimer)
  
    for i,s in enumerate(self.stb):
      s.clicked.connect(partial(self.updateStage, i));
  
    self.updatePlot();


     
  def updateScaleBar(self):
    t0 = int(self.spin.val);
    self.sb.setValue(t0);
    self.updatePlot();

  def animate(self):
    ta = int(self.spin.val);
    ta += int(self.spin3.val);
    if ta > self.t1:
      ta = self.t0;
    self.sb.setValue(ta);
    self.spin.setValue(ta);
    self.updatePlot();
  
  def toggleTimer(self):
    if self.cb.checkState():
      self.timer.start(int(self.spin2.val));
    else:
      self.timer.stop();

  def updateTimer(self):
    self.timer.setInterval(int(self.spin2.val));
  
  def updateStage(self,i):
    self.spin.setValue(self.stage_times[i]);
    self.updatePlot();
  
  # Callbacks for handling user interaction
  def updatePlot(self):
    #global strain, wid, img_data, xy, roi, border, sb, feature_data, pf, img, history, history_delta
    self.time = t0 = self.sb.value();
    self.spin.setValue(self.time);
    
    # Generate image data
    #wimg = wormimage(strain = strain, wid = wid, t = t0, xy = xy[t0], roi = roi, border = border, worm = img_data[t0]); 
    #img.setImage(wimg, levels = levels);
    if self.image_source is not None:
      self.image_data = self.image_source[t0].copy();
      if self.image_background is not None:
        self.image_data[self.image_data > self.image_background] = self.image_background;
    
    self.img.setImage(self.image_data.T, levels = self.image_levels, pos = self.xy[t0] - self.image_size2, autoRange = False)

    #history
    hsteps = int(t0 / self.trajectory_delta) + 1; # number of available steps in the past
    if hsteps - self.trajectory_length > 0:
      ts = t0 - self.trajectory_delta * (self.trajectory_length - 1);
    else:
      ts = t0 - self.trajectory_delta * (hsteps - 1);
    te = t0 + 1;
    
    # update xy positions
    x = self.xy[ts:te:self.trajectory_delta,0];
    y = self.xy[ts:te:self.trajectory_delta,1];
    self.pxy.setData(x,y);
    self.pxy.setBrush(self.brushes[-len(x):]);

    # feature traces
    #for fd, pl in zip(feature_data, pf):
    #  pl.plot(fd[ts:te:history_delta],clear=True);
    # 
    #if feature_indicator is not None:
    #  if feature_indicator[t0]:
    #    indicator.setBrush(pg.mkBrush('r'))
    #  else:
    #    indicator.setBrush(pg.mkBrush('w'))  
  



#  
#
#def generate_feature_data(strain = 'n2', wid = 0, features = [], feature_filters = None):
#  """Generate feature data"""
#  if feature_filters is None:
#    feature_filters = [None for f in features];
#  
#  feature_data = [];
#  feature_label = [];
#  for f,ff in zip(features, feature_filters):
#    if isinstance(f, str):
#      feature_label.append(f);
#      dat = exp.load(strain = strain, wid = wid, dtype = f);
#      if ff is not None:
#        dat = ff(dat);
#      feature_data.append(dat);
#    elif isinstance(f, tuple):
#      feature_label.append(f[0]);
#      feature_data.append(f[1]);
#    else:
#      feature_label.append('feature');
#      feature_data.append(f);
#  
#  return (feature_label, feature_data);
#  
#def generate_feature_indicator(strain = 'n2', wid = 0, feature_indicator = None):
#  """Generate data for the feature indicator"""
#  if isinstance(feature_indicator, basestring):
#    feature_indicator = exp.load(strain = strain, wid = wid, dtype = feature_indicator);
#    feature_indicator[np.isnan(feature_indicator)] = 0.0;
#    feature_indicator = np.array(feature_indicator, dtype = bool);
#  return feature_indicator;
#
#
#def animate_frames(strain = 'n2', wid = 0,  
#                size = None, xy = None, roi = None, 
#                background = None, sigma = 1.0, border = 75, vmin = 60, vmax = 90, cmap = plt.cm.gray,
#                times = all,
#                features = ['speed', 'rotation'], feature_filters = None, history = 10, history_delta = 1, linecolor = 'gray', 
#                feature_indicator = 'roam',
#                time_data = None, time_cmap = plt.cm.rainbow, time_size = 30,
#                time_stamp = True, sample_rate = 3, tref = None,
#                stage_stamp = True, stage_indicator = True, stage_colors = None,
#                
#                font_size = 16, add_below = 75,
#                pause = 0.01, legend = None,
#                save = None, fps = 20, dpi = 300,
#                verbose = True):
#  """Animate frames and generate video"""         
#  if times is all:
#    t = 0;
#  else:
#    t = times[0];
#  
#  xynans = np.nan * np.ones(history);
#  
#  # memmap to images
#  worm = exp.load_img(strain = strain, wid = wid);
#  
#  # xy positions
#  if xy is None:
#    xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
#
#  if times is all:    
#    times = (0, len(xy));
#  
#  if isinstance(time_data, basestring):
#    time_data = exp.load(strain = strain, wid = wid, dtype = time_data);
#  
#  # initialize image data
#  wormt, xyt, roi, xylim, extent, plot_range = generate_image_data(strain = strain, wid = wid, t = times[0], size = size, 
#                                                        xy = xy[t], worm = worm[t], roi = roi, 
#                                                        background = background, sigma = sigma, border = border);
#                                                        
#  # initialize feature data                                                      
#  feature_label, feature_data = generate_feature_data(strain, wid, features, feature_filters);
#  nplt = len(feature_data)
#  ncols = 1 + int(nplt > 0);
#  
#  # create plot
#  fig = plt.gcf(); plt.clf();
#  ax = plt.subplot(1,ncols,1);
#  ax.set_xticks([])
#  ax.set_yticks([])
#  ax.set_axis_off();
#  
#  if legend is not None:
#    for li,l in enumerate(legend):
#      font = {'size': font_size }
#      plt.text(10, plot_range[1]-100 + li * 50, l[1], color = l[0], fontdict = font)
#  
#  # draw plate outline
#  center = np.array(np.array(plot_range) /2, dtype = int);
#  r = center[0] - border;
#  ax.add_artist(plt.Circle((center[0], center[1]), r, color = 'black', fill = False, linewidth = 0.5))
#  ax.set_xlim(0, plot_range[0] + 0* 50);
#  ax.set_ylim(-add_below, plot_range[1] + 0);
#  
#  # place worm image
#  wimg = ax.imshow(wormt, extent = extent, vmin = vmin, vmax = vmax, cmap = cmap);
#    
#  # plot history of trace
#  def generate_xy(t):
#    hsteps = int(t / history_delta) + 1; # number of available steps in the past
#    if hsteps - history > 0:
#      t0 = t - history_delta * (history - 1);
#      t1 = 0;
#    else:
#      t0 = t - history_delta * (hsteps - 1);
#      t1 = -hsteps;
#    
#    xdat = xynans.copy();
#    ydat = xynans.copy();
#    xdat[t1:] = xy[t0:t+1:history_delta,0] - xylim[0][0]
#    ydat[t1:] = xy[t0:t+1:history_delta,1] - xylim[1][0]
#    return [xdat, ydat], t0, t1;
#  
#  if history > 0:
#    xydat, t0, t1 = generate_xy(t);
#    if time_data is None:
#      cdat = np.linspace(0,1,history);
#      vmin = 0;
#      vmax = 1;
#    else:
#      cdat = xynans.copy();
#      cdat[t1:] = time_data[t0:t+1:history_delta];
#      vmin = np.nanmin(time_data);
#      vmax = np.nanmax(time_data);
#    
#    print time_cmap.name
#    scatter = plt.scatter(xydat[0], xydat[1], c = cdat, cmap = time_cmap, edgecolor = 'none', vmin = vmin, vmax = vmax, s = time_size)  ;
#    
#    # feature plots
#    iplt = 0;
#    pl = [];
#    for fl,fd in zip(feature_label, feature_data):
#      iplt += 1;
#      plt.subplot(nplt, ncols, iplt * 2);
#      pl.append(plt.plot(fd[t0:t+1:history_delta], c = linecolor));
#      #plt.scatter(np.arange(nt), fd[t0:t+1:history_delta], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
#      plt.title(fl);      
#  
#  else:
#      scatter = None;
#  
#  # plot feature indicator
#  feature_indicator = generate_feature_indicator(strain, wid, feature_indicator);
#  if feature_indicator is not None:
#    if feature_indicator[t]:
#      fc = 'r';
#    else:
#      fc = 'w';
#    posfac = 0.95; sizefac = 0.03;
#    feature_rect = patches.Rectangle((posfac * plot_range[0], posfac * plot_range[1]),  sizefac * plot_range[0], sizefac * plot_range[1], edgecolor = None, facecolor = fc);
#    ax.add_patch(feature_rect);
#  
#  # time stamp text  
#  if time_stamp:
#    font = {'size': font_size }
#    tt = generate_time_str(0);
#    time_stamp = plt.text(10, 10, tt, fontdict = font)
#    if tref is None:
#      tref = times[0];
#
#  if stage_colors is None:
#    stage_colors = ['#ed1c24', '#231f20', '#2e3192', '#27963c', '#918f8f', '#918f8f'];
#    
#  # stage stamp text
#  if stage_stamp:
#    stage_times = exp.load_stage_times(strain = strain, wid = wid);
#    font = {'size': font_size }
#    tt = generate_stage_str(1);
#    stage_stamp = plt.text(plot_range[0] - 10, 10, tt, fontdict = font, horizontalalignment='right')
#  
#  
#  if stage_indicator:
#    stage_times = exp.load_stage_times(strain = strain, wid = wid);
#    stage_time_min = stage_times[0]
#    stage_time_max = stage_times[-2];
#    #print stage_time_min, stage_time_max
#    
#    sizefac = 0.9; stage_off = plot_range[0] * (1 - sizefac) / 2.0;
#    #print plot_range, stage_times;
#    for s in range(5):
#      y = plot_range[1] + 15;
#      y = -20;
#      x = 1.0 * (stage_times[s] - stage_time_min) / (stage_time_max - stage_time_min) * plot_range[0] * sizefac + stage_off;
#      h = 10;
#      w = 1.0 * (stage_times[s+1] - stage_times[s]) / (stage_time_max - stage_time_min) * plot_range[0] * sizefac;
#      #print x,y,w,h
#      ax.add_patch(patches.Rectangle((x,y), w, h, edgecolor = None, facecolor = stage_colors[s]));
#      
#      plt.text(x + w / 2.0, y - 50, generate_stage_str(s+1), fontdict = font, horizontalalignment='center', color = stage_colors[s])
#
#    
#    r = 7.5;
#    x = (t0 - stage_time_min) / (stage_time_max - stage_time_min) * plot_range[1] * sizefac + stage_off;
#    stage_pointer = plt.Circle((x, y + h/2.0), r, color = 'black', fill = False, linewidth = 0.5)
#    ax.add_artist(stage_pointer);
#
#  
#  # movie saver
#  if save is not None:
#    FFMpegWriter = manimation.writers['ffmpeg']
#    #FFMpegWriter = manimation.writers['mencoder']
#    metadata = dict(title='Strain %s Worm %d', artist='Chirstoph Kirst',
#                    comment='C Elegans Dataset, Bargmann Lab')
#    writer = FFMpegWriter(fps=15, metadata=metadata);
#    writer.setup(fig, save, dpi = dpi)
#    
#  
#  
#  # loop over times
#  for t in times:
#    if verbose:
#      print '%d / %d' % (t, times[-1]);
#    xyt = round_xy(xy[t]);
#    
#    # update worm image and position 
#    wormt = exp.smooth_image(worm[t], sigma = sigma);
#    wormt = wormt[::-1, :];
#    wimg.set_data(wormt);
#    extent = generate_image_extent(xyt, xylim, wormt.shape, plot_range);
#    #extent = np.array(extent); extent = extent[[2,3,0,1]];
#    wimg.set_extent(extent)
#    
#    # update scatter plot data
#    if scatter is not None:
#      #plt.scatter(xy[t0:t+1,1] - ymm[0], xy[t0:t+1,0] - xmm[0], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
#      xydat, t0, t1 = generate_xy(t);
#      scatter.set_offsets(np.array(xydat).T);
#      if time_data is not None:
#        cdat = xynans.copy();
#        cdat[t1:] = time_data[t0:t+1:history_delta];
#        #cdat[cdat == np.nan] = (vmax+vmin)/2.0;
#      scatter.set_array(cdat);
#        
#      for p,fd in zip(pl, feature_data):
#        p[0].set_ydata(fd[t0:t+1:history_delta]);
#        
#    if feature_indicator is not None:
#      if feature_indicator[t]:
#        feature_rect.set_color('r');
#      else:
#        feature_rect.set_color('w');
#    
#    if time_stamp:
#      time_stamp.set_text(generate_time_str((t-tref)/sample_rate));
#    
#    if stage_stamp:     
#      stage = np.sum(t >= stage_times);
#      stage_stamp.set_text(generate_stage_str(stage));
#      stage_stamp.set_color(stage_colors[stage-1]);
#      
#    if stage_indicator:
#      x = 1.0 * (t - stage_time_min) / (stage_time_max - stage_time_min) * plot_range[0] * sizefac + stage_off;
#      c = stage_pointer.center; 
#      stage_pointer.center = (x, c[1]);
#      stage_pointer.set_fill(True);
#    
#    
#    fig.canvas.draw();
#    fig.canvas.flush_events()   
#    
#    if pause is not None:
#      #plt.show();
#      plt.pause(pause);
#    
#    if save is not None:
#      writer.grab_frame();
#      #fig.savefig(save % t, dpi = 'figure', pad_inches = 0);
#  
#  if save is not None:
#    writer.cleanup();
#    
#
#
#
#
#
#def animate_worms(strain = 'n2', wid = 0, times = all, worm_image = True,
#                size = None, xy = None, roi = None, 
#                background = None, sigma = 1.0, border = 75, vmin = 60, vmax = 90, cmap = plt.cm.gray,
#                time_data = None, time_cmap = plt.cm.rainbow, time_size = 30, history = 10, history_delta = 1, 
#                time_stamp = True, sample_rate = 3, stamp_off = 0,
#                stage_stamp = True, stage_indicator = True, stage_below = 100, stage_colors = None,
#                font_size = 16, font_size_title = 16, line_width = 0.5, extra_title = '',
#                pause = 0.01, save = None, fps = 20, dpi = 300,
#                verbose = True):
#  """Animate multiple worms and generate video"""
#  
#  # worms to plot
#  wid = np.array([wid]).flatten();  
#  nworms = len(wid);
#  nrows,ncols = arange_plots(nworms)#
#
#  # draw worms       
#  xynans = np.nan * np.ones(history);
#  
#  # memmap to images
#  worms = [exp.load_img(strain = strain, wid = w) for w in wid];
#  #print worms  
#  
#  # xy positions
#  if xy is None:
#    xys = [exp.load(strain = strain, wid = w, dtype = 'xy') for w in wid];
#  else:
#    xys = np.array(xy);
#  #print xys
#
#  if times is all:
#    tm = np.min([len(x) for x in xys]);
#    times = np.array([[0, tm] for w in wid]);
#  elif isinstance(times, np.ndarray) and times.ndim == 1:
#    times = np.array([times for w in wid]);
#  else:
#    times = np.array(times);
#  t0s = times[:,0];
#  #print t0s
#  
#  if isinstance(time_data, basestring):
#    time_data = [exp.load(strain = strain, wid = w, dtype = time_data) for w in wid];
#  elif time_data is None:
#    time_data = [None for w in wid];
#  #print time_data
#  
#  #if time_stamp is True:
#  #  time_stamp = [None for w in wid];
#  #  time_stamp[0] = True;
#  
#  # initialize image data
#  #print len(wid), len(t0s), len(xys), len(worms)
#  worm0s = []; xy0s = []; xylims = []; rois = []; extents = []; plot_ranges = [];
#  i = 0;
#  for w,t,x,ws in zip(wid, t0s, xys, worms):
#    print i; i+= 1;
#    wormt, xyt, roiw, xylim, extent, plot_range = generate_image_data(strain = strain, wid = w, t = t, size = size, 
#                                                                     xy = x[t], worm = ws[t], roi = roi, 
#                                                                     background = background, sigma = sigma, border = border);
#    worm0s.append(wormt); xy0s.append(xyt); xylims.append(xylim); rois.append(roiw); extents.append(extent); plot_ranges.append(plot_range);
#  
#  # helper to create xy data
#  def generate_xy(wi, t):
#    hsteps = int(t / history_delta) + 1; # number of available steps in the past
#    if hsteps - history > 0:
#      t0 = t - history_delta * (history - 1);
#      t1 = 0;
#    else:
#      t0 = t - history_delta * (hsteps - 1);
#      t1 = -hsteps;
#          
#    xdat = xynans.copy();
#    ydat = xynans.copy();
#    xdat[t1:] = xys[wi][t0:t+1:history_delta,0] - xylims[wi][0][0]
#    ydat[t1:] = xys[wi][t0:t+1:history_delta,1] - xylims[wi][1][0]
#    return [xdat, ydat], t0, t1;  
#  
#  # create figure
#  fig = plt.gcf(); plt.clf();
#  axs = [];
#  wimgs = []; scatters = []; stage_times = []; stage_stamps = []; stage_rects = []; stage_pointers = [];
#  for wi, w in enumerate(wid):
#    ax = plt.subplot(nrows,ncols,wi+1);
#    axs.append(ax);
#    ax.set_xticks([])
#    ax.set_yticks([])
#    ax.set_axis_off();
#    ax.set_aspect('equal');
#    #ax.set_title('%d' % w, fontsize = font_size);
#    
#    # draw plate outline
#    center = np.array(np.array(plot_ranges[wi]) /2, dtype = int);
#    r = center[0] - border;
#    ax.add_artist(plt.Circle((center[0], center[1]), r, color = 'black', fill = False, linewidth = line_width))
#    ax.set_xlim(0, plot_ranges[wi][0]);
#    ax.set_ylim(-stage_below, plot_ranges[wi][1]);
#  
#    # place worm image
#    if worm_image:
#      wimgs.append(ax.imshow(worm0s[wi], extent = extents[wi], vmin = vmin, vmax = vmax, cmap = cmap));
#      #print wimgs    
#    
#    if history > 0:
#      xydat, t0, t1 = generate_xy(wi, t);
#      if time_data[wi] is None:
#        cdat = np.linspace(0,1,history);
#        hvmin = 0;
#        hvmax = 1;
#      else:
#        cdat = xynans.copy();
#        cdat[t1:] = time_data[wi][t0:t+1:history_delta];
#        hvmin = np.nanmin(time_data[wi]);
#        hvmax = np.nanmax(time_data[wi]);
#      
#      scatters.append(plt.scatter(xydat[0], xydat[1], c = cdat, cmap = time_cmap, edgecolor = 'none', vmin = hvmin, vmax = hvmax, s = time_size));
#    else:
#      scatters.append(None);
#    #print scatters
#  
#    # plot feature indicator
#    #feature_indicator = generate_feature_indicator(strain, wid, feature_indicator);
#    #if feature_indicator is not None:
#    #if feature_indicator[t]:
#    #  fc = 'r';
#    #else:
#    #  fc = 'w';
#    #posfac = 0.95; sizefac = 0.03;
#    #feature_rect = patches.Rectangle((posfac * plot_range[0], posfac * plot_range[1]),  sizefac * plot_range[0], sizefac * plot_range[1], edgecolor = None, facecolor = fc);
#    #ax.add_patch(feature_rect);
#  
#    # time stamp text
#
#    #if time_stamp[wi] is not None:
#    #  font = {'size': font_size }
#    #  tt = generate_time_str(0);
#    #  time_stamps.append(plt.text(10, 10, tt, fontdict = font));
#    
#    
#    if stage_colors is None:
#      stage_colors = ['#ed1c24', '#231f20', '#2e3192', '#27963c', '#918f8f', '#918f8f'];
#    
#    # stage stamp text
#    if stage_stamp:
#      stage_times.append(exp.load_stage_times(strain = strain, wid = w));
#      font = {'size': font_size}; #, 'weight' : 'bold' }
#      tt = generate_stage_str(1);
#      stage_stamps.append(plt.text(plot_range[0] - 50, 80 - stamp_off, tt, fontdict = font, horizontalalignment='right',
#                                   bbox = {'pad' : 0.3, 'facecolor' : 'none', 'edgecolor' : 'none'}));
#      # stage_colors = plt.cm.Paired(np.linspace(0,1,12))[::2]
#      #stage_colors = plt.cm.nipy_spectral(np.linspace(0,1,12))[[10,8,5,2,1,0]]
#      #stage_colors = plt.cm.nipy_spectral(np.linspace(0,1,20))[[18, 18, 15, 9,3,1,0]];
#      
#      #stage_rects.append(patches.Rectangle((plot_ranges[wi][0]-120, 12), 100, 65, edgecolor = None, facecolor = stage_colors[0]));
#      #ax.add_patch(stage_rects[wi]);
#      
#    if stage_indicator:
#      if not stage_stamp:
#        stage_times.append(exp.load_stage_times(strain = strain, wid = w));
#
#      sizefac = 0.9; stage_off = plot_ranges[wi][0] * (1 - sizefac) / 2.0;
#      #print plot_range, stage_times;
#      for s in range(5):
#        y = plot_range[1] + 15;
#        y = -20;
#        x = 1.0 * (stage_times[wi][s] - stage_times[-1][0]) / (stage_times[wi][-2] - stage_times[wi][0]) * plot_ranges[wi][0] * sizefac + stage_off;
#        h = 10;
#        w = 1.0 * (stage_times[wi][s+1] - stage_times[-1][s]) / (stage_times[wi][-2] - stage_times[wi][0])  * plot_ranges[wi][0] * sizefac;
#        #print x,y,w,h
#        ax.add_patch(patches.Rectangle((x,y), w, h, edgecolor = None, facecolor = stage_colors[s]));
#        
#        #plt.text(x + w / 2.0, y - 50, generate_stage_str(s+1), fontdict = font, horizontalalignment='center', color = stage_colors[s])
#      
#      r = 7.5;
#      x = (t0s[wi] -  stage_times[wi][0]) /  (stage_times[wi][-2] - stage_times[wi][0]) * plot_range[1] * sizefac + stage_off;
#      stage_pointers.append(plt.Circle((x, y + h/2.0), r, color = 'black', fill = False, linewidth = 0.5));
#      ax.add_artist(stage_pointers[wi]);
#
#  
#  if time_stamp:
#    font = {'size': font_size_title}
#    tt = 'N2 %s %s' % (generate_time_str(0), extra_title);
#    #time_stamp = fig.suptitle(tt, fontdict = font);
#    time_stamp = fig.suptitle(tt, fontsize = font_size_title);
#  
#  # movie saver
#  if save is not None:
#    FFMpegWriter = manimation.writers['ffmpeg']
#    metadata = dict(title='Strain %s Worm %d', artist='Chirstoph Kirst',
#                    comment='C Elegans Dataset, Bargmann Lab')
#    writer = FFMpegWriter(fps=15, metadata=metadata);
#    writer.setup(fig, save, dpi = dpi)
#  
#  # loop over times
#  times_len = [len(ts) for ts in times];
#  nsteps = np.min(times_len);
#  #print plot_ranges
#  #print xylims
#  #print xys
#  for s in range(nsteps):
#    for wi,w in enumerate(wid):
#      t = times[wi][s];
#      if verbose:
#        print '%d / %d' % (t, times[wi][-1]);        
#      
#      xyt = round_xy(xys[wi][t]);
#      #print xyt
#    
#      # update worm image and position 
#      if worm_image:
#        wormt = exp.smooth_image(worms[wi][t], sigma = sigma);
#        wormt = wormt[::-1, :];
#        if wormt.min() != 0:
#          #wormt[:,:] = vmax;
#          wimgs[wi].set_data(wormt);
#          extent = generate_image_extent(xyt, xylims[wi], wormt.shape, plot_ranges[wi]);
#          #print wi, xylims[wi], plot_ranges[wi]
#          #print extent;
#          wimgs[wi].set_extent(extent)
#          wimgs[wi].set_clim(vmin, vmax)
#    
#      # update scatter plot data
#      if scatters[wi] is not None:
#        #plt.scatter(xy[t0:t+1,1] - ymm[0], xy[t0:t+1,0] - xmm[0], c = np.arange(nt), cmap = cmap, edgecolor = 'face')
#        xydat, t0, t1 = generate_xy(wi,t);
#        scatters[wi].set_offsets(np.array(xydat).T);
#        if time_data[wi] is not None:
#          cdat = xynans.copy();
#          cdat[t1:] = time_data[wi][t0:t+1:history_delta];
#        scatters[wi].set_array(cdat);
#        #print scatters[wi];
#      
#      #if time_stamp[wi]:
#      #  time_stamps[wi].set_text(generate_time_str((t-times[wi][0])/sample_rate));
#    
#      if stage_stamps[wi]:
#        stage = np.sum(t >= stage_times[wi]);
#        stage_stamps[wi].set_text(generate_stage_str(stage));
#        #stage_stamps[wi].set_color(stage_colors[stage-1]);
#        stage_stamps[wi].set_color('w');
#        stage_stamps[wi].set_backgroundcolor(stage_colors[stage-1]);
#        #stage_rects[wi].set_color(stage_colors[stage-1]);
#        
#      if stage_indicator:
#        stage_off = plot_ranges[wi][0] * (1 - sizefac) / 2.0;
#        x = 1.0 * (t - stage_times[wi][0]) / (stage_times[wi][-2] - stage_times[wi][0]) * plot_ranges[wi][0] * sizefac + stage_off;
#        sp = stage_pointers[wi];
#        c = sp.center; 
#        sp.center = (x, c[1]);
#        sp.set_fill(True);
#    
#    
#    if time_stamp:
#      time_stamp.set_text('N2 %s %s' % (generate_time_str((t-times[wi][0])/sample_rate), extra_title));    
#    
#    fig.canvas.draw();
#    fig.canvas.flush_events()   
#    
#    if pause is not None:
#      #plt.show();
#      plt.pause(pause);
#    
#    if save is not None:
#      writer.grab_frame();
#      #fig.savefig(save % t, dpi = 'figure', pad_inches = 0);
#  
#  if save is not None:
#    writer.cleanup();
#
#
################################################################################
#### Gui
################################################################################
#
#
#import pyqtgraph as pg
#from functools import partial
#
#def worm_feature_gui(strain = 'n2', wid = 0, times = all, tstart = None, xy = None, roi = None,
#                     size = None, cmap = 'rainbow', linecolor = 'gray', border = 75, levels = None, background = 90, sigma = 1.0,
#                     features = ['speed', 'rotation', 'roam'], feature_filters = None, history = 10, history_delta = 1, 
#                     feature_indicator = None, stage_times = None):
#
#  # load feature data
#  feature_label, feature_data = generate_feature_data(strain, wid, features, feature_filters);
#  feature_indicator = generate_feature_indicator(strain, wid, feature_indicator = feature_indicator);
#  
#  if xy is None:
#    xy = exp.load(strain = strain, wid = wid, dtype = 'xy');
#  
#  if stage_times is None:
#    stage_times = exp.load_stage_times(strain = strain, wid = wid)[:-1];  
#  
#  if times is all:
#    times = (0, len(xy));
#  t0 = times[0]; t1 = times[1];
#  
#  if tstart is None:
#    tstart = stage_times[0];
#  tstart = min(max(tstart, t0), t1);
#
#  # memmap to images
#  worm = exp.load_img(strain = strain, wid = wid);
#  
#  # generate image data
#  wormt, xyt, roi, xylim, extent, plot_range = generate_image_data(strain = strain, wid = wid, t = tstart, size = size, 
#                                                                   xy = xy[tstart], worm = worm[tstart], roi = roi, 
#                                                                   background = background, sigma = sigma, border = border);
#  
#  img_size = wormt.shape;
#  img_size2 = (img_size[0]/2, img_size[1]/2);
#  
#  
#  # create the gui
#  pg.mkQApp()  
#  
#  widget = pg.QtGui.QWidget();
#  widget.setWindowTitle('Feature analysis: Strain: %s, Worm %d' % (strain, wid));
#  widget.resize(1000,800)  
#  
#  layout = pg.QtGui.QVBoxLayout();
#  layout.setContentsMargins(0,0,0,0)        
#   
#  splitter0 = pg.QtGui.QSplitter();
#  splitter0.setOrientation(pg.QtCore.Qt.Vertical)
#  splitter0.setSizes([int(widget.height()*0.99), int(widget.height()*0.01)]);
#  layout.addWidget(splitter0);
#   
#   
#  splitter = pg.QtGui.QSplitter()
#  splitter.setOrientation(pg.QtCore.Qt.Horizontal)
#  splitter.setSizes([int(widget.width()*0.5), int(widget.width()*0.5)]);
#  splitter0.addWidget(splitter);
#  
#  
#  #  Image plot
#  img = pg.ImageView();
#  img.view.setXRange(0, plot_range[0]);
#  img.view.setYRange(0, plot_range[1]);
#  
#  # xy history
#  x = np.zeros(history);
#  fade = np.array(np.linspace(5, 255, history), dtype = int)[::-1];
#  #brushes = np.array([pg.QtGui.QBrush(pg.QtGui.QColor(255, i, i)) for i in fade]);
#  brushes = np.array([pg.QtGui.QBrush(pg.QtGui.QColor(255, 0, 0)) for i in fade]);
#  pxy = pg.ScatterPlotItem(x, x, size = 2, pen=pg.mkPen(None), brush = brushes)  # brush=pg.mkBrush(255, 255, 255, 120))
#  img.addItem(pxy)
#  
#  # circle
#  k = 100;
#  center = np.array(plot_range)/2.0;
#  x = roi[2] * np.cos(np.linspace(0, 2*np.pi, k)) + center[0];  
#  y = roi[2] * np.sin(np.linspace(0, 2*np.pi, k)) + center[1];
#  circle = pg.PlotCurveItem(x,y, pen = pg.mkPen(pg.QtGui.QColor(0,0,0)));
#  img.addItem(circle);
#  
#  # feature indicator
#  if feature_indicator is not None:
#    posfac = 0.9; sizefac = 0.03;
#    indicator = pg.QtGui.QGraphicsRectItem(int(posfac * plot_range[0]), int(posfac * plot_range[1]), int(sizefac * plot_range[0])+1, int(sizefac * plot_range[1])+1);
#    indicator.setPen(pg.mkPen(None))
#    img.addItem(indicator);
#  
#  # Contrast/color control
#  #hist = pg.HistogramLUTItem()
#  #hist.setImageItem(img)
#  #win.addItem(hist)
#  #splitter.addWidget(gl1);
#  splitter.addWidget(img);
#  
#  # Feture data plots
#  gl2 = pg.GraphicsLayoutWidget(border=(50,50,50))
#  pf = [];
#  for f in feature_label:
#    pf.append(gl2.addPlot(title = f));
#    gl2.nextRow();
#    
#  splitter.addWidget(gl2);
#  
#  
#  # counter and Scroll bar
#  widget_tools = pg.QtGui.QWidget();
#  layout_tools = pg.QtGui.QGridLayout()
#
#  spin = pg.SpinBox(value=tstart, int = True, bounds=[t0,t1], decimals = 10);
#  spin.setMaximumWidth(200);
#  layout_tools.addWidget(spin,0,0);
#  
#  sb = pg.QtGui.QScrollBar(pg.QtCore.Qt.Horizontal);
#  sb.setMinimum(t0); sb.setMaximum(t1);
#  sb.setValue(tstart);
#  layout_tools.addWidget(sb,0,1);
#  
#  # add stage times
#  iitem = 1;
#  stb = [];
#  for st in stage_times:
#    b = pg.QtGui.QPushButton('L%d - %d' % (iitem, st));
#    b.setMaximumWidth(100);
#    iitem += 1;
#    layout_tools.addWidget(b,0,iitem);
#    stb.append(b);
#  
#  cb = pg.QtGui.QCheckBox('>'); 
#  cb.setCheckState(False);
#  cb.setMaximumWidth(50);
#  layout_tools.addWidget(cb,0,iitem+1);
#  
#  spin2 = pg.SpinBox(value=1, int = True, bounds=[1,1000], decimals = 3, step = 1);
#  spin2.setMaximumWidth(100);
#  layout_tools.addWidget(spin2,0,iitem+2);  
#  
#  spin3 = pg.SpinBox(value=1, int = True, bounds=[1,10000], decimals = 3, step = 1);
#  spin3.setMaximumWidth(100);
#  layout_tools.addWidget(spin3,0,iitem+3);  
#  
#  widget_tools.setLayout(layout_tools);
#  splitter0.addWidget(widget_tools);
#  
#  widget.setLayout(layout)
#  widget.show();
#  
#  # Callbacks for handling user interaction
#  def updatePlot():
#    #global strain, wid, img_data, xy, roi, border, sb, feature_data, pf, img, history, history_delta
#    t0 = sb.value();    
#    spin.setValue(t0);
#    
#    # Generate image data
#    #wimg = wormimage(strain = strain, wid = wid, t = t0, xy = xy[t0], roi = roi, border = border, worm = img_data[t0]); 
#    #img.setImage(wimg, levels = levels);
#    wimg = worm[t0].copy();
#    if background is not None:
#      wimg[wimg > background] = background;
#    
#    x0 = max(xy[t0,0] - xylim[0][0] - img_size2[0], 0);
#    y0 = max(xy[t0,1] - xylim[1][0] - img_size2[1], 0);
#    img.setImage(wimg.T, levels = levels, pos = round_xy([x0,y0]), autoRange = False)
#
#    #history
#    hsteps = int(t0 / history_delta) + 1; # number of available steps in the past
#    if hsteps - history > 0:
#      ts = t0 - history_delta * (history - 1);
#    else:
#      ts = t0 - history_delta * (hsteps - 1);
#    te = t0 + 1;
#    
#    # update xy positions
#    x = xy[ts:te:history_delta,0] - roi[0] + center[0];
#    y = xy[ts:te:history_delta,1] - roi[1] + center[1];
#    pxy.setData(x,y);
#    pxy.setBrush(brushes[-len(x):]);
#
#    # feature traces
#    for fd, pl in zip(feature_data, pf):
#      pl.plot(fd[ts:te:history_delta],clear=True);
#    
#    if feature_indicator is not None:
#      if feature_indicator[t0]:
#        indicator.setBrush(pg.mkBrush('r'))
#      else:
#        indicator.setBrush(pg.mkBrush('w'))
#      
#  def updateScaleBar():
#    t0 = int(spin.val);
#    sb.setValue(t0);
#    updatePlot();
#
#  def animate():
#    ta = int(spin.val);
#    ta += int(spin3.val);
#    if ta > t1:
#      ta = t0;
#    sb.setValue(ta);
#    spin.setValue(ta);
#    updatePlot();
#   
#  timer = pg.QtCore.QTimer();
#  timer.timeout.connect(animate);
#  
#  def toggleTimer():
#    if cb.checkState():
#      timer.start(int(spin2.val));
#    else:
#      timer.stop();
#
#  def updateTimer():
#    timer.setInterval(int(spin2.val));
#  
#  def updateStage(i):
#    spin.setValue(stage_times[i]);
#    updatePlot();
#  
#  
#  sb.valueChanged.connect(updatePlot);
#  spin.sigValueChanged.connect(updateScaleBar);
#  
#  cb.stateChanged.connect(toggleTimer);
#  spin2.sigValueChanged.connect(updateTimer)
#  
#  for i,s in enumerate(stb):
#    s.clicked.connect(partial(updateStage, i));
#  
#  updatePlot();
#  
#  return widget;
#
#
#
### Start Qt event loop unless running in interactive mode or using pyside.
#
#if __name__ == '__main__':
#  pass
# 