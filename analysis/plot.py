# -*- coding: utf-8 -*-
"""
Fast plotting of large matrices via pyqtgraph

This is module provides scripts to plot large matrices or point clouds in a fast way
using pyqtgraph and opengl.
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
import pyqtgraph as pg
import pyqtgraph.exporters as pgexp

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

#import matplotlib.pyplot as plt
import matplotlib.cm as cm

def colormap_lut(color = 'viridis', ncolors = None):
   # build lookup table
  if color == 'r': 
    pos = np.array([0.0, 1.0])
    color = np.array([[0,0,0,255], [255,0,0,255]], dtype=np.ubyte)
    ncolors = 512;
  elif color =='g':
    pos = np.array([0.0, 1.0])
    color = np.array([[0,0,0,255], [0,255,0,255]], dtype=np.ubyte)
    ncolors = 512;
  elif color =='b':
    pos = np.array([0.0, 1.0])
    color = np.array([[0,0,0,255], [0,0,255,255]], dtype=np.ubyte)
    ncolors = 512;
  else:
    #pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    #color = np.array([[0,0,255,255], [0,255,255,255],  [0,255,0,255], [255,255,0,255], [255,0,0,255]], dtype=np.ubyte)
    #color = np.array([[0,0,128,255], [0,255,255,255],  [0,255,0,255], [255,255,0,255], [128,0,0,255]], dtype=np.ubyte)
    cmap = cm.get_cmap(color);
    if ncolors is None:
      ncolors = cmap.N;
    pos = np.linspace(0.0, 1.0, ncolors);
    color = cmap(pos, bytes = True);
  
  cmap = pg.ColorMap(pos, color)
  return cmap.getLookupTable(0.0, 1.0, ncolors);


class plot_array():
  """Plots a 2d matrix"""
  def __init__(self, data, title = None, color = 'viridis', ncolors = None):
    #self.app = pg.QtGui.QApplication([])
    #self.win = pg.GraphicsLayoutWidget()
    #self.win.resize(1200, 800)
    lut = colormap_lut(color, ncolors);
 
    self.img = pg.ImageItem()
    self.img.setLookupTable(lut)
    self.img.setLevels([0,1])        
    
    #self.plot = self.win.addPlot()
    self.plot = pg.plot(title = title);
    self.plot.addItem(self.img)
    
    #self.timer = QtCore.QTimer()
    #self.timer.timeout.connect(self.check_for_new_data_and_replot)
    #self.timer.start(100)
    self.img.setImage(data.T)
    #self.win.show()
    
    

import pyqtgraph.opengl as gl

class plot3d():
  """Plots a 3d could of points"""
  def __init__(self, points, title = None):
    pg.mkQApp();
    self.w = gl.GLViewWidget()
    self.w.opts['distance'] = 20
    self.w.show()
    self.w.setWindowTitle(title)

    self.g = gl.GLGridItem()
    self.w.addItem(self.g)
    self.sp = gl.GLScatterPlotItem(pos=points, color=(1,1,1,1), pxMode= True)
    self.w.addItem(self.sp);
    #self.plot.addItem(self.w);
      
    #
    ### create three grids, add each to the view
    #xgrid = gl.GLGridItem()
    #ygrid = gl.GLGridItem()
    #zgrid = gl.GLGridItem()
    #view.addItem(xgrid)
    #view.addItem(ygrid)
    #view.addItem(zgrid)
    #
    ### rotate x and y grids to face the correct direction
    #xgrid.rotate(90, 0, 1, 0)
    #ygrid.rotate(90, 1, 0, 0)
    #
    ### scale each grid differently
    #xgrid.scale(0.2, 0.1, 0.1)
    #ygrid.scale(0.2, 0.1, 0.1)
    #zgrid.scale(0.1, 0.2, 0.1)


def savefig(plot, filename, width = None):
  """Export plot to file"""
  exporter = pgexp.ImageExporter(plot.img);
  if width is not None:
    exporter.parameters()['width'] = width   # (note this also affects height parameter)
  
  # save to file
  exporter.export(filename)



def plot_trace(xy, ids = None, depth = 0, colormap = 'rainbow', line_color = 'k', line_width = 1, point_size = 5, title = None):
  """Plot trajectories with positions color coded according to discrete ids"""
  
  #if ids is not None:
  uids = np.unique(ids);
  
  cmap = cm.get_cmap(colormap);
  n = len(uids);
  colors = cmap(range(n), bytes = True);
  
  #lines
  if line_width is not None:
    #plt.plot(xy[:,0], xy[:,1], color = lines);    
    plot = pg.plot(xy[:,0], xy[:,1], pen = pg.mkPen(color = line_color, width = line_width))    
  else:
    plot = pg.plot(title = title);
    
  if ids is None:
    sp = pg.ScatterPlotItem(pos = xy, size=point_size, pen=pg.mkPen(colors[0])); #, pxMode=True);
  else:
    sp = pg.ScatterPlotItem(size=point_size); #, pxMode=True);
    spots = [];
    for j,i in enumerate(uids):
      idx = ids == i;
      spots.append({'pos': xy[idx,:].T, 'data': 1, 'brush':pg.mkBrush(colors[j])}); #, 'size': point_size});
    sp.addPoints(spots)
  
  plot.addItem(sp);
  
  return plot;

  
#  legs = [];
#  for k,i in enumerate(uids):
#    ii = np.where(ids == i)[0];
#    if depth > 0:
#      ii = [ii-d for d in range(depth)];
#      ii = np.unique(np.concatenate(ii));
#    
#    plt.plot(data[ii, 0], data[ii, 1], '.', color = color[k]);
#
#    legs.append(mpatches.Patch(color=color[k], label= str(i)));
#  
#  plt.legend(handles=legs);













    
def test():
  import numpy as np
  import plot as p;
  reload(p)
  data = np.random.rand(100,200);
  p.plot(data, 'test')
  
  reload(p)
  pts  = np.random.rand(10000,3);
  p.plot3d(pts);
  
if __name__ == '__main__':
  test();