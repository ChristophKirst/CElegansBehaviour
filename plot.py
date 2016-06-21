"""
Module handling visualization of data and analysis results

Long Term Behaviour Analysis of C-elegans
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np

import experiment as exp
import analysis


 ############################################################################
  ## Plotting 
  ############################################################################
  
def plotData(self, data = all, label = all, stage = all, valid = False, lines = True, **args):
  """Plot each data coordinate"""
  
  if data is all or None:
    lids = self.labelid(label);
    lids = np.array(lids).flatten();
    label = [self.label()[i] for i in self.labelid(label)];
  else:
    if data.ndim == 1:
      data = data.reshape((data.shape[0],1));  
    lids = range(data.shape[-1]);
    if label is all:
      label = ['' for i in lids];
  
  fig = plt.gcf();
  #fig, axes = fig.subplots(nrows = len(lids), ncols = 1);
  
  stages = self.stageIndex(stage = stage, valid = valid);
  
  pid = 0;
  for i,l in zip(lids, label):
    #ax.set_title('Test Axes {}'.format(i))
    #ax.set_xlabel('X axis')
    pid += 1;
    ax = plt.subplot(len(lids), 1, pid)
    ax.set_ylabel(l);
    
    if data is all or None:
      datap = self.data(label = i, stage = stage, valid = valid);
    else:
      datap = data[:,i];
    ax.plot(datap, **args);
    
    if lines:
      for s in stages:
        ax.axvline(s, linewidth = 1, color = 'k');
  
  fig.tight_layout();
  plt.show();
  
  
def plotDataColor(self, data = all, label = all, stage = all, valid = False, lines = True, **args):
  """Plot each data coordinate"""
  
  if data is all or None:
    lids = self.labelid(label);
    lids = np.array(lids).flatten();
    label = [self.label()[i] for i in self.labelid(label)];
  else:
    if data.ndim == 1:
      data = data.reshape((data.shape[0],1));  
    lids = range(data.shape[-1]);
    if label is all:
      label = ['' for i in lids];
  
  fig = plt.gcf();
  #fig, axes = fig.subplots(nrows = len(lids), ncols = 1);
  
  stages = self.stageIndex(stage = stage, valid = valid);
  
  pid = 0;
  for i,l in zip(lids, label):
    #ax.set_title('Test Axes {}'.format(i))
    #ax.set_xlabel('X axis')
    pid += 1;
    ax = plt.subplot(len(lids), 1, pid)
    ax.set_ylabel(l);
    
    if data is all or None:
      datap = self.data(label = i, stage = stage, valid = valid);
    else:
      datap = data[:,i];
    ax.scatter(np.arange(datap.shape[0]), datap, **args);
    
    if lines:
      for s in stages:
        ax.axvline(s, linewidth = 1, color = 'k');
  
  fig.tight_layout();
  plt.show();
  

def plotTrajectory(self, stage = all, valid = False, colordata = 'time', cmap = cm.jet, size = 20, ax = None):
  """Plot trajectories with color using time or specified values"""
  #    points = np.array([x, y]).T.reshape(-1, 1, 2)
  #    segments = np.concatenate([points[:-1], points[1:]], axis=1)
  #
  #    lc = LineCollection(segments, cmap=plt.get_cmap('Spectral'))
  #    lc.set_array(z)
  #    #lc.set_linewidth(2)
  #
  #    plt.gca().add_collection(lc)
  
  data = self.data(label = ['x', 'y'], stage = stage, valid = valid);
  if isinstance(colordata, basestring) and colordata in ['time']:
    c = np.linspace(0, len(data[:,0]), len(data[:,0]));
  else:
    #def normalize(v):
    #  v = v - np.min(v);
    #  norm=np.linalg.norm(v)
    #  if norm==0: 
    #    return v
    #  return v/norm
    #c = normalize(colordata); 
    c = colordata;
  
  if ax is None:
    ax = plt.gca();
  s = ax.scatter(data[:,0], data[:,1], c = c, cmap = cmap, s = size, marker = 'o', lw = 0);
  ax.plot(data[:,0], data[:,1], color = 'gray');    
  ax.get_figure().colorbar(s, ax = ax);


def plotTrace(self, ids = None, depth = 0, stage = all, valid = False, color = all):
  """Plot trajectories with positions color coded according to discrete ids"""

  data = self.data(label = ['x', 'y'], stage = stage, valid = valid);
  
  uids = np.unique(ids);
  
  if color is all:
    c = cm.rainbow;
    n = len(uids);
    color = [np.array(c(c.N * i/(n-1)))[:-1] for i in range(n)];

  #lines
  plt.plot(data[:,0], data[:,1], color = 'black');    
  
  legs = [];
  for k,i in enumerate(uids):
    ii = np.where(ids == i)[0];
    if depth > 0:
      ii = [ii-d for d in range(depth)];
      ii = np.unique(np.concatenate(ii));
    
    plt.plot(data[ii, 0], data[ii, 1], '.', color = color[k]);

    legs.append(mpatches.Patch(color=color[k], label= str(i)));
  
  plt.legend(handles=legs);