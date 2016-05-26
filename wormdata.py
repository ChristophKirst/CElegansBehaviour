"""
Base Data Class for C-elegans Worm Data

Long Term Behaviour Analysis of C-elegans

Experimental Data: 
Shay Stern, Bargman lab, The Rockefeller University 2016
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np
import matplotlib.pyplot as plt;
import matplotlib.cm as cm;
import matplotlib.patches as mpatches;
import scipy.io as io

from utils import printProgress

from parallel_tools import parallelIterateOnMemMap

from scipy import stats


def shiftData(data, s, reverse = True, nan = -1):
  s = min(max(int(s),0), data.shape[0]);
  if s == 0:
    return data;
  
  ds = np.zeros_like(data);  
  if reverse:
      ds[:-s,...] = data[s:,...];
      ds[-s:,...] = np.NaN;
  else:
    ds[s:,...] = data[:-s,...];
    ds[:s,...] = np.NaN;
    
  if nan is not None:
    ds[np.isnan(ds)] = nan;
  
  return ds;



#helper used for parallel processing
def calcDiff(dists, diffs, i):
    if i % 100 == 0:
      print("[Worker %d] processing iteration %d" % (os.getpid(), i))
    
    if np.any(np.isnan(dists[i,:])):
      diffs[i,:] = np.NaN;
    else:
      try:
        #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
        diffs[i,:] = np.array(stats.linregress(np.log(range(1,dists.shape[1]+1)),np.log(dists[i,:])));
      except:
        diffs[i,:] = np.NaN;



#
#class Experiment:
#  """Base class for a set of worm traces"""
#  
#  
#  def __init__(self, filename = None, name = None):
#    """Constructor"""
#    
#    if filename is not None:
#      data = 
#    self.data = {};
#    self.id = id;


class WormData:
  """Base class for Worm Data, single individuum"""
  
  def __init__(self, data, label = None, stage = None, valid = all, wid = None):
    """Constructor"""
    
    self._data = data;  # array [time, label]
    self._wid = wid;
    
    
    if label is None:
      label = ('x', 'y', 'stage');
    if isinstance(label, dict):
      self._label = label;
    else:
      self._label = {l : i for l,i in zip(label, range(len(label)))};
    
    if stage is None:
      if 'stage' in self.label():
        self._stage = self._data[:, self._label['stage']];
      else:
        self._stage = np.zeros(self.length());
    else:
      if len(stage) != self.length():
        raise RuntimeError("stage and data have inconsistent sizes: %d, %d" % (len(stage), self.length()));
      self._stage = np.array(stage);
    
    if valid is all:
      self._valid = np.ones(self.length(), dtype = bool);
    else:
      if len(valid) != self.length():
        raise RuntimeError("valid specification and data have inconsistent sizes: %d, %d" % (len(valid), self.length()));
      self._valid = np.array(valid);   
  
  def __str__(self):
    return "WormData#%d label:%s length:%d valid:%d" % (self.wid(), str(self.label()), self.length(), np.sum(self.valid()));
 
  def __repr__(self):
    return self.__str__();
     
  def length(self):
    """Length of data points in each entry"""
    return self._data.shape[0];
    
  def wid(self):
    """Worm id"""
    if self._wid is None:
      return -1;
    else:
      return self._wid;
    
  def valid(self):
    """Valid data entries"""
    return self._valid;
    
  def stage(self, stage = all, valid = False):
    """Stage of data entries"""

    if stage is all:
      if valid:
        return self._stage[self._valid];
      else:
        return self._stage;
    
    else:
      sids = np.in1d(self._stage, np.array(stage));
      if valid:
        return self._stage[np.logical_and(self._valid, sids),:];
      else:
        return self._stage[sids];
  
    
  def label(self, i = all):
    """Returns string label of the data"""
    if i is all:
      return self._label.keys();
    else:
      if isinstance(i, int):
        for (k,v) in self._label.items():
          if i in np.array(v).flatten():
            return k;
      else:
        l = [];
        for (k,v) in self._label.items():
          for ii in i:
            if i in np.array(v).flatten():
              l.append(k);
        return l;
  
  
  def labelid(self, label = all):
    """Return label ids of label given as strings or ids"""
    
    def makelab(lab):
      if isinstance(lab, str):
        return self._label[lab];
      else:
        return lab;
    
    if label is all:
      return range(self._data.shape[-1]);
    else:
      if isinstance(label, str) or isinstance(label, int):
        return makelab(label);
      else:
        return np.array([makelab(l) for l in label]).flatten();
    
  def data(self, label = all, stage = all, valid = False):
    """Return specified data as array"""

    lids = self.labelid(label);

    if stage is all:
      if valid:
        return self._data[self._valid, :][:,lids];
      else:
        return self._data[:, lids];
    
    else:
      sids = np.in1d(self._stage, np.array(stage));
      if valid:
        return self._data[np.logical_and(self._valid, sids),:][:,lids];
      else:
        return self._data[sids,:][:,lids];
     
  ############################################################################
  ## Util 
  ############################################################################      
  
  def replaceInvalid(self, replace = np.NaN):
    """Replace invalid data points with specified values"""
    self._data[np.logical_not(self._valid), :] = replace;
    return;
    
    
  def stageIndex(self, stage = all, valid = False):
    """Returns indices of developmental stage changes"""
    
    return np.argwhere(np.diff(self.stage(stage = stage, valid = valid)));  
    
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

  
  ############################################################################
  ## Analysis 
  ############################################################################
  
  def _accumulate(self, data, n = 10, offset = 0):
    """Accumulate entries in data back in history, NaN if history is not available"""

    d = data.copy();

    a = np.zeros((data.shape[0], n));  
    a[:,offset] = d;
    
    for i in range(1, n-offset):
      d = np.concatenate([[np.NaN], d[:-1]]);
      a[:,i+offset] = a[:,i+offset-1] + d;
    
    if offset > 0:
      d = data.copy();
      for i in range(1, offset+1):
        d = np.concatenate([d[1:], [np.NaN]]);
        a[:,offset-i] = a[:,offset-i+1] - d;
    
    return a;
  
  def calculateDistances(self, n = 10, offset = 0, stage = all, valid = False):
    """Create the distances moved in the last n time steps"""
    
    data = self.data(label = ['x', 'y'], stage = stage, valid = valid);
    dx = np.concatenate([[np.NaN], np.asarray(np.diff(data[:,0]), dtype = float)]);
    dy = np.concatenate([[np.NaN], np.asarray(np.diff(data[:,1]), dtype = float)]);
    
    dx0 = dx.copy();
    dy0 = dy.copy();
    
    delta = np.zeros((data.shape[0], n));      
    delta[:,offset] = np.sqrt(dx * dx + dy * dy);
    
    for i in range(1,n-offset):
      dx0 = np.concatenate([[np.NaN], dx0[:-1]]);
      dy0 = np.concatenate([[np.NaN], dy0[:-1]]);
      dx += dx0;
      dy += dy0;
      delta[:,i+offset] = np.sqrt(dx * dx + dy * dy);      
    
    if offset > 0:
      dx = np.concatenate([[np.NaN], np.asarray(np.diff(data[:,0]), dtype = float)]);
      dy = np.concatenate([[np.NaN], np.asarray(np.diff(data[:,1]), dtype = float)]);
      dx0 = dx.copy();
      dy0 = dy.copy();

      delta[:,offset-1] = np.sqrt(dx * dx + dy * dy);
      
      for i in range(1,offset):
        dx0 = np.concatenate([dx0[1:], [np.NaN]]);
        dy0 = np.concatenate([dy0[1:], [np.NaN]]);
        dx += dx0;
        dy += dy0;
        delta[:,offset-i-1] = np.sqrt(dx * dx + dy * dy);   
    
    return delta;
  
  
  def calculateLengths(self, n = 10, offset = 0, stage = all, valid = False):
    """Create the path lengths moved in the last n time steps"""
    delta = self.calculateDistances(n = 1, offset = 0, stage = stage, valid = valid);
    return self._accumulate(delta[:,0], n = n, offset = offset);
  
    
  def calculateAngles(self, n = 10, offset = 0, stage = all, valid = False):
    """Calculate angles between two translation vectors"""
    
    data = self.data(label = ['x', 'y'], stage = stage, valid = valid);
    dx = np.diff(data[:,0]);
    dy = np.diff(data[:,1]);
    
    dx1 = dx[:-1];
    dx2 = dx[1:];
    
    dy1 = dy[:-1];
    dy2 = dy[1:];
    
    ang = np.arctan2(dx2, dy2) - np.arctan2(dx1, dy1);
    ang = np.mod(ang + np.pi, np.pi * 2) - np.pi;
    ang = np.concatenate([[np.NaN], ang, [np.NaN]]);
    
    #history of angles;
    angc = np.zeros((ang.shape[0], n));
    ang0 = ang.copy();    
    angc[:,offset] = ang0;
    
    for i in range(1, n-offset):
      ang0 = np.concatenate([[np.NaN],ang0[:-1]]);
      angc[:,i+offset] = angc[:,i+offset-1] + ang0;
      
    if offset > 0:
       ang0 = ang.copy(); 
       for i in range(1,offset+1):
         ang0 = np.concatenate([ang0[1:], [np.NaN]]);
         angc[:,offset-i] = angc[:, offset-i+1] - ang0;
    
    return angc;
  
  def calculateRotations(self, n = 10, offset = 0, stage = all, valid = False):
    """Calculate accumulated absolute rotation of trajectories"""

    ang = self.calculateAngles(n = 1, offset = 0, stage = stage, valid = valid);
    return self._accumulate(np.abs(ang)[:,0], n = n, offset = offset);
  
  def calculateDiffusionParameter(self, n = 20, offset = 0, stage = all, valid = False, distances = None, p0 = None, parallel = True):
    """Fits a x**g onto the time evolution of distances"""
    #from scipy.optimize import curve_fit
    #from scipy import stats;

    #def f(x, c, b, a):
    #  return a + b * np.power(x,c);
    
    if distances is None:
      distances = self.calculateDistances(n= n, offset = offset, stage = stage, valid = valid);
    
    diff = np.zeros((distances.shape[0], 5));    
    x = np.array(range(1, n + 1));
    logx = np.log(x);
    
    if parallel:
      
      diff = parallelIterateOnMemMap(calcDiff, distances, diff, range(distances.shape[0]));
       
    else:
      for i in range(distances.shape[0]):
        printProgress(i, distances.shape[0], prefix = 'Diffusion:', suffix = '', barLength = 80)
        
        if np.any(np.isnan(distances[i,:])):
          diff[i,:] = np.NaN;
        else:
          try:
            #slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
            diff[i,:] = np.array(stats.linregress(logx,np.log(distances[i,:])));
          except:
            diff[i,:] = np.NaN;
    
    return diff;
    


  def calculateConvexHullParameter(self, n = 20, offset = 0, stage = all, valid = False):
    """Calculates convex hull derived parameters on the trajectories time evolution"""
    pass
    
  def calculateFeatures(self, n = 10, offset = 0, features = ['rotations', 'angles', 'distances', 'lengths'], stage = all, valid = False):
    """Calculate feature vectors for trajectory"""

    # rotate features so that at t=0 direction is (0,y)
    f = np.zeros((self.length(), 0));
    if 'distances' in features:
      f = np.hstack((f, self.calculateDistances(n = n, offset = offset, stage = stage, valid = valid)));
    if 'lengths' in features:
      f = np.hstack((f, self.calculateLengths(n = n, offset = offset, stage = stage, valid = valid)));   
    if 'rotations' in features:
      f = np.hstack((f, self.calculateRotations(n = n, offset = offset, stage = stage, valid = valid)));
    if 'angles' in features:
      f = np.hstack((f, self.calculateAngles(n = n, offset = offset, stage = stage, valid = valid)));
    
    return f;





def _test(): 
  xyd = np.array([[0,0],[0,1],[1,1],[1,1],[1,2],[5,2],[10,10]])
  w1 = WormData(xyd, stage = np.array([0,0,0,0,1,1,1]))

  def pinfo(label, res):
    print "------------"
    print label;
    print res
    
  def mkfig(n, s = None):
    plt.figure(n); plt.clf();
    plt.title(s)

  pinfo('data', w1.data())
  pinfo('data stage = 0', w1.data(stage = 0));
  pinfo('data stage = 1', w1.data(stage = 1));

  mkfig(0, 'plotData');
  w1.plotData()
  
  mkfig(1, 'plotTrajectory');  
  w1.plotTrajectory()
  
  mkfig(2, 'plotTrace');
  w1.plotTrace()
  
  mkfig(3, 'plotTrace with ids');
  w1.plotTrace(ids = w1.stage())

  pinfo('angles', w1.calculateAngles(n=2));
  pinfo('distances', w1.calculateDistances(n=2));
  pinfo('lengths', w1.calculateLengths(n=2));

  pinfo('angles offset=1', w1.calculateAngles(n=3, offset = 1));
  pinfo('rotations offset = 1', w1.calculateRotations(n=3, offset = 1));

  mkfig(4, 'plotTrajectory with feature coloring');
  ang = w1.calculateAngles(n=1)[:,0]; 
  w1.plotTrajectory(colordata = ang)
  
  mkfig(5, 'plotTrajectory with feature coloring and sizing');
  w1.plotTrajectory(colordata = ang, size = w1.calculateDistances(n=1)[:,0] * 100)

def _testWithData():

  basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/';
  filename = os.path.join(basedir, 'Experiment/individuals_N2_X_Y.mat')

  data = io.loadmat(filename);

  XYdata = data['individual_X_Y'][0];

  print XYdata.shape
  print XYdata[0].shape  
  XYwdata = XYdata[0];
  
  wd = WormData(XYwdata[:,0:2], stage = XYwdata[:,-1], valid = XYwdata[:,0] != 1)

   