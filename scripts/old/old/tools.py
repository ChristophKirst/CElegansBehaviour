"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Experimental Data: 
Shay Stern, Bargman lab, The Rockefeller University 2016

Plotting Routines
"""


import matplotlib.pyplot as plt
import numpy as np
import itertools

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection

def stageIndex(data, cls = -1):
  """Returns indices at which the developmental stage changes"""
  
  return np.argwhere(np.diff(data[:,cls]));  


def shiftData(data):
  """Center the AV data"""
  data[:,1] = np.mod(data[:,1] + 90, 90)


def smoothIndividualData(data, bins = (5,5)):
  """Smooth the data by binning and averaging"""
  
  for i in range(len(bins)):
    if bins[i] > 0:
        w = np.ones(bins[i]);
        data[:,i] = np.convolve(w/w.sum(),data[:,i],mode='same');
  
  return data;

def plotIndividualData(data, idx = 0, cls = -1):
  """Plots the individual data including the markers for the different developmetnal stages"""
  p = plt.plot(data[:,idx]);
  
  ## add developmental stages
  # detect switches
  stages = stageIndex(data, cls = cls);
  for s in stages:
    plt.axvline(s, linewidth = 1, color = 'k');
  
  return p;


def plotAllIndividualData(data, idx = (0,1), cls = -1):
  """ Plot all the individual data traces with stage identifier"""
  n = len(idx);
  for i in range(n):  
    plt.subplot(n,1,1+i);
    plotIndividualData(data, idx = idx[i], cls = cls);



def plotHistogramStages(data, idx = (0,1), cls = -1, **args):
  """Plots joint distributions at each stage for an individual"""
  
  stages = stageIndex(data, cls = cls);  
  stages = np.insert(np.append(stages, data.shape[0]), 0, 0);
  nstages = len(stages) - 1;
  
  for i in range(nstages):
    plt.subplot(1, nstages, i);
    plt.hexbin(data[stages[i]:stages[i+1],idx[0]], data[stages[i]:stages[i+1],idx[1]], **args);
  



def histogram2dstd(data, std = 6, bins = 50):
  """Create histogram resolving distribution according to std"""
  
  ## calculate standard deviations in each direction
  stds = np.std(data[:,0:-2], axis = 0);
  means = np.mean(data[:,0:-2], axis = 0);

  rngs = [[m- std * s, m + std * s] for m,s in itertools.izip(means,stds)];  
  
  hist = np.histogram2d(data[:,0], data[:,1], bins = bins, range = rngs);
  
  return hist;
  


def correlationIndividual(data, idx = (0,1), cls = -1, delay = (-100, 100)):
  """Calculate corrs and auto correlation in time between the various measures"""

  n = len(idx);  
  means = np.mean(data[:,:-1], axis = 0);
  
  nd = delay[1] - delay[0] + 1;
  
  cc = np.zeros((nd,n,n))
  for i in range(n):
    for j in range(n):
        if delay[0] < 0:
          cm = np.correlate(data[:, i] - means[i], data[-delay[0]:, j] - means[j]);
        else:
          cm = [0];
        
        if delay[1] > 0:
          cp = np.correlate(data[:, j] - means[j], data[delay[1]:, i] - means[i]);
        else:
          cp = [0];
        
        ca = np.concatenate((cm[1:], cp[::-1]));
        
        if delay[0] > 0:
          cc[:,i,j] = ca[delay[0]:];
        elif delay[1] < 0:
          cc[:,i,j] = ca[:-delay[1]];
        else:
          cc[:,i,j] = ca;
  
  return cc;
  
  

def correlationIndividualStages(data, idx = (0,1), cls = -1, delay = (-100, 100)):
  """Calculate correlation functions in time between the various measures in each stage"""

  stages = stageIndex(data, cls = cls);
  stages = np.insert(np.append(stages, data.shape[0]), 0, 0);
  ns = len(stages) - 1;
  

  n = len(idx);  
  means = np.mean(data[:,:-1], axis = 0);
  
  nd = delay[1] - delay[0] + 1;
  
  cc = np.zeros((nd,n,n,ns))
  for s in range(ns):
    dat = data[stages[s]:stages[s+1],:];
    for i in range(n):
      for j in range(n):
          if delay[0] < 0:
            cm = np.correlate(dat[:, i] - means[i], dat[-delay[0]:, j] - means[j]);
          else:
            cm = [0];
          
          if delay[1] > 0:
            cp = np.correlate(dat[:, j] - means[j], dat[delay[1]:, i] - means[i]);
          else:
            cp = [0];
          
          ca = np.concatenate((cm[1:], cp[::-1]));
          
          if delay[0] > 0:
            cc[:,i,j,s] = ca[delay[0]:];
          elif delay[1] < 0:
            cc[:,i,j,s] = ca[:-delay[1]];
          else:
            cc[:,i,j,s] = ca;
  
  return cc;



def plotCorrelationIndividual(cc):
  n1 = cc.shape[1]; n2 = cc.shape[2];
  k = 1;
  for i in range(n1):
    for j in range(n2):
      plt.subplot(n1, n2, k)
      plt.plot(cc[:,i,j]);
      k += 1;
      
 
def plotCorrelationIndividualStages(cc):
  n1 = cc.shape[1]; n2 = cc.shape[2]; ns = cc.shape[3];

  for s in range(ns):
    k = 1;
    for i in range(n1):
      for j in range(n2):
        plt.subplot(n1, n2 * ns, (k - i * n2) + s * n2 + n2 * ns * i);
        plt.plot(cc[:,i,j,s]);
        k += 1;

  
  
def removeUndefined(data):
  return data[data[:,0] > 1,:];

def replaceUndefined(data, replace = np.NaN):
  ids = data[:,0] == 1;
  data[ids, :-1] = replace;
  return data;





    

def calculateDeltas(data, n = 10):
  """create the distnances travelled since the last n time steps"""
  
  dx = np.diff(data[:,0]);
  dy = np.diff(data[:,1]);
  
  dx0 = dx.copy();
  dy0 = dy.copy();
  
  delta = np.zeros((dx.shape[0], n));
  for i in range(n):
    delta[:,i] = np.sqrt(dx * dx + dy * dy);
    if i < n-1:
      dx0 = np.concatenate([[np.NaN], dx0[:-1]]);
      dy0 = np.concatenate([[np.NaN], dy0[:-1]]);
      dx += dx0;
      dy += dy0;
  
  return delta;
  
  
def calculateAngles(data):
  """calculate angle between two translation vectors"""
  
  dx = np.diff(data[:,0]);
  dy = np.diff(data[:,1]);
  
  dx1 = dx[:-1];
  dx2 = dx[1:];
  
  dy1 = dy[:-1];
  dy2 = dy[1:];
  
  ang =  np.arctan2(dx2, dy2) - np.arctan2(dx1, dy1);
  return np.mod(ang + np.pi, np.pi * 2)-np.pi;
  

def accumulate(data, n = 10):
  d0 = data.copy();
  d = data.copy();
  a = np.zeros((data.shape[0], n));
  for i in range(n):
    a[:,i] = d0;
    if i < n-1:
      d = np.concatenate([[np.NaN], d[:-1]]);
      d0 += d;
      
  return a;
      


  
def plotTrajectory(data):
  #    points = np.array([x, y]).T.reshape(-1, 1, 2)
  #    segments = np.concatenate([points[:-1], points[1:]], axis=1)
  #
  #    lc = LineCollection(segments, cmap=plt.get_cmap('Spectral'))
  #    lc.set_array(z)
  #    #lc.set_linewidth(2)
  #
  #    plt.gca().add_collection(lc)
  
  import matplotlib.cm as cm
  cmap = cm.jet;
  c = np.linspace(0, 10, len(data[:,0]));
  plt.scatter(data[:,0], data[:,1], c = c, cmap = cmap, marker = '+');

  

def plotTrace(data, ids = None):
  """Plot Trajectories color ids"""

  idsall = np.where(ids);
  idsall = [idsall- i for i in np.array(range(kk+1))];
  idsall = np.unique(np.concatenate(idsall));
  np.put(ids, idsall, True);

  plt.figure(32); plt.clf();
  plt.plot(dd[:,0], dd[:,1], color = 'black');
  plt.plot(dd[~ids,0], dd[~ids,1], '.');
  plt.plot(dd[ids, 0], dd[ids, 1], '.', color = 'red')



def plotTrace(data, stage = True):
  
  mxstage = 5;
  cm = plt.get_cmap('rainbow');
  cdat = cm(data[:,-1] / mxstage);
  plt.scatter(data[:, 0], data[:,1], c = cdat);
