# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 01:57:00 2016

@author: ckirst
"""


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
    
  def calculatePathEntropy(self, radius, n = 20, offset = 0, stage = all, valid = False):
    """Calculates path entropy of the trajectories time evolution"""
    
    import scipy.spatial 
    #from joblib import Parallel, delayed, cpu_count
    #import multiprocessing
    #from parallel_tools import createSharedNumpyArray
    #import ctypes
    
    #for each point caluclate distances to n-1 previous positions
    data = self.data(label = ['x', 'y'], stage = stage, valid = valid);
    nd = data.shape[0];
    
    dists = np.zeros((nd,n-1)); # exclude self-distance = 0 -> n-1 distances for path of length n
    for i in range(n-1, nd):    # distances from not i -> (i-(n-1), i-(n-1)+1, ... i-1)
        if i % 1000 == 0:
          print '%d / %d' % (i, nd);
        dists[i:i+1,:] = scipy.spatial.distance.cdist(data[i:i+1,:],data[i-n+1:i,:]);
    for i in range(1,n-1):      # fill remaining distances
      dists[i,:] = np.nan;
      dists[i:i+1,n-1-i:n-1] = scipy.spatial.distance.cdist(data[i:i+1,:],data[0:i,:]);
    dists[0,:] = np.nan;
    #print dists;

    # calculate number of neighbours
    
    # for the path of length n calculate entropy: min number of balls of radius r centered at positions convering the path
    if not isinstance(radius, list):
      radius = [radius];
    nr = len(radius);
    
    entropies = np.zeros((nd, nr))
    #br = createSharedNumpyArray(dists.shape, ctypes.c_double);
    for ri,r in enumerate(radius):
      br = dists < r;
      #print br
      #br_shared = multiprocessing.Array(br);
      
      #entropies[n-1:,ri] = Parallel(n_jobs = cpu_count(), verbose = 5)(delayed(calcEntropy)(br, i) for i in range(n-1,nd));
      for i in range(n-1, nd): # loop over different radii 
        if i % 1000 == 0:
          print '%d / %d' % (i, nd);
          
        #build local connectivity matrix on distance thresholding
        mat = np.zeros((n,n), dtype = bool);
        for j in range(n-1):
          mat[j,j+1:] = br[i-j,j:][::-1];
        mat = mat + mat.T;
        np.fill_diagonal(mat, 1);        
        #print mat;
        
        #calcualte number of covering balls
        e = 0;
        while mat.shape[0] > 0:
          deg = np.sum(mat, axis = 0);
          #print 'deg', deg
          mi = np.argmax(deg);
          #print 'max i = %d' % mi;
          mi = mat[mi,:];
          mat = mat[:,~mi][~mi,:];
          #print 'mat: ', mat
          #print '-------'
          e += 1;
        
        entropies[i,ri] = e;
    
    entropies[:n-1,:] = np.nan;
    return entropies;
  
  
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

   