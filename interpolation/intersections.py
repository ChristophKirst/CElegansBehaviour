# -*- coding: utf-8 -*-
"""
Routines to find intersections between curves
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np

from scipy.linalg import lu;

def curve_intersections_discrete(xy1, xy2, robust  = True):
  """Returns intersection points and indices between two curves
  
  Arguments:
    xy1,xy2 (nx2 array): sample points along curve
    robust (bool): if true additional checks for duplication on boundary are performed
  
  Returns:
    mx2 array: intersection points
    m arrays: indices of the reference points for the intersecting segments in xy1 and xy2
    m arrays: distance of intersection point within the reference segments
    
  Notes: Based on intersections by Douglas M. Schwarz
  """

  x1 = xy1[:,0];
  y1 = xy1[:,1];
  x2 = xy2[:,0];
  y2 = xy2[:,1];

  n1 = len(x1) - 1;
  n2 = len(x2) - 1;
  
  dxy1 = np.diff(xy1, axis = 0);
  dxy2 = np.diff(xy2, axis = 0);

  #determine the combinations of i and j where the rectangle enclosing the
  #i'th line segment of curve 1 overlaps with the rectangle enclosing the
  #j'th line segment of curve 2
  minx1 = np.min([x1[:-1], x1[1:]], axis = 0);
  minx2 = np.min([x2[:-1], x2[1:]], axis = 0);
  miny1 = np.min([y1[:-1], y1[1:]], axis = 0);
  miny2 = np.min([y2[:-1], y2[1:]], axis = 0);
  
  maxx1 = np.max([x1[:-1], x1[1:]], axis = 0);
  maxx2 = np.max([x2[:-1], x2[1:]], axis = 0);
  maxy1 = np.max([y1[:-1], y1[1:]], axis = 0);
  maxy2 = np.max([y2[:-1], y2[1:]], axis = 0);
  
  
  i,j = np.where((np.repeat(minx1[:,np.newaxis], n2, axis = 1) <= np.repeat(maxx2[np.newaxis,:], n1, axis = 0)) &  
                 (np.repeat(minx2[np.newaxis,:], n1, axis = 0) <= np.repeat(maxx1[:,np.newaxis], n2, axis = 1)) &
                 (np.repeat(miny1[:,np.newaxis], n2, axis = 1) <= np.repeat(maxy2[np.newaxis,:], n1, axis = 0)) &  
                 (np.repeat(miny2[np.newaxis,:], n1, axis = 0) <= np.repeat(maxy1[:,np.newaxis], n2, axis = 1)));

  if len(i) == 0:
    return np.zeros((0,2)), i,i,j,j;
  
  n = len(i);
  T = np.zeros((4,n));
  AA = np.zeros((4,4,n));
  AA[[0,1],2,:] = -1;
  AA[[2,3],3,:] = -1;
  AA[[0, 2],0,:] = dxy1[i,:].T;
  AA[[1, 3],1,:] = dxy2[j,:].T;
  B = - np.vstack([x1[i], x2[j], y1[i], y2[j]]);

  if robust:
    overlap = np.zeros(n, dtype = bool);
    for k in range(n):
      try:
        T[:,k] = np.linalg.solve(AA[:,:,k],B[:,k]);
      except np.linalg.linalg.LinAlgError: #AA is signular
        T[0,k] = np.NaN;
        # determine if these segments overlap or are just parallel.
        overlap[k] = np.linalg.cond(np.vstack([dxy1[i[k],:], xy2[j[k],:] - xy1[i[k],:]])) < np.finfo(float).eps;

    # Find where t1 and t2 are between 0 and 1 and return the corresponding x0 and y0 values.
    in_range = (T[0,:] >= 0) & (T[1,:] >= 0) & (T[0,:] <= 1) & (T[1,:] <= 1);
    
    # For overlapping segment pairs the algorithm will return an
    # intersection point that is at the center of the overlapping region.
    if np.any(overlap):
      ia = i[overlap];
      ja = j[overlap];
      # set x0 and y0 to middle of overlapping region.
      T[2,overlap] = (np.max([np.min([x1[ia],x1[ia+1]],axis=0), np.min([x2[ja],x2[ja+1]],axis=0)],axis = 0) +
                      np.min([np.max([x1[ia],x1[ia+1]],axis=0), np.max([x2[ja],x2[ja+1]],axis=0)],axis=0) ) / 2;
      T[3,overlap] = (np.max([np.min([y1[ia],y1[ia+1]],axis=0), np.min([y2[ja],y2[ja+1]],axis=0)],axis = 0) +
                      np.min([np.max([y1[ia],y1[ia+1]],axis=0), np.max([y2[ja],y2[ja+1]],axis=0)],axis=0) ) / 2;              
      selected = in_range | overlap;
    else:
      selected = in_range;
  
    xy0 = T[2:4,selected].T;
    
    # Remove duplicate intersection points.  
    xy0c = np.ascontiguousarray(xy0).view(np.dtype((np.void, xy0.dtype.itemsize * xy0.shape[1])))
    _, index = np.unique(xy0c, return_index=True);
    xy0 = xy0[index];
  
    sel_index= np.where(selected)[0];
    sel = sel_index[index];
    
    return xy0, i[sel], j[sel], T[0,sel], T[1,sel];
  
  else: # non-robust option
    for k in range(n):
      [L,U] = lu(AA[:,:,k]);
      T[:,k] = np.linalg.solve(U, np.linalg.solve(L, B[:,k]));
    
    # Find where t1 and t2 are between 0 and 1 and return the corresponding
    # x0 and y0 values.
    in_range = (T[0,:] >= 0) & (T[1,:] >= 0) & (T[0,:] < 1) & (T[1,:] < 1);
    
    xy0 = T[2:4,in_range].T;
    
    return xy0, i[sel], j[sel], T[0,sel], T[1,sel];


def test():
  import numpy as np
  import matplotlib.pyplot as plt;
  import interpolation.intersections as ii;
  reload(ii)
  
  s = 2 * np.pi* np.linspace(0,1,150);
  xy1 = np.vstack([np.cos(s), np.sin(2*s)]).T;
  xy2 = np.vstack([0.5 * s, 0.2* s]).T + [-2,-0.5];
  
  xy0,i,j,di,dj = ii.curve_intersections_discrete(xy1, xy2);
  
  plt.figure(1); plt.clf();
  plt.plot(xy1[:,0], xy1[:,1]);
  plt.plot(xy2[:,0], xy2[:,1]);
  plt.scatter(xy0[:,0], xy0[:,1], c = 'm', s = 40);
  plt.axis('equal')
  
  
  # non robust  
  xy0,i,j,di,dj = ii.curve_intersections_discrete(xy1, xy2, robust = False);
  
  plt.figure(1); plt.clf();
  plt.plot(xy1[:,0], xy1[:,1]);
  plt.plot(xy2[:,0], xy2[:,1]);
  plt.scatter(xy0[:,0], xy0[:,1], c = 'm', s = 40);
  plt.axis('equal')
  

if __name__ == "__main__":
  test();
