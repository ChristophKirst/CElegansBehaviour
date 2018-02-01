# -*- coding: utf-8 -*-
"""
Contour Module

Routines for contour detection using basic matplotlib functionality
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
#import matplotlib._contour as _contour
import matplotlib._cntr as _contour


from matplotlib.path import Path
from scipy.spatial.distance import pdist, squareform

import cv2

def detect_contour_old(img, level, with_hierarchy = False):
  """Returns list of vertices of contours at a given level
  
  Arguments:
    img (array): the image array
    level (number): the level at which to create the contour
  
  Returns:
    (list of nx2 arrays): list of list of vertices of the different contours
  
  Note:
    The contour detection is based on matplotlib's QuadContourGenerator
  """

  #img, contours, hierarchy  = cv2.findContours((blur[f_id] < 68).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  #img, contours, hierarchy  = cv2.findContours((blur[f_id] < 65).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

  #res, contours, hierarchy  = cv2.findContours((img >= level).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
  res, contours, hierarchy  = cv2.findContours((img >= level).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
  #res, contours, hierarchy  = cv2.findContours((img >= level).view('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
  contours = [c[:,0,:] for c in contours];  
  #contours = [np.vstack([c, c[[0]]]) for c in contours]; # make periodic
  if with_hierarchy:
    return contours, hierarchy[0]
  else:
    return contours;



def detect_contour(img, level, with_hierarchy = False):
  #z = ma.asarray(img, dtype=np.float64); 
  ny, nx = img.shape;
  x, y = np.meshgrid(np.arange(nx), np.arange(ny));

  #find contour
  #contour_generator = _contour.QuadContourGenerator(x, y, img, None, True, 0)
  contour_generator = _contour.Cntr(x, y, img)
  contours = contour_generator.trace(level);
  contours = contours[:len(contours)/2];
  contours = [np.asarray(c, dtype = 'float32') for c in contours];
  contours = [c[np.hstack([np.linalg.norm(c[1:] - c[:-1], axis = 1)>0,[True]])] for c in contours];
  
  if not with_hierarchy:
    return contours;

  n = len(contours);
  h = np.zeros((n,n), dtype = bool);
  for i,c in enumerate(contours):
    for j,c2 in enumerate(contours):
      if i==j:
        h[i,j] = 0;
      else:
        if inside(c, c2[0]):
          h[i,j] = 1;
        else:
          h[i,j] = 0;
  return contours, h
      



def inside(contour, point): 
    c = np.reshape(contour, (contour.shape[0], 1, contour.shape[1]));
    #print point
    return cv2.pointPolygonTest(c,(point[0], point[1]),False) > 0;


def inside_polygon(vertices, point):
  """Checks if a point is inside a polygon
  
  Arguments:
    vertices (nx2 array): vertices of the polygon
    point (2 array or dx2 array): coordinates of the point
    
  Returns:
    bool: True if point is inside the polygon
  """
  
  p = Path(vertices);
  if point.ndim == 1:
    return p.contains_point(point);
  else:
    return p.contains_points(point);
    
    
def sort_points_to_line(vertices, start = 0):
  """Sorts points to a line by sequentiall connecting nearest points
  
  Arguments:
    vertices (nx2 array): vertices of the line
    start (int): start index
  
  Returns:
    nx2 array: sorted points
  """
  
  d = squareform(pdist(vertices));
  
  i = start;
  n = vertices.shape[0];
  uidx = np.ones(n, dtype = bool);
  uidx[i] = False;
  sidx = [i];  
    
  while np.sum(uidx) > 0:
    i = np.argmin(d[i][uidx]);
    i = np.where(uidx)[0][i];
    sidx.append(i);
    uidx[i] = False;
  
  return vertices[sidx];
