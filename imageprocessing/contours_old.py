# -*- coding: utf-8 -*-
"""
Contour Module

Routines for contour detection using basic matplotlib functionality
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import numpy as np
from numpy import ma
import matplotlib._contour as _contour
from matplotlib.path import Path
from scipy.spatial.distance import pdist, squareform

def detect_contour(img, level):
  """Returns list of vertices of contours at a given level
  
  Arguments:
    img (array): the image array
    level (number): the level at which to create the contour
  
  Returns:
    (list of nx2 arrays): list of list of vertices of the different contours
  
  Note:
    The contour detection is based on matplotlib's QuadContourGenerator
  """
  #parameter
  mask = None;
  corner_mask = True;
  nchunk = 0;

  #prepare image data
  z = ma.asarray(img, dtype=np.float64); 
  ny, nx = z.shape;
  x, y = np.meshgrid(np.arange(nx), np.arange(ny));

  #find contour
  contour_generator = _contour.QuadContourGenerator(x, y, z.filled(), mask, corner_mask, nchunk)
  vertices = contour_generator.create_contour(level);
  
  return vertices;


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
