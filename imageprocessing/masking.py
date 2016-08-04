# -*- coding: utf-8 -*-
"""
Masking Module

Routines for masking and conversions to level sets
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
import scipy.ndimage as nd

eps = np.finfo(float).eps


def mask_to_dist(mask):
  """Returns distance transform on a mask
  
  Arguments:
    mask (array): the mask for which to calculate distance transform
    
  Returns:
    array: distance transform of the mask
  """
  return nd.distance_transform_edt(mask == 0)


def mask_to_phi(mask):
  """Returns distance transform to the contour of the mask
  
  Arguments:
    mask (array): the mask for which to calculate distance transform
  
  Returns:
    array: distance transform to the contour of the mask
  """
  
  phi = mask_to_dist(mask) - mask_to_dist(1 - mask) + np.asarray(mask, dtype = float)/mask.max() - 0.5
  return phi   
    

# Compute curvature
def curvature_from_phi(phi, idx = all):
  """Returns curvature of a level set surface
  
  Arguments:
    phi (array): level set surface
    idx (list or all): indices of pixel to compute curvature for
    
  Returns:
    list: list of curvatures at the speicifed indices
  """
  
  if idx is all:
    idx = range(phi.size);
  
  dimy, dimx = phi.shape
  yx = np.array([np.unravel_index(i, phi.shape) for i in idx])  # subscripts
  y = yx[:, 0]
  x = yx[:, 1]

  # Get subscripts of neighbors
  ym1 = y - 1
  xm1 = x - 1
  yp1 = y + 1
  xp1 = x + 1

  # Bounds checking
  ym1[ym1 < 0] = 0
  xm1[xm1 < 0] = 0
  yp1[yp1 >= dimy] = dimy - 1
  xp1[xp1 >= dimx] = dimx - 1

  # Get indexes for 8 neighbors
  idup = np.ravel_multi_index((yp1, x), phi.shape)
  iddn = np.ravel_multi_index((ym1, x), phi.shape)
  idlt = np.ravel_multi_index((y, xm1), phi.shape)
  idrt = np.ravel_multi_index((y, xp1), phi.shape)
  idul = np.ravel_multi_index((yp1, xm1), phi.shape)
  idur = np.ravel_multi_index((yp1, xp1), phi.shape)
  iddl = np.ravel_multi_index((ym1, xm1), phi.shape)
  iddr = np.ravel_multi_index((ym1, xp1), phi.shape)

  # Get central derivatives of SDF at x,y
  phi_x = -phi.flat[idlt] + phi.flat[idrt]
  phi_y = -phi.flat[iddn] + phi.flat[idup]
  phi_xx = phi.flat[idlt] - 2 * phi.flat[idx] + phi.flat[idrt]
  phi_yy = phi.flat[iddn] - 2 * phi.flat[idx] + phi.flat[idup]
  phi_xy = 0.25 * (- phi.flat[iddl] - phi.flat[idur] +
                   phi.flat[iddr] + phi.flat[idul])
  phi_x2 = phi_x**2
  phi_y2 = phi_y**2

  # Compute curvature (Kappa)
  curvature = ((phi_x2 * phi_yy + phi_y2 * phi_xx - 2 * phi_x * phi_y * phi_xy) /
               (phi_x2 + phi_y2 + eps) ** 1.5) * (phi_x2 + phi_y2) ** 0.5

  return curvature
  
