# -*- coding: utf-8 -*-
"""
Curve module to handle mappings from parameter to curve spaces

This module provides a base class to handle curves and their parameterization
used to capter the pose of worms
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


class Curve:
  """Class providing basic interface to a parameterized curve"""
  
  def __init__(self, *args, **kwargs):
    """Constructor """
    self.nparameter = 0;
    self.parameter = None;
    self.npoints = 0;
    self.points = None;
    self.values = None;
  
  def get_nparameter(self):
    """Number of parameter to describe curve"""
    return self.nparameter;    
    
  def get_parameter(self):
    """Parameter of the curve"""
    return self.parameter;
    
  def get_npoints(self):
    """Number of sample points for the curve values"""
    return self.npoints;
  
  def get_points(self):
    """Coordinate points along which the curve is sampled"""
    return self.points;
  
  def get_values(self, points = None, **kwargs):
    """Returns the values of the curve at the sample points"""
    return None;
  
  def from_values(self, values, points = None, **kwargs):
    """Updates the parameter of the curve given new values"""
    pass
    
  def from_parameter(self, parameter, **kwargs):
    """Updates the parameter of the curve given new parameter"""
    pass
    
#  def to_shape(self, points = None, width = None, with_normals = False):
#    """Returns the boundary curves af the shape with a width and the curve as center line"""
#    return None;
#    
#  def from_shape(self, left, right, points = None, with_width = False):
#    """Determines parameter of the center curve given the left and right boundary curves"""
#    return None;