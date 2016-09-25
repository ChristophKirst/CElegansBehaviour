# -*- coding: utf-8 -*-
"""
Curve module to handle mappings from parameter to curve spaces

This module provides a base class to handle curves and their parameterization
used to capture the center line of a worm and its bending 
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


class Curve:
  """Basic interface to a parameterized curve in 1d"""
  
  def __init__(self, *args, **kwargs):
    """Constructor """
    self.nparameter = 0;
    self.parameter = None;
    self.npoints = 0;
    self.points = None;
    self.values = None;
  
  def get_values(self, points = None, **kwargs):
    """Returns the values of the curve at the sample points"""
    return None;
  
  def from_values(self, values, points = None, **kwargs):
    """Updates the parameter of the curve given new values"""
    pass
    
  def from_parameter(self, parameter, **kwargs):
    """Updates the parameter of the curve given new parameter"""
    pass
  
  def integral(self):
    """Returns the integral of the curve along all dimensions"""
    pass
    
  def derivative(self):
    """Returns the derivative of the curve along all dimensions"""
    pass
  

