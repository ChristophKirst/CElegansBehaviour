# -*- coding: utf-8 -*-
"""
Analysis module

Module to handle reading and writing of analysis results to disk using tags
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import types
from collections import OrderedDict
import inspect
import numpy

import experiment as exp


############################################################################
### Base Classes
############################################################################    


class Data:
  """Data base class"""
  
  def __init__(self, name = None, data = None):
    """Constructor for Data """
    self.name = name;
    self.data = data;
  
  def tag(self):
    """Tag for this data"""
    if self.name is None:
      return '';
    else:
      return self.name;
      
  def data(self):
    """Return data"""
    return self.data;
    

class Analysis:
  """Analysis base class"""
  
  def __init__(self, name = None, function = None, parameter = None, with_default_parameter = True):
    self.name = name;
    self.function = function;
    self.parameter = parameter;
    self.with_default_parameter = with_default_parameter;
  
  def tag(self, **kwargs):
    """Tag for this analysis"""
    if isinstance(name, basestring):
      tag = name;
    else:
      tag = '';
      
    tagfunc = tag_from_function(function = self.function,  with_default_parameter = self.with_default_parameter, parameter = self.parameter, **kwargs);
    
    return tag_join(tag, tagfunc);

     
  def run(self, data = None):
    """Run the analysis"""
    



############################################################################
### Tags
############################################################################    

def tag_join(tag, tag2, join = '_'):
  """Joins two tag expressions"""
  if tag != '':
    if tag2 != '':
      tag = tag + join + tag2;
  else:
    if tag2 != '':
      tag = tag2;
  
  return tag;

def dict_to_string(head = None, **args):
    """Convert dictionary in to a formatted string
    
    Arguments:
        head (str or None): prefix of each line
        **args: the parameter values as key=value arguments
    
    Returns:
        str or None: a formated string with parameter info
    """
    if head is None:
        tag = '';
    else:
        tag = head;
        
    keys = args.keys();
    vals = args.values();
    
    for i in range(len(keys)):
      tag = tag_join(tag, keys[i] + '=' + str(vals[i]));
    
    return s;
        
        
def dict_join(*args):
    """Joins dictionaries in a consitent way
    
    For multiple occurences of a key the  value is defined by the last key : value pair.
    
    Arguments:
        *args: list of parameter dictonaries
    
    Returns:
        dict: the joined dictionary
    """
    
    keyList = [x.keys() for x in args];
    n = len(args);
    
    keys = [];
    values = [];
    for i in range(n):
        values = values + [args[i][k] for k in keyList[i] if k not in keys];
        keys   = keys + [k for k in keyList[i] if k 
        not in keys];
    
    return OrderedDict(zip(keys, values));


def is_file(source):
    """Checks if filename is a real file, returns false if it is directory or regular expression
    
    Arguments:
        source (str): source file name
        
    Returns:
        bool: true if source is a real file   
    """
    
    if not isinstance(source, basestring):
        return False;
  
    if os.path.exists(source):
        if os.path.isdir(source):
            return False;
        else:
            return True;
    else:
        return False;



def tag_from_function(function, with_default_parameter = True, parameter = None, **kwargs):
  """Generate tag from a function with parameters
  
  Arguments:
      function (function):  the function to generate tag for
      with_default_parameter (bool): add default parameter of function to tag
      parameter (dict): add these parameter to the tag
      **kwargs: additional key value pairs to add 

  Returns:
      string: tag
  """
  
  if isinstance(function, types.FunctionType):
    a = inspect.getargspec(function);
    if with_default_parameter:
      defparam = OrderedDict(zip(a.args[-len(a.defaults):],a.defaults));
    else:
      defparam = dict();
    name = function.__name__;
  elif isinstance(function, basestring):
    name = function;
    defparam = dict();
  else:
    name = '';
    defparam = dict();
   
  if name != '':
    tag = name;
  else:
    tag = '';
  
  if isinstance(parameter, dict):
    tagpar = dict_to_string(dict_join(defparam, parameter, kwargs)) + '_';
  elif isinstance(parameter, basestring) and parameter != '':
    if len(kwargs) > 0 or len(defparam) > 0:
      tagpar = parameter + '_' + dict_to_string(dict_join(defparam, kwargs)) + '_';
    else:
      tagpar = parameter + '_';
  else:
    if len(kwargs) > 0 or len(defparam) > 0:
      tagpar = dict_to_string(dict_join(defargs, kwargs)) + '_';
    else:
      tagpar = '';
  
  return tag_join(tag, tagpar);



def tag(data = None, analysis = None, with_default_parameter = True, parameter = None, tags = None, **kwargs):
  """Generates a tag from data sepcification and analysis tags
  
  Arguments:
    data (object): a class handling data with tagname() routine
    analysis (object): a class handling analysis with tagname() routine
    tags (dict): optional tag values
  
  Returns:
    string: the tag
  """

  #analysis tag (first as this allows for different analysus subfolders)
  if isinstance(analysis, Analysis):
    tag = analysis.tagname();
  elif isinstance(analysis, types.FunctionType) or isinstance(analysis, basestring):
    tag = tag_from_function(analysis, with_default_parameter = with_default_parameter, parameter = parameter, kwargs);
  else:
    tag = '';
    
  #data tag
  if isinstance(data, Data):
    tagdata = data.tagname();
  elif isinstance(data, basestring):
    tagdata = data;
  else:
    tagdata = '';
  
  tag = tag_join(tag, tagdata);
      
  if isinstance(tags, dict):
    tagtags = dict_to_string(tags);
  elif isinstance(tags, basestring):
    tagtags = tags;
  else:
    tagtags = '';
  
  tag = tag_join(tag, tagtags);
  
  return tag;

 
def tagfile(data = None, analysis = None, with_default_parameter = True, parameter = None, tags = None, directory = exp.data_directory, extension = 'npy', **kwargs):
  """Return a file name genereated from the tags"""
  tn = tag(data = data, analysis = analysis, parameter = parameter, with_default_parameter = with_default_parameter, tags = tags, **kwargs);
  if isinstance(extension, basestring) and len(extension) > 0:
    tn = tn + '.' + extension;
  return os.path.join(directory, tn);




############################################################################
### Analysis
############################################################################    


def analyze_data(data = None, analysis = None, parameter = None, tags = None, redo = False, overwrite = True, memmap = 'r+', **kwargs):
  """Analyze the data and save under a tagged file name"""
  tn = tagfile(data = data, analysis = analysis, parameter = parameter, tags = tags, **kwargs);
  if not redo and isFile(tn):
    return numpy.load(tn, mmap_mode = memmap);
  else:
    if tags is None:
      tags = dict();
    tags = dictJoin(tags, kwargs);
    result = analysis(wormdata, **tags);
    if overwrite:
      numpy.save(tn, result);
    
    if memmap is not None:
      return numpy.load(tn, mmap_mode = memmap);
    else:
      return result;
    

def loadAnalysis(analysis, wormdata, tags = None, mmap_mode = 'r+', **kwargs):
  tn = tagfilename(wormdata = wormdata, analysis = analysis, tags = tags, **kwargs);
  if isFile(tn):
    return numpy.load(tn, mmap_mode = mmap_mode);
  else:
    raise RuntimeError('cannot find analysis %s' % tn);


def saveAnaysis(result, analysis, wormdata, tags = None, **kwargs):
  tn = tagfilename(wormdata = wormdata, analysis = analysis, tags = tags, **kwargs);
  if tags is None:
    tags = dict();
  tags = dictJoin(tags, kwargs);
  numpy.save(tn, result);
  return tn;