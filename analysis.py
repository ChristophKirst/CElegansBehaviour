# -*- coding: utf-8 -*-
"""
Analysis module

Module to manage reading and writing of analysis results to disk using tags
This module allows to analyze data in a systematic and parallel way, storing
the results in specific tagged files that are accessible as memmory maps

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
  
  def run(self, data = None, *args, **kwargs):
    """Run the analysis"""
    param = dict_join(self.parameter, kwargs);
    return function(data, *args, **param);
    
  def __call__(self, data = None, *args, **kwargs):
    return self.run(data, *args, **kwargs);
    



############################################################################
### Tags
############################################################################    

def stra(s):
  """Convert values to string including special built in symbols"""
  if s is all:
    return 'all'
  else:
    return str(s);

def tag_join(tag, tag2, join = '_'):
  """Joins two tag expressions"""
  if tag != '':
    if tag2 != '':
      tag = tag + join + tag2;
  else:
    if tag2 != '':
      tag = tag2;
  
  return tag;


def dict_to_string(parameter = None, head = None):
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
    
    if isinstance(parameter, dict):
      keys = parameter.keys();
      vals = parameter.values();
    
      for i in range(len(keys)):
        tag = tag_join(tag, keys[i] + '=' + stra(vals[i]));
    
    return tag;
        
        
def dict_join(*args):
    """Joins dictionaries in a consitent way
    
    For multiple occurences of a key the  value is defined by the last occurence of the key : value pair.
    
    Arguments:
        *args: list of parameter dictonaries
    
    Returns:
        dict: the joined dictionary
    """
    
    n = len(args);
    
    keys = [];
    values = [];
    for i in range(n):
      if isinstance(args[i], dict):
        ks = args[i].keys();
        values = values + [args[i][k] for k in ks if k not in keys];
        keys   = keys + [k for k in ks if k not in keys];
    
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
      tagpar = dict_to_string(dict_join(defparam, kwargs)) + '_';
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


def analyze(data = None, analysis = None, parameter = None, tags = None, redo = False, overwrite = True, memmap = 'r+', **kwargs):
  """Analyze the data and save under a tagged file name
  
  Arguments:
    data (object, array, None): data to analyze
    analysis (object, function, None): analysis function
    parameter (dict or None): analysis parameter
    tags (dict, str, or None): additional tag specifications
    redo (bool): if True redo analysis even if results exists in a file
    overwrite (bool): if True overwrite tagged file
    memmap (str or None): return result as memmap, if 't' return tagged file name
    **kwargs: additional parameter for the analysis overwriting parameter
  
  Returns:
    array or memmap or file name: analysis results
  
  """
  
  tn = tagfile(data = data, analysis = analysis, parameter = parameter, tags = tags, **kwargs);
  isfile = isFile(tn);
  
  if not redo and isfile: # load tagged file
    if memmap == 't' or memmap == 'tag':
      return tn;
    else:
      return numpy.load(tn, mmap_mode = memmap);
  
  else: # do analysis
    param = dict_join(parameter, kwargs);
    
    result = analysis(data, **param);
    
    if overwrite or ~isfile:
      numpy.save(tn, result);
      
      if memmap == 't' or memmap == 'tag':
        return tn;
      elif memmap is not None:
        return numpy.load(tn, mmap_mode = memmap);
      else:
        return result;  
    else:
      return result;


        
def load(data = None, analysis = None, parameter = None, tags = None, memmap = 'r+', **kwargs):
  """Load results from an analysis
  
  Arguments:  
    data (object, array, None): data to analyze  
    analysis (object, function, None): analysis function
    parameter (dict or None): analysis parameter
    tags (dict, str, or None): additional tag specifications
    memmap (str or None): return result as memmap, if 't' return tagged file name
    **kwargs: additional parameter for the analysis overwriting parameter
  
  Returns:
    array or memmap: analysis results
  """
  tn = tagfilename(wormdata = wormdata, analysis = analysis, tags = tags, **kwargs);
  if isFile(tn):
    if memmap == 't' or memmap == 'tag':
      return tn;
    else:
      return numpy.load(tn, mmap_mode = mmap_mode);
  else:
    raise RuntimeError('cannot find analysis %s' % tn);



def save(result, data = None, analysis = None, parameter = None, tags = None, overwrite = True, memmap = 't', **kwargs):
  """Save results under tagged file corresponding to analysis and data
  
  Arguments:  
    result (array): data to save
    data (object, array or None): data supposedly being analyzed
    analysis (object, function, None): analysis function
    parameter (dict or None): analysis parameter
    tags (dict, str, or None): additional tag specifications
    **kwargs: additional parameter for the analysis overwriting parameter
  
  Returns:
    str, memmap or array: tagged filename that stores the result, result array or memmap
  """
  
  tn = tagfile(data = data, analysis = analysis, parameter = parameter, tags = tags, **kwargs);

  if not overwrite and isFile(tn):
    raise RuntimeWarning('Tagged file %s exists, results not saved to the file as overwrite=False' % tn);
    return tn;
  
  else:
    numpy.save(tn, result);
  
    if memmap == 't' or memmap == 'tag':
      return tn;
    elif memmap is not None:
      return numpy.load(tn, mmap_mode = memmap);
    else:
      return result;  



if __name__ == "__main__":
  #some tests
  pass
