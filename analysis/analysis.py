# -*- coding: utf-8 -*-
"""
Analysis module

Module to manage reading and writing of analysis results to disk using tags.

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
    """Constructor for Analysis
    
    Arguments:
      name (str or None): analysis name
      function (function): analysis function
      parameter (dict): parameter settings passed to analysis function
      with_default_parameter (bool): if True add defaulta parameter of the analysis function to the tag
    """
    self.name = name;
    self.function = function;
    self.parameter = parameter;
    self.with_default_parameter = with_default_parameter;
  
  
  def tag(self, **kwargs):
    """Tag for this analysis
    
    See also:
      :func:`tag_from_function`   
    """      
    tagfunc = tag_from_function(function = self.function, parameter = self.parameter, with_name = False, 
                                with_default_parameter = self.with_default_parameter, **kwargs);
    return tag_join(self.name, tagfunc);
  
  
  def tag_filename(self, tags = None, directory = exp.analysis_directory, extension = 'npy', prefix = None, **kwargs):
    """Tag for this analysis
    
    See also:
      :func:`tag_filename`    
    """
    return tag_filename(analysis = self, parameter = self.parameter, 
                        with_default_parameter=self.with_default_parameter, tags = tags, 
                        directory = directory, extension = extension, prefix = prefix **kwargs)
  
  
  def run(self, data = None, *args, **kwargs):
    """Run the analysis"""
    param = dict_join(self.parameter, kwargs);
    return self.function(data, *args, **param);
  
  
  def __call__(self, data = None, *args, **kwargs):
    """Run the analysis"""
    return self.run(data, *args, **kwargs);
  
  
  def analyze(self, data = None, tags = None, redo = False, overwrite = True, memmap = 'r+', **kwargs):
    """Run the analysis and save in tagged file
    
    See also:
      :func:`analyze`
    """
    return analyze(data = data, analysis = self, parameter = self.parameter, 
                   with_default_parameter = self.with_default_parameter, tags = tags,
                   redo = redo, overwrite = overwrite, memmap = memmap, **kwargs)
  

    



############################################################################
### Tags
############################################################################    

def stra(s):
  """Convert values to string including special built in symbols
  
  Arguments:
    s (object): object to convert to string
  
  Returns:
    str: string version of the object
  """
  
  if s is all:
    return 'all'
  else:
    return str(s);

def stra_inverse(s):
  """Convert a string to the most likely object it was converted from
  
  Arguments:
    s (str): string to convert back to an object
  
  Returns:
    object: the object corresponding to the string
  """
  
  if s == 'all':
    val = all;
  elif s == 'None':
    val = None;
  else:
      try:
        val = int(s);
      except:
        try:
          val = float(s);
        except:
          val = s;
  return val;


def tag_join(tag1, tag2, join = '_'):
  """Joins two tag expressions
  
  Arguments:
    tag* (str): tags
    join (str): join symbol
  
  Returns:
    str: joined tag string
  """
  if not isinstance(tag1, basestring):
    tag1 = '';
  if not isinstance(tag2, basestring):
    tag2 = '';
  
  if tag1 != '':
    if tag2 != '':
      tag1 = tag1 + join + tag2;
  else:
    if tag2 != '':
      tag1 = tag2;
  
  return tag1;


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

  
def dict_from_string(string, head = None):
  """Convert formatted string to dictionary 
  
  Arguments:
    head (str or None): prefix of each line
    string (str): string to parse back into dictionary
  
  Returns:
    dict: dictionary obtained from the string
  """

  ls = len(string);    
  if isinstance(head, basestring) and head != '':
    lh = len(head);
    if ls > lh+1 and string[:lh] == head and string[lh] == '_':
      string = string[(lh+1):];
      ls = len(string);  
    else:
      raise RuntimeError('head of string "%s" does not match required head "%s"' % (string[:lh], head));
  
  d = {};
  i = 0;
  while i < ls:
    j = i;
    while j < ls and string[j] != '=':
      j+=1;
    if j==ls:
      raise RuntimeError('incomplete parameter=value pair!');
    
    name = string[i:j];
    k = j+1;
    while k < ls and string[k] != '_':
      k+=1;
    val = string[(j+1):k];
    
    d[name] = stra_inverse(val);
    i = k+1;
  
  return d;
      
        
def dict_join(*args):
  """Joins dictionaries in a consitent way
  
  For multiple occurences of a key the  value is defined by the last occurence of the key : value pair.
  
  Arguments:
    *args: list of parameter dictonaries
  
  Returns:
    dict: the joined dictionary
  """
  overwrite = True;
  n = len(args);
      
  if overwrite:
    d = OrderedDict();
    for i in range(n):
      if isinstance(args[i], dict):
        keys = d.keys();
        for k,v in args[i].items():
            d[k] = v;
    return d;
  
  else:  
    keys = [];
    vals = [];
    for i in range(n):
      if isinstance(args[i], dict):
        ks = args[i].keys();
        keys = keys + [k for k in ks if k not in keys];
        vals = vals + [args[i][k] for k in ks if k not in keys];
    return OrderedDict(zip(keys, vals));


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



def tag_from_function(function, with_name = True, with_default_parameter = True, parameter = None, **kwargs):
  """Generate tag from a function with parameters
  
  Arguments:
    function (function):  the function to generate tag for
    with_default_parameter (bool): add default parameter of function to tag
    with_name (bool): include function name
    parameter (dict): add these parameter to the tag
    **kwargs: additional parameter or overwrites

  Returns:
    string: tag
  """
  
  if isinstance(function, types.FunctionType):
    a = inspect.getargspec(function);
    if with_default_parameter:
      defparam = OrderedDict(zip(a.args[-len(a.defaults):],a.defaults));
    else:
      defparam = dict();
    if with_name:
      name = function.__name__;
    else:
      name = '';
  
  elif isinstance(function, basestring):
    name = function;
    defparam = dict();
  else:
    name = '';
    defparam = dict();
   
  tag = name
  
  if isinstance(parameter, dict):
    tagpar = dict_to_string(dict_join(defparam, parameter, kwargs));
  elif isinstance(parameter, basestring):
    if len(kwargs) > 0 or len(defparam) > 0:
      tagpar = tag_join(parameter, dict_to_string(dict_join(defparam, kwargs)));
    else:
      tagpar = parameter;
  else:
    if len(kwargs) > 0 or len(defparam) > 0:
      tagpar = dict_to_string(dict_join(defparam, kwargs));
    else:
      tagpar = '';
  
  return tag_join(tag, tagpar);



def tag(data = None, analysis = None, parameter = None,  tags = None,
        with_default_parameter = True, prefix = None, postfix = None, **kwargs):
  """Generates a tag from data specification and analysis tags
  
  Arguments:
    data (object): a class handling data with tagname() routine
    analysis (object): a class handling analysis with tagname() routine
    parameter (dict): specific analysis parameter to use     
    tags (dict): optional tag values
    with_default_parameter (bool): if True add default analysis parameter
    prefix (str or None): an additional prefix
    postfix (str or None): an additional postfix
    **kwargs: additional parameter for the analysis or parameter overwrites
  
  Returns:
    string: the tag
  """

  #analysis tag (first as this allows for different analysus subfolders)
  if isinstance(analysis, Analysis):
    tag = analysis.tag();
  elif isinstance(analysis, types.FunctionType) or isinstance(analysis, basestring):
    tag = tag_from_function(analysis, parameter = parameter,
                            with_default_parameter = with_default_parameter, **kwargs);
  else:
    tag = '';
  
  #data tag
  if isinstance(data, Data):
    tagdata = data.tag();
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
    
  return tag_join(tag_join(prefix, tag), postfix);

 
def tag_filename(data = None, analysis = None, with_default_parameter = True, parameter = None, 
                 tags = None, directory = exp.analysis_directory, extension = 'npy', **kwargs):
  """Returns a file name generated from the tags
  
  Arguments:
    data (object): a class handling data with tagname() routine
    analysis (object): a class handling analysis with tagname() routine
    with_default_parameter (bool): if True add default analysis parameter
    parameter (dict): specific analysis parameter to use     
    tags (dict): optional tag values
    directory (str): base directory
    extension (str): file extension
    **kwargs: additional parameter for the analysis or parameter overwrites
  
  Returns:
    string: tagged file name
  """  
  
  tn = tag(data = data, analysis = analysis, parameter = parameter, 
           with_default_parameter = with_default_parameter, tags = tags, **kwargs);
  
  if isinstance(extension, basestring) and len(extension) > 0:
    tn = tn + '.' + extension;
  
  if isinstance(directory, basestring):
    return os.path.join(directory, tn);
  else:
    return tn;




############################################################################
### Analysis
############################################################################    


def analyze(data = None, analysis = None, parameter = None, with_default_parameter = True, tags = None, 
            redo = False, overwrite = True, memmap = 'r+', **kwargs):
  """Analyze the data and save under a tagged file name
  
  Arguments:
    data (object, array, None): data to analyze
    analysis (object, function, None): analysis function
    parameter (dict or None): analysis parameter
    tags (dict, str, or None): additional tag specifications
    redo (bool): if True redo analysis even if results exists in a file
    overwrite (bool): if True overwrite tagged file
    memmap (str or None): return result as memmap, if 't' return tagged file name
    **kwargs: additional parameter for the analysis or parameter overwrites
  
  Returns:
    array or memmap or file name: analysis results
  
  """
  
  tn = tag_filename(data = data, analysis = analysis, parameter = parameter, 
                    with_default_parameter =  with_default_parameter, tags = tags, **kwargs);
  isfile = is_file(tn);
  
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


        
def load(data = None, analysis = None, parameter = None, with_default_parameter = True, 
         tags = None, memmap = 'r+', **kwargs):
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
  tn = tag_filename(data = data, analysis = analysis, parameter = None, 
                    with_default_parameter =  with_default_parameter, tags = tags, **kwargs);
  
  if is_file(tn):
    if memmap == 't' or memmap == 'tag':
      return tn;
    else:
      return numpy.load(tn, mmap_mode = memmap);
  else:
    raise RuntimeError('cannot find analysis %s' % tn);



def save(result, data = None, analysis = None, parameter = None, with_default_parameter = True, 
         tags = None, overwrite = True, memmap = 't', **kwargs):
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
  
  tn = tag_filename(data = data, analysis = analysis, parameter = parameter,
                    with_default_parameter =  with_default_parameter, tags = tags, **kwargs);

  if not overwrite and is_file(tn):
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


def test():
  import analysis.analysis as als;
  
  ## tag strings
  reload(als);
  als.tag_join('a', 'b')
  
  d = {'a' : 100, 'b' : 'abc'};
  s = als.dict_to_string(d, head = 'hello');
  print '%s as dict is %s' % (s, str(als.dict_from_string(s, head = 'hello')));

  als.dict_join({'a':100, 'b' : 'abc'},{'b' : 20, 'c': 'hello'})
 

  ## tags
  def add(x,y, offset = 0):
    return x + y + offset;
  a = als.Analysis(name = None, function = add)
  
  print a.tag()
  print a(5,6, offset = 10)
  
  reload(als)
  import numpy as np
  def add(x, offset = 0):
    return x[:,0] + x[:,1] + offset;
  a = als.Analysis(name = 'test', function = add)
  
  print a.tag()
  x = np.random.rand(5,2);  
  print add(x) - a(x)
  print als.tag(analysis = a)  
  
  print als.tag_filename(analysis=a)
  print a.tag_filename()
  a.analyze(data = x)

  print als.load(analysis = a)
    
  import os
  os.remove(a.tag_filename())


if __name__ == "__main__":
  test();
