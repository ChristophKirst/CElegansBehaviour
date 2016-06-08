# -*- coding: utf-8 -*-
"""
Tagged Analysis

Module to handle reading and writing of analysis results to disk
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import types
from collections import OrderedDict
import inspect
import numpy

from settings import datadir

def dictString(head = None, **args):
    """Convert dictionary in to a formatted string
    
    Arguments:
        head (str or None): prefix of each line
        **args: the parameter values as key=value arguments
    
    Returns:
        str or None: a formated string with parameter info
    """
    if head is None:
        head = '';
    else:
        head = head + '_';
        
    keys = args.keys();
    vals = args.values();
    
    s = [head + keys[i] + '=' + str(vals[i]) + '_' for i in range(len(keys))];
    s = ''.join(s);
    s = s[:-1];
    
    return s;
        
        
def dictJoin(*args):
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


def isFile(source):
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


def tagname(data = None, analysis = None, tags = None, **kwargs):
  """Generates tile tag from worm data sepcification and analysis tags"""

  #data tag
  if data is not None:
    tn = data.tagname() + '_';
  else:
    tn = '';
  
  #analysis tag
  defargs = dict();
  if analysis is not None:
    if isinstance(analysis, types.FunctionType):
      a = inspect.getargspec(analysis);
      defargs = OrderedDict(zip(a.args[-len(a.defaults):],a.defaults));
      analysis = analysis.__name__;
    
    tn = tn + analysis + '_';
    
  if tags is not None:
    tn = tn + dictString(dictJoin(defargs, tags, kwargs)) + '_';
  else:
    tn = tn + dictString(dictJoin(defargs, kwargs)) + '_';\
  
  return tn[:-1];

 
def tagfilename(wormdata = None, analysis = None, tags = None, datadir = datadir, **kwargs):
  tn = tagname(wormdata = wormdata, analysis = analysis, tags = tags, **kwargs);
  return os.path.join(datadir, tn);


def getAnalysis(analysis, wormdata, tags = None, redo = False, overwrite = True, memmap = 'r+', **kwargs):
  tn = tagfilename(wormdata = wormdata, analysis = analysis, tags = tags, **kwargs);
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