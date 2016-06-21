"""
Base Data Class for C-elegans Worm Data


"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np

import experiment as exp
import analysis

class WormData(analysis.Data):
  """Base Class to handle worm data"""
  
  def __init__(self, name = None, strain = 'n2', dtype = 'xy', wid = 0, stage = all, 
               valid_only = False, replace_invalid = None, memmap = 'r'):
    super(self.__class__, self).__init__(name = name);
    
    self.strain = strain;
    self.dtype = dtype;
    self.wid = wid;
    self.stage = stage;
    self.valid_only = valid_only;
    self.replace_invalid = replace_invalid;
    self.memmap = memmap;
  
  def data(self):
    """Returns worm data as array"""
    return exp.load(strain = self.strain, dtype = self.dtype, wid = self.wid, stage = self.stage, label = self.label, 
                    valid_only = self.valid_only, replace_invalid = self.replace_invalid, memmap = None);
  
  def load(self):
    """Returns worm data as memmap or array"""
    return exp.load(strain = self.strain,  dtype = self.dtype, wid = self.wid, stage = self.stage, label = self.label, 
                    valid_only = self.valid_only, replace_invalid = self.replace_invalid, memmap = self.memmap);
  
  def tag(self):
    """Tag for this data"""
    tag = super(self.__class__, self).tag();
    
    tag = analysis.tag_join(tag, analysis.stra(self.strain));
    
    tag = analysis.tag_join(tag, analysis.stra(self.dtype));
    
    tag = analysis.tag_join(tag, 'w=%s' % analysis.stra(self.wid));   
    
    tag = analysis.tag_join(tag, 's=%s' % analysis.stra(self.stage));
    
    #tag = analysis.tag_join(tag, 'l=%s' % analysis.stra(self.label));    

    return tag;

  
  def __str__(self):
    info = tuple([analysis.stra(x) for x in [self.strain, self.wid, self.stage, self.label]]);
    return "WormData %s w%s s%s l%s" % info;
 
  def __repr__(self):
    return self.__str__();
     
  def length(self):
    """Length of data points in each entry"""
    mmap = self.memmap;
    self.memmap = 'r';
    data = self.load();
    self.memmap = mmap;
    return data.shape[0];
