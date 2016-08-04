"""
Module Utils

Utility routines
"""

__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'

import numpy as np
import numbers

eps = np.finfo(float).eps
"""Epsilon small float"""

def isnumber(x):
  """Checks if argument is a number"""
  return isinstance(x, numbers.Number);

# Print iterations progress
def print_progress (iteration, total, prefix = '', postfix = '', decimals = 2, bar_length = 100, out = None):
    """Print progress of an iteration
    
    Arguments:
        iteration (int): current iteration 
        total (int): total iterations 
        prefix (str): prefix string 
        postfix (str): postfix string
        decimals (int): number of decimals in percent complete 
        bar_length (int): character length of bar 
        out (stream): a stream to write to
    """
    
    filled   = int(round(bar_length * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar      = '#' * filled + '-' * (bar_length - filled)
    
    if out is None:
      print '%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', postfix)
      if iteration == total:
        print("\n")
      
    else:    
      out.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', postfix)),
      out.flush()
      
      if iteration == total:
        out.write("\n");
        out.flush();
        
      #sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', postfix)),
      #sys.stdout.flush()

