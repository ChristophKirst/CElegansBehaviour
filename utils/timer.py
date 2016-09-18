# -*- coding: utf-8 -*-
"""
Time decorator

Example:

  >>>from timer import timeit
  >>>@timeit
  >>>def test():
  >>>  return 3+4;
  >>>test()
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


import time

def timeit(method):
  def timed(*args, **kw):
      ts = time.time()
      result = method(*args, **kw)
      te = time.time()
      
      print '%r(%r, %r) took %2.3f ms' % (method.__name__, args, kw, (te-ts) * 1000)
      return result

  return timed
  