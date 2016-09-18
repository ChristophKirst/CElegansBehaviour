# -*- coding: utf-8 -*-
"""
Simple Memorization decorator

Acts as a cache
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'


def memorize(f):
    """ Memorization decorator for functions taking one or more arguments. """
    
    class memodict(dict):
        def __init__(self, f):
            self.f = f
        def __call__(self, *args):
            return self[args]
        def __missing__(self, key):
            ret = self[key] = self.f(*key)
            return ret 
    
    return memodict(f)

