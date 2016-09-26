# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 20:42:17 2016

@author: ckirst
"""

class C(object):
    def __init__(self):
        self._x = None

    @property
    def x(self):
        """I'm the 'x' property."""
        if self._x is None:
          return 100;
        return self._x

    @x.setter
    def x(self, value):
        print 'setting'
        self._x = value

    @x.deleter
    def x(self):
        del self._x
        
        

c = C();
c.x

c.x = 10

c.x