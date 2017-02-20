# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 23:53:20 2016

@author: ckirst
"""
import matplotlib.pyplot as plt

## test shapel intersection
from shapely.geometry.polygon import LinearRing, LineString

contour = LinearRing([(0, 0), (2, 0), (2,2), (1,2)]);
line = LineString([(0,0), (3,4)]);

plt.figure(100); plt.clf();
x, y = contour.xy
plt.plot(x, y, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
x,y = line.xy;
plt.plot(x, y, color='b', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

x = line.intersection(contour);

for ob in x:
    x, y = ob.xy
    if len(x) == 1:
        plt.plot(x, y, 'o', color='m', zorder=2)
    else:
        plt.plot(x, y, color='m', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
        
        

# test erosion of polygon
from shapely.geometry.polygon import Polygon


from shapely.geometry import Polygon

p = Polygon([(0, 0), (1, 1), (1, 0)])

x,y = p.boundary.xy
plt.plot(x, y, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


p2 = p.buffer(-0.2);
x,y = p2.boundary.xy
plt.plot(x, y, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)

b = p2.boundary;

for o in b:
  x,y = o.xy
  plt.plot(x, y, color='b', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


import numpy as np
t = np.linspace(0,10,50);
aline = np.vstack([t, np.sin(t)+0.5]).T;
aline[0] = [0,0];
aline[-1] = [10,-0.2];
bline = np.vstack([t, np.sin(t)]).T;

plt.figure(9); plt.clf();
p = Polygon(np.vstack([aline, bline[::-1,:]]));
x,y = p.boundary.xy
plt.plot(x, y, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)
p2 = p.buffer(-0.175);
x,y = p2.boundary.xy
plt.plot(x, y, color='r', alpha=0.7, linewidth=3, solid_capstyle='round', zorder=2)


