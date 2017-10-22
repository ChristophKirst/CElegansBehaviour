

#%% Roaming Dwelling example

import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;

plt.rcParams['axes.facecolor'] = 'w'
plt.rcParams['figure.facecolor'] = 'w'
 

import analysis.video as v;
reload(v);


#%% Gui

reload(v)
td = exp.load(wid = 35, dtype = 'roam');
td[np.isnan(td)] = 0.5;

#td = np.random.rand(td.shape[0]);

dt = 3;
g = v.WormGui(strain = 'n2', wid = 35, time = 400000, trajectory_delta=dt, trajectory_length = 3*60*60 / dt, trajectory_data=td, trajectory_cmap=plt.cm.bwr, trajectory_vmin=0, trajectory_vmax=1);


#%%

#t = 381500;

td = exp.load(wid = 35, dtype = 'roam');
td[np.isnan(td)] = 0;

np.mean(td[(t-3*60*60 ):t]) 


plt.figure(11); plt.clf();
plt.plot(td[(t-3*60*60 ):t])

#%%  

reload(v)

t = 391097;
t = 518922;

t = 361000;
#t = 490000;
#t = 387414;
t = 507927
t = 490000

t = 527020;
t = 528020;

t = 539020;

t = 385100;

t = 382100;
#t = 381500;

t = 373000;

t = 515000;

t = 535000;

#t = 560000;

#t = 615000;

dt = 3;
th = 3*60*60 / dt; 

fig = plt.figure(1, figsize = (3,3), dpi = 300, facecolor = 'w'); plt.clf();
ax = plt.subplot(1,1,1);

td = exp.load(wid = 35, dtype = 'roam');
td[np.isnan(td)] = 0.0;
#td = 8 - td * 8;
td = td * 8;

stage_colors = ['#8b4726', '#231f20', '#2e3192', '#458b00', '#918f8f', '#918f8f'];

def stage_indicator_position(time, time_max, time_min, xlim, ylim, time_end = None):
  sizefac= 0.9;
  plot_range = xlim[1] - xlim[0];
  off = xlim[0] + (1 - sizefac) * plot_range / 2.0;
  x = sizefac * (time - time_min) / (time_max - time_min) * plot_range + off;
  y = -15 + ylim[1];
  if time_end is None:
    return [x,y];
  else:
    h = 10;
    w = sizefac * (time_end - time) / (time_max - time_min) * plot_range;
    return x,y,w,h;

tjs = 2.0;

p = v.WormPlot(ax, strain = 'n2', wid = 35, time = t,   
           image = False,
           
           plate_color = 'black',
           plate_linewidth = 0.5,

           trajectory = True,
           trajectory_length=th, trajectory_delta= dt, trajectory_size=tjs,
           trajectory_cmap=plt.cm.bwr, trajectory_vmin=0, trajectory_vmax=8,
           trajectory_data = td,
           stage_stamp_font_size=2.8,
           stage_stamp_colors = stage_colors,
           
           stage_indicator = False,
           stage_indicator_label_font_size=6,
           stage_indicator_colors = stage_colors,
           stage_indicator_position = stage_indicator_position,
           stage_indicator_label_offset = 60,
           
           stage_stamp = False,
           stage_stamp_position = None,   
           
           time_stamp = False,
           time_stamp_font_size=12,
           time_stamp_offset= 34200,           
           
           border = [[15,15],[15,15]]);       


x, y, c = p.generate_trajectory(t);

#%%
ids = c == 0;
plt.scatter(x[ids], y[ids], color = plt.cm.bwr(0.0), edgecolor = 'none',  s = tjs+0.0);

fig.canvas.draw()
fig.show()

#%%
fig.savefig('worm_35_roaming_dwelling_%f.pdf' % tjs)