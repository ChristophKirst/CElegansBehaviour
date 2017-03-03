# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 01:28:27 2017

@author: ckirst
"""

import os
import numpy as np
import analysis.experiment as exp;
import matplotlib.pyplot as plt;


import analysis.video as v;
reload(v);

plt.figure(1); plt.clf();
plt.xticks([])
plt.yticks([])
v.plot_worm(strain = 'n2', wid = 0, t = 600000);


### Make a Movie
reload(v);

plt.figure(1, figsize = (2,2), dpi = 400); plt.clf();
plt.xticks([]); plt.yticks([]);
plt.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0)

wid = 38;

fname = os.path.join('/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Movies/', 'test_wid=%d.mp4' % (wid,));

stimes = exp.load_stage_times(wid = wid);
t0 = stimes[0];
t1 = stimes[-2];

t0 = 552061;
t0 = 34200;
t1 = t0 + 520000;
dt = 250;
sp = 10;

hist = 120*60 / dt * sp;


v.animate_frames(strain = 'n2', wid = wid, 
                 times = range(t0, t1, dt), pause = None, 
                 history_delta = dt/sp, history = hist, 
                 features = [], feature_indicator= None,
                 time_data = None, time_size = 10, time_cmap = plt.cm.Reds,
                 border = 20, 
                 save = None, fps = 10, dpi = 600,
                 time_stamp = True)




t0 = 34200+5000;
t0 = 70000 - 1000;
t0 = 180000
t0 = 263000;


t1 = t0 + 520000;
dt = 1;
sp = 1;

hist = 60 / dt * sp;


v.animate_frames(strain = 'n2', wid = wid, 
                 times = range(t0, t1, dt), pause = None, 
                 history_delta = dt/sp, history = hist, 
                 features = [], feature_indicator= None,
                 time_data = 'roam', time_size = 10, time_cmap = plt.cm.bwr,
                 border = 20, 
                 save = None, fps = 10, dpi = 600,
                 time_stamp = True)



#os.system("mencoder 'mf://movie_wid=%d_*.png' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o worm_%d.mpg" % (wid,wid))
#import glob
#files = np.sort(np.unique(glob.glob('movie_wid=%d_*.png' % wid)))
#for f in files:
#  os.remove(f)




import numpy as np
import os
import analysis.experiment as exp
import analysis.plot as fplt
import matplotlib.pyplot as plt  

import analysis.video as vd
reload(vd)

from analysis.features import moving_average
  
dat_v = exp.load(wid =0, strain = 'n2', dtype = 'speed')    
dat_v = moving_average(dat_v, bin_size=30);

dat_r = exp.load(wid =0, strain = 'n2', dtype = 'rotation')    
dat_r = moving_average(dat_r, bin_size=30);

w = vd.worm_feature_gui(strain = 'n2', wid = 0, history = 500, features = [('speed', dat_v), ('rotation', dat_r), 'roam'], levels = (60,90), background = None, feature_indicator="roam");











plt.figure(15); plt.clf();

roam = exp.load(wid = 0, dtype ='roam');
xy = exp.load(wid = 0, dtype = 'xy');

t0 = 550000;
t1 = t0 + 10000;
sh = 0;


plt.plot(xy[t0:t1:3, 0], xy[t0:t1:3,1], c = 'gray')
plt.scatter(xy[t0:t1:3, 0], xy[t0:t1:3,1], c = roam[t0+sh:t1+sh:3], s = 20)













xy = exp.load(wid = 0, dtype = 'xy')
t0 = 591963;
t0 = 600000;
t0 = 600702;
t0 = 600499;

t0 = 517694; # shift right -> 

print('>')
print(xy[t0])
print(xy[t0+1])

t0 = 517742; # shfit down v

print('v')
print(xy[t0])
print(xy[t0+1])





w1 = exp.load_img(t = t0);
w2 = exp.load_img(t = t0+1);
w3 = exp.load_img(t = t0+2);

plt.figure(1); plt.clf();

plt.imshow(w1)
plt.imshow(w2)



plt.imshow(w1[0:, :-1])
plt.imshow(w2[:, 1:])
plt.imshow(w3)


t0 = 517742;
w0 = np.array(exp.load_img(t = t0), dtype = float);
for i,d in enumerate([[0,0], [-1,0], [1,0], [0,-1], [0,1], [-1,-1], [-1,1], [1,-1], [1,1]]):
  plt.subplot(3,3,i+1);
  ww = w0.copy();
  ww = np.roll(ww, d, axis = [0,1]);
  plt.imshow(w0-ww)
  plt.title('%r, %d' % (d, np.sum(np.square(w0-ww))));







strain = 'n2'; wid = 0;

trans = np.load(os.path.join(exp.data_directory, 'transitions_times.npy'))
trans = trans[wid];
t0 = trans[0];

reload(vd)


speed = exp.load(strain = strain, wid = wid, dtype = 'speed')
plt.figure(6); plt.clf();
plt.plot(speed)
plt.ylim(0,20)

t0= 350600
vid = [vd.wormimage(t = t0+ti) for ti in range(0,1000,5)]
vid = np.array(vid)

fplt.pg.image(vid)  
plt.figure(7); plt.clf();
plt.imshow(vid[0], cmap = plt.cm.gray)

t0= 350600
img = vd.wormimage(t = t0, strain = strain, wid = wid, border = 75)
plt.figure(7); plt.clf(); 
plt.subplot(1,2,1)
plt.imshow(img, cmap = plt.cm.gray)
sh = img.shape;
plt.gca().add_artist(plt.Circle((sh[0]/2,sh[1]/2),sh[0]/2-75, color = 'black',fill = False, linewidth = 1))

reload(vd)
plt.figure(10); plt.clf();
vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+500, 5),
                  history= 50, history_delta=5)


file_format = 'movie_%d.png';
plt.figure(11);
t0 = 350600;
vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+5000, 1),
                  history= 150, history_delta=1, save = file_format);





#os.system("convert _tmp*.png animation.mng")

# cleanup
#for fname in files:
#  os.remove(fname)



#  import matplotlib
#  matplotlib.use("Agg")
#  import matplotlib.animation as manimation
#
#  FFMpegWriter = manimation.writers['ffmpeg']
#  metadata = dict(title='Worm %d t0 = %d' % (wid, t0), artist='Christoph Kirst', comment='CElegans Behaviour Project')
#  writer = FFMpegWriter(fps=15, metadata=metadata)
#
#  fig = plt.figure(11);
#  writer.saving(fig, "writer_test.mp4", 100);
#  vd.compose_frames(strain = strain, wid = wid, xy = all, speed = all, rotation = all, roam = all, times = range(t0, t0+500, 5),
#                    history= 50, history_delta=5, writer= writer)