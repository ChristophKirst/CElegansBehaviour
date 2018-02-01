# -*- coding: utf-8 -*-
"""
Generate Worm Images from Movie
"""

import os
import shutil
import glob
import natsort

try:
  from importlib import reload
except:
  pass

import numpy as np

import imageio as iio

import matplotlib.pyplot as plt

import cv2

import ClearMap.GUI.DataViewer as dv


import scripts.process_movie_plot as pmp;
import scripts.process_movie_util as pmu;

reload(pmp); reload(pmu);

import worm.geometry_new as wgn
reload(wgn);


#%% Movie files

movie_dir = '/run/media/ckirst/My Book/'

#movie_name = 'CAM207_2017-01-30-172321'
movie_name = 'CAM800_2017-01-30-171215'
#movie_name = 'CAM807_2017-01-30-171140'
#movie_name = 'CAM819_2017-01-30-172249'

data_dir = '/home/ckirst/Data/Science/Projects/CElegans/Experiment/Movies/'
data_dir = '/home/ckirst/Movies'

data_name = '%s_%s_%s.npy' % (movie_name, '%d', '%s');

region_id = 0;

data_image_file  = os.path.join(data_dir, data_name % (region_id, 'images'));
data_info_file   = os.path.join(data_dir, data_name % (region_id, 'info'));
data_meta_file   = os.path.join(data_dir, data_name % (region_id, 'meta'));

data_shape_file = os.path.join(data_dir, data_name % (region_id, 'shapes'));
data_contour_file = os.path.join(data_dir, data_name % (region_id, 'contours'));

data_shape_info_file = os.path.join(data_dir, data_name % (region_id, 'shapes_info'));

#
#data_image_file  = os.path.join(data_dir, data_name % (region_id, 'images_new'));
#data_info_file   = os.path.join(data_dir, data_name % (region_id, 'info_new'));
#data_meta_file   = os.path.join(data_dir, data_name % (region_id, 'meta_new'));
#
#data_shape_file = os.path.join(data_dir, data_name % (region_id, 'shapes_new'));
#data_contour_file = os.path.join(data_dir, data_name % (region_id, 'contours_new'));
#
#data_shape_info_file = os.path.join(data_dir, data_name % (region_id, 'shapes_info_new'));




#%% Detect worms in frames

overwrite = False;

data_meta  = np.load(data_meta_file);

data_images = np.lib.format.open_memmap(data_image_file, mode = 'r+');
n_frames_total = data_images.shape[0];

worm_shape = (151,151);

n_points = 50;
n_contour = 100;

if overwrite or not os.path.isfile(data_shape_file):
  data_shape = np.lib.format.open_memmap(data_shape_file, 'w+', shape = (n_frames_total, 3, n_points),  dtype = 'float32', fortran_order = False);
else:
  data_shape = np.lib.format.open_memmap(data_shape_file, mode = 'r+');

  
if overwrite or not os.path.isfile(data_contour_file):
  data_contour = np.lib.format.open_memmap(data_contour_file, 'w+', shape = (n_frames_total, 4, n_contour),  dtype = 'float32', fortran_order = False);
else:
  data_contour = np.lib.format.open_memmap(data_contour_file, mode = 'r+');  
  
  

if overwrite or not os.path.isfile(data_shape_info_file):
  data_shape_info = np.lib.format.open_memmap(data_shape_info_file, 'w+', shape = (n_frames_total,),  dtype = [('center', '2float32'), ('success', 'int32')], fortran_order = False);
  #data_info = np.zeros(n_frames_total, dtype = [('origin', '2int32'), ('size', 'int32'), ('center', '2float32'), ('failed' , 'uint8'), ('objects', 'int32')]);
else:
  data_shape_info = np.lib.format.open_memmap(data_shape_info_file, mode = 'r+');
  #data_info  = np.load(data_info_file);  


#%%


#%%
reload(wgn);

import warnings
warnings.filterwarnings("ignore")

verbose = True;

#frame_batch = 1000;
frame_batch = 1;
frame_ids = range(0, n_frames_total, frame_batch);
#frame_ids = failed;
#frame_ids = fids[3:4]
#frame_ids = [534440];
#frame_ids = [320579-20000-10];
#frame_ids = [524431];
#frame_ids = [514425];
#frame_ids = [320579-12];
#frame_ids = [494428+2];
#frame_ids = [474424];
frame_ids = [604384];
frame_ids = [520049];
frame_ids = [521277];


parallel = True;
parallel = False;
verbose = False if parallel else verbose;

n_points2 = n_points//2;


import scripts.parallel_memmaps as pmm

#for m in range(n_movies):
def analyze_shape(fid):
  #load memmaps
  fid2 = min(fid+frame_batch, n_frames_total);
  images = np.lib.format.open_memmap(data_image_file, mode = 'r');
  
  shape = pmm.open_memmap(data_shape_file, arange = (fid, fid2), mode = 'r+');
  shape_info = pmm.open_memmap(data_shape_info_file, arange = (fid, fid2), mode = 'r+');
  contour = pmm.open_memmap(data_contour_file, arange = (fid, fid2), mode = 'r+');
  
  #smooth
  print('processing %d / %d' % (fid, n_frames_total));

  for i,f in enumerate(range(fid, fid2)):
    #print('success status: %d' % shape_info[i]['success']);   
    if i % 100 == 0:
      print('processing %d / %d' % (f, n_frames_total));
    blur = cv2.GaussianBlur(np.asarray(images[f], dtype = float), ksize = (5,5), sigmaX = 0);  
    #if i > 0:
    #  head_tail_hint = shape[i-1,:2,[0,-1]];
    #  print head_tail_hint.shape
    #else:
    head_tail_hint = None;
    #print i, head_tail_hint
        
    if verbose:
      plt.clf();
    #try:
    res = wgn.shape_from_image(blur, absolute_threshold = 136, sigma = None,
                               smooth_head_tail = 10, smooth_left_right = 5.0, smooth_center = 10.0, npoints = n_points, ncontour = n_contour, center_offset = 1,
                               head_tail_hint = head_tail_hint,
                               verbose = verbose);
    #print(res[0]);
    #except:
    #  print('something went wrong');
    #  res = (-200000, np.zeros((n_contour,2)), np.zeros((n_contour,2)), np.zeros((n_points,2)), np.zeros(n_points));
    
    if verbose:
      plt.draw(); plt.pause(0.5); 

    shape_info[i] = (res[-2][n_points2], res[0]);
    shape[i] = np.vstack([res[-2].T, res[-1]]);
    contour[i] = np.vstack([res[1].T, res[2].T]);
    #print('success status after: %d vs %d' % (shape_info[i]['success'], res[0]));  
    #print res[0]

  #save info after each batch
  shape.flush();
  shape_info.flush();
  
  if not parallel:
    return blur, res;


if parallel:
  import multiprocessing as mp
  pool = mp.Pool(processes = mp.cpu_count());
  #pool = mp.Pool(processes = 1);
  pool.map(analyze_shape, frame_ids)
else:
  for f in frame_ids:
   blur, res =  analyze_shape(f);



#%%
import worm.model as wmod;

#del data_shape
#del data_shape_info
data_shape = np.lib.format.open_memmap(data_shape_file, mode = 'r+');
data_shape_info = np.lib.format.open_memmap(data_shape_info_file, mode = 'r+');

failed = np.where(data_shape_info['success']<0)[0]
print('failed: %d' % len(failed));

fids = np.arange(6) + 516000;
#fids = failed[10871];
fids = 534426-60000;
fids = failed[65];

fids = range(fids-6, fids+6);


fig = 21; link = True;
plt.figure(fig); plt.clf();
m,k = pmp.arrange_plots(len(fids));
fig, axs = plt.subplots(k,m, sharex = link, sharey = link, num = fig);
if not isinstance(axs, np.ndarray):
  axs = np.array([axs]);
axs = axs.flatten();
for j,i in enumerate(fids):
  print 'frame %d success = %d' % (i, data_shape_info[i]['success'])
  wm = wmod.WormModel(center = data_shape[i,:2].T, width = data_shape[i,2]);
  wm.plot(data_images[i], ax = axs[j]);
  axs[j].set_title('%d' % i);
plt.tight_layout();
plt.draw();


print('failed: %d' % len(failed));
print(np.where(np.diff(failed)>5));


#%% no worm images

d0 = data_images[:,0,0];
detect_failed = np.where(d0 == 0)[0]
failed_shape = np.setdiff1d(failed, detect_failed)

#%%

fids = failed_shape

fig = 21; link = True;
plt.figure(fig); plt.clf();
m,k = pmp.arrange_plots(len(fids));
fig, axs = plt.subplots(k,m, sharex = link, sharey = link, num = fig);
if not isinstance(axs, np.ndarray):
  axs = np.array([axs]);
axs = axs.flatten();
for j,i in enumerate(fids):
  print 'frame %d success = %d' % (i, data_shape_info[i]['success'])
  wm = wmod.WormModel(center = data_shape[i,:2].T, width = data_shape[i,2]);
  wm.plot(data_images[i], ax = axs[j]);
  axs[j].set_title('%d' % i);
plt.tight_layout();
plt.draw();

#%% statistics

s = data_shape_info['success'];
su = np.unique(s)

for i in su:
  print('success %8d: %d' % (i, np.sum(s == i)))


#%%

failed = np.where(data_shape_info['success']<0)[0]
print('failed: %d' % len(failed));

fids = failed[np.where(np.diff(failed)>1)[0]+1][:];

s = data_shape_info['success'];
fids = np.where(s == 1612)[0];
fids = np.where(s == 1605)[0];
fids = np.where(s == 1205)[0][:48];

fig = 21; link = True;
plt.figure(fig); plt.clf();
m,k = pmp.arrange_plots(len(fids));
fig, axs = plt.subplots(k,m, sharex = link, sharey = link, num = fig);
if not isinstance(axs, np.ndarray):
  axs = np.array([axs]);
axs = axs.flatten();
for j,i in enumerate(fids):
  print 'frame %d success = %d' % (i, data_shape_info[i]['success'])
  wm = wmod.WormModel(center = data_shape[i,:2].T, width = data_shape[i,2]);
  wm.plot(data_images[i], ax = axs[j]);
  axs[j].set_title('%d' % i);
plt.tight_layout();
plt.draw();


print('failed: %d' % len(failed));
#print(np.where(np.diff(failed)>5));


#looks good ! => no obvious failures anymore !

#%% area / length
import worm ,geometry as wgeo;

length = wgeo.length_from_center_discrete(data_shape[f][:2].T)

left = data_contour[f][:2,:].T;
right = data_contour[f][2:,:].T;

cont = np.vstack([left, right[-2:0:-1]]);

import cv2
area = cv2.contourArea(cont)
perimeter = cv2.arcLength(cont, True)
r = perimeter/2/np.pi;
a = np.pi * r**2;

print a,area


#%% cureled




#%%


blur2 = np.zeros(turn2.shape);
for i in range(blur2.shape[0]):
  blur2[i] = cv2.GaussianBlur(np.asarray(turn2[i], dtype = float), ksize = (5,5), sigmaX = 0);
  

dv.plot(blur2);




#%%
plt.figure(7); plt.clf();
for d in range(3):
  plt.subplot(1,3,d+1); plt.imshow(frame_color[:,:,d])


#%%

dv.plot(data_images.transpose([1,2,0]));


#%%


def fast_hist(image):
  h = np.zeros(255);
  for i in image.flatten():
    h[i] += 1;
  return h;
  
h = fast_hist(b)


#%%

i = 242000;
#i = 56463;
#i = 500000;
#i = 20000;
#i = 38271;
i = 38549;
#i = 41450;
i = 147968;
i = 148000;
i = 147883;
i = 172504 + 2;
s = 100
#s = 800

b = cv2.GaussianBlur(data_images[i], ksize = (3,3), sigmaX = 0);
h = fast_hist(b);
t = 255 - np.where(np.cumsum(h[::-1]) > s)[0][0];
dv.multiPlot([data_images[i], b, b > t, b > 135])
print t

#%% ni

plt.figure(18);
plt.plot(h)



#%%

reload(wgn);
plt.figure(19); plt.clf();
plt.imshow(blur);

left = data_contour[f][:2,:].T;
right = data_contour[f][2:,:].T;

left = left[::-1]; right = right[::-1];

center, width = wgn.center_from_sides_min_projection(left, right, npoints = n_points, nsamples = n_contour, with_width = True, smooth = 1.0, center_offset = 3, verbose = True);
  
#%%
  
cont = np.vstack([left, right[-2:0:-1]]);
cont  = np.reshape(cont, (cont.shape[0], 1, cont.shape[1]));
cont = np.asarray(cont, dtype = 'int32');


mask = np.zeros_like(blur);

cv2.drawContours(mask, [cont], 0, 255, -1);


dv.dualPlot(blur, mask)

#%%


dist = cv2.distanceTransform(mask, cv2.DIST_L2, maskSize = 3)