# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:15:17 2018

@author: ckirst
"""


#%% Find turns

dv.plot(data_image.transpose([1,2,0]))


#%%

tid = 544871;
tw = 50;
turn = data_image[tid-tw:tid+tw];

np.save(turn_file, turn);

#%%

turn = np.load(turn_file);
n_frames_total = turn.shape[0];

#%%

dv.plot(turn);

#%%
blur = np.zeros(turn.shape, dtype = float);
for i in range(turn.shape[0]):
  blur[i] = cv2.GaussianBlur(np.asarray(turn[i], dtype = float), ksize = (5,5), sigmaX = 0);


#%%

import scripts.detect_shape_turn_code as dst;
reload(dst);


#%% get previous center line / shape

fid = 31;
n_points = 45;

status_prev, center_prev, _, _, width_prev = wgeo.shape_from_image(blur[fid-1], absolute_threshold = 135, sigma =  None, verbose = True,
                                                                   smooth = 20, smooth_center = 2.5, npoints = n_points, ncontour = 80, center_offset = 3)




#%%



fid = 81;
n_points = 45;
plt.clf();
status_end, center_end, _, _, width_end = wgeo.shape_from_image(blur[fid-1], absolute_threshold = 135, sigma =  None, verbose = True,
                                                                   smooth = 20, smooth_center = 2.5, npoints = n_points, ncontour = 80, center_offset = 3)

#%%

np.mean(np.linalg.norm(center_end[1:] - center_end[:-1],axis = 1))

#%%

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)

#%%
reload(dst);
center_new = center_prev;
#for f in range(fid, fid+25):
#centers = [center_new];

centers = [];


#%%

fig = plt.figure(23);
writer.setup(fig, "turning.mp4", dpi = 300)


#%%
reload(dst);

fid = 31;
f0 = fid-fid;

#writer = None;

#center_new = centers[f0];
center_new = center_prev;
centers_2 = [];
for i,f in enumerate(range(fid+f0, fid+60)):
  print i,f
  if f < 58:
    orient = -167;
  else:
    orient = 164;
  if f in [56,57]:
    orient = -162;
  if f in [54,55]:
    orient = -164;
  orient = 0;
    
  r = 1.0 * 4.686105205771127  + 1.15 * (4.904232701508219 - 4.686105205771127)/60. * i;
  r = 2.239730611858905 * (1 + (1.2-1) * i/60.);
    
  
  center_new = dst.unroll_center(blur[f], level = 135, center_guess = center_new, search_angle = 90, remove = 128, remove_factor = 0.5, prior_kappa = 0.0, orient = orient, verbose = True, writer= writer, radius = r);
  plt.title('%d' % f);
  plt.draw(); plt.pause(0.05);
  #if len(centers) < i:
  centers_2.append(center_new);
  #else:
  #  centers[i] = center_new;

#%%


writer.cleanup();

writer.finish();

#%%

np.save('centers_2.npy', np.array(centers_2))


#%%

plt.plot(width_prev)




import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='Movie support!')
writer = FFMpegWriter(fps=15, metadata=metadata)


#%%
import worm.model as wmod;


# create worms
fig = plt.figure(9);

writer.setup(fig, "turning_resul_2t.mp4", dpi = 300)

plt.clf(); plt.subplot(1,1,1);
ax = plt.gca();
ax.set_xlim(0, 151)
ax.set_ylim(0, 151)
for i,c in enumerate(centers_2):
  print i
  #if i == 54-fid:
  #  continue;
  #if i == 65 - fid:
  #  break;
  wm = wmod.WormModel(center = c, width = width_prev);
  plt.clf();
  wm.plot(blur[i+fid], ax = plt.gca(), cmap = 'gray');
  plt.xlim(0,151); plt.ylim(0,151);
  plt.draw(); plt.pause(0.05);
  writer.grab_frame();
  

writer.cleanup();



#%%

reload(dst);

f0 = 55 -  fid;
center_new = centers[f0];
#for f in range(fid, fid+25):

#for f in range(fid+f0, fid+f0+7):
for f in range(fid+f0, fid+f0+17):  
  center_new = dst.unroll_center(blur[f], level = 135, center_guess = center_new, search_angle = 70, remove = 128, remove_factor = 0.55, prior_kappa = 0.0, orient = 1, verbose = True);
  plt.title('%d' % f);
  plt.draw(); plt.pause(0.05);



#%% worm flipping 

plt.figure(8);
plt.clf();
for c in centers[54-fid:61-fid]:
  plt.plot(*(c.T));  
  

#%% unknow it ???

#%% match centers




#%%  rolling worm along shape

fid = 31;
n_points = 45;

cnts, hrchy = wgeo.detect_contour(blur[fid], 135, with_hierarchy=True)


plt.figure(13); plt.clf();
plt.imshow(blur[fid], interpolation = 'none');
for c in cnts:
  plt.plot(c[:,0], c[:,1]);
  
  

#%%

f = 58-fid;

plt.figure(190);
cc = centers[f]
plt.clf();
plt.plot(cc[:,0], cc[:,1])
plt.clf(); plt.subplot(1,2,1);
plt.plot(cc[:,0], cc[:,1])
plt.imshow(blur[fid + f])
plt.scatter(cc[0,0], cc[0,1], color = 'red', s = 30);
dc = cc[1:]- cc[:-1];
alpha = dst.angle(dc[1:], dc[:-1]);
plt.subplot(1,2,2);
plt.plot(alpha)














#%%
import worm.geometry as wgeo

verbose = True;

n_points = 45;

frame_batch = 1;
frame_ids = range(30, 32); #n_frames_total, frame_batch);
#frame_ids = failed;
#frame_ids = failed[:1];
#frame_ids = fids;

parallel = False;
verbose = False if parallel else verbose;


import scripts.parallel_memmaps as pmm

#for m in range(n_movies):
def analyze_shape(fid):
  #load memmaps
  fid2 = min(fid+frame_batch, n_frames_total);
  #images = np.lib.format.open_memmap(data_image_file, mode = 'r');
  images = turn;
  
  #shape = pmm.open_memmap(data_shape_file, arange = (fid, fid2), mode = 'r+');
  #shape_info = pmm.open_memmap(data_shape_info_file, arange = (fid, fid2), mode = 'r+');
  
  #smooth
  print('processing %d / %d' % (fid, n_frames_total));

  for i,f in enumerate(range(fid, fid2)):
    #print('success status: %d' % shape_info[i]['success']);    
    blur = cv2.GaussianBlur(images[f], ksize = (5,5), sigmaX = 0);  
    if verbose:
      plt.clf();
    try:
      res = wgeo.shape_from_image(blur, absolute_threshold = 135, sigma =  None, verbose = verbose,
                                  smooth = 20, smooth_center = 2.5, npoints = n_points, ncontour = 80, center_offset = 3)
      #print(res[0]);
    except:
      print('something went wrong');
      #res = (-1, np.zeros((n_points,2)), 0, 0, np.zeros(n_points));
    if verbose:
      plt.draw(); plt.pause(0.5);

    #shape_info[i] = (res[1][n_points2], res[0]);
    #shape[i] = np.vstack([res[1].T,res[-1]]);
    
    #print('success status after: %d vs %d' % (shape_info[i]['success'], res[0]));  
    #print res[0]

  #save info after each batch
  #shape.flush();
  #shape_info.flush();


if parallel:
  import multiprocessing as mp
  pool = mp.Pool(processes = mp.cpu_count());
  #pool = mp.Pool(processes = 1);
  pool.map(analyze_shape, frame_ids)
else:
  for f in frame_ids:
    analyze_shape(f);


#%%

blur = np.zeros(turn.shape);
for i in range(turn.shape[0]):
  blur[i] = cv2.GaussianBlur(turn[i], ksize = (5,5), sigmaX = 0);


#%%  rolling worm along shape

fid = 31;
n_points = 45;

cnts, hrchy = wgeo.detect_contour(blur[fid], 135, with_hierarchy=True)


plt.figure(13); plt.clf();
plt.imshow(blur[fid], interpolation = 'none');
for c in cnts:
  plt.plot(c[:,0], c[:,1]);
  
  
#%% get previous center line / shape

status_prev, center_prev, _, _, width_prev = wgeo.shape_from_image(blur[fid-1], absolute_threshold = 135, sigma =  None, verbose = True,
                                                                   smooth = 20, smooth_center = 2.5, npoints = n_points, ncontour = 80, center_offset = 3)

 
#%% decide inner vs outer
 

n_cnts =  len(cnts);
print('Number of contours: %d' % n_cnts);

# check if inner outer contour
# find children of each contour
children = [];
for cid in range(n_cnts):
  children.append(np.where(hrchy[:,-1]==cid)[0]);
# outer vs inner 
nc = [len(c) for c in children]
oid = np.argmax(nc);
iid = np.argmin(nc);


plt.figure(13); plt.clf();
plt.imshow(blur[fid], interpolation = 'none');
for c,co in zip([cnts[i] for i in [oid, iid]], ['red', 'blue']):
  plt.plot(c[:,0], c[:,1], color = co);
  

#%% max curvature of outer contour

pts = cnts[oid];

smooth = 20;
delta = 0.3;
ncontour = 80;

nextra = min(len(pts)-1, 20); # pts[0]==pts[-1] !!
ptsa = np.vstack([pts[-nextra-1:-1], pts, pts[1:nextra+1]]);
cinterp, u = wgeo.splprep(ptsa.T, u = None, s = smooth, per = 0, k = 4) 
u0 = u[nextra];
u1 = u[-nextra-1];
us = np.linspace(u0, u1, ncontour+1)
dx, dy = wgeo.splev(us, cinterp, der = 1)
d2x, d2y = wgeo.splev(us, cinterp, der = 2)
k = - (dx * d2y - dy * d2x)/np.power(dx**2 + dy**2, 1.5);
#kk = np.hstack([k[-nextra-1:-1], k, k[1:nextra+1]]);

imax = np.argmax(k);
x,y = wgeo.splev(us, cinterp, der = 0);

plt.figure(13); plt.clf();
plt.subplot(1,2,1);
plt.plot(k)
#plt.scatter(imax, k[imax], c = 'r', s= 100)
plt.scatter(imax, k[imax], c = 'm', s= 40);
plt.subplot(1,2,2);
plt.imshow(blur[fid]);
plt.plot(pts[:,0], pts[:,1]);
plt.scatter(x[imax], y[imax], c = 'r', s= 40);
plt.scatter(y[[0]], x[[0]], c = 'b', s= 60);

#%% is it close to head / tail

ht = center_prev[[0,-1]];

ht_new = np.array([x[imax], y[imax]]);

i_prev = np.argmin(np.linalg.norm(ht -ht_new, axis = 1));

if i_prev == 1:
  center_ref = center_prev[::-1];
else:
  center_ref = center_prev;


plt.figure(22); plt.clf();
plt.imshow(blur[fid]);
plt.plot(center_ref[:,0], center_ref[:,1], color = 'white');
plt.scatter(*center_ref[0], color = 'white');
plt.scatter(*ht_new, color = 'red');


#%% optimize center line step by step

verbose = True

import scipy.ndimage as ndi

num = 360
p1 = center_prev[1:]; p0 = center_prev[:-1];

r = np.mean(np.linalg.norm(p1-p0, axis = 1));
t = np.linspace(0, 2 * np.pi, num);

ref = ht_new;

sin = r*np.sin(t);
cos = r*np.cos(t);

dd = 90;

new_centers = [];

center_step = center_prev;
center_step[0] = ht_new;

dp = p1-p0;
d0 = np.arctan2(dp[0,1], dp[0,0]) * 180 / np.pi;
d0 = int(d0);

blur2 = cv2.GaussianBlur(blur[fid], ksize = (5,5), sigmaX = 0);

center_new = np.zeros_like(center_ref);
center_new[0] = ht_new;

for kk in range(len(p1)):

  x, y = cos[dr] + ref[0],  sin[dr] + ref[1];
  
  #intensity
  zi = ndi.map_coordinates(blur2, np.vstack((y,x)), order = 5)
  zmax = np.argmax(zi);
  d0 = int(np.arctan2(y[zmax]-ref[1],  x[zmax]- ref[0]) / np.pi * 180);
  ref = [x[zmax], y[zmax]];
  center_new[kk+1] = ref;  
  
  if verbose:
    if kk == 0:
      plt.figure(23); 
      plt.clf(); 
      plt.subplot(1,2,1);
      plt.imshow(blur[fid]);
    plt.subplot(1,2,1);
    plt.scatter(x,y, color = 'white', s = 2);
    plt.scatter(x[zmax], y[zmax], color = 'red', s = 2);
    plt.subplot(1,2,2);
    plt.plot(zi);
    plt.scatter(zmax, zi[zmax], color = 'red', s = 2);



#%% width profile ignore for now -> find closest boundary







#%%


test = np.zeros((30,30));
test[5:25, 5:25] = 1;
test[7:9,7:9] = 0
test[15:18, 15:18] = 0
q = wgeo.detect_contour(test, 1, with_hierarchy = True)



#%%

failed_id = np.where(data_info['failed'])[0]
print('failed: %d' % len(failed_id));


#%%

movie_files = pmu.get_movie_files(os.path.join(movie_dir, movie_name));
n_frames = pmu.get_n_frames(movie_files)


#%%

pmu.get_movie_from_frame(movie_files, failed_id[0], n_frames=n_frames)

#%% 

plt.figure(1); plt.clf();

plt.imshow(data_image[263332-37443])

#%%


dv.plot(data_image.transpose([2,1,0]))


#%% curled
263332-37520



#%% turns

i = 337714;
i = 109715;

turn = data_image[i-75:i+150];

dv.plot(turn)

#%%

np.save(os.path.join(data_dir, 'Tests/turn_001.npy'), turn)


#%%


turn = np.load(os.path.join(data_dir, 'Tests/turn_000.npy'))



#%%


#%%

i = 320579;

b = cv2.GaussianBlur(data_image[i], ksize = (3,3), sigmaX = 0);


dv.plot(b> 133)


#%%

wgeo.detect_contour(

#%%

