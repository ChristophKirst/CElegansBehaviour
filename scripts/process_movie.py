# -*- coding: utf-8 -*-
"""
Generate Worm Images from Movie
"""

import os
import shutil
import glob
import natsort

import numpy as np

import imageio as iio

import matplotlib.pyplot as plt

import cv2

import ClearMap.GUI.DataViewer as dv


import scripts.process_movie_plot as pmp;
import scripts.process_movie_util as pmu;

reload(pmp); reload(pmu);

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


shape_file = os.path.join(data_dir, data_name % (region_id, 'shapes'));


#%% Detect worms in frames

overwrite = False;


worm_shape = (151,151);
worm_shape_2 = tuple(np.array(worm_shape, dtype = int) // 2);

if overwrite or not os.path.isfile(data_image_file):  
  data_image = np.lib.format.open_memmap(data_image_file, 'w+', shape = (n_frames_total,) + worm_shape, dtype = 'uint8', fortran_order = False);
else:
  data_image = np.lib.format.open_memmap(data_image_file, mode = 'r+');

if overwrite or not os.path.isfile(data_info_file):
  data_info = np.lib.format.open_memmap(data_info_file, 'w+', shape = (n_frames_total,),  dtype = [('origin', '2int32'), ('size', 'int32'), ('center', '2float32'), ('failed' , 'uint8'), ('objects', 'int32')], fortran_order = False);
  #data_info = np.zeros(n_frames_total, dtype = [('origin', '2int32'), ('size', 'int32'), ('center', '2float32'), ('failed' , 'uint8'), ('objects', 'int32')]);
else:
  data_info = np.lib.format.open_memmap(data_info_file, mode = 'r+');
  #data_info  = np.load(data_info_file);  

if overwrite or not os.path.isfile(data_meta_file):
  data_meta = np.array([(region_id, region_origin, region_shape, worm_shape, n_frames_total, norm_ids)], 
                      dtype = [('plate_id', 'int32'), ('plate_origin', '2int32'), ('plate_shape', '2int32'), ('image_shape', '2int32'), ('n_frames', 'int32'), ('norm_ids', '%dint32'%len(norm_ids))]);
  np.save(data_meta_file, data_meta)
else:
  data_meta  = np.load(data_meta_file);


#%%

threshold = 8;
verbose = True;
search_shape = (350, 350);
norm_ids_center = m_range/2 + norm_ids;
#norm_ids_center = norm_ids;

movie_ids = [31];
movie_ids = range(n_movies);
frame_ids = None;
#frame_ids = [1900];
parallel = True;
verbose = False if parallel else verbose;

worm_sizes_stage = np.array([0, 100, 180]);
worm_sizes_min = [7, 80, 200];
worm_sizes_max = [300, 1500, 1500]; 

#for m in range(n_movies):
def analyze_movie(m):
  
  #load memmaps
  data_image = np.lib.format.open_memmap(data_image_file, mode = 'r+');
  data_info = np.lib.format.open_memmap(data_info_file, mode = 'r+');  
  
  #video reader
  reader = cv2.VideoCapture(movie_files[m]);
  nf = int(reader.get(cv2.CAP_PROP_FRAME_COUNT));
  assert nf == n_frames[m]
  i_frame_0 = np.hstack([[0], np.cumsum(n_frames)])[m];
  
  #normalization and masking
  norm_id = np.argmin(np.abs(m - norm_ids_center))
  norm = norms[norm_id];
  
  norm_threshold = 60;
  plate_radius = 475;
  circles = cv2.HoughCircles(np.asarray(norm , dtype ='uint8'), cv2.HOUGH_GRADIENT, dp = 3, minDist = 300, minRadius = 455, maxRadius = 505, param1 = 60, param2 = 80)[0];
  if len(circles) == 0:
    print('could not detect plate border!');
    mask = norm < norm_threshold;
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.dilate(mask.view(dtype = 'uint8'), kernel, iterations = 1);
    mask = np.logical_not(mask.view(dtype = bool))  
  else:
    sort_id = np.argsort(np.abs(plate_radius - circles[:,-1]));
    circle = circles[sort_id[0]];
    mask = np.zeros_like(norm, dtype = bool);
    mask = cv2.circle(mask.view('uint8'), tuple(circle[:2]), circle[2], color = 1,thickness = -1);
    mask = np.logical_and(mask, norm >= norm_threshold);
    mask[0,0] = 0; #ensure background label = 0;
    
    #remove smaller components not part o the plate
    r,l,s,c = cv2.connectedComponentsWithStats(mask.view('uint8'));
    sort_id = np.argsort(-s[:,-1]);
    for si in sort_id:
      if si != 0:
        break;
    mask = l == si;

    if verbose:    
      circles = np.array([tuple(x) for x in circles], dtype = [('x', 'int'),('y', 'int'), ('r', 'float')]);
      pmp.plot_plate(norm, circles = circles, mask = mask)
  
  #trigger reset after each movie
  center_previous = None;  
  worm_size_previous = None;

  stage = np.where(m >= worm_sizes_stage)[0][-1];
  worm_size_min = worm_sizes_min[stage];
  worm_size_max = worm_sizes_max[stage];
  
  print('processing movie %d/%d with %d frames, using norm %d' % (m, n_movies, nf, norm_ids[norm_id]));    
  # movie loop
  if frame_ids is not None:
    f_ids = frame_ids;
  else:
    f_ids = range(nf);
  
  for f in f_ids:
    i_frame = i_frame_0 + f;
    print('processing movie %d/%d, frame %d/%d' % (m, n_movies, f, nf));  
    
    #frame_color = reader.get_data(f)[region_slice_color];
    _ = reader.set(cv2.CAP_PROP_POS_FRAMES, f);
    _, frame_color = reader.read();
    frame_color = frame_color[region_slice_color];
    
    if center_previous is not None:
      search_origin = pmu.get_origin(center_previous, search_shape, frame_color.shape);
      search_slice = [slice(search_origin[d], search_origin[d] + search_shape[d]) for d in [1,0]];
    else:
      search_origin = (0,0);
      search_slice = [slice(None)] * 2;
    
    frame = to_gray(frame_color[search_slice]);
    
    norm_search = norm[search_slice];
    mask_search = mask[search_slice];
    
    # detect worm forgroung
    detect = norm_search-frame > threshold;
    #detect[detect < 0] = 0;
    detect = np.logical_and(mask_search, detect);

    # center of worm
    r,l,s,c = cv2.connectedComponentsWithStats(np.asarray(detect, dtype = 'uint8'))

    if r == 1:  
      print('No worm found');
      # save       
      data_info[i_frame] = ([0,0], 0, [0,0], 1, 0);
      center_previous = None;
      if verbose:
        pmp.plot_worm_detection(frame, norm_search, norm_search-frame, detect);            
      continue;
      
    #get rid of largest background component first
    valid_id = np.argsort(-s[:,-1]);
    valid_id = valid_id[1:];
    
    #sort by closest to last position
    if center_previous is None:
      worm_center_guess = np.array(frame.shape, dtype = int)//2;
    else:
      worm_center_guess = np.array(center_previous, dtype = int) - np.array(search_origin, dtype = int);
    
    center_sort_id = np.argsort(np.linalg.norm(c[valid_id]-worm_center_guess, axis = 1));
    
    #sort by closest to last size
    #if worm_size_previous is not None:
    #  worm_size_guess = worm_size_previous;
    #else:
    #  worm_size_guess = 600.0 * i_frame / n_frames_total + 15;
    #size_sort_id = np.argsort(np.abs(s[valid_id]-worm_size_guess, axis = 1));
    
    worm_center = None;
    for si in valid_id[center_sort_id]:
      if worm_size_previous is not None:
        if np.abs(s[si,-1] - worm_size_previous) < 0.2 * worm_size_previous:
          worm_center = np.asarray(c[si], dtype = int);
          worm_size = s[si,-1];
          print('Worm found at (%d,%d)!' % tuple(worm_center + search_origin));
          break;
    
    if worm_center is None:
      for si in valid_id[center_sort_id]:     
        if worm_size_min < s[si,-1] < worm_size_max:
          worm_center = np.asarray(c[si], dtype = int);
          worm_size = s[si,-1];
          print('Worm found at (%d,%d)!' % tuple(worm_center + search_origin));
          break;
    
    if worm_center is None:
      print('No worm center found in first run');
      
      #dilate binary and try again
      kernel = np.ones((3,3),np.uint8)
      detect = cv2.dilate(detect.view(dtype = 'uint8'), kernel, iterations = 1);     
      
      # center of worm
      r,l,s,c = cv2.connectedComponentsWithStats(detect)

      if r == 1:  
        print('No worm found in second run');
        # save       
        data_info[i_frame] = ([0,0], 0, [0,0], 2, 0);
        center_previous = None;
        if verbose:
          pmp.plot_worm_detection(frame, norm_search, norm_search-frame, detect);            
        continue;
    
      #get rid of largest background component first
      valid_id = np.argsort(-s[:,-1]);
      valid_id = valid_id[1:];
    
      # sort by closest to last position
      center_sort_id = np.argsort(np.linalg.norm(c[valid_id]-worm_center_guess, axis = 1));

    
      worm_center = None;
      for si in valid_id[center_sort_id]:
        if worm_size_previous is not None:
          if np.abs(s[si,-1] - worm_size_previous) < 0.2 * worm_size_previous:
            worm_center = np.asarray(c[si], dtype = int);
            worm_size = s[si,-1];
            print('Worm found at (%d,%d)!' % tuple(worm_center + search_origin));
            break;
      
      if worm_center is None:
        for si in valid_id[center_sort_id]:     
          if worm_size_min < s[si,-1] < worm_size_max:
            worm_center = np.asarray(c[si], dtype = int);
            worm_size = s[si,-1];
            print('Worm found at (%d,%d)!' % tuple(worm_center + search_origin));
            break;
      
      if worm_center is None:
        print('No worm center found in second run');
        # save       
        data_info[i_frame] = ([0,0], 0, [0,0], 3, len(center_sort_id));
        center_previous = None;
        if verbose:
          pmp.plot_worm_detection(frame, norm_search, norm_search-frame, detect);      
        continue;
    
    # worm origin and search image
    worm_origin = pmu.get_origin(worm_center, worm_shape, frame.shape);
    #worm_region = worm_origin + worm_shape;
    worm_slice = [slice(worm_origin[d], worm_origin[d] + worm_shape[d]) for d in [1,0]]
    #worm_slice_color = worm_slice + [slice(None)];
    
    worm = 128 + np.array(norm_search[worm_slice], dtype = int) - np.asarray(frame[worm_slice], dtype = int);
    worm[worm < 0] = 0;
    worm[worm > 255] = 255;
    #worm_color = frame_color[worm_slice_color];
    #worm_norm = norm[worm_slice] - worm;
    #worm_norm_color = np.asarray(norm_color[worm_slice_color], dtype = float) - worm_color;
    #worm_norm_color[worm_norm_color < 0] = 0;
    
    #correct for search region
    worm_origin = tuple(np.array(worm_origin) + np.array(search_origin))
    worm_center = tuple(np.array(worm_center) + np.array(search_origin));
    center_previous = worm_center;
    
    data_image[i_frame] = worm;
    data_info[i_frame]  = (worm_origin, worm_size, worm_center, 0, len(center_sort_id));
    
    if verbose:
      pmp.plot_worm_detection(frame, norm_search, detect, worm); 
  
  #save info after each movie
  data_image.flush();
  data_info.flush();


if parallel:
  import multiprocessing as mp
  pool = mp.Pool(processes = mp.cpu_count());
  pool.map(analyze_movie, movie_ids)
else:
  for m in movie_ids:
    analyze_movie(m);





#%%
plt.figure(7); plt.clf();
for d in range(3):
  plt.subplot(1,3,d+1); plt.imshow(frame_color[:,:,d])
