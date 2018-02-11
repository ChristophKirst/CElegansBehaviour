# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 02:06:18 2018

@author: ckirst
"""

import os
import shutil
import glob
import natsort

import numpy as np

import cv2


def get_origin(center, shape, full_shape = None):
  shape2 = np.array(shape, dtype = int) // 2;
  origin = [0,0];
  for d in range(2):
    if center[d] - shape2[d] < 0:
      origin[d] = 0;
    else:
      origin[d] =  center[d] - shape2[d];
    if full_shape is not None:
      if origin[d] + shape[d] > full_shape[d]:
        origin[d] = full_shape[d] - shape[d];
  return tuple(origin);



def get_movie_files(name, check = True, verbose = True):
  movie_files = os.path.join(name + '*.avi');
  movie_files = natsort.natsorted(glob.glob(movie_files))
  
  if verbose:
    print('%d potential movie files found!' % len(movie_files));
  
  try:
    #reader = iio.get_reader(movie_files[-1]);
    reader = cv2.VideoCapture(movie_files[-1])
    ret, frame = reader.read();
    if ret is False:
      if verbose:
        print('Warning: last movie file seems corrupted!');  
      movie_files = movie_files[:-1];
  except:
    if verbose:
      print('Warning: last movie file seems corrupted!');
    movie_files = movie_files[:-1];
  
  if verbose:
    print('%d valid movie files found!' % len(movie_files));
  
  return movie_files;


def get_n_frames(movie_files, verbose = True):  
  n_movies = len(movie_files);
  n_frames = np.zeros(n_movies, dtype = int);
  for m in range(n_movies):
    reader = cv2.VideoCapture(movie_files[m]);
    n_frames[m] = int(reader.get(cv2.CAP_PROP_FRAME_COUNT));
    if verbose:
      print('Movie %d/%d %d frames' % (m, n_movies, n_frames[m]));    
    
  if verbose:
    print('Total number of frames %d' % np.sum(n_frames));
    
  return n_frames;


def get_ids_from_frame(movie_files, frame, n_frames = None, verbose = False):
  if n_frames is None:
    n_frames = get_n_frames(movie_files, verbose = verbose);
  
  n_frames_sum = np.cumsum(n_frames);
  m = np.where(frame < n_frames_sum)[0][0];
  return m, frame - np.hstack([[0],n_frames_sum])[m];


def get_data_from_frame(movie_files, frame, n_frames = None, verbose = False):
  if n_frames is None:
     n_frames = get_n_frames(movie_files, verbose = verbose);
  m, f = get_ids_from_frame(movie_files, frame, n_frames = n_frames, verbose = verbose);
  reader = cv2.VideoCapture(movie_files[m]);
  _ = reader.set(cv2.CAP_PROP_POS_FRAMES, f);
  _, data = reader.read();
  return data;
  