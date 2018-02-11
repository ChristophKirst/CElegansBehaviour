# -*- coding: utf-8 -*-
"""
Data Manager
"""

import os
#import shutil
import glob
import natsort

import numpy as np

import cv2


#%% Memmory mappping

def open_memmap(file_name, mode = 'r+', shape = None, dtype = None):
  return np.lib.format.open_memmap(file_name, mode = mode, shape = shape, dtype = dtype, fortran_order = False);

def memmap(file_name, shape = None, dtype  = None, overwrite = False):  
  if os.path.isfile(file_name):
    if overwrite:
      if shape is None or dtype is None:
        d = open_memmap(file_name);
        if shape is None:
          shape = d.shape;
        if dtype is None:
          dtype = d.dtype;
      return open_memmap(file_name, mode = 'w+', shape = shape, dtype = dtype);
    else:
      return open_memmap(file_name, mode = 'r+', shape = shape, dtype = dtype); 
  else: 
    return open_memmap(file_name, mode = 'w+', shape = shape, dtype = dtype);


#%% Data files

movie_dir = '/run/media/ckirst/My Book/'

data_names = ['CAM207_2017-01-30-172321', 'CAM800_2017-01-30-171215', 'CAM807_2017-01-30-171140', 'CAM819_2017-01-30-172249'];

default_data_name = data_names[1];
default_region_id  = 0;

#hostname = socket.gethostname();
#if hostname == 'ChristophsLaptop.home':  #Christoph's Laptop 
  #data_dir = '/home/ckirst/Data/Science/Projects/CElegans/Experiment/Movies/  
#elif hostname == 'ChristophsComputer.rockefeller.edu':  #Christoph's Desktop 
  #data_dir = '/home/ckirst/Science/Data/CElegans/'

data_dir = '/home/ckirst/Science/Data/CElegans/'
data_file_name = '%s_%d_%s.npy';

def data_file(type_name, data_name = default_data_name, region_id = default_region_id):
  return os.path.join(data_dir, data_file_name % (data_name, region_id, type_name));
  

#%% Movie Files

#cache
last_movie_name = None;
last_movie_files = None;
last_movie_n_frames = None;


def movie_files(data_name = default_data_name, check = True, verbose = True):
  global last_movie_name, last_movie_files, last_movie_n_frames;
  
  if data_name == last_movie_name and last_movie_files is not None:
    return last_movie_files;
  
  movie_files = os.path.join(movie_dir, data_name + '*.avi');
  movie_files = natsort.natsorted(glob.glob(movie_files))
  
  if check:
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
  
  last_movie_name  = data_name;
  last_movie_files = movie_files;
  last_movie_n_frames = None;
  
  return movie_files;


def movie_n_frames(data_name = default_data_name, verbose = True):
  global last_movie_name, last_movie_n_frames  
  
  if data_name == last_movie_name and last_movie_n_frames is not None:
    return last_movie_n_frames;

  m_files = movie_files(data_name, verbose = verbose);
  n_movies = len(m_files);
  n_frames = np.zeros(n_movies, dtype = int);
  for m in range(n_movies):
    reader = cv2.VideoCapture(m_files[m]);
    n_frames[m] = int(reader.get(cv2.CAP_PROP_FRAME_COUNT));
    if verbose:
      print('Movie %d/%d %d frames' % (m, n_movies, n_frames[m]));    
    
  if verbose:
    print('Total number of frames %d' % np.sum(n_frames));
  
  last_movie_n_frames = n_frames;
  
  return n_frames;


def movie_file_from_frame(frame, data_name = default_data_name, verbose = False):
  global last_movie_files;
  n_frames = movie_n_frames(data_name, verbose = verbose);  
  n_frames_sum = np.cumsum(n_frames);
  m = np.where(frame < n_frames_sum)[0][0];
  return last_movie_files[m], frame - np.hstack([[0],n_frames_sum])[m];


def movie_data(frame, data_name = default_data_name, verbose = False):
  m_file, m_frame = movie_file_from_frame(frame = frame, data_name = data_name, verbose = verbose);
  reader = cv2.VideoCapture(m_file);
  _ = reader.set(cv2.CAP_PROP_POS_FRAMES, m_frame);
  _, data = reader.read();
  return data;


#%% Plate Data

def data_meta_file(data_name = default_data_name, region_id = default_region_id):
  return data_file('meta', data_name = data_name, region_id = region_id);


def plate_region(data_name = default_data_name, region_id = default_region_id):
  meta_file = data_meta_file(data_name = data_name, region_id = region_id);
  meta = np.load(meta_file);
  return meta['plate_origin'][0], meta['plate_shape'][0];  


def plate_data(frame, data_name = default_data_name, region_id = default_region_id, verbose = False):
  data = movie_data(frame, data_name = data_name, verbose = verbose);
  origin, shape = plate_region(data_name = data_name, region_id = region_id);
  sl = [slice(origin[1], origin[1]+shape[0]), slice(origin[0], origin[0] + shape[1])];
  return data[sl];


#%% Raw Image Data

def data_info_file(data_name = default_data_name, region_id = default_region_id):
  return data_file('info', data_name = data_name, region_id = region_id);

  
def worm_data(frame,  data_name = default_data_name, region_id = default_region_id, verbose = False):
  info_file = data_info_file(data_name = data_name, region_id = region_id);
  info_data = open_memmap(info_file);
  data = plate_data(frame, data_name = data_name, region_id = region_id, verbose = verbose);
  meta_file = data_meta_file(data_name = data_name, region_id = region_id);
  meta = np.load(meta_file);
  origin, shape = info_data['origin'][frame], meta['image_shape'][0];
  sl = [slice(origin[1], origin[1]+shape[0]), slice(origin[0], origin[0] + shape[1])];
  return data[sl];


#%% Image Data
  
def data_image_file(data_name = default_data_name, region_id = default_region_id):
  return data_file('images', data_name = data_name, region_id = region_id);


def image_data(frame = None, data_name = default_data_name, region_id = default_region_id, smooth = None, dtype = None):
  image_file = data_image_file(data_name = data_name, region_id = region_id);
  image_data = open_memmap(image_file);
  if frame is None:
        return image_data;
  else:
    if dtype is  None:
      return smooth_image(image_data[frame], smooth = smooth);
    else:
      return np.asarray(smooth_image(image_data[frame], smooth = smooth), dtype = dtype);

 
def smooth_image(image, smooth = None):
  if smooth is None:
    return image;
  if smooth is True:
    smooth = (5,5)
  return cv2.GaussianBlur(np.asarray(image, dtype = float), ksize = smooth, sigmaX = 0);
 

#%% Shape Data
  
def data_shape_file(data_name = default_data_name, region_id = default_region_id):
  return data_file('shapes', data_name = data_name, region_id = region_id);


def shape_data(frame = None, data_name = default_data_name, region_id = default_region_id):
  shape_file = data_shape_file(data_name = data_name, region_id = region_id);
  shape_data = open_memmap(shape_file);
  if frame is None:
    return shape_data;
  else:
    return shape_data[frame];
    
    
#%% Contour Data
  
def data_contour_file(data_name = default_data_name, region_id = default_region_id):
  return data_file('contours', data_name = data_name, region_id = region_id);


def contour_data(frame = None, data_name = default_data_name, region_id = default_region_id):
  contour_file = data_shape_file(data_name = data_name, region_id = region_id);
  contour_data = open_memmap(contour_file);
  if frame is None:
    return contour_data;
  else:
    return contour_data[frame];
    

