# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 21:11:10 2016

@author: ckirst
"""

import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import tensorflow as tf

import analysis.experiment as exp

import interpolation.resampling as ir

import worm.geometry as wgeo



class ImageGenerator(object):
  """Generate Images and Phi's from data set"""
  def __init__(self, wids = [80,96], strains = ['n2']*2, tmins = [400000]*2, tmaxs = [500000]*2, smooth = 1.0, contour_size = 70, contour_inner = 10):
    self.wids = wids;
    self.strains = strains;
    self.tmins = tmins;
    self.tmaxs = tmaxs;
    self.nwids = len(self.wids);
    assert len(self.strains) == self.nwids;
    assert len(self.tmins) == self.nwids;
    assert len(self.tmaxs) == self.nwids;
    
    self.t_counter = tmins[0];
    self.wid_counter = 0;
    
    img = self.get_image(random = True);
    self.image_size = img.shape[1:3];
    
    self.contour_size = contour_size;
    self.contour_inner = contour_inner;

   
  def increase_counter(self, stride = 1):
    tc = self.t_counter;
    wc = self.wid_counter;
    tm = self.tmaxs[wc];
    if tc + stride > tm:
      wc += 1;
      if wc > self.nwids:
        wc = 0;
      self.t_counter = self.tmins[wc] + stride + tc - tm;
      self.wid_counter = wc;
    else:
      self.t_counter += stride;  
  
  def get_image(self, stride = 1, random = False, sigma = 1.0):
    if random:
      wc = np.random.randint(0, self.nwids);
      t = np.random.randint(self.tmins[wc], self.tmaxs[wc]);
      img = exp.load_img(strain = self.strains[wc], wid = self.wids[wc], t = t, sigma = sigma);
    else:
      t = self.t_counter;
      wid = self.wids[self.wid_counter];
      img = exp.load_img(wid = wid, t = t);
      self.increase_counter(stride = stride);
    
    img.shape = (1,) + img.shape + (1,);
    return img;
  
  def contour_from_image(self, img, threshold = 75):
    cntrs = wgeo.contours_from_image(img, sigma = 1, absolute_threshold = threshold, verbose = False, save = None);
    nc = len(cntrs);
    if nc == 1:
      return ir.resample(cntrs[0], self.contour_size);
    else:
      return np.vstack([ir.resample(cntrs[0], self.contour_size-self.contour_inner), ir.resample(cntrs[1], self.contour_inner)]);
  
  def get_batch(self, nimages = 1, stride = 1, random = False, threshold = 75):
    imgs = np.zeros((nimages,) + self.image_size + (1,));
    cntrs = np.zeros((nimages, self.contour_size, 2));
    for i in range(nimages):
      imgs[i,:,:,:] = self.get_image(random = random, stride = stride)[0];
      cntrs[i] = self.contour_from_image(imgs[i,:,:,0]);
      #n = min(c.shape[0], self.contour_size);
      #cntrs[i,:n,:] = c[:n, :];

    return (imgs, cntrs)



def create_left_right(c, w): 
  #tanget vectors
  c_shape = c.get_shape().as_list();
  c1 = tf.slice(c, [1,0], [-1,-1]);
  c2 = tf.slice(c, [0,0], [c_shape[0]-1,-1]);
  dc = tf.sub(c1,c2);
  
  #normalized tangent vectors
  nr = tf.sqrt(tf.reduce_sum(tf.square(dc), reduction_indices = 1));
  dcn = tf.transpose(tf.div(tf.transpose(dc), nr));
  
  #average tangent
  dc1 = tf.concat(0, [tf.slice(dcn, [0,0], [1,-1]), dcn]);
  dc2 = tf.concat(0, [dcn, tf.slice(dcn, [c_shape[0]-2,0], [1,-1])]);
  av = tf.scalar_mul(0.5, tf.add(dc1,dc2));
  nr = tf.sqrt(tf.reduce_sum(tf.square(av), reduction_indices = 1));
  av = tf.transpose(tf.div(tf.transpose(av), nr));
    
  #normal
  nrm = tf.mul(av, tf.constant([-1,1], "float32"));
  nrm = tf.reverse(nrm, [False, True]);
  
  left = tf.transpose(tf.mul(tf.transpose(nrm), w));
  right= tf.scalar_mul(-1.0, left);
  
  left = tf.add(left, c);
  right= tf.add(right,c);
  
  return left,right


def create_cost_distance(l, r, d):
  dd = tf.sqrt(tf.reduce_sum(tf.squared_difference(l,r), reduction_indices=1));
  dd = tf.squared_difference(dd, d);
  return tf.reduce_mean(dd);

def create_cost_soft_min_distance(c, s, k = 2.0):
  """Creates a soft-min distance of the centers to the points"""
  c_shape = c.get_shape().as_list();        
  s_shape = s.get_shape().as_list();
  
  #expand matrices
  cc = tf.reshape(c, [c_shape[0], c_shape[1], 1]);    
  ss = tf.reshape(s, [s_shape[0], s_shape[1], 1]);
  ss = tf.transpose(ss, perm = [2,1,0]);
  cc = tf.tile(cc, [1, 1, s_shape[0]]);
  ss = tf.tile(ss, [c_shape[0], 1, 1]);
  #cc = tf.transpose(cc, perm = [2,1,0]);
  #cc = tf.tile(cc, [s_shape[0], 1, 1]);
  #ss = tf.tile(ss, [1, 1, c_shape[0]]); 
  
  #pairwise distances
  dist2 = tf.sqrt(tf.reduce_sum(tf.squared_difference(cc,ss), reduction_indices = 1));
    
  #softmin
  softmin = tf.reduce_sum(tf.mul(tf.nn.softmax(tf.scalar_mul(tf.constant(-k,"float32"), dist2)), dist2),reduction_indices = 1);
  
  return tf.reduce_mean(softmin);

def create_cost_spacing( c, length, normalized = True):
  c_shape = c.get_shape().as_list();
  c1 = tf.slice(c, [1,0], [-1,-1]);
  c2 = tf.slice(c, [0,0], [c_shape[0]-1,-1]);
  d = tf.sqrt(tf.reduce_sum(tf.squared_difference(c1,c2), reduction_indices = 1));
  if normalized:
    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length / (c_shape[0]-1), "float32")));
  else:
    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length, "float32")));
  
def create_cost_bending(c):
  c_shape = c.get_shape().as_list();
  c1 = tf.slice(c, [1,0], [-1,-1]);
  c2 = tf.slice(c, [0,0], [c_shape[0]-1,-1]);
  dc = tf.sub(c1,c2);
  
  #normalized tangent vectors
  nr = tf.sqrt(tf.reduce_sum(tf.square(dc), reduction_indices = 1));
  dcn = tf.transpose(tf.div(tf.transpose(dc), nr));  
  
  dc1 = tf.slice(dcn, [1,0], [-1,-1]);
  dc2 = tf.slice(dcn, [0,0], [c_shape[0]-2,-1]);
  dp = tf.reduce_sum(tf.mul(dc1, dc2), reduction_indices = 1);

  return tf.mul(tf.constant(-1.0, "float32"), tf.reduce_mean(dp));

def create_cost_side(s, b, length = 1.0, weight_spacing = 1.0, weight_bending = 1.0):
  cost = create_cost_soft_min_distance(s, b);
  if weight_spacing != 0:
    cost_spacing = create_cost_spacing(s, length);
    cost = tf.add(cost, tf.mul(tf.constant(weight_spacing, "float32"), cost_spacing));
  if weight_bending != 0:
    cost_bending = create_cost_bending(s);
    cost = tf.add(cost, tf.mul(tf.constant(weight_bending, "float32"), cost_bending));
  return cost;

def create_cost(l, r, b, w, length, weight_spacing = 1.0, weight_bending = 1.0, weight_distance = 1.0):
  #left right boundaries
  cost_left = create_cost_side(l, b, length, weight_spacing = weight_spacing, weight_bending = weight_bending);
  cost_right= create_cost_side(r, b, length, weight_spacing = weight_spacing, weight_bending = weight_bending);
  cost = tf.add(cost_left, cost_right);
  
  #keep distance between sides of worm
  if weight_distance != 0:
    cost_dist = create_cost_distance(l,r, w);
    cost = tf.add(cost, tf.mul(tf.constant(weight_distance, "float32"), cost_dist));  
  
  return cost;

def create_variable(shape, value = 0.01):
  """Create variable"""
  v = tf.constant(value, shape = shape)
  return tf.Variable(v);


def test():
  import numpy as np
  import tensorflow as tf
  import matplotlib.pyplot as plt;
  import worm.model as wm;
  import worm.machine_vision_b as wmv
  import worm.geometry as wgeo
  
  reload(wgeo)
  reload(wmv)  
  
  ### Prepare optimization task
  
  # work shape
  w = wm.WormModel(length = 80);
  ig = wmv.ImageGenerator();
    
  ig.t_counter = 500000 + 25620 - 5;
  ig.wid_counter = 0;

  imgs, cntrs = ig.get_batch(nimages = 1);
  img = imgs[0,:,:,0]; cntr = cntrs[0];
  w.from_image(img)  

  plt.figure(20); plt.clf();
  #wgeo.skeleton_from_image_discrete(img, verbose = True, with_head_tail=True)
  w.plot(image = img);
  plt.scatter(cntr[:,0], cntr[:,1])
  
  # target
  ig.t_counter = 500000 + 25620 - 1;
  ig.wid_counter = 0;
  #imgs, skels, valids, hts, htvs = ig.get_batch(nimages = 1);
  imgs, cntrs = ig.get_batch(nimages = 1);
  imgt = imgs[0,:,:,0]; cntrt = cntrs[0];
  plt.figure(21); plt.clf();
  w.plot(image = imgt);
  plt.scatter(cntrt[:,0], cntrt[:,1])
  
  #Test center to left,right
  wh = w.width; wh[wh < 1.0] = 1.0; 
  l1,r1 = w.shape();
  length = w.length;
  
  width = tf.constant(wh /2.0, "float32");
  center = tf.constant(w.center, "float32");
  
  left,right = wmv.create_left_right(center, width)
  
  sess = None;
  sess = tf.InteractiveSession()
  init = tf.initialize_all_variables();
  sess.run(init)   
  l2= sess.run(left);
  r2= sess.run(right);
  
  plt.figure(1); plt.clf();
  plt.scatter(l1[:,0], l1[:,1], c='g');
  plt.scatter(r1[:,0], r1[:,1], c='r');
  plt.scatter(l2[:,0], l2[:,1], c='k');
  plt.scatter(r2[:,0], r2[:,1], c='m'); 
  
  
  ### Cost functions 
  wb = 0; ws = 0.1; wd = 0.1
  
  npts = w.npoints;
  left = wmv.create_variable([npts, 2]);
  right = wmv.create_variable([npts, 2]);
  
  
  wh = w.width; wh[wh < 1.0] = 1.0;
  width = tf.constant(wh, "float32");
  length = w.length;
  contour = tf.constant(cntrt, "float32");
  
  cost = wmv.create_cost(left, right, contour, width, length, weight_bending = wb, weight_spacing = ws, weight_distance = wd);
  
  cost_left = wmv.create_cost_side(left, contour, length, weight_spacing = ws, weight_bending = wb);
  cost_right = wmv.create_cost_side(right, contour, length, weight_spacing = ws, weight_bending = wb); 
  cost_left_bend = wmv.create_cost_bending(left);
  cost_right_bend = wmv.create_cost_bending(right);
  cost_left_spacing = wmv.create_cost_spacing(left, length);
  cost_right_spacing = wmv.create_cost_spacing(right, length);
  
  cost_dist = wmv.create_cost_distance(left, right, width);
  cost_left_contour = wmv.create_cost_soft_min_distance(left, contour);
  cost_right_contour = wmv.create_cost_soft_min_distance(right, contour);
  
  costs = [cost, cost_left, cost_right, cost_left_bend, cost_right_bend, cost_left_spacing, cost_right_spacing, 
               cost_dist, cost_left_contour, cost_right_contour ];
  costs_names = ['cost', 'cost_left', 'cost_right', 'cost_left_bend', 'cost_right_bend', 'cost_left_spacing', 'cost_right_spacing', 
                 'cost_dist', 'cost_left_contour','cost_right_contour' ];
  costs_weights = [1, 1, 1, wb, wb, ws, ws, wd, 1, 1];
  
  grad = tf.gradients(cost, [left,right]);
  
  
  ### Tensoroflow  - Session 
  sess = None;
  sess = tf.InteractiveSession()
  init = tf.initialize_all_variables();
  sess.run(init)      
  
  l,r = w.shape();
  assign_left  = left.assign(l);
  assign_right = right.assign(r);
  sess.run(assign_left); sess.run(assign_right);
  g = sess.run(grad);
  print g

  ### Compare costs
  for c,n,ww in zip(costs, costs_names, costs_weights):
    print '%20s: %f' % (n, ww * sess.run(c));

  
  ### Manual Gradient descent
  
  l1,r1 = w.shape();
  sg = .75;
  nsteps = 100;
  for i in range(nsteps): 
    sess.run(left.assign(l1));
    sess.run(right.assign(r1));
    g = sess.run(grad);
    l1 = l1 - sg * g[0] / np.sqrt(np.sum(g[0]*g[0]));
    r1 = r1 - sg * g[1] / np.sqrt(np.sum(g[1]*g[1]));
    
    plt.figure(10); plt.clf();
    plt.imshow(imgt, cmap = 'jet');
    plt.scatter(l1[:,0], l1[:,1], s = 60, c = 'r');
    plt.scatter(r1[:,0], r1[:,1], s = 60, c = 'g');
    plt.plot(l1[:,0], l1[:,1], c = 'r');
    plt.plot(r1[:,0], r1[:,1], c = 'g');
    plt.scatter(l1[0,0], l1[0,1], c = 'k', s = 150)
    plt.scatter(cntrt[0,0], cntrt[0,1], c = 'k', s = 150)
    plt.scatter(cntrt[:,0], cntrt[:,1])   

    plt.title('cost %f' % sess.run(cost));
    plt.draw();
    plt.pause(0.05);
  
  
  
  
  c = left; s = contour;
  c_shape = c.get_shape().as_list();        
  s_shape = s.get_shape().as_list();
  
  #expand matrices
  cc = tf.reshape(c, [c_shape[0], c_shape[1], 1]);    
  ss = tf.reshape(s, [s_shape[0], s_shape[1], 1]);
  ss = tf.transpose(ss, perm = [2,1,0]);
  cc = tf.tile(cc, [1, 1, s_shape[0]]);
  ss = tf.tile(ss, [c_shape[0], 1, 1]);
  #cc = tf.transpose(cc, perm = [2,1,0]);
  #cc = tf.tile(cc, [s_shape[0], 1, 1]);
  #ss = tf.tile(ss, [1, 1, c_shape[0]]);  
  
  #pairwise distances
  dist2 = tf.sqrt(tf.reduce_sum(tf.squared_difference(cc,ss), reduction_indices = 1));
    
  #softmin
  k = 100000.0
  softmin = tf.mul(tf.nn.softmax(tf.scalar_mul(tf.constant(-k,"float32"), dist2)), dist2);
  softmin2 = tf.reduce_sum(tf.mul(tf.nn.softmax(tf.scalar_mul(tf.constant(-k,"float32"), dist2)), dist2),reduction_indices = 1);  
    
  plt.figure(11); plt.clf();
  plt.subplot(2,2,1);
  plt.imshow(dist2.eval(), interpolation = 'none'); plt.colorbar()
  plt.subplot(2,2,2);
  plt.imshow(softmin.eval(), interpolation = 'none'); plt.colorbar()
  plt.subplot(2,2,3);
  plt.plot(softmin2.eval())
  
  
  
  
  ### Tensorflow optimization
  #trainer = tf.train.AdadeltaOptimizer().minimize(cost, var_list=[par]);
  trainer = tf.train.GradientDescentOptimizer(learning_rate=2.0).minimize(cost);
  init = tf.initialize_all_variables();
  sess.run(init)   
  
  sess.run(assign_op)
  
  nsteps = 1000;
  for i in range(nsteps):
    trainer.run(session = sess, feed_dict={});
    p1 = par.eval(session = sess);
    
    if i%10 == 0:
      plt.figure(10); plt.clf();
      w.center = p1[0];
      w.plot(image= imgt);
      plt.scatter(skels[0,:,0], skels[0,:,1], c= 'm')
      plt.xlim(0,151); plt.ylim(0,151)
      plt.draw();
      plt.pause(0.1);