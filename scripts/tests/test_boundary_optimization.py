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

def create_cost_distance(self, l, r, d):
  dd = tf.reduce_sum(tf.squared_difference(l,r), reduction_indices=1);
  dd = tf.squared_difference(dd, d);
  return tf.reduce_mean(dd);

def create_cost_soft_min_distance(self, c, s):
  """Creates a soft-min distance of the centers to the points"""
  c_shape = c.get_shape().as_list();        
  s_shape = s.get_shape().as_list();
  
  #expand matrices
  cc = tf.reshape(c, [c_shape[0], c_shape[1], 1]);    
  ss = tf.reshape(s, [s_shape[0], s_shape[1], 1]);
  ss = tf.transpose(ss, perm = [0,2,1]);
  cc = tf.tile(cc, [1, 1, s_shape[0]]);
  ss = tf.tile(ss, [c_shape[0], 1, 1]);
  
  #pairwise distances
  dist2 = tf.sqrt(tf.reduce_sum(tf.squared_difference(cc,ss), reduction_indices = 1));
  dist2 = tf.reduce_mean(dist2, reduction_indices=0); # hack: get rid of batches here 
  
  #softmin
  return tf.reduce_sum(tf.mul(tf.nn.softmax(tf.scalar_mul(tf.constant(-1.0,"float32"), dist2)), dist2),reduction_indices = 0);

def create_cost_spacing(self, c, length, normalized = True):
  c_shape = c.get_shape().as_list();
  c1 = tf.slice(c, [1,0], [-1,-1]);
  c2 = tf.slice(c, [0,0], [c_shape[0]-1,-1]);
  d = tf.sqrt(tf.reduce_sum(tf.squared_difference(c1,c2), reduction_indices = 1));
  if normalized:
    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length / (c_shape[0]-1), "float32")));
  else:
    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length, "float32")));
  
def create_cost_bending(self, c):
  c_shape = c.get_shape().as_list();
  c1 = tf.slice(c, [1,0], [-1,-1]);
  c2 = tf.slice(c, [0,0], [c_shape[0]-1,-1]);
  dc = tf.sub(c1,c2);
  dc1 = tf.slice(dc, [1,0], [-1,-1]);
  dc2 = tf.slice(dc, [0,0], [c_shape[0]-2,-1]);
  dn1 = tf.sqrt(tf.reduce_sum(tf.square(dc1), reduction_indices = 1));
  dn2 = tf.sqrt(tf.reduce_sum(tf.square(dc2), reduction_indices = 1));
  dp = tf.reduce_sum(tf.mul(dc1, dc2), reduction_indices = 1);
  dp = tf.div(tf.div(dp, dn1), dn2);
  return tf.mul(tf.constant(-1.0, "float32"), tf.reduce_mean(dp));

def create_cost_side(self, s, b, length = 1.0, weight_spacing = 1.0, weight_bending = 1.0):
  cost = self.create_cost_soft_min_distance(s, b);
  if weight_spacing != 0:
    cost_spacing = self.create_cost_spacing(s, length);
    cost = tf.add(cost, tf.mul(tf.constant(weight_spacing, "float32"),  cost_spacing));
  if weight_bending != 0:
    cost_bending = self.create_cost_bending(s);
    cost = tf.add(cost, tf.mul(tf.constant(weight_bending, "float32"), cost_bending));
  return cost;

def create_cost(self, l, r, b, w, length, weight_spacing = 1.0, weight_bending = 1.0, weight_distance = 1.0):
  #left right boundaries
  #length = self.model.length;
  cost_left = self.create_cost_side(l,b, length, weight_spacing = weight_spacing, weight_bending = weight_bending);
  cost_right= self.create_cost_side(r,b, length, weight_spacing = weight_spacing, weight_bending = weight_bending);
  
  #keep distance between sides of worm
  cost_dist = self.create_cost_distanace(l,r, w);
  
  cost = tf.add(cost_left, cost_right);
  cost = tf.add(cost, tf.mul(tf.constant(weight_distance, "float32"), cost_dist));
  return cost;




def test():
  import numpy as np
  import tensorflow as tf
  import matplotlib.pyplot as plt;
  import worm.model as wm;
  import worm.machine_vision_3 as wmv
  import worm.geometry as wgeo
  
  reload(wgeo)
  reload(wmv)  
  
  ### Prepare optimization task
  
  # work shape
  w = wm.WormModel(length = 80);
  ig = wmv.ImageGenerator();
  net = wmv.WormVision(model = w, images = ig);  
  
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
  
  
  ### Cost functions 
  wb = 2; ws = 1.0; wd = 2.0
  
  npts = w.npoints;
  left = net.create_variable([1, npts]);
  right = net.create_variable([1, npts]);
  
  width = tf.constant(w.width, "float32");
  length = tf.constant(w.length, "float32");
  contour = tf.constant(cntrs, "float32");
  
  cost = net.create_cost(left, right, contour, width,  weight_bending = wb, weight_spacing = ws);
  cost_bend = net.create_cost_bending(par);
  cost_spacing = net.create_cost_spacing(par);
  cost_dist = net.create_cost_soft_min_distance(par, skel);

  grad = tf.gradients(cost, [par]);
  
  ### Tensoroflow 
  ### Session 
  sess = None;
  sess = tf.InteractiveSession()
  init = tf.initialize_all_variables();
  sess.run(init)      
  
  assign_op = par.assign(w.center[None,:,:]);
  sess.run(assign_op)

  ### Compare costs

  cb = wb * sess.run(cost_bend);
  cs = ws * sess.run(cost_spacing);
  cd = sess.run(cost_dist);
  c  = sess.run(cost);
  print 'Costs: full: %f;  dist: %f;  bend: %f;  spacing :%f' % (c, cd, cb, cs);
  
  
  ### Manual Gradient descent
  
  p1 = w.center;
  sg = .75;
  nsteps = 100;
  for i in range(nsteps): 
    sess.run(par.assign(p1[None,:,:]));
    g = sess.run(grad)[0][0];
    p1 = p1 - sg * g / np.sqrt(np.sum(g*g));
    
    
    plt.figure(10); plt.clf();
    #plt.subplot(1,2,1)
    #w.plot(image = img);

    w.center = p1;
    #plt.subplot(1,2,2);
    w.plot(image= imgt);

    plt.title('cost %f' % sess.run(cost));
    
    plt.draw();
    plt.pause(0.05);
      
  
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