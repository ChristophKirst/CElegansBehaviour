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
    
    #img.shape = (1,) + img.shape + (1,);
    return img;
  
  def contour_from_image(self, img, threshold = 75):
    cntrs = wgeo.contours_from_image(img, sigma = 1, absolute_threshold = threshold, verbose = False, save = None);
    nc = len(cntrs);

    if nc == 1:
      cntrs = (ir.resample(cntrs[0], self.contour_size),);
    else:
      cntrs = (ir.resample(cntrs[0], self.contour_size-self.contour_inner), ir.resample(cntrs[1], self.contour_inner));
    
    #calculate normals
    nrmls = [wgeo.normals_from_contour_discrete(c) for c in cntrs];
    cntrs = tuple([c[:-1] for c in cntrs]);
    
    return np.vstack(cntrs), np.vstack(nrmls)
    

  
  def get_batch(self, nimages = 1, stride = 1, random = False, threshold = 75):
    img = self.get_image(random = random, stride = stride);
    cntr, nrml = self.contour_from_image(img);
    return (img, cntr, nrml)


def create_tangent(c):
  #tanget vectors
  c_shape = c.get_shape().as_list();
  c1 = tf.slice(c, [1,0], [-1,-1]);
  c2 = tf.slice(c, [0,0], [c_shape[0]-1,-1]);
  return tf.sub(c1,c2);
  
def create_normalize_tangent(t):
  #normalized tangent vectors
  nr = tf.sqrt(tf.reduce_sum(tf.square(t), reduction_indices = 1));
  return tf.transpose(tf.div(tf.transpose(t), nr));


def create_normal(tn):
  tn_shape = tn.get_shape().as_list();
  
  #average tangent
  tn1 = tf.concat(0, [tf.slice(tn, [0,0], [1,-1]), tn]);
  tn2 = tf.concat(0, [tn, tf.slice(tn, [tn_shape[0]-1,0], [1,-1])]);
  av = tf.add(tn1,tn2);
  nr = tf.sqrt(tf.reduce_sum(tf.square(av), reduction_indices = 1));
  av = tf.transpose(tf.div(tf.transpose(av), nr));
  
  #normal
  nrm = tf.mul(av, tf.constant([-1,1], "float32"));
  nrm = tf.reverse(nrm, [False, True]);
  return nrm;

def create_left_right(c, w, nrm): 
  left = tf.transpose(tf.mul(tf.transpose(nrm), w));
  right= tf.scalar_mul(-1.0, left);
  
  left = tf.add(left, c);
  right= tf.add(right,c);
  
  return left,right

#def create_dot(v1, v2):
#  return tf.reduce_sum(tf.mul(v1, v2), reduction_indices = 1);

def create_soft_min(d, k = 2.0):    
  softmin = tf.reduce_sum(tf.mul(tf.nn.softmax(tf.scalar_mul(-k, d)), d),reduction_indices = 1);
  return softmin;

def create_aligned_distance(d, a, gamma = 1.0, k = 2.0): #alignment [0,1]
  dm = tf.reduce_sum(tf.mul(tf.nn.softmax(d), d));
  da = tf.add(d, tf.scalar_mul(gamma, tf.mul(dm,a)));
  return create_soft_min(da, k);

def create_cost_distance(l, r, d):
  dd = tf.sqrt(tf.reduce_sum(tf.squared_difference(l,r), reduction_indices=1));
  dd = tf.squared_difference(dd, d);
  return tf.reduce_mean(dd);

def create_pair_wise_distances(x, y):
  x_shape = x.get_shape().as_list();        
  y_shape = y.get_shape().as_list();
  
  #expand matrices
  xx = tf.reshape(x, [x_shape[0], x_shape[1], 1]);    
  yy = tf.reshape(y, [y_shape[0], y_shape[1], 1]);
  yy = tf.transpose(yy, perm = [2,1,0]);
  xx = tf.tile(xx, [1, 1, y_shape[0]]);
  yy = tf.tile(yy, [x_shape[0], 1, 1]);
  #cc = tf.transpose(cc, perm = [2,1,0]);
  #cc = tf.tile(cc, [s_shape[0], 1, 1]);
  #ss = tf.tile(ss, [1, 1, c_shape[0]]); 
  
  #pairwise distances
  dist = tf.sqrt(tf.reduce_sum(tf.squared_difference(xx,yy), reduction_indices = 1));
  return dist;

def create_pair_wise_dots(x, y):
  x_shape = x.get_shape().as_list();        
  y_shape = y.get_shape().as_list();
  
  #expand matrices
  xx = tf.reshape(x, [x_shape[0], x_shape[1], 1]);    
  yy = tf.reshape(y, [y_shape[0], y_shape[1], 1]);
  yy = tf.transpose(yy, perm = [2,1,0]);
  xx = tf.tile(xx, [1, 1, y_shape[0]]);
  yy = tf.tile(yy, [x_shape[0], 1, 1]);
  
  #pairwise dot products
  dots = tf.reduce_sum(tf.mul(xx,yy), reduction_indices = 1);
  return dots;


def create_cost_soft_min_distance(x, y, k = 2.0):
  d = create_pair_wise_distances(x,y,k);
  return  tf.reduce_mean(create_soft_min(d, k));


def create_cost_soft_min_aligned_distance(x,y,nx,ny, k = 2.0, gamma = 1.0):
  d = create_pair_wise_distances(x, y);
  a = create_pair_wise_dots(nx, ny);
  a = tf.scalar_mul(-0.5, tf.add(a, -1.0)); # [0,1] 0 = aligned
  return tf.reduce_mean(create_aligned_distance(d, a, k = k, gamma = gamma));


#def create_cost_spacing(c, length, normalized = True):
#  c_shape = c.get_shape().as_list();
#  c1 = tf.slice(c, [1,0], [-1,-1]);
#  c2 = tf.slice(c, [0,0], [c_shape[0]-1,-1]);
#  d = tf.sqrt(tf.reduce_sum(tf.squared_difference(c1,c2), reduction_indices = 1));
#  if normalized:
#    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length / (c_shape[0]-1), "float32")));
#  else:
#    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length, "float32")));


def create_cost_spacing(t, length, normalized = True):
  d = tf.sqrt(tf.reduce_sum(tf.square(t), reduction_indices = 1));
  if normalized:
    s = t.get_shape().as_list();
    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length / s[0], "float32")));
  else:
    return tf.reduce_mean(tf.squared_difference(d, tf.constant(length, "float32")));

def create_cost_bending(tn):
  tn_shape = tn.get_shape().as_list();
  tn1 = tf.slice(tn, [1,0], [-1,-1]);
  tn2 = tf.slice(tn, [0,0], [tn_shape[0]-1,-1]);
  dp = tf.reduce_sum(tf.mul(tn1, tn2), reduction_indices = 1);
  return tf.scalar_mul(-1.0, tf.reduce_mean(dp));

#def create_cost_side(s, b, length = 1.0, weight_spacing = 1.0, weight_bending = 1.0):
#  cost = create_cost_soft_min_distance(s, b);
#  if weight_spacing != 0:
#    cost_spacing = create_cost_spacing(s, length);
#    cost = tf.add(cost, tf.mul(tf.constant(weight_spacing, "float32"), cost_spacing));
#  if weight_bending != 0:
#    cost_bending = create_cost_bending(s);
#    cost = tf.add(cost, tf.mul(tf.constant(weight_bending, "float32"), cost_bending));
#  return cost;

def create_cost(c, w, b, nb, length, weight_spacing = 1.0, weight_bending = 1.0, gamma = 1.0, kappa = 2.0):
  #tangents
  t  = create_tangent(c);
  tn = create_normalize_tangent(t);
  nl = create_normal(tn);
  nr = tf.scalar_mul(-1.0, nl);
  
  l,r = create_left_right(c,w,nl);

  cost_left = create_cost_soft_min_aligned_distance(l, b, nl, nb, k = kappa, gamma = gamma);
  cost_right= create_cost_soft_min_aligned_distance(r, b, nr, nb, k = kappa, gamma = gamma);
  cost = tf.add(cost_left, cost_right);
  
  #spacing and bending
  if weight_spacing != 0:
    cost_spacing = tf.scalar_mul(weight_spacing, create_cost_spacing(t, length));
    cost = tf.add(cost, cost_spacing);
  else:
    cost_spacing = tf.constant(0);
  
  if weight_bending != 0:
    cost_bending = tf.scalar_mul(weight_bending, create_cost_bending(tn));
    cost = tf.add(cost, cost_bending);
  else:
    cost_bending = tf.constant(0);
  
  return (cost, cost_left, cost_right, cost_spacing, cost_bending, nl, l, r);

def create_variable(shape, value = 0.01):
  """Create variable"""
  v = tf.constant(value, shape = shape)
  return tf.Variable(v);


def test():
  import numpy as np
  import tensorflow as tf
  import matplotlib.pyplot as plt;
  import worm.model as wm;
  import worm.machine_vision_c as wmv
  import worm.geometry as wgeo
  
  reload(wgeo)
  reload(wmv)  
  
  ### Prepare optimization task
  
  # work shape
  w = wm.WormModel(length = 80);
  ig = wmv.ImageGenerator();
    
  ig.t_counter = 500000 + 25620 - 5;
  ig.wid_counter = 0;

  img, cntr, nrml = ig.get_batch(nimages = 1);
  w.from_image(img)  

  plt.figure(20); plt.clf();
  #wgeo.skeleton_from_image_discrete(img, verbose = True, with_head_tail=True)
  w.plot(image = img);
  plt.scatter(cntr[:,0], cntr[:,1])
  
  # target
  ig.t_counter = 500000 + 25620 - 1;
  ig.wid_counter = 0;
  #imgs, skels, valids, hts, htvs = ig.get_batch(nimages = 1);
  imgt, cntrt, nrmlt = ig.get_batch(nimages = 1);
  plt.figure(21); plt.clf();
  w.plot(image = imgt);
  plt.scatter(cntrt[:,0], cntrt[:,1])
  
  w2 = w.copy();
  
  
  ### Cost functions 
  wb = 1.0; ws = 0.5; gamma = 1.0; kappa = 10.0
  
  npts = w.npoints;
  center = wmv.create_variable([npts, 2]);
  
  wh = w.width; wh[wh < 1.0] = 1.0;
  width = tf.constant(wh / 2.0, "float32");
  length = w.length + 20;
  contour = tf.constant(cntrt, "float32");
  contour_normals = tf.constant(nrmlt, "float32");
  
  cost, cost_left, cost_right, cost_spacing, cost_bending, normals, left, right = wmv.create_cost(center, width, contour, contour_normals, length,
                                                                            weight_bending = wb, weight_spacing = ws, gamma = gamma, kappa = kappa);
  
  costs       = [ cost, cost_left, cost_right, cost_spacing, cost_bending];
  costs_names = ['cost', 'cost_left', 'cost_right', 'cost_spacing', 'cost_bending'];
  
  grad = tf.gradients(cost, [center]);
  
  
  ### Tensoroflow  - Session 
  sess = None;
  sess = tf.InteractiveSession()
  init = tf.initialize_all_variables();
  sess.run(init)      
  
  assign = center.assign(w.center);
  sess.run(assign); 
  g = sess.run(grad);
  print g

  ### Compare costs
  for c,n in zip(costs, costs_names):
    print '%20s: %f' % (n, sess.run(c));
  
  ### Manual Gradient descent
  
  c1 = w.center;
  w2 = w.copy();
  sg = .75;
  nsteps = 100;
  for i in range(nsteps): 
    sess.run(center.assign(c1));
    g = sess.run(grad);
    c1 = c1 - sg * g[0] / np.sqrt(np.sum(g[0]*g[0]));

    w2.center = c1;
    plt.figure(10); plt.clf();
    w2.plot(image = imgt);
    plt.scatter(cntrt[0,0], cntrt[0,1], c = 'k', s = 150)
    plt.scatter(cntrt[1,0], cntrt[1,1], c = 'r', s = 150)
    plt.scatter(w2.center[0,0], w2.center[0,1], c = 'r', s = 150)
    plt.scatter(cntrt[:,0], cntrt[:,1])
    for cc,nn in zip(cntrt, nrmlt):
      cn = cc + nn;
      plt.plot([cc[0], cn[0]], [cc[1], cn[1]], c = 'k')
    
    l,r = left.eval(), right.eval();
    nrmls = normals.eval();
    for cc,nn in zip(l, nrmls):
      cn = cc + nn;
      plt.plot([cc[0], cn[0]], [cc[1], cn[1]], c = 'k')

    for cc,nn in zip(r, -nrmls):
      cn = cc + nn;
      plt.plot([cc[0], cn[0]], [cc[1], cn[1]], c = 'k')


    plt.title('cost %f' % sess.run(cost));
    plt.xlim(40,120); plt.ylim(40, 120)
    plt.draw();
    plt.pause(0.05);
  
  
  
  
  # test routines
  t  = wmv.create_tangent(center);
  tn = wmv.create_normalize_tangent(t);
  nl = wmv.create_normal(tn);  
  l,r = wmv.create_left_right(center, width, nl);

  d = wmv.create_pair_wise_distances(l, contour);
  a = wmv.create_pair_wise_dots(nl, contour_normals);
  a = tf.scalar_mul(-0.5, tf.add(a, -1.0)); # [0,1] 0 = aligned
  da= wmv.create_aligned_distance(d,a,gamma = gamma, k = kappa);
  
  plt.figure(6); plt.clf();
  plt.subplot(2,2,1);
  plt.imshow(d.eval(), interpolation = 'none');
  plt.subplot(2,2,2);
  plt.imshow(a.eval(), interpolation = 'none');
  plt.subplot(2,2,3);
  plt.imshow(d.eval() + a.eval() * np.max(d.eval()), interpolation = 'none');
  plt.subplot(2,2,4);
  plt.plot(da.eval());
  
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