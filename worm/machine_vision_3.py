"""
Deep Network to map images to worm shapes

Note:
  Network structure img -> conv -> conv -> hidden -> worm parameter
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import tensorflow as tf

import analysis.experiment as exp

import interpolation.resampling as ir

import worm.geometry as wgeo

class ImageGenerator(object):
  """Generate Images and Phi's from data set"""
  def __init__(self, wids = [80,96], strains = ['n2']*2, tmins = [400000]*2, tmaxs = [500000]*2, smooth = 1.0, contour_size = 200, contour_inner = 50):
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



#register python function and gradient in tensor flow


class WormVision(object):
  """Neural Network to map images to worm shape parameter"""
  
  def __init__(self, model, images):
    self.model = model;
    self.images = images;
    self.image_size = self.images.image_size;
    self.nparameter = model.nparameter;
    self.npoints = self.nparameter /2;
    
    self.contour_size = images.contour_size;    

    self.create_network(nparameter = self.nparameter, image_size = self.image_size);
    self.create_training();
    
    self.saver = tf.train.Saver();
        
    self.session = tf.Session()
    self.session.run(tf.initialize_all_variables())
  
  
  def create_network(self, nparameter, image_size = [151,151]):
    """Creates the neural network"""
    
    # input
    self.input = tf.placeholder("float32",[1,image_size[0],image_size[0],1]);
    #self.input = tf.placeholder("float",[1,image_size[0],image_size[0],1]);
    
    # convolution layer
    self.conv1_w = w = self.create_weights([8,8,1,32])
    self.conv1_b = b = self.create_bias([32])
    self.conv1 = self.create_conv2d(self.input, w, b, stride = 4, name = 'Layer1');
    
    self.conv2_w = w = self.create_weights([4,4,32,64])
    self.conv2_b = b = self.create_bias([64])
    self.conv2 = self.create_conv2d(self.conv1, w, b, stride = 2, name = 'Layer2');
    
    self.conv3_w = w = self.create_weights([3,3,64,64])
    self.conv3_b = b = self.create_bias([64])
    self.conv3 = self.create_conv2d(self.conv2, w, b, stride = 1, name = 'Layer2');    
    
    # hidden layer
    conv3_shape = self.conv3.get_shape().as_list();
    conv3_n = conv3_shape[1] * conv3_shape[2] * conv3_shape[3];
    conv3_flat = tf.reshape(self.conv3,[-1,conv3_n]);
    #conv3_flat = tf.reshape(self.conv3,[conv3_n]);
    self.fc4_w = w = self.create_weights([conv3_n, 512]);
    self.fc4_b = b = self.create_bias([512]);
    self.fc4 = tf.nn.relu(tf.matmul(conv3_flat, w) + b)    
    
    # output layer
    self.output_w = w = self.create_weights([512, nparameter]);
    self.output_b = b = self.create_bias([nparameter]);
    
    out = tf.nn.tanh(tf.matmul(self.fc4, w) + b);
    self.output = tf.reshape(out, [1, self.npoints, 2]);
  
  
  def create_weights(self, shape, stddev = 0.01):
    """Create a weight matrix for the network"""
    w = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(w);

  def create_bias(self, shape, value = 0.01):
    """Create bias es for the network"""
    b = tf.constant(value, shape = shape)
    return tf.Variable(b)
    
  def create_variable(self, shape, value = 0.01):
    """Create variable"""
    v = tf.constant(value, shape = shape)
    return tf.Variable(v);

  def create_conv2d(self, x, w, b, stride = 1, name = None):
    """Creates a convolutional layer for the network"""
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='VALID', name = name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)    
  

  def create_cost_distance(self, l, r, d):
    dd = tf.reduce_sum(tf.squared_difference(l,r), reduction_indices=1);
    dd = tf.squared_difference(dd, d);
    return tf.reduce_mean(dd);
  
  def create_cost_soft_min_distance(self, c, s):
    """Creates a soft-min distance of the centers to the points"""
    c_shape = c.get_shape().as_list();        
    s_shape = s.get_shape().as_list();
    
    #expand matrices
    cc = tf.reshape(c, [c_shape[0], c_shape[1], c_shape[2], 1]);    
    ss = tf.reshape(s, [s_shape[0], s_shape[1], s_shape[2], 1]);
    ss = tf.transpose(ss, perm = [0,3,2,1]);
    cc = tf.tile(cc, [1, 1, 1, s_shape[0]]);
    ss = tf.tile(ss, [1, c_shape[0], 1, 1]);
    
    #pairwise distances
    dist2 = tf.sqrt(tf.reduce_sum(tf.squared_difference(cc,ss), reduction_indices = 2));
    dist2 = tf.reduce_mean(dist2, reduction_indices=0); # hack: get rid of batches here 
    
    #softmin
    distmin = tf.reduce_sum(tf.mul(tf.nn.softmax(tf.scalar_mul(tf.constant(-1.0,"float32"), dist2)), dist2),reduction_indices = 1);
    return tf.reduce_mean(distmin);
  
  def create_cost_spacing(self, c, length, normalized = True):
    c_shape = c.get_shape().as_list();
    c1 = tf.slice(c, [0,1,0], [-1,-1,-1]);
    c2 = tf.slice(c, [0,0,0], [-1,c_shape[1]-1,-1]);
    d = tf.sqrt(tf.reduce_sum(tf.squared_difference(c1,c2), reduction_indices = 2));
    if normalized:
      return tf.reduce_mean(tf.squared_difference(d, tf.constant(length / (c_shape[1]-1), "float32")));
    else:
      return tf.reduce_mean(tf.squared_difference(d, tf.constant(length, "float32")));
    
  def create_cost_bending(self, c):
    c_shape = c.get_shape().as_list();
    c1 = tf.slice(c, [0,1,0], [-1,-1,-1]);
    c2 = tf.slice(c, [0,0,0], [-1,c_shape[1]-1,-1]);
    dc = tf.sub(c1,c2);
    dc1 = tf.slice(dc, [0,1,0], [-1,-1,-1]);
    dc2 = tf.slice(dc, [0,0,0], [-1,c_shape[1]-2,-1]);
    dn1 = tf.sqrt(tf.reduce_sum(tf.square(dc1), reduction_indices =2));
    dn2 = tf.sqrt(tf.reduce_sum(tf.square(dc2), reduction_indices =2));
    dp = tf.reduce_sum(tf.mul(dc1, dc2), reduction_indices =2);
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
  
  def create_training(self, weight_spacing = 1.0, weight_bending = 1.0, weight_head_tail = 1.0):
    """Create the cost function and trainer"""
    #self.left = tf.placeholder("float32", [1, self.npoints, 2]);
    #self.right = tf.placeholder("float32", [1, self.npoints, 2]);
    self.width = tf.placeholder("float32", [self.npoints]);
    self.contour = tf.placeholder("float32", [1, self.contour_size, 2]);

    #self.cost = self.create_cost(self.output, self.skeleton, weight_spacing = weight_spacing, weight_bending = weight_bending); 
    
    #trainer
    #self.trainer = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)
  
  #def get_cost(self, images, skeleton, skeleton_valid, head_tail, head_tail_valid):
  #  return self.cost.eval(session = self.session, feed_dict = {self.input : images, self.skeleton : skeleton, self.skeleton_valid : skeleton_valid, 
  #                                                             self.head_tail : head_tail, self.head_tail_valid : head_tail_valid});    

  #def get_cost(self, images, skeleton):
  #  return self.cost.eval(session = self.session, feed_dict = {self.input : images, self.skeleton : skeleton});    
  
  #def train(self, nsamples = 1, verbose = False, random = True):
  #  """Train the network on a batch of images"""
  #  imgs, skels = self.images.get_batch(nimages = nsamples, random = random);
  #  self.trainer.run(session = self.session, feed_dict={self.input : imgs, self.skeleton : skels})
  #  if verbose:
  #    self.plot_results(imgs);
  
  
  def set_parameter(self, output):
    """Sets the model parameter from the network output"""
    self.model.set_parameter(output);
  
  def get_parameter(self, images):
    """Returns the worm model parameter"""
    return self.output.eval(session = self.session, feed_dict = {self.input : images});
  
    
  def plot_results(self,images):
    nsamples = images.shape[0];
    pars = self.get_parameter(images);
    for i in range(nsamples):
      plt.subplot(1,nsamples,i+1);
      self.model.set_parameter(pars[i]);
      self.model.plot(image = images[i,:,:,0]);
  
  def save(self, filename):
    #save tf graph
    save_path = self.saver.save(self.session, filename);
    # save class pareter
    filename_class = filename + '.pickle';
    pickle.dump([self.model, self.images], open(filename_class, 'wb'));
    print("WormVision saved to files: %s, %s" % (save_path, filename_class));
  
  
  def load(self, filename):
    #load class
    filename_class = filename + '.pickle'; 
    mi = pickle.load(open(filename_class, 'rb'));
    self.model  = mi[0]; self.images = mi[1];
    #load tf graph
    self.saver.restore(self.session, filename);


      


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


  
  import shapely.geometry as geom
  cntrs = wgeo.contours_from_image(imgt)
  poly = geom.Polygon(cntrs[0], [cntrs[i] for i in range(1,len(cntrs))]);
  poly2 = poly.buffer(-3)
  bdr = poly.boundary;
  
  from descartes.patch import PolygonPatch
  patch = PolygonPatch(poly, facecolor = 'b', edgecolor='k', alpha=0.5, zorder=2)
  patch2= PolygonPatch(poly2, facecolor = 'r', edgecolor='k', alpha=0.5, zorder=2)
  
  fig = plt.figure(28);
  ax = fig.add_subplot(111)
  ax.add_patch(patch)
  ax.add_patch(patch2)
  plt.xlim(0,151); plt.ylim(1,151)
  plt.show()
  
  
  
  ### make boundary pts in tensorflow
  
  npoints = 21;
  width = tf.placeholder("float32", shape = [npoints,2]);
  
  
  

  
  
  










