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
import tempfile
import dill as pickle
import tensorflow as tf

import analysis.experiment as exp
import imageprocessing.masking as msk


class ImageGenerator(object):
  """Generate Images and Phi's from data set"""
  def __init__(self, wids = [80,96], strains = ['n2']*2, tmins = [400000]*2, tmaxs = [500000]*2, smooth = 1.0):
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
  
  def get_image(self, stride = 1, random = False, smooth = 1.0):
    if random:
      wc = np.random.randint(0, self.nwids);
      t = np.random.randint(self.tmins[wc], self.tmaxs[wc]);
      img = exp.load_img(strain = self.strains[wc], wid = self.wids[wc], t = t, smooth = smooth);
    else:
      t = self.t_counter;
      wid = self.wids[self.wid_counter];
      img = exp.load_img(wid = wid, t = t);
      self.increase_counter(stride = stride);
    
    img.shape = (1,) + img.shape + (1,);
    return img;
  
  def phi_from_image(self, img, threshold = 75):
    return msk.mask_to_phi(img < threshold);
  
  def get_batch(self, nimages, stride = 1, random = False, threshold = 75):
    imgs = np.zeros((nimages,) + self.image_size + (1,));
    phis = np.zeros((nimages,) + self.image_size + (1,));
    for i in range(nimages):
      imgs[i,:,:,:] = self.get_image(random = random, stride = stride)[0];
      phis[i,:,:,0] = self.phi_from_image(imgs[i,:,:,0], threshold = threshold);
    return (imgs, phis);



#register python function and gradient in tensor flow



def py_func(func, inp, Tout, name=None, grad=None):
  """Redfine tf.py_func to include gradients"""
  temp_name = next(tempfile._get_candidate_names())
  _name = 'PyFuncGrad%s' %temp_name;
  tf.RegisterGradient(_name)(grad)
  g = tf.get_default_graph()
  with g.gradient_override_map({"PyFunc": _name}):
    return tf.py_func(func, inp, Tout, name=name)



class WormVision(object):
  """Neural Network to map images to worm shape parameter"""
  
  def __init__(self, model, images):
    self.model = model;
    self.images = images;
    self.image_size = self.images.image_size;
    self.nparameter = model.nparameter; # + 1; # orientation is a phase variable -> phi = np.arctan2(phi_x,phi_y)
    self.create_network(nparameter = self.nparameter, image_size = self.image_size);
    self.create_training(image_size = self.image_size);
    
    self.saver = tf.train.Saver();
        
    self.session = tf.Session()
    self.session.run(tf.initialize_all_variables())

    
  
  def create_network(self, nparameter, image_size = [151,151]):
    """Creates the neural network"""
    
    # input
    self.input = tf.placeholder("float32",[None,image_size[0],image_size[0],1]);
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
    self.output = tf.nn.tanh(tf.matmul(self.fc4, w) + b);
  
  
  def create_weights(self, shape, stddev = 0.01):
    """Create a weight matrix for the network"""
    w = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(w);

  def create_bias(self, shape, value = 0.01):
    """Create bias es for the network"""
    b = tf.constant(value, shape = shape)
    return tf.Variable(b)

  def create_conv2d(self, x, w, b, stride = 1, name = None):
    """Creates a convolutional layer for the network"""
    x = tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='VALID', name = name)
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)    
  
  
  def create_training(self, image_size = [151,151]):
    """Create the cost function and trainer"""
    self.phi_input = tf.stop_gradient(tf.placeholder("float32", [None, image_size[0], image_size[1], 1]));
    def cost(output, phi_in):
       #return np.array([self.cost(o, phi_in) for o in output]);
      return np.sum(self.cost_func(output, phi_in));
    
    def cost_grad(op, grad):
      #print op
      output = op.inputs[0];
      phi = op.inputs[1];
      grad = tf.py_func(self.cost_func_grad, [output, phi], [tf.float32])[0];
      #return [self.cost_func_grad(output, phi_in, epsilon = 0.01), np.zeros((phi_in.shape))];
      return [grad, None];
      
    self.cost_tf = py_func(cost, [self.output, self.phi_input], [tf.float32], grad = cost_grad)[0];
    #self.cost_tf = tf.py_func(cost, [self.output, self.phi_input], [tf.float64])[0];
    #self.phi = tf.py_func(phi_func, [self.output], [tf.float64]);  
    #self.cost = tf.reduce_mean(tf.squared_difference(self.phi_input, self.phi));
    
    self.train_tf = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost_tf)

  def cost(self, images, phis):
    return self.cost_tf.eval(session = self.session, feed_dict = {self.input : images, self.phi_input : phis});
  
  def train(self, nsamples = 1, verbose = False, random = True):
    """Train the network on a batch of images"""
    imgs, phis = self.images.get_batch(nimages = nsamples, random = random);
    self.train_tf.run(session = self.session, feed_dict={self.input : imgs, self.phi_input : phis})
    if verbose:
      self.plot_results(imgs);
  
  
  def cost_func(self, output, phi):
    """Cost function"""
    nsamples = output.shape[0];
    cost = np.zeros(nsamples, dtype = 'float32');
    for i in range(nsamples):
      self.set_parameter(output[i]);
      phi_o = self.model.phi();
      cost[i] = np.mean(np.square(phi_o - phi[i,:,:,0]));
    return cost;
  
  def cost_func_grad(self, output, phi, epsilon = 0.1):
    """Numerical approximation of the gradient of the cost function"""
    nsamples,noutput = output.shape;
    grad = np.zeros((nsamples, noutput), dtype = 'float32');
    c0s = self.cost_func(output, phi);
    epsilon = np.array(epsilon);
    if epsilon.ndim == 0:
      epsilon = np.ones(noutput) * epsilon;
    for i in range(noutput):
      o = output.copy(); o[:,i] += epsilon[i];
      cs = self.cost_func(o, phi);
      grad[:,i] = (cs - c0s) / epsilon[i];
    return grad;
  
  def parameter_from_output(self, output):
    """Converts network output to worm model parameter"""
    i = self.model.theta.shape[0];
    if output.ndim == 1:
      return np.hstack([np.pi * output[:i], [np.arctan2(output[i], output[i+1])], self.image_size * (output[i+2:] + 1) * 0.5, [0]]);
    else:
      return np.concatenate([np.pi * output[:, :i], np.atleast_2d(np.arctan2(output[:,i], output[:,i+1])), self.image_size * (output[:,i+2:] + 1) * 0.5, [[0]]], axis = 1);
  
  def set_parameter(self, output):
    """Sets the model parameter from the network output"""
    self.model.set_parameter(self.parameter_from_output(output));
  
  def get_output(self, images):
    """Returns the worm model parameter"""
    return self.output.eval(session = self.session, feed_dict = {self.input : images});
    
  def get_parameter(self, images):
    return self.parameter_from_output(self.get_output(images));
  
  def get_phi(self, images):
    """Returns the phi generated from the worm model parameter inffered form the image input"""
    nsamples = images.shape[0];
    pars = self.get_parameter(images);
    phis = np.zeros((nsamples,) + self.image_size);
    for i in range(nsamples):
      self.model.set_parameter(pars[i]);
      phis[i] = self.model.phi();
    return phis;
    
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
  import matplotlib.pyplot as plt;
  import worm.model as wm;
  import worm.machine_vision as wmv
  reload(wmv)
  
  w = wm.WormModel(length = 80);
  ig = wmv.ImageGenerator();
  net = wmv.WormVision(model = w, images = ig);
  
  #test network
  o = net.output.eval(session = net.session, feed_dict = {net.input : ig.get_image()})
  
  phi = net.get_phi(ig.get_image());
  plt.figure(1); plt.clf();
  plt.imshow(phi[0]);
  
  
  plt.figure(2); plt.clf();
  net.plot_results(ig.get_image());
  
  
  plt.figure(3); plt.clf();
  i = net.model.theta.shape[0];
  oo = o[0];
  oo[i] = 1; oo[i+1] = 1;
  oo[:i] = 2 * np.random.rand(i) - 1;
  net.set_parameter(oo);
  net.model.plot();
  
  # get cost
  reload(wmv)
  
  w = wm.WormModel(length = 80);
  ig = wmv.ImageGenerator();
  net = wmv.WormVision(model = w, images = ig);  
  
  imgs, phis = ig.get_batch(nimages = 1);
   
  net.cost(imgs, phis)
  
  output = net.get_output(imgs)
  eps = 0.1 * np.ones(output.shape[1]);
  ik = net.model.theta.shape[0];
  eps[:ik] = 1;
  eps[:] = 0.1;
  g = net.cost_func_grad(output, phis, epsilon = eps)
  print g

  # try gradient descent
  nsteps = 100;
  sw = 0.1;
  for i in range(nsteps):
    out_0 = output[0];
    gg = g[0];
    gg[ik:] *= 0.1;
    out_1 = out_0 - sw * (gg / np.abs(gg).sum());
    output = np.array([out_1]);
    g = net.cost_func_grad(output, phis, epsilon = eps)
    
    
    plt.figure(10); plt.clf();
    plt.subplot(2,4,1)
    net.set_parameter(out_0);
    dd = np.square(net.model.phi()-phis[0,:,:,0]);
    c0 = dd.sum();
    net.model.plot(image = imgs[0,:,:,0])

    plt.subplot(2,4,2);
    #plt.imshow(net.model.phi()-phis[0,:,:,0]);
    plt.imshow(dd);
    plt.title('cost %f' % c0);

    plt.subplot(2,4,3);
    plt.imshow(net.model.phi());


    plt.subplot(2,4,4);
    plt.imshow(phis[0,:,:,0]);

    plt.subplot(2,4,5)
    net.set_parameter(out_1);
    dd = np.square(net.model.phi()-phis[0,:,:,0]);
    c1 = dd.sum();
    net.model.plot(image = imgs[0,:,:,0])

    plt.subplot(2,4,6);
    #plt.imshow(net.model.phi()-phis[0,:,:,0]);
    plt.imshow(dd);
    plt.title('cost %f' % c1);

    plt.subplot(2,4,7);
    plt.imshow(net.model.phi());

    plt.subplot
    plt.draw();
    plt.pause(0.1);
      
    
  

  

    
  plt.figure(3); plt.clf();
  n = imgs.shape[0];
  for i in range(imgs.shape[0]):
    plt.subplot(2,n,i+1);
    plt.imshow(imgs[i,:,:,0]);
    plt.subplot(2,n,i+n+1);
    plt.imshow(phis[i,:,:,0]);
    
    
  ### test loading saving
  import matplotlib.pyplot as plt;
  import worm.model as wm;
  import worm.machine_vision as wmv
  reload(wmv)
  
  w = wm.WormModel(length = 80);
  ig = wmv.ImageGenerator();
  net = wmv.WormVision(model = w, images = ig);  
  
  net.save('test.ckpt');
    
  ### time training the network
  import matplotlib.pyplot as plt;
  import worm.model as wm;
  import worm.machine_vision as wmv
  reload(wmv)
  
  w = wm.WormModel(length = 80);
  ig = wmv.ImageGenerator();
  net = wmv.WormVision(model = w, images = ig);  
  
  from utils.timer import timeit
  
  @timeit
  def train():
    nsteps = 100;
    for s in range(nsteps):
      print 'training %d/%d' % (s, nsteps);
      if s % 10 == 0:
        plt.figure(1); plt.clf();
        net.train(nsamples = 1, verbose=True);
        plt.title('training %d/%d' % (s, nsteps));
        plt.draw();
        plt.pause(0.01)
      else:
        net.train(nsamples = 1);
      if s % 10000 == 0:
        net.save('wormvision.ckpt');
    
  train()
  
  
  ### train the network
  
  import matplotlib.pyplot as plt;
  import worm.model as wm;
  import worm.machine_vision as wmv
  reload(wmv)
  
  w = wm.WormModel(length = 80);  
  ig = wmv.ImageGenerator();

  img = ig.get_image()[0,:,:,0];
  w.from_image(img)  
  plt.figure(1); plt.clf();
  w.plot(image=img)  
  
  net = wmv.WormVision(model = w, images = ig);  
  
  nsteps = 100000;
  for s in range(nsteps):
    print 'training %d/%d' % (s, nsteps);
    if s % 10 == 0:
      plt.figure(1); plt.clf();
      net.train(nsamples = 1, verbose=True);
      plt.title('training %d/%d' % (s, nsteps));
      plt.draw();
      plt.pause(0.01)
    else:
      net.train(nsamples = 1);
    if s % 10000 == 0:
      net.save('wormvision.ckpt');