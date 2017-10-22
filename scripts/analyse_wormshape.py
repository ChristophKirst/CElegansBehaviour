# -*- coding: utf-8 -*-
"""
Worm Shapes

Long Term Behaviour Analysis of C-elegans

Detect worm shapes / postures
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'

import os
import numpy as np
import matplotlib.pyplot as plt

import analysis.experiment as exp
import analysis.plot as aplt
import worm.model as wmodel


npts = 22;
w = wmodel.WormModel(npoints = npts);

fn = exp.filename(strain = 'n2', wid = 80, dtype = 'img')
imgdata = np.load(fn, mmap_mode = 'r');

ntimes = imgdata.shape[0];

theta, orientation, xy, length = w.theta();
ntheta = theta.shape[0];

data_path = '/home/ckirst/Desktop';
data_names = ['theta', 'orientation', 'xy', 'length', 'center', 'width'];
data_sizes = [(ntimes, ntheta), ntimes, (ntimes,2), ntimes, (ntimes, npts, 2), (ntimes, npts)];

data_files = [os.path.join(data_path, 'analysis_2016_10_26_' + n + '.npy') for n in data_names];


for fn,s in zip(data_files, data_sizes):
  #fn = os.path.join(data_path, 'analysis_2016_10_25_' + n + '.npy');
  #np.lib.format.open_memmap(fn, mode = 'w+', shape  = s);
  d = np.zeros(s);
  np.save(fn, d);

import warnings
warnings.filterwarnings("ignore")

def process(i):
  img = exp.load_img(t=i);
  data = [np.load(fn, mmap_mode = 'r+') for fn in data_files];
  if i % 1000 == 0:
    print 'processing %d/%d...' % (i, ntimes);
  try:
    w.from_image(img, verbose = False);
    results = list(w.theta());
    results.append(w.center);
    results.append(w.width);
    
    for d,r in zip(data, results):
      d[i] = r;
    
    return True;
  except:
    data[-1][i] = -1;
    return False;

from multiprocessing import Pool, cpu_count;
pool = Pool(processes = cpu_count());
#pool.map(process, range(2));
pool.map(process, range(ntimes));



data = [np.load(f, mmap_mode = 'r+') for f in data_files];

### Failed frames
nfailed = np.sum(data[3] == -1);
print 'percentage failed %f' % (100.0 * nfailed/data[3].shape[0])


# plot some of the failed images

failed_ids = np.where(data[3] == -1)[0];
good_ids = data[3] != -1

plt.figure(1); plt.clf();
iids= np.array(np.linspace(0, 8000, 36), dtype = int);
for i,j in enumerate(iids):
  plt.subplot(6,6,i+1)
  img = exp.load_img(t = failed_ids[j]);
  plt.imshow(img)
  

fig_dir = '/home/ckirst/Desktop/labmeeting'
os.mkdir(fig_dir)

#make some plots of failed cases

for i in failed_ids[iids]:
  fig = plt.figure(2); plt.clf();
  plt.imshow(exp.load_img(t = i));
  plt.tight_layout();
  fig.savefig(os.path.join(fig_dir, 'sample_shape_detection_failed_%d.png' % i))



#w80 = exp.load(wid = 80)
#w80s = np.load(exp.filename(strain = 'n2', dtype = 'stage', wid = 80));
#
#plt.figure(1); plt.clf();
#plt.plot(data[3])
#
#plt.figure(2); plt.clf();


#shape parameter

theta = np.load(data_files[0]);
theta *= 1.0/(theta.shape[1]+1)

#kick out failed ones

theta = theta[good_ids];


theta.shape
theta_median = np.median(np.abs(theta))

# curvature of the worm

curv = np.sum(theta, axis = 1);
bend = np.sum(np.abs(theta), axis = 1);

plt.figure(6); plt.clf();
plt.plot(curv);
plt.plot(bend)


plt.figure(1); plt.clf();
plt.hist(curv, bins = 256)

plt.figure(1); plt.clf();
plt.hist(bend, bins = 256)


# resolve for stages etc


mt = np.max(np.abs(theta), axis = 1)
idx = mt < 1;

theta_red= theta[idx]
theta_red.shape

aplt.plot_array(theta_red.T)


### PCA
plt.figure(12); plt.clf();
aplt.plot_pca(theta)

pca = aplt.PCA(theta)


pca_comp = pca.Wt;

import worm.model as wm
import worm.geometry as wgeo
w = wm.WormModel(npoints = 22)

fig = plt.figure(17); plt.clf();
for i in range(len(pca_comp)):
  tt = pca_comp[i] * 21;
  tta = tt - np.mean(tt);
  if i == 0:
    oo = 0.2;
  elif i == 1:
    oo = .9 + np.pi;
  else:
    oo = 0;
  w.center = wgeo.center_from_theta_discrete(tta, length = 95, orientation = oo);
  plt.subplot(5,4,i+1)
  plt.title('PCA %d' % (i+1))
  w.plot()
  plt.axis('equal')
fig.savefig(os.path.join(fig_dir, 'theta_pca.png'), facecolor='white')
    

fig = plt.figure(18); plt.clf();
for i in range(len(pca_comp)):
  tt = pca_comp[i] * 21;
  tta = tt - np.mean(tt);
  plt.subplot(5,4,i+1)
  plt.plot(tta, linewidth = 2, c = 'r')
  plt.title('PCA %d' % (i+1))

fig.savefig(os.path.join(fig_dir, 'theta_pca_plot.png'),  facecolor='white')
      

### stage resolved pca

theta = np.load(data_files[0]);
theta *= 1.0/(theta.shape[1]+1)

w80s = np.load(exp.filename(strain = 'n2', dtype = 'stage', wid = 80));

valid = data[3] != -1;

for s in range(1,5):
  idx = np.logical_and(valid, w80s == s);
    
  pca = aplt.PCA(theta[idx]);
  pca_comp = pca.Wt;

  w = wm.WormModel(npoints = 22)

  plt.figure(20+s); plt.clf();
  for i in range(len(pca_comp)):
    tt = pca_comp[i] * 21;
    tta = tt - np.mean(tt);
    w.center = wgeo.center_from_theta_discrete(tta, length = 95, orientation = 0);
    plt.subplot(5,4,i+1)
    plt.title('PCA %d' % (i+1))
    w.plot()
    plt.axis('equal')
  
  fig.savefig(os.path.join(fig_dir, 'theta_pca_stage=%d.png' % s),  facecolor='white'); 





# movement movie does not work ???

# segment thetas into blocks of 5

ntw = 10;
ntheta = theta.shape[1];

tt = theta[good_ids];

nm = tt.shape[0]/ntw

tm = np.zeros((nm, ntw, ntheta))

for i in range(nm):
  tm[i, :, :] = tt[i*ntw:(i+1)*ntw,:];

tm = np.reshape(tm, (nm, ntw*ntheta))

pca = aplt.PCA(tm);
pca_comp = pca.Wt;

w = wm.WormModel(npoints = 22)

npca = 3
fig = plt.figure(70); plt.clf();
for ip in range(npca):
  tmi = np.reshape(pca_comp[ip], (ntw, ntheta));
  for i in range(ntw):
    tt = tmi[i] * 21;
    tta = tt - np.mean(tt);
    w.center = wgeo.center_from_theta_discrete(tta, length = 95, orientation = oo);
    plt.subplot(npca,ntw,ip*ntw+i+1)
    plt.title('PCA %d t=%d' % (ip, i+1))
    w.plot()
    plt.axis('equal')

#fig.savefig(os.path.join(fig_dir, 'theta_pca.png'), facecolor='white')







