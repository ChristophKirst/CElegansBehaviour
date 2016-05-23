"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Clustering / Feature Detection

Experimental Data: 
Shay Stern, cori Bargman lab, The Rockefeller University 2016
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt

import wormdata as wd;
reload(wd)

from wormdata import shiftData

basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/';
filename = os.path.join(basedir, 'Experiment/individuals_N2_X_Y.mat')

data = io.loadmat(filename);
XYdata = data['individual_X_Y'][0];

print XYdata.shape
print XYdata[0].shape  

wid = 71;
#wid = 1;
XYwdata = XYdata[wid].copy();

reload(wd)
w = wd.WormData(XYwdata[:,0:2], stage = XYwdata[:,-1], valid = XYwdata[:,0] != 1, label = ('x', 'y'), wid = wid);
w.replaceInvalid();

plt.figure(1); plt.clf(); 
w.plotData();

plt.figure(2); plt.clf();
w.plotTrajectory()

plt.figure(3); plt.clt();
w.plotTrace(ids = w.stage())


##############################################################################
# small test set
reload(wd)
tmin = 200000;
tmax = tmin + 50000;
w = wd.WormData(XYwdata[tmin:tmax,0:2], stage = XYwdata[tmin:tmax,-1], valid = XYwdata[tmin:tmax,0] != 1, label = ('x', 'y'), wid = wid);
w.replaceInvalid();

plt.figure(1); plt.clf(); 
w.plotTrajectory(size = 50);


def plotData(data, iters = [0,1,10,20], name = None):
  #plt.figure(10); plt.clf();
  #plt.plot(np.log(range(1,nn+1)), np.log(rots[3200:3300,:].T))  

  #fig = plt.figure(10); plt.clf();
  n = len(iters);
  n2 = int(np.ceil(n/2.));
  fig, axs = plt.subplots(2, n2, sharex=True, sharey=True);
  for k,i in enumerate(iters):
    #plt.subplot(2,2,k+1);
    w.plotTrajectory(colordata = shiftData(data, i/2, reverse = True)[:,i], ax = axs.flat[k]);
    axs.flat[k].set_title(str(i));
   
  if name is not None:
    fig = plt.gcf();
    fig.canvas.set_window_title(name);
  
  fig = plt.figure(); plt.clf();
  for k,i in enumerate(iters):
    plt.subplot(2,n2,k+1);
    plt.hist(data[~np.isnan(data[:,i]),i], bins = 50)
    plt.title(str(i));
  
  if name is not None:
    fig = plt.gcf();
    fig.canvas.set_window_title(name);


##############
#distances
nns = [0,1,5,10,20,40,100,200];
nn = max(nns)+1;
dists = w.calculateDistances(n= nn);
#plt.figure(10); plt.clf();
#plt.plot(np.log(range(1,nn+1)), np.log(dists[3200:3300,:].T))

plotData(dists, iters = nns, name = 'Distances')


#################
#rotations

rots = w.calculateRotations(n= nn);
#plt.figure(10); plt.clf();
#plt.plot(np.log(range(1,nn+1)), np.log(rots[3200:3300,:].T))

plotData(rots, iters = nns, name = 'Rotations')


##############
#angles

angs = w.calculateAngles(n = nn);
#plt.figure(10); plt.clf();
#plt.plot(range(1,nn+1), angs[3200:3300,:].T)

plotData(angs, iters = nns, name = 'Angles')



##############
#diffusions
nn = 50;
diffs = w.calculateDiffusionParameter(n=nn, parallel = True)

plt.figure(20); plt.clf();
diffsshift = shiftData(diffs, nn/2, reverse = True);
w.plotTrajectory(colordata = diffsshift[:,0], size = 100)
plt.figure(21); plt.clf();
w.plotTrajectory(colordata = diffsshift[:,0], size = diffsshift[:,-1]* 1000)

plt.figure(22); plt.clf();
w.plotTrajectory(colordata = diffsshift[:,1], size = 60)
#w.plotTrajectory(colordata = -diffsshift[:,-1], size = diffsshift[:,0]* 1000)

#histogram
nbins = 50;
plt.figure(13); plt.clf();
plt.subplot(2,2,1);
plt.hist(diffs[~np.isnan(diffs[:,0]),0], bins = nbins)
plt.subplot(2,2,2);
plt.hist(diffs[~np.isnan(diffs[:,0]),-1], bins = nbins)
plt.subplot(2,1,2);
plt.hexbin(diffs[:,0], diffs[:,-1])


##############################################################################
#joint distributions

##############
## load entire data set
reload(wd)
wid = 71;
XYwdata = XYdata[wid].copy();
tmin = 0 * 100000;
tmax = tmin + 50000;
tmax = -1;
w = wd.WormData(XYwdata[tmin:tmax,0:2], stage = XYwdata[tmin:tmax,-1], valid = XYwdata[tmin:tmax,0] != 1, label = ('x', 'y'), wid = wid);
w.replaceInvalid();



##############
## dists and rots

k = [1, 5, 10, 20, 30, 40, 60, 80, 100, 200, 300, 400];
nn = max(k)+1
dists = w.calculateDistances(n = nn);
rots  = w.calculateRotations(n = nn)

n2 = int(np.ceil(len(k)/2.))
fig, axs = plt.subplots(2, n2);

for i,j in enumerate(k):
  axs.flat[i].hexbin(np.log(dists[:,j]), (rots[:,j]));
  axs.flat[i].set_title(str(j));


##############
# GMM
from sklearn import mixture
from matplotlib.colors import LogNorm

j = 200;
gmmdata = np.vstack([np.log(dists[:,j]), (rots[:,j])]).T
#gmmdata = np.vstack([dists[:,j], (rots[:,j])]).T
gmmdata.shape

nanids = np.any(np.isnan(gmmdata), axis=1);
gmmdata = gmmdata[~nanids,:];
gmmdata.shape

clf = mixture.GMM(n_components=3, covariance_type='full',  n_init = 5, verbose = 2)
clf.fit(gmmdata)

# display
npts = 100;
x = np.linspace(gmmdata[:,0].min(), gmmdata[:,0].max(), npts);
y = np.linspace(gmmdata[:,1].min(), gmmdata[:,1].max(), npts);
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)[0]
Z = Z.reshape(X.shape)


import matplotlib as mpl
import itertools

plt.figure(100); plt.clf();
CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 1, 10))
CB = plt.colorbar(CS, shrink=1, extend='both')

pred =-np.ones(rots.shape[0])
pred[~nanids] = clf.predict(gmmdata);

color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])
#plt.scatter(gmmdata[:, 0], gmmdata[:, 1], c = pred[~nanids], s = 10, marker = 'o', lw = 0)
# Plot an ellipse to show the Gaussian component

ax = plt.gca();
for i, (mean, covar, color) in enumerate(zip(clf.means_, clf._get_covars(), color_iter)):
    pids = pred[~nanids] == i;
    ax.scatter(gmmdata[pids, 0], gmmdata[pids, 1], color = color, s = 10, marker = 'o', lw = 0)

    v, ww = np.linalg.eigh(covar)
    u = ww[0] / np.linalg.norm(ww[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
    ell.set_clip_box(ax.bbox)
    ell.set_alpha(0.5)
    ax.add_artist(ell)

plt.title('GMM fit to %d' % j)
plt.axis('tight')
plt.show()


plt.figure(101); plt.clf();
w.plotTrace(ids = shiftData(pred, j/2, nan = -1))



###############################################################################
# smooth the pdf + watershed
from utils import smooth
nbins = (400,400);


j = 400;
gmmdata = np.vstack([np.log(dists[:,j]), (rots[:,j])]).T
#gmmdata = np.vstack([dists[:,j], (rots[:,j])]).T
gmmdata.shape

nanids = np.any(np.isnan(gmmdata), axis=1);
gmmdata = gmmdata[~nanids,:];
gmmdata.shape


imgbin = smooth(gmmdata, nbins = nbins, sigma = None)
img = smooth(gmmdata, nbins = nbins, sigma = (7,7))
#plt.matshow(img)

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max

local_maxi = peak_local_max(img, indices=False, footprint=np.ones((3, 3)))
markers = ndi.label(local_maxi)[0]
labels = watershed(-img, markers, mask = None)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})
ax0, ax1 = axes
ax0.imshow(img, cmap=plt.cm.jet, interpolation='nearest')
ax0.set_title('PDF')
labels[imgbin==0] = 0;
labels[0,0] = -1;
ax1.imshow(labels, cmap=plt.cm.rainbow, interpolation='nearest')
ax1.set_title('Segmentation')


#classification
pred2 =-np.ones(rots.shape[0])
pred2nn = pred2[~nanids];
d = gmmdata.copy();
pred2nn.shape
d.shape
d.dtype

for i in range(2):
  d[:,i] = d[:,i] - d[:,i].min();
  d[:,i] = (d[:,i] / d[:,i].max()) * (nbins[i]-1);
  
d = np.asarray(d, dtype = int);
  
for i in range(2):
  d[d[:,i] > (nbins[i]-1), i] = nbins[i]-1;

for i in xrange(d.shape[0]):          ## draw pixels
  pred2nn[i] = labels[d[i,0], d[i,1]];
  
pred2[~nanids] = pred2nn;
pred2nn.max();


plt.figure(506); plt.clf();
w.plotTrace(ids = shiftData(pred2, j/2, nan = -1))


dist2 = w.calculateDistances(n=200);
plt.figure(510); plt.clf();
w.plotDataColor(data = dist2[:, -1], c = pred2, lw = 0, s = 20)



dist2 = w.calculateLengths(n=200);
plt.figure(511); plt.clf();
w.plotDataColor(data = dist2[:, -1], c = pred2, lw = 0, s = 20)


#nice


##############################################################################
#GMM / AIC segmentation based on rots 

from sklearn import mixture
from matplotlib.colors import LogNorm

j = 200;
rnanids = np.isnan(rots[:,j]);
gmmrots = rots[~rnanids, j];

plt.figure(300); plt.clf();
plt.hist(gmmrots, bins = 256)


N = np.arange(1, 10)
models = [None for i in range(len(N))]

gmmrots.shape =  (gmmrots.shape[0],1);
for i in range(len(N)):
    print 'fitting %d'% i;
    models[i] = mixture.GMM(N[i], verbose = 2).fit(gmmrots);

# compute the AIC and the BIC
AIC = [m.aic(gmmrots) for m in models]
BIC = [m.bic(gmmrots) for m in models]

plt.figure(400); plt.clf();
plt.subplot(2,1,1);
plt.plot(AIC); plt.title('AIC')
plt.subplot(2,1,2);
plt.plot(BIC); plt.title('BIC')

print "Best AIC: %d" % np.argmin(AIC)
print "Best BIC: %d" % np.argmin(BIC)

mbest = np.argmin(AIC);
mbest = 3;

#------------------------------------------------------------
# Plot the results
#  We'll use three panels:
#   1) data + best-fit mixture
#   2) AIC and BIC vs number of components
#   3) probability that a point came from each component

fig = plt.figure(309)

# plot 1: data + best-fit mixture
ax = fig.add_subplot(131)
M_best = models[mbest]

x = np.linspace(gmmrots.min(), gmmrots.max(), 1000)
x.shape = (x.shape[0], 1);
logprob, responsibilities = M_best.score_samples(x)
pdf = np.exp(logprob)
pdf_individual = responsibilities * pdf[:, np.newaxis]

ax.hist(gmmrots, 30, normed=True, histtype='stepfilled', alpha=0.4, color = 'gray')
ax.plot(x, pdf, '-k')
ax.plot(x, pdf_individual, '--k')
ax.text(0.04, 0.96, "Best-fit Mixture", ha='left', va='top', transform=ax.transAxes)
ax.set_xlabel('$x$')
ax.set_ylabel('$p(x)$')


# plot 2: AIC and BIC
ax = fig.add_subplot(132)
ax.plot(N, AIC, '-k', label='AIC')
ax.plot(N, BIC, '--k', label='BIC')
ax.set_xlabel('n. components')
ax.set_ylabel('information criterion')
ax.legend(loc=2)


# plot 3: posterior probabilities for each component
ax = fig.add_subplot(133)

x.shape = (x.shape[0],1);
p = M_best.predict_proba(x)
p = p.cumsum(1).T

x.shape = (x.shape[0],);
for i in range(p.shape[0]):
  print i
  if i == 0:
    ax.fill_between(x, 0, p[i,:], color='gray', alpha=(i + 1.)/p.shape[0])
  elif i == p.shape[0]-1:
    ax.fill_between(x, p[i,:], 1, color='gray', alpha=(i + 1.)/p.shape[0])
  else:
    ax.fill_between(x, p[i,:], p[i+1,:], color='gray', alpha=(i + 1.)/p.shape[0])

plt.ylim([0,1])
plt.show()


## Classify
rpred =-np.ones(rots.shape[0])
rpred[~rnanids] = M_best.predict(gmmrots);

plt.figure(310); plt.clf();
w.plotTrace(ids = shiftData(rpred, j/2, nan = -1))

## plot class as fucntion of time

plt.figure(311)
w.plotDataColor(c = rpred, lw = 0, s = 20)





















##################################################################################################
##################################### Old


import numpy as np

np.random.seed(1)
g = mixture.GMM(n_components=2)
# Generate random observations with two modes centered on 0
# and 10 to use for training.
obs = np.concatenate((np.random.randn(100, 1),
                      10 + np.random.randn(300, 1)))
g.fit(obs) 



np.round(g.weights_, 2)

np.round(g.means_, 2)


np.round(g.covars_, 2) 


g.predict([[0], [2], [9], [10]]) 

np.round(g.score([[0], [2], [9], [10]]), 2)

# Refit the model on new data (initial parameters remain the
# same), this time with an even split between the two modes.
g.fit(20 * [[0]] +  20 * [[10]]) 



np.round(g.weights_, 2)

















## some basic analysis

nhist = 31;
dist = w.calculateDistances(n = nhist);
rots = w.calculateRotations(n = nhist);

plt.figure(10); plt.clf();
for dd in range(10):
  plt.subplot(2,5,dd+1);
  #plt.plot(np.log(dd[1:,k]), ac[:,k] / ((k+1) * np.pi),'.');
  d = 3 * dd;
  #plt.hexbin(np.log(dd[1:,k]), ac[:,k] / ((k+1) * np.pi));
  #plt.hexbin(dist[1:,d], rots[:,d] / ((d + 1) * np.pi), bins = 'log');
  #plt.hexbin(np.log(dist[1:,d]), rots[:,d] / ((d + 1) * np.pi), bins = 'log');
  plt.hexbin(np.log(dist[1:,d]), rots[:,d] / ((d + 1) * np.pi));
  #plt.xlabel('distance');
  #plt.ylabel('rotation');



## generat feature data

from scipy.cluster.vq import whiten, kmeans, kmeans2
from scipy.cluster.hierarchy import linkage, dendrogram;

feat = w.calculateFeatures(n = 30, features = ['rotations']);
#feat = w.calculateFeatures(n = 20, features = ['rotations', 'distances', 'lengths']);
#feat = w.calculateFeatures(n = 20, features = ['rotations']);
#feat = feat[:,-5:];
feat.shape


#feat = feat[:,::5];
#feat[:,:15] = np.log10(feat[:,:15]);
feat.shape


#feat = w.calculateFeatures(n = 15, features = ['rotations', 'distances', 'lengths']);
#feat = feat[:,::5];
#feat.shape


nnids = ~np.isnan(feat).any(axis=1);
featn = feat[nnids];

featw = whiten(featn);
#featw = featn;

km = kmeans(featw,  k_or_guess = 3);
kmv = km[0];

#km = kmeans2(featw,  k = 2);
#kmv = km[0];

#distances to cluster center
from scipy.spatial.distance import pdist, cdist, squareform
dd = cdist(featw, kmv);

#max dist index
clidsn = np.argmax(dd, axis = 1);
clids = -np.ones(feat.shape[0]);
clids[nnids] = clidsn;
clids = np.concatenate([[-1,-1], clids]);

plt.figure(40); plt.clf();
w.plotData(data = clids, marker = '.', linestyle = '')

plt.figure(41); plt.clf(); #plt.tight_layout();
w.plotTrace(ids = clids)



nn = 20;
diffs = w.calculateDiffusionParameter(n=nn, parallel = False)

plt.figure(2); plt.clf();
diffsshift = np.zeros_like(diffs);
diffsshift[nn/2:, :] = diffs[:-nn/2,:];
diffsshift[:nn/2. :] = np.NaN;
w.plotTrajectory(colordata = diffsshift[:,0], size = diffsshift[:,-1]* 1000)


## Feature evolution

feat = w.calculateFeatures(n = 50, features = ['distances']);

plt.figure(50); plt.clf();
plt.plot(feat[::500,:].T)


feat = w.calculateFeatures(n = 50, features = ['distances']);

plt.figure(51); plt.clf();
plt.plot(feat[::500,:].T)



xpos = np.array(range(50));
xpos.shape = (1, xpos.shape[0]);
xpos = np.repeat(xpos, feat.shape[0], axis = 0);

plt.figure(51); plt.clf();
plt.hexbin(xpos.flatten(), feat.flatten(),bins = 'log')



feat = w.calculateFeatures(n = 50, features = ['rotations']);

plt.figure(52); plt.clf();
plt.plot(feat[::150,:].T)

xpos = np.array(range(50));
xpos.shape = (1, xpos.shape[0]);
xpos = np.repeat(xpos, feat.shape[0], axis = 0);

plt.figure(53); plt.clf();
plt.hexbin(xpos.flatten(), feat.flatten(),bins = 'log')







## Density estimation

from sklearn import mixture

n_samples = 300

# generate random sample, two components
np.random.seed(0)

# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])


clf = mixture.GMM(n_components=2, covariance_type='full')
clf.fit(X_train)


# display predicted scores by the model as a contour plot
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)[0]
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 10))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()



## Linkage analysis
#Z = linkage(featw[::1000,:]);
#plt.figure(20); plt.clf();
#dn = dendrogram(Z)

##pca on this data


## cluster analysis on the data


## feature analysis on the data

  






### some basic measures of differences


import scipy.stats.entropy as kld

# loop over individuals
for i in range(6):
  
  #distributions
  


##
dat = XYdata[60];
dat = replaceUndefined(dat);

dd = calculateDeltas(dat, n = 40);
aa = calculateAngles(dat);
ac = accumulate(np.abs(aa), n = 40)


plt.figure(35); plt.clf();
for kk in range(10):
  plt.subplot(2,5,kk+1);
  #plt.plot(np.log(dd[1:,k]), ac[:,k] / ((k+1) * np.pi),'.');
  k = 3 * kk;
  #plt.hexbin(np.log(dd[1:,k]), ac[:,k] / ((k+1) * np.pi));
  plt.hexbin((dd[1:,k]), ac[:,k] / ((k+1) * np.pi));

plt.figure(31); plt.clf();
for kk in range(5):
  plt.subplot(1,5,kk+1);
  k = 2 * kk;
  plt.hist(ac[~np.isnan(ac[:,k]),k], bins = 100);


kk = 0;
#ids = ac[:,kk] < 5.5 + 0 * np.pi/2;
#ids = ac[:,kk]  / ((kk+1) * np.pi)) < 0.05 + 0 * np.pi/2;
ids = ac[:,kk] < np.pi /2;

dd = dat[1:-1,:].copy();


idsall = np.where(ids);
idsall = [idsall- i for i in np.array(range(kk+1))];
idsall = np.unique(np.concatenate(idsall));
np.put(ids, idsall, True);

plt.figure(32); plt.clf();
plt.plot(dd[:,0], dd[:,1], color = 'black');
plt.plot(dd[~ids,0], dd[~ids,1], '.');
plt.plot(dd[ids, 0], dd[ids, 1], '.', color = 'red')



##
plt.figure(501); plt.clf();
sdat1 = individual_speed_AV[0]
plotIndividualData(sdat1);


plt.figure(500); plt.clf();
sdat1 = smoothIndividualData(individual_speed_AV[0], bins = (120,120));
plotIndividualData(sdat1);