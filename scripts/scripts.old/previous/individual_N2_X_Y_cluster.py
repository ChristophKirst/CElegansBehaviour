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

basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/';
filename = os.path.join(basedir, 'Experiment/individuals_N2_X_Y.mat')

data = io.loadmat(filename);
XYdata = data['individual_X_Y'][0];

print XYdata.shape
print XYdata[0].shape  

wid = 71;
wid = 1;
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