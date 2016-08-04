"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

Experimental Data: 
Shay Stern, cori Bargman lab, The Rockefeller University 2016
"""

__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np
import scipy.io as io

from tools import *

basedir = '/home/ckirst/Science/Projects/CElegansBehaviour/';
filename = os.path.join(basedir, 'Experiment/individuals_N2_X_Y.mat')

data = io.loadmat(filename);

XYdata = data['individual_X_Y'][0];

print XYdata.shape
print XYdata[0].shape

#### investiage individual data

## plot data

dat = XYdata[2];

plt.figure(1); plt.clf();
plotAllIndividualData(dat);



for k in range(5):
  plt.figure(k); plt.clf();
  dat = XYdata[k];
  plotAllIndividualData(dat);


for k in range(6,10):
  plt.figure(k); plt.clf();
  dat = individual_speed_AV[k];

plt.figure(100);
plotAllIndividualData(individual_speed_AV[4]);

plt.figure(101);
plotAllIndividualData(individual_speed_AV[5]);



## correlation structure

for k in range(5):
  dat = individual_speed_AV[k];
  cc = correlationIndividual(dat);

  plt.figure(20+k); plt.clf();  
  plotCorrelationIndividual(cc);

for k in range(6,10):
  dat = individual_speed_AV[k];
  cc = correlationIndividualStages(dat);

  plt.figure(20+k); plt.clf();  
  plotCorrelationIndividualStages(cc);



## smooth data
dat = individual_speed_AV[72];
sdat = [];
for bs,k in zip([1, 5, 10, 20, 30], range(5)):
  print bs, k
  sdat.append(dat.copy());
  sdat[k] = smoothIndividualData(sdat[k], bins = (bs,bs));

  plt.figure(k); plt.clf();  
  plotHistogramStages(sdat[k]);
  
  plt.figure(200+k); plt.clf(); 
  plotAllIndividualData(sdat[k]);
  

## make histograms

plt.figure(2); plt.clf();
plt.subplot(1,2,1)
plt.hexbin(dat[:,0],dat[:,1], bins = 'log')
plt.subplot(1,2,2);
plt.hexbin(sdat[:,0],sdat[:,1], bins = 'log')
plt.show()


plt.figure(3); plt.clf();
plotHistogramStages(dat, bins = 'log');
plt.figure(4); plt.clf();
plotHistogramStages(sdat, bins = 'log');


## histograms at developmental stages








for i in range(2):
  plt.figure(600+i); plt.clf();
  dat = individual_speed_AV[i];
  plotHistogramStages(dat, bins = 'log');
  
for i in range(5):
  plt.figure(12 + i); plt.clf();
  dat = individual_speed_AV[i];
  plotAllIndividualData(dat)


## average angular velovity data ?



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