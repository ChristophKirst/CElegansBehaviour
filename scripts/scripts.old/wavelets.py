# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 22:42:16 2016

@author: ckirst
"""

from scipy import signal
import matplotlib.pyplot as plt
t = np.linspace(-2, 2, 200, endpoint=False)
sig  = 0.2*np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.ricker, widths)

plt.figure(1); plt.clf();
plt.subplot(1,2,1);
plt.plot(t, sig)
plt.subplot(1,2,2)
plt.imshow(cwtmatr, extent=[-2, 2, 31, 1], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()




### Test on Speed data



# load data
#column 1 - speed
#column 2 - angular velocity 
#column 5- life stage
#column 6- roaming/dwelling classification (1- roaming 0-dwelling)

import numpy as np
from scipy import signal, io


n2 = io.loadmat('/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/DwellingRoaming/N2_1.mat');

# build classification matrix

n2data = n2['individual_Speed_AV2'][0];

n = n2data.shape[0];

lens = [n2data[i].shape[0] for i in range(n)]
maxlen = max(lens)  

ii = 0;
speed = n2data[ii][:,0];
roam = n2data[ii][:,5];
stage = n2data[ii][:,4];


st = 350000;
ed = st + 100000
speed1 = speed[st:ed];
roam1  = roam[st:ed];
stage1 = stage[st:ed];

widths = np.arange(1, 4000, 50)
cwtmatr = signal.cwt(speed1, signal.ricker, widths)

plt.figure(1); plt.clf();
ax1 = plt.subplot(2,1,1);
plt.plot(roam1 * 1)
plt.plot(speed1)
plt.plot(stage1);



ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.imshow(cwtmatr, cmap='PRGn', aspect='auto',# interpolation = 'none',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), extent = [0, len(speed1), 500, 1])
#plt.pcolorfast(cwtmatr, vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.imshow(cwtmatr, aspect='auto',# interpolation = 'none',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max(), extent = [0, len(speed1), 500, 1])
#plt.pcolorfast(cwtmatr, vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

plt.show()


 

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]


ac = autocorr(speed);

plt.figure(19);
plt.plot(ac)


# auto correlation


plt.plot(np.log(speed))



cwtmatr2 = signal.cwt(speed1, signal.morlet, widths)

plt.figure(2); plt.clf();
ax1 = plt.subplot(2,1,1);
plt.plot(speed1)
#plt.plot(roam1 * 10)
ax2 = plt.subplot(2,1,2, sharex=ax1)
plt.imshow(cwtmatr2, cmap='PRGn', aspect='auto',# interpolation = 'none',
           vmax=abs(cwtmatr2).max(), vmin=-abs(cwtmatr2).max(), extent = [0, len(speed1), 500, 1])
#plt.pcolorfast(cwtmatr, vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()



### Stages

nstages = 4;
stage = np.zeros((n,nstages), dtype = int);
n - n2data.shape[0];
for i in range(n):
  stage[i,:] = np.where(np.diff(n2data[i][:,4])>0)[0];

roam_all = np.zeros((n, maxlen));
for i in range(n):
  roam_all[i, 0:lens[i]] = n2data[i][:,5];

plt.figure(3); plt.clf(); 
import os
os.chdir('/data/Science/Projects/CElegansBehaviour/Experiment/DwellingRoaming')
import timeseries as ts
for i in range(stage.shape[1]):
  a = ts.align_series(roam_all, stage[:,i]);
  plt.subplot(nstages,1,i+1);
  plt.matshow(a, fignum = False, aspect = 'auto')
plt.tight_layout()





import analysis.experiment as exp;
