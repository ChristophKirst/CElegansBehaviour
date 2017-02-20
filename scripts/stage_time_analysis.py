# -*- coding: utf-8 -*-
"""
Compare manual and automatic detected life stage durations

@author: ckirst
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt

dir_roaming = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/DwellingRoaming/'
dir_behaviour  = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Code/'
dir_mapping = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Code/scripts/preprocessing/';
dir_data = '/run/media/ckirst/My Book'

base_directory = '/run/media/ckirst/My Book'

os.chdir(dir_behaviour);
import analysis.experiment as exp
exp.experiment_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Experiment/ImageData'


fig_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Figures/2017_01_20_TimeAnalysis/'

### Match file names

worm_directories = sorted(glob.glob(os.path.join(base_directory, 'Results*/*/')))

os.chdir(dir_mapping);
from file_ordering import file_ordering

rd_name = [f[5:37] for f in file_ordering];

xy_name = [glob.glob(os.path.join(d, '*'))[0][-41:-9] for d in worm_directories];

rd_name = np.array(rd_name);
xy_name = np.array(xy_name);

rd_to_xy = - np.ones(len(rd_name), dtype = int);
for i in range(len(rd_name)):
  pos = np.nonzero(xy_name == rd_name[i])[0];
  if len(pos) > 0:
    rd_to_xy[i] = pos;

xy_to_rd = - np.ones(len(xy_name), dtype = int);
for i in range(len(xy_name)):
  pos = np.nonzero(rd_name == xy_name[i])[0];
  if len(pos) > 0:
    xy_to_rd[i] = pos;

# correct for restarted exp 20/21






### Get stage durations from Roaming Dwelling data set

os.chdir(dir_roaming);
import experiment as exprd

strain = 'N2';

rd_data = exprd.load_data(strain);
assert rd_data.stage_durations.shape[0] == rd_name.shape[0]

rd_stage_ids = rd_data.stage_switch;
rd_stage_dur = rd_data.stage_durations;

### Get stage durations from automatic detection

xy_stage_ids = np.load(os.path.join(exp.experiment_directory, 'transitions_times.npy'))
xy_stage_dur = np.diff(xy_stage_ids, axis = 1)


### Plot Stage Durations 


plt.figure(1); plt.clf();

rate = 3.0 * 60;
valid = rd_to_xy >= 0
for s in range(4):
  plt.subplot(2,2, s+1);
  plt.plot(xy_stage_dur[rd_to_xy[valid], s]/rate, rd_stage_dur[valid, s]/rate,'*')
  plt.title('L%d' % (s+1));
  plt.xlabel('t [min]');
  plt.ylabel('t [min]');



### Plot Roaming / Dwelling using the xy transitions
valid = rd_to_xy >= 0
n = np.sum(valid);
rd = rd_data.roam[valid].copy();
nworms = rd.shape[0];
rd_switch = rd_data.stage_switch[valid];

xy_switch = (xy_stage_ids[rd_to_xy[valid]].T - xy_stage_ids[rd_to_xy[valid]][:,1] + rd_switch[:,1]).T

ww = 500;
for wid in range(nworms):
  for s in range(1,6):
    rd[wid, max(rd_switch[wid, s]-ww,0):min(rd_switch[wid,s]+ww, rd.shape[1])] = 3;
    rd[wid, max(xy_switch[wid, s]-ww,0):min(xy_switch[wid,s]+ww, rd.shape[1])] = 4;
    
import plot as fplt
#import analysis.plot as fplt;
fplt.plot_array(rd)
#plt.figure(10); plt.clf();
#plt.imshow(rd, aspect = 'auto', cmap = plt.cm.viridis)






rate = 3.0 * 60 * 60;
valid = rd_to_xy >= 0

xy_switch_hr = np.round(xy_switch / rate);
rd_switch_hr = np.round(rd_switch / rate);

#xy_switch_hr = xy_switch / rate
#rd_switch_hr = rd_switch / rate

fig = plt.figure(2); plt.clf();
plt.plot(xy_switch_hr[:,:-1], rd_switch_hr[:,:-1],'*')
plt.plot(np.linspace(0, 50, 10), np.linspace(0,50,10), 'k')
plt.title('L%d' % (s+1));
plt.xlabel('t [hr] (automatic)');
plt.ylabel('t [hr] (manual)');
plt.grid(True)
fig.savefig(os.path.join(fig_directory, 'manual_vs_hand_stage_annotation.pdf'))



import scipy.stats as st

lr = st.linregress(xy_switch_hr[:,:-1].flatten(), rd_switch_hr[:,:-1].flatten())



    

    
    
    






