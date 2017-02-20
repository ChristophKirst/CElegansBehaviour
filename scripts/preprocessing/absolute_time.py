"""
Absolute Time Stamps

Long Term Behaviour Analysis of C-elegans
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import glob
import matplotlib.pyplot as plt;
import numpy as np

import analysis.experiment as exp
import scripts.preprocessing.file_order as fo

fig_directory = '/home/ckirst/Science/Projects/CElegansBehaviour/Analysis/WormBehaviour/Figures/2017_01_20_TimeAnalysis/'

worm_directories = fo.directory_names;

### Time stamps of the data

import datetime, time, calendar

exp_times = np.zeros(len(worm_directories));

for wid, wdir in enumerate(worm_directories):
  print '%s / %s' % (wid, fo.nworms);
  file_pattern = os.path.join(wdir, 'corrd*.mat');
  fns = sorted(glob.glob(file_pattern));
  fn = fns[0]; 
  date = fn[-26:-9];
  t = time.strptime(date, '%Y-%m-%d-%H%M%S');
  secs = calendar.timegm(t);
  
  secs_24h = t.tm_hour * 60 * 60 + t.tm_min * 60 + t.tm_sec;
  
  exp_times[wid] = secs_24h;  

exp_times = np.array(exp_times, dtype = int)

fig = plt.figure(1); plt.clf();
plt.plot(exp_times)
plt.xlim(0, len(exp_times));
plt.ylabel('day time [sec]')

fig.savefig(os.path.join(fig_directory, 'experiments_starting_times_24hr.pdf'))


fn_times = os.path.join(exp.data_directory, 'n2_time.npy');
np.save(fn_times, exp_times);