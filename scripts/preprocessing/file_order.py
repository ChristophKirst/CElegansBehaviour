# -*- coding: utf-8 -*-
"""
Behavioural Development

Long Term Behaviour Analysis of C-elegans

The mapping of roaming dwelling data set to the raw image / trajectory data
"""
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__author__ = 'Christoph Kirst <ckirst@rockefeller.edu>'
__docformat__ = 'rest'

import os
import numpy as np;

import analysis.experiment as exp

files_raw = np.array(['corrdCAM207A1CAM207_2015-11-20-174842-0059_L1_bursts.mat',
'corrdCAM207A1CAM207_2016-04-29-172141-0058_L1_bursts.mat',
'corrdCAM207A1CAM207_2016-10-14-164337-0065_L1_bursts.mat',
'corrdCAM207A2CAM207_2015-11-20-174842-0059_L1_bursts.mat',
'corrdCAM207A2CAM207_2016-04-29-172141-0058_L1_bursts.mat',
'corrdCAM207A2CAM207_2016-08-24-192135-0053_L1_bursts.mat',
'corrdCAM207A2CAM207_2016-10-14-164337-0065_L1_bursts.mat',
'corrdCAM207A2CAM207_2016-10-21-092134-0024_L1_bursts.mat',
'corrdCAM207A3CAM207_2015-11-20-174842-0059_L1_bursts.mat',
'corrdCAM207A3CAM207_2016-04-29-172141-0058_L1_bursts.mat',
'corrdCAM207A3CAM207_2016-08-24-192135-0053_L1_bursts.mat',
'corrdCAM207A3CAM207_2016-10-14-164337-0065_L1_bursts.mat',
'corrdCAM207A3CAM207_2016-10-21-092134-0024_L1_bursts.mat',
'corrdCAM207A4CAM207_2015-11-20-174842-0059_L1_bursts.mat',
'corrdCAM207A4CAM207_2016-04-29-172141-0058_L1_bursts.mat',
'corrdCAM207A4CAM207_2016-08-24-192135-0053_L1_bursts.mat',
'corrdCAM207A4CAM207_2016-10-14-164337-0065_L1_bursts.mat',
'corrdCAM207A4CAM207_2016-10-21-092134-0024_L1_bursts.mat',
'corrdCAM207A5CAM207_2015-11-20-174842-0059_L1_bursts.mat',
'corrdCAM207A5CAM207_2016-04-29-172141-0058_L1_bursts.mat',
'corrdCAM207A5CAM207_2016-08-24-192135-0053_L1_bursts.mat',
'corrdCAM207A5CAM207_2016-10-14-164337-0065_L1_bursts.mat',
'corrdCAM207A5CAM207_2016-10-21-092134-0024_L1_bursts.mat',
'corrdCAM207A6CAM207_2015-11-20-174842-0059_L1_bursts.mat',
'corrdCAM207A6CAM207_2016-04-29-172141-0058_L1_bursts.mat',
'corrdCAM207A6CAM207_2016-08-24-192135-0053_L1_bursts.mat',
'corrdCAM207A6CAM207_2016-10-14-164337-0065_L1_bursts.mat',
'corrdCAM207A6CAM207_2016-10-21-092134-0024_L1_bursts.mat',
'corrdCAM800A1CAM800_2016-04-29-171717-0058_L1_bursts.mat',
'corrdCAM800A1CAM800_2016-10-14-163342-0066_L1_bursts.mat',
'corrdCAM800A1CAM800_2016-10-20-193442-0081_L1_bursts.mat',
'corrdCAM800A2CAM800_2015-11-20-174411-0059_L1_bursts.mat',
'corrdCAM800A2CAM800_2016-04-29-171717-0058_L1_bursts.mat',
'corrdCAM800A2CAM800_2016-10-20-193442-0081_L1_bursts.mat',
'corrdCAM800A3CAM800_2015-11-20-174411-0059_L1_bursts.mat',
'corrdCAM800A3CAM800_2016-08-24-191319-0059_L1_bursts.mat',
'corrdCAM800A4CAM800_2015-11-20-174411-0059_L1_bursts.mat',
'corrdCAM800A4CAM800_2016-04-29-171717-0058_L1_bursts.mat',
'corrdCAM800A4CAM800_2016-10-20-193442-0081_L1_bursts.mat',
'corrdCAM800A5CAM800_2015-11-20-174411-0059_L1_bursts.mat',
'corrdCAM800A5CAM800_2016-04-29-171717-0058_L1_bursts.mat',
'corrdCAM800A5CAM800_2016-08-24-191319-0059_L1_bursts.mat',
'corrdCAM800A5CAM800_2016-10-14-163342-0066_L1_bursts.mat',
'corrdCAM800A5CAM800_2016-10-20-193442-0081_L1_bursts.mat',
'corrdCAM800A6CAM800_2016-04-29-171717-0058_L1_bursts.mat',
'corrdCAM800A6CAM800_2016-08-24-191319-0059_L1_bursts.mat',
'corrdCAM800A6CAM800_2016-10-14-163342-0066_L1_bursts.mat',
'corrdCAM800A6CAM800_2016-10-20-193442-0081_L1_bursts.mat',
'corrdCAM807A1CAM807_2015-11-20-174323-0059_L1_bursts.mat',
'corrdCAM807A1CAM807_2016-04-29-171758-0058_L1_bursts.mat',
'corrdCAM807A1CAM807_2016-08-24-191439-0059_L1_bursts.mat',
'corrdCAM807A1CAM807_2016-10-20-193403-0081_L1_bursts.mat',
'corrdCAM807A2CAM807_2016-08-24-191439-0059_L1_bursts.mat',
'corrdCAM807A3CAM807_2015-11-20-174323-0059_L1_bursts.mat',
'corrdCAM807A3CAM807_2016-04-29-171758-0058_L1_bursts.mat',
'corrdCAM807A3CAM807_2016-10-20-193403-0081_L1_bursts.mat',
'corrdCAM807A4CAM807_2016-04-29-171758-0058_L1_bursts.mat',
'corrdCAM807A4CAM807_2016-10-20-193403-0081_L1_bursts.mat',
'corrdCAM807A5CAM807_2015-11-20-174323-0059_L1_bursts.mat',
'corrdCAM807A5CAM807_2016-08-24-191439-0059_L1_bursts.mat',
'corrdCAM807A5CAM807_2016-10-14-163417-0066_L1_bursts.mat',
'corrdCAM807A5CAM807_2016-10-20-193403-0081_L1_bursts.mat',
'corrdCAM807A6CAM807_2016-04-29-171758-0058_L1_bursts.mat',
'corrdCAM807A6CAM807_2016-08-24-191439-0059_L1_bursts.mat',
'corrdCAM807A6CAM807_2016-10-14-163417-0066_L1_bursts.mat',
'corrdCAM813A1CAM813_2015-11-20-174728-0059_L1_bursts.mat',
'corrdCAM813A1CAM813_2016-04-29-172224-0058_L1_bursts.mat',
'corrdCAM813A1CAM813_2016-10-14-164222-0065_L1_bursts.mat',
'corrdCAM813A1CAM813_2016-10-21-092034-0026_L1_bursts.mat',
'corrdCAM813A2CAM813_2016-04-29-172224-0058_L1_bursts.mat',
'corrdCAM813A2CAM813_2016-08-24-192101-0053_L1_bursts.mat',
'corrdCAM813A2CAM813_2016-10-21-092034-0026_L1_bursts.mat',
'corrdCAM813A3CAM813_2015-11-20-174728-0059_L1_bursts.mat',
'corrdCAM813A3CAM813_2016-08-24-192101-0053_L1_bursts.mat',
'corrdCAM813A3CAM813_2016-10-21-092034-0026_L1_bursts.mat',
'corrdCAM813A4CAM813_2015-11-20-174728-0059_L1_bursts.mat',
'corrdCAM813A4CAM813_2016-04-29-172224-0058_L1_bursts.mat',
'corrdCAM813A4CAM813_2016-08-24-192101-0053_L1_bursts.mat',
'corrdCAM813A4CAM813_2016-10-21-092034-0026_L1_bursts.mat',
'corrdCAM813A5CAM813_2015-11-20-174728-0059_L1_bursts.mat',
'corrdCAM813A5CAM813_2016-10-21-092034-0026_L1_bursts.mat',
'corrdCAM813A6CAM813_2015-11-20-174728-0059_L1_bursts.mat',
'corrdCAM813A6CAM813_2016-10-14-164222-0065_L1_bursts.mat',
'corrdCAM814A1CAM814_2016-04-29-171847-0058_L1_bursts.mat',
'corrdCAM814A1CAM814_2016-08-24-191224-0059_L1_bursts.mat',
'corrdCAM814A2CAM814_2015-11-20-174505-0059_L1_bursts.mat',
'corrdCAM814A2CAM814_2016-08-24-191224-0059_L1_bursts.mat',
'corrdCAM814A2CAM814_2016-10-14-163452-0066_L1_bursts.mat',
'corrdCAM814A2CAM814_2016-10-20-193530-0081_L1_bursts.mat',
'corrdCAM814A3CAM814_2015-11-20-174505-0059_L1_bursts.mat',
'corrdCAM814A3CAM814_2016-04-29-171847-0058_L1_bursts.mat',
'corrdCAM814A3CAM814_2016-10-14-163452-0066_L1_bursts.mat',
'corrdCAM814A3CAM814_2016-10-20-193530-0081_L1_bursts.mat',
'corrdCAM814A4CAM814_2015-11-20-174505-0059_L1_bursts.mat',
'corrdCAM814A4CAM814_2016-04-29-171847-0058_L1_bursts.mat',
'corrdCAM814A4CAM814_2016-08-24-191224-0059_L1_bursts.mat',
'corrdCAM814A4CAM814_2016-10-14-163452-0066_L1_bursts.mat',
'corrdCAM814A4CAM814_2016-10-20-193530-0081_L1_bursts.mat',
'corrdCAM814A5CAM814_2015-11-20-174505-0059_L1_bursts.mat',
'corrdCAM814A5CAM814_2016-04-29-171847-0058_L1_bursts.mat',
'corrdCAM814A5CAM814_2016-10-14-163452-0066_L1_bursts.mat',
'corrdCAM814A5CAM814_2016-10-20-193530-0081_L1_bursts.mat',
'corrdCAM814A6CAM814_2015-11-20-174505-0059_L1_bursts.mat',
'corrdCAM814A6CAM814_2016-04-29-171847-0058_L1_bursts.mat',
'corrdCAM814A6CAM814_2016-10-14-163452-0066_L1_bursts.mat',
'corrdCAM814A6CAM814_2016-10-20-193530-0081_L1_bursts.mat',
'corrdCAM819A1CAM819_2015-11-20-174925-0059_L1_bursts.mat',
'corrdCAM819A1CAM819_2016-04-29-172306-0058_L1_bursts.mat',
'corrdCAM819A1CAM819_2016-08-24-192029-0054_L1_bursts.mat',
'corrdCAM819A1CAM819_2016-10-14-164252-0065_L1_bursts.mat',
'corrdCAM819A2CAM819_2015-11-20-174925-0059_L1_bursts.mat',
'corrdCAM819A2CAM819_2016-04-29-172306-0058_L1_bursts.mat',
'corrdCAM819A2CAM819_2016-10-14-164252-0065_L1_bursts.mat',
'corrdCAM819A3CAM819_2015-11-20-174925-0059_L1_bursts.mat',
'corrdCAM819A3CAM819_2016-10-21-092103-0024_L1_bursts.mat',
'corrdCAM819A4CAM819_2015-11-20-174925-0059_L1_bursts.mat',
'corrdCAM819A4CAM819_2016-04-29-172306-0058_L1_bursts.mat',
'corrdCAM819A4CAM819_2016-10-14-164252-0065_L1_bursts.mat',
'corrdCAM819A5CAM819_2015-11-20-174925-0059_L1_bursts.mat',
'corrdCAM819A5CAM819_2016-08-24-192029-0054_L1_bursts.mat',
'corrdCAM819A5CAM819_2016-10-14-164252-0065_L1_bursts.mat',
'corrdCAM819A5CAM819_2016-10-21-092103-0024_L1_bursts.mat',
'corrdCAM819A6CAM819_2015-11-20-174925-0059_L1_bursts.mat',
'corrdCAM819A6CAM819_2016-04-29-172306-0058_L1_bursts.mat',
'corrdCAM819A6CAM819_2016-08-24-192029-0054_L1_bursts.mat']);


### List matching util

def identify(names1, names2):
  """Matches a list of names to a second one"""
  import numpy as np
  n = len(names1);
  match = -np.ones(n, dtype = int);
  for i in range(n):
    pos = np.nonzero(names2 == names1[i])[0];
    if len(pos) > 0:
      match[i] = pos[0];
  return match;

### Identify roaming / Dwelling data with Image data

def match_image_data_to_roaming_data(raw_directory = exp.raw_data_directory):
  import os
  import glob 
  import numpy as np
  
  # get file names from experiments
  data_directories = np.array(sorted(glob.glob(os.path.join(raw_directory, 'Results*/*/'))));
  
  # file ordering (roaming dwelling data set)
  fns = files_raw;
  
  order_name = np.array([f[5:37] for f in fns]);
  data_name = np.array([glob.glob(os.path.join(d, '*'))[-1][-41:-9] for d in data_directories]);
  
  for i,d in enumerate(data_directories):
    print i,d
    glob.glob(os.path.join(d, '*'))[-1][-41:-9]
  
  
  data_to_order = identify(order_name, data_name);
  order_to_data = identify(data_name, order_name);

  # failed mappings
  failed_order = order_name[data_to_order==-1];
  failed_data = data_name[order_to_data==-1];

  # match 2016-10-20 to 2016-10-21
  match_fail = -1 * np.ones(len(failed_order), dtype = int);
  for i,f in enumerate(failed_order):
    ff = f[:14];
    for j,g in enumerate(failed_data): 
      if g[:14] == ff:
        match_fail[i] = j;
        break;
  
  # replace names in file ordering
  order_name_new = order_name.copy();
  order_name_new[data_to_order==-1] = data_name[order_to_data==-1][match_fail];
  
  data_to_order_new = identify(order_name_new, data_name);  
  order_to_data_new = identify(data_name, order_name_new);
  
  #failed_order_new = order_name_new[data_to_order_new==-1];
  #failed_data_new = data_name[order_to_data_new==-1];
  #failed_ids = np.where(order_to_data_new==-1)[0]
  
  data_directories_valid = data_directories[order_to_data_new>=0]
  print '%d valid directories found!' % len(data_directories_valid)

  data_directories_ordered = data_directories[data_to_order_new]
  data_names_ordered = data_name[data_to_order_new]

  #check 
  assert np.all(order_name_new == data_name[data_to_order_new])
  
  return (data_directories_ordered, data_names_ordered)

(file_names, experiment_names) = match_image_data_to_roaming_data();
subdirectory_names = np.array([os.path.join(f.split('/')[-3], f.split('/')[-2]) for f in file_names]);

def create_directory_names(base_directory = exp.raw_data_directory):
  return np.array([os.path.join(base_directory, sd) for sd in subdirectory_names])

directory_names = create_directory_names();

nworms = len(directory_names);
