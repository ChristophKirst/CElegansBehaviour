# -*- coding: utf-8 -*-
"""
Binned Roaming Data

Binn roaming data using automatic detected stage transitions
"""

__author__  = 'Christoph Kirst <ckirst@rockefeller.edu>'
__license__ = 'MIT License <http://www.opensource.org/licenses/mit-license.php>'
__docformat__ = 'rest'

import numpy as np
import scipy.io as io
import os

import analysis.experiment as exp;
import analysis.plot as plt




dat = exp.load_stage_binned(dtype = 'roam', nbins_per_stage=75);
dat[np.isnan(dat)] = 0;


plt.plot_array(dat)

io.savemat()