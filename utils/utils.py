# -*- coding: utf-8 -*-
"""
Created on Sun May 22 21:42:04 2016

@author: ckirst
"""

import sys 
# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int) 
        barLength   - Optional  : character length of bar (Int) 
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    #sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    #sys.stdout.flush()
    print '%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)
    if iteration == total:
        print("\n")


