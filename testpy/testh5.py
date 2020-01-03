import h5py
import sys
sys.path.insert(0,'./build')
import pyabcranger

import numpy as np

f = h5py.File('./test/data/reftable.h5','r')

rf = pyabcranger.reftable(
    f['nrec'][0],
    f['nrecscen'],
    f['nparam'],
    f['params'].attrs['params_names'],
    f['stats'].attrs['stats_names'],
    f['stats'],
    f['params'],
    f['scenarios']
    )

statobs = np.loadtxt('./test/data/statobsRF.txt',skiprows=2)
pyabcranger.modelchoice(rf, statobs,"bleuargh",True)

