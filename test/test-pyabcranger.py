import subprocess
import os
import glob
import time
import h5py
import pyabcranger

import numpy as np

f = h5py.File('reftable.h5','r')
statobs = np.loadtxt('statobsRF.txt',skiprows=2)

stats_mc = np.transpose(f['stats'])
params_mc = np.transpose(f['params'])

rf_mc = pyabcranger.reftable(
    f['nrec'][0],
    f['nrecscen'],
    f['nparam'],
    f['params'].attrs['params_names'],
    f['stats'].attrs['stats_names'],
    stats_mc,
    params_mc,
    f['scenarios']
    )

def test_modelchoice():
    """Run basic Model choice example
    """
    for filePath in glob.glob('modelchoice_out.*'):
        os.remove(filePath)
    postres = pyabcranger.modelchoice(rf_mc, statobs,"",False)
    assert len(postres.votes) == 6

# def test_estimparam(path):
#     """Run basic Parameter estimation example
#     """
#     for filePath in glob.glob('estimparam_out.*'):
#         os.remove(filePath)
#     subprocess.call([path,"--parameter","ra","--chosenscen","3","--noob","50"])

# def test_parallel(path):
#     """Check multithreaded performance
#     """
#     for filePath in glob.glob('estimparam_out.*'):
#         os.remove(filePath)
#     time1 = time.time()
#     subprocess.call([path,"--parameter","ra","--chosenscen","3","--noob","50","-j","1"])
#     time2 = time.time()
#     subprocess.call([path,"--parameter","ra","--chosenscen","3","--noob","50","-j","8"])
#     time3 = time.time()
#     assert (time2-time1) > 1.5 * (time3 - time2)
