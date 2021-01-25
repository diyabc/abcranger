import subprocess
import os
import glob
import time
import h5py
import pyabcranger

import numpy as np

f = h5py.File('reftable.h5','r')
statobs = np.loadtxt('statobsRF.txt',skiprows=2)

stats_mc = np.array(f['stats'])
params_mc = np.array(f['params'])

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

selected = np.array(f['scenarios']) == 3
stats_ep = np.array(stats_mc[selected,:])
params_ep = np.array(params_mc[selected,:])

rf_ep = pyabcranger.reftable(
    np.sum(selected),
    f['nrecscen'],
    f['nparam'],
    f['params'].attrs['params_names'],
    f['stats'].attrs['stats_names'],
    stats_ep,
    params_ep,
    f['scenarios']
    )

def test_modelchoice():
    """Run basic Model choice example
    """
    for filePath in glob.glob('modelchoice_out.*'):
        os.remove(filePath)
    postres = pyabcranger.modelchoice(rf_mc, statobs,"",False)
    assert len(postres.votes[0]) == 6

def test_estimparam(path):
    """Run basic Parameter estimation example
    """
    for filePath in glob.glob('estimparam_out.*'):
        os.remove(filePath)
    pyabcranger.estimparam(rf_ep,statobs,"--parameter ra --chosenscen 3 --noob 50",False,False)

def test_parallel(path):
    """Check multithreaded performance
    """
    for filePath in glob.glob('estimparam_out.*'):
        os.remove(filePath)
    time1 = time.time()
    pyabcranger.estimparam(rf_ep,statobs,"--parameter ra --chosenscen 3 --noob 50 --threads 1",False,False)
    time2 = time.time()
    pyabcranger.estimparam(rf_ep,statobs,"--parameter ra --chosenscen 3 --noob 50 --threads 8",False,False)
    time3 = time.time()
    assert (time2-time1) > 1.5 * (time3 - time2)
