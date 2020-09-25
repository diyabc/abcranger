# %%
import h5py
import pyabcranger

import numpy as np


# %%
f = h5py.File('modelchoice-reftable.h5','r')
statobs = np.loadtxt('modelchoice-statobs.txt',skiprows=2)


# %%
stats = np.transpose(f['stats'])
params = np.transpose(f['params'])

rf = pyabcranger.reftable(
    f['nrec'][0],
    f['nrecscen'],
    f['nparam'],
    f['params'].attrs['params_names'],
    f['stats'].attrs['stats_names'],
    stats,
    params,
    f['scenarios']
    )


# %%
ntree = 500

postres = pyabcranger.modelchoice(rf, statobs,"--ntree "+str(ntree),False)
