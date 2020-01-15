#%%
import h5py
import sys
sys.path.insert(0,'../build')
import pyabcranger

import numpy as np

#%%
f = h5py.File('../../test/data/reftable.h5','r')

#%%
statobs = np.loadtxt('../test/data/statobsRF.txt',skiprows=2)

#%%

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

statobs = np.loadtxt('../test/data/statobsRF.txt',skiprows=2)

#%%
pyabcranger.modelchoice(rf, statobs,"",False)
#%%
f = h5py.File('../../diyabc-global/reftable.h5','r')
statobs = np.loadtxt('../../diyabc-global/statobsRF.txt',skiprows=2)

#%%
selected = np.array(f['scenarios']) == 1
nref = np.sum(selected)
np.array(f['params']).shape
#%%
rf = pyabcranger.reftable(
    nref,
    f['nrecscen'],
    f['nparam'],
    f['params'].attrs['params_names'],
    f['stats'].attrs['stats_names'],
    f['stats'][selected,:],
    f['params'][selected,:],
    f['scenarios']
    )

#%%
#f['stats'][np.array(f['scenarios']) == 1,:]
#np.sum(np.array(f['scenarios']) == 1)
# f['scenarios'] == 1
# np.array(f['params'])[np.array(f['scenarios']) == 1][:,f['params'].attrs['params_names'] == b'ra']

#%%

postres = pyabcranger.estimparam(rf,statobs," ".join(["--ntree","500","--nref",str(nref),"--parameter","ra","--noob","50","--chosenscen","1","--nolinear"]),False)

# %%
postres.point_estimates

# %%
postres.values_weights

# %%
import matplotlib.pyplot as plt
import seaborn as sns
x,y = np.asanyarray(postres.values_weights)[:,0],np.asanyarray(postres.values_weights)[:,1]
plt.figure()
sns.distplot(x,y,hist=False,kde=True);

# %%
