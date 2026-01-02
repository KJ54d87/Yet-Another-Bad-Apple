import sys
import abtem
import ase
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from scipy.ndimage import rotate

SIZE = int(sys.argv[1])
FRAME_NUMBER = sys.argv[2]

print(SIZE, FRAME_NUMBER)

with h5py.File(f"atom_frames/frame_{FRAME_NUMBER}.h5","r") as f:
    atoms = f['atoms'][:]

#print(atoms)

ducky_atoms = ase.Atoms(
    numbers=np.full(atoms.shape[0], 6),
    positions=atoms[:,:3],
    cell=np.array([SIZE,SIZE,atoms[:,2].max() + atoms[:,2].min()])
)

del ducky_atoms[(ducky_atoms.positions[:,0]<0) | (ducky_atoms.positions[:,0]>SIZE)]
del ducky_atoms[(ducky_atoms.positions[:,1]<0) | (ducky_atoms.positions[:,1]>SIZE)]
#ducky_atoms.translate([-50,-50,0])

#fig, axs = plt.subplots(1,2,figsize=(8,4))
#
#abtem.show_atoms(ducky_atoms,ax=axs[0],tight_limits=True)
#abtem.show_atoms(ducky_atoms,plane='xz',ax=axs[1],tight_limits=True)

#fig.tight_layout()
#fig.show()

potential = abtem.Potential(
    ducky_atoms,
    gpts=(SIZE*10,SIZE*10),
    slice_thickness=0.83875,
).build()

probe = abtem.Probe(
    energy=80e3,
    semiangle_cutoff=20,
    defocus=500,
).match_grid(
    potential
)

#fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))

#potential.project().show(
#    ax=ax1,
#    cmap='magma',
#)

#probe.show(ax =ax2);

#fig.tight_layout()

pixelated_detector = abtem.PixelatedDetector(
    max_angle=None,
)

grid_scan = abtem.GridScan(
    (0, 0), (SIZE,SIZE),
    sampling=4,
    endpoint=True,
)

measurement = probe.scan(
    potential=potential,
    scan=grid_scan,
    detectors=pixelated_detector,
).compute(
)

sx, sy = grid_scan.shape
bin_factor = 4

# plane-wave descan
#sx, sy = grid_scan.shape
#bin_factor = 4

#x = np.linspace(-50,50,sx)
#y = np.linspace(-50,50,sy)

#descan_x = np.round((y[None,:]-2*x[:,None]) / grid_scan.sampling[0])
#descan_y = np.round((-y[None,:]-3*x[:,None]) / grid_scan.sampling[1])

#print(potential.gpts)
#crop_x, crop_y = (np.array(potential.gpts) - 100 * bin_factor) // 2
#array = np.zeros((sx,sy,100,100),dtype=measurement.array.dtype)
#for i in range(sx):
#    for j in range(sy):
#
#        dp = measurement.array[i,j]
#        dp = rotate(dp,15,reshape=False,order=1) # rotate
#        dp = np.roll(dp,(int(descan_x[i,j]),int(descan_y[i,j])),axis=(0,1)) # descan
#        dp = dp[crop_x:-crop_x,crop_y:-crop_y].reshape((100,bin_factor,100,bin_factor)).sum((1,3)) # bin
#        array[i,j] = dp

#array_sums = array.sum((-2,-1))
#print(f"Smallest array sum (should be close to 1): {array_sums.min()}")

import h5py

samples = measurement.to_data_array()
#print(type(samples))
samples.to_netcdf(f"dp/frame{FRAME_NUMBER}.h5", engine="h5netcdf", invalid_netcdf=True)

#print(type(samples))
#sample = xr.DataArray(array)
#samples.to_netcdf("dp/test.h5", engine="h5netcdf", invalid_netcdf=True)

#samples.
#files.download('/data2.h5')

