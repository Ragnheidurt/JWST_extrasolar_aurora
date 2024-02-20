from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math


# Read in files from both sensors
nrs1 = get_pkg_data_filename('MAST_2024-02-20T10_10_20.787Z\MAST_2024-02-20T10_10_20.787Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs1_s3d.fits')
nrs2 = get_pkg_data_filename('MAST_2024-02-20T10_10_20.787Z\MAST_2024-02-20T10_10_20.787Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs2_s3d.fits')


# Extract the data from the files
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)

# Extract header for both data sets
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)

print(nrs1_header)
print(nrs2_header)

# Get the start and end wavelengths for both sensors
nrs1_wavestart = nrs1_header['WAVSTART']
nrs1_wavend = nrs1_header['WAVEND']
nrs2_wavestart = nrs2_header['WAVSTART']
nrs2_wavend = nrs2_header['WAVEND']

# Calculate the interval and size of wavelength data
nrs1_dist = nrs1_wavend-nrs1_wavestart
nrs1_interval = nrs1_dist/nrs1_data.shape[0]
nrs2_dist = nrs2_wavend-nrs2_wavestart
nrs2_interval = nrs2_dist/nrs2_data.shape[0]

# Create array with wavelength
nrs1_wave = np.zeros(nrs1_data.shape[0])
nrs2_wave = np.zeros(nrs2_data.shape[0])

for i in range(nrs1_data.shape[0]):
    if i != 0: 
        nrs1_wave[i] = nrs1_wave[i-1] + nrs1_interval
    else:
        nrs1_wave[i] = nrs1_wavestart

for i in range(nrs2_data.shape[0]):
    if i != 0:
        nrs2_wave[i] = nrs2_wave[i-1] + nrs2_interval
    else:
        nrs2_wave[i] = nrs2_wavestart

# Find the brightest spot for each data set
        
nrs1_slice = nrs1_data[1500,:,:]
nrs1_maxindex = np.unravel_index(np.nanargmax(nrs1_slice), nrs1_slice.shape)
nrs2_slice = nrs2_data[2500,:,:]
nrs2_maxindex = np.unravel_index(np.nanargmax(nrs2_slice),nrs2_slice.shape)

print(nrs1_maxindex)
print(nrs2_maxindex)

# Take the average of the brightest pixel and the eight pixels around it

nrs1_avg = np.zeros(nrs1_wave.shape[0])
a = nrs1_maxindex[0]-1
b = nrs1_maxindex[0]
c = nrs1_maxindex[0]+1
d = nrs1_maxindex[1]-1
e = nrs1_maxindex[1]
f = nrs1_maxindex[1]+1
for i in range(nrs1_avg.shape[0]):
    nrs1_avg[i] = (nrs1_data[i,a,d]+nrs1_data[i,a,e]+nrs1_data[i,a,f]+nrs1_data[i,b,d]+nrs1_data[i,b,e]+nrs1_data[i,b,f]+nrs1_data[i,c,d]+nrs1_data[i,c,e]+nrs1_data[i,c,f])/9
    if abs(nrs1_avg[i]) > 1e5 and i != 0:
        nrs1_avg[i] = nrs1_avg[i-1]


nrs2_avg = np.zeros(nrs2_wave.shape[0])
a = nrs2_maxindex[0]-1
b = nrs2_maxindex[0]
c = nrs2_maxindex[0]+1
d = nrs2_maxindex[1]-1
e = nrs2_maxindex[1]
f = nrs2_maxindex[1]+1
for i in range(nrs2_wave.shape[0]):
    nrs2_avg[i] = (nrs2_data[i,a,d]+nrs2_data[i,a,e]+nrs2_data[i,a,f]+nrs2_data[i,b,d]+nrs2_data[i,b,e]+nrs2_data[i,b,f]+nrs2_data[i,c,d]+nrs2_data[i,c,e]+nrs2_data[i,c,f])/9
    if abs(nrs2_avg[i]) > 1e5 and i != 0:
        nrs2_avg[i] = nrs2_avg[i-1]


# Calculating the Planck curve
nrs1_Planck = np.zeros(nrs1_wave.shape[0])
Temp = 600
h = 6.62607015e-34
c = 3e8
kb = 1.380649e-23

for i in range(nrs1_wave.shape[0]):
    nrs1_Planck[i] = 2e-6*(2*h*c**2)/(nrs1_wave[i]**5)*1/(math.exp((h*c)/(nrs1_wave[i]*kb*Temp))-1)

nrs2_Planck = np.zeros(nrs2_wave.shape[0])

for i in range(nrs2_wave.shape[0]):
    nrs2_Planck[i] = 5e-7*(2*h*c**2)/(nrs2_wave[i]**5)*1/(math.exp((h*c)/(nrs2_wave[i]*kb*Temp))-1)




plt.figure(1)
plt.imshow(nrs1_data[1500,:,:])
plt.colorbar()

plt.figure(2)
plt.imshow(nrs2_data[2500,:,:])
plt.colorbar()

plt.figure(3)
plt.plot(nrs1_wave,nrs1_avg)
plt.plot(nrs1_wave,nrs1_Planck)

plt.figure(4)
plt.plot(nrs2_wave,nrs2_avg)
plt.plot(nrs2_wave,nrs2_Planck)

plt.figure(5)
plt.plot(nrs1_wave,nrs1_avg)
plt.plot(nrs2_wave,nrs2_avg)
plt.plot(nrs1_wave,nrs1_Planck)
plt.plot(nrs2_wave,nrs2_Planck)

plt.show()

