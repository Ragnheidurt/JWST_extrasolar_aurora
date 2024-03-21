from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy import signal
import sys
sys.path.append('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Functions')
from average_errors import average_errors


# Read in files from both sensors
nrs1 = get_pkg_data_filename('MAST_2024-02-27T16_24_08.087Z\MAST_2024-02-27T16_24_08.087Z\JWST\jw01189-o004_t002_nirspec\jw01189-o004_t002_nirspec_g395h-f290lp_s3d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-27T16_24_08.087Z\MAST_2024-02-27T16_24_08.087Z\JWST\jw01189-o004_t002_nirspec\jw01189-o004_t002_nirspec_g235m-f170lp_s3d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-27T16_24_08.087Z\MAST_2024-02-27T16_24_08.087Z\JWST\jw01189-o004_t002_nirspec\jw01189-o004_t002_nirspec_g140m-f100lp_s3d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-01-31T15_21_00.167Z\MAST_2024-01-31T15_21_00.167Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs1_s3d.fits')

#
#nrs1 = get_pkg_data_filename('Data\MAST_2024-03-03T22_04_44.899Z\MAST_2024-03-03T22_04_44.899Z\JWST\jw01189-o004_t002_nirspec\jw01189-o004_t002_nirspec_g395h-f290lp_s3d.fits')


# Extract the data from the files
nrs1_data = fits.getdata(nrs1,ext=1)

# Extract header for both data sets
nrs1_header = fits.getheader(nrs1,ext=1)

# Get the start and end wavelengths for both sensors
nrs1_wavestart = nrs1_header['WAVSTART']
nrs1_wavend = nrs1_header['WAVEND']


# Calculate the interval and size of wavelength data
nrs1_dist = nrs1_wavend-nrs1_wavestart
nrs1_interval = nrs1_dist/nrs1_data.shape[0]


# Create array with wavelength
nrs1_wave = np.zeros(nrs1_data.shape[0])


for i in range(nrs1_data.shape[0]):
    if i != 0: 
        nrs1_wave[i] = nrs1_wave[i-1] + nrs1_interval
    else:
        nrs1_wave[i] = nrs1_wavestart



# Find the brightest spot for each data set
        
nrs1_slice = nrs1_data[500,:,:]
nrs1_maxindex = np.unravel_index(np.nanargmax(nrs1_slice), nrs1_slice.shape)


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

# Mask the max value and the 8 values around it
nrs1_blocked_indices = [(a,d),(a,e),(a,f),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f)]
nrs1_mask = np.ones_like(nrs1_slice, dtype=bool)
for idx in nrs1_blocked_indices:
    nrs1_mask[idx] = False




# Calculating the second brightest spot
nrs1_secmaxindex = np.unravel_index(np.nanargmax(nrs1_slice*nrs1_mask), nrs1_slice.shape)

# Take the average of the second brightest pixel and the eight pixels around it
nrs1_secavg = np.zeros(nrs1_wave.shape[0])
a = nrs1_secmaxindex[0]-1
b = nrs1_secmaxindex[0]
c = nrs1_secmaxindex[0]+1
d = nrs1_secmaxindex[1]-1
e = nrs1_secmaxindex[1]
f = nrs1_secmaxindex[1]+1
for i in range(nrs1_avg.shape[0]):
    nrs1_secavg[i] = (nrs1_data[i,a,d]+nrs1_data[i,a,e]+nrs1_data[i,a,f]+nrs1_data[i,b,d]+nrs1_data[i,b,e]+nrs1_data[i,b,f]+nrs1_data[i,c,d]+nrs1_data[i,c,e]+nrs1_data[i,c,f])/9
    if abs(nrs1_secavg[i]) > 1e5 and i != 0:
        nrs1_secavg[i] = nrs1_secavg[i-1]



# Average all pixels 
nrs1_allavg = np.nanmean(nrs1_data,axis=(1,2))

# Remove the outliers
for i in range(nrs1_allavg.shape[0]):
    if i!= 0 and abs(nrs1_allavg[i])> 10:
        nrs1_allavg[i] = nrs1_allavg[i-1]



# Calculating the Planck curve
nrs1_Planck = np.zeros(nrs1_wave.shape[0])
Temp = 700
h = 6.62607015e-34
c = 3e8
kb = 1.380649e-23

for i in range(nrs1_wave.shape[0]):
    nrs1_Planck[i] = 6e-7*(2*h*c**2)/(nrs1_wave[i]**5)*1/(math.exp((h*c)/(nrs1_wave[i]*kb*Temp))-1)
   # nrs1_Planck[i] = (2*h*c**2)/(nrs1_wave[i]**5)*(1/(math.exp((h*c)/(nrs1_wave[i]*kb*Temp))-1))



# Looking at a specific interval

low_limit = 3.92e-06
up_limit = 4.02e-06
nrs1_lengd = 0
for i in range(nrs1_wave.shape[0]):
    if(nrs1_wave[i]>=low_limit and nrs1_wave[i]<= up_limit):
        nrs1_lengd = nrs1_lengd +1



nrs1_close_wave = np.zeros(nrs1_lengd,dtype='float32')
nrs1_close_value = np.zeros(nrs1_lengd,dtype='float32')
count = 0
for i in range(nrs1_wave.shape[0]):
    if nrs1_wave[i] >= 3.92e-06 and nrs1_wave[i] <= 4.02e-06:
        nrs1_close_wave[count] = nrs1_wave[i]
        nrs1_close_value[count] = nrs1_data[i,nrs1_maxindex[0],nrs1_maxindex[1]]
        count = count+1



# Smoothing the data
nrs1_window_all = int(nrs1_avg.shape[0]*0.2)
nrs1_window_close = int(nrs1_close_value.shape[0]*0.2)
nrs1_smooth_avg = signal.savgol_filter(nrs1_avg, window_length=nrs1_window_all, polyorder=3, mode="nearest")
nrs1_smooth_close = signal.savgol_filter(nrs1_close_value,window_length=nrs1_window_close, polyorder=3, mode="nearest")
#nrs1_smooth_close = signal.savgol_filter(nrs1_close_value,window_length = 5, polyorder=3, mode="nearest")

#nrs1_wave_average, nrs1_average, nrs_error_average = 


plt.figure(1)
plt.imshow(nrs1_data[500,:,:])
plt.colorbar()


plt.figure(3)
plt.plot(nrs1_wave,nrs1_avg)
plt.plot(nrs1_wave,nrs1_smooth_avg)

plt.figure(4)
plt.plot(nrs1_close_wave,nrs1_close_value)
plt.plot(nrs1_close_wave,nrs1_smooth_close)
plt.axvline(x=3953e-9, color='r', linestyle='--', label='Vertical Line')
plt.axvline(x=3985.5e-9, color='r', linestyle='--', label='Vertical Line')

plt.figure(5)
plt.plot(nrs1_wave,nrs1_secavg)

plt.figure(6)
plt.plot(nrs1_wave,nrs1_allavg)
plt.axvline(x=3953e-9, color='r', linestyle='--', label='Vertical Line')
plt.axvline(x=3985.5e-9, color='r', linestyle='--', label='Vertical Line')




plt.show()

