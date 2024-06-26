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
from short_interval import short_interval
from model_data import model_data
from model_data import model_shortening
from scale_model import scale_model
from scale_model import scale_correctly


# Read in files from both sensors
#nrs1 = get_pkg_data_filename('MAST_2024-01-31T15_21_00.167Z\MAST_2024-01-31T15_21_00.167Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-01-31T15_21_00.167Z\MAST_2024-01-31T15_21_00.167Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs2_s3d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-20T10_10_20.787Z\MAST_2024-02-20T10_10_20.787Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-02-20T10_10_20.787Z\MAST_2024-02-20T10_10_20.787Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs2_s3d.fits')

#
#nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs2_s3d.fits')
#nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_21_56.858Z\MAST_2024-03-03T22_21_56.858Z\JWST\jw01189004001_03106_00002\jw01189004001_03106_00002_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_21_56.858Z\MAST_2024-03-03T22_21_56.858Z\JWST\jw01189004001_03106_00002\jw01189004001_03106_00002_nrs2_s3d.fits')
#nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs2_s3d.fits')
nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_40_03.977Z\MAST_2024-03-03T22_40_03.977Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs1_s3d.fits')
nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_40_03.977Z\MAST_2024-03-03T22_40_03.977Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs2_s3d.fits')

# Extract the data from the files
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)
nrs1_error = fits.getdata(nrs1,ext=2)

# Extract header for both data sets
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)
error_header = fits.getheader(nrs1,ext=2)

print(nrs1_header)
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
    elif nrs1_avg[i]<10 and i != 0:
        nrs1_avg[i] = nrs1_avg[i-1]

# Mask the max value and the 8 values around it
nrs1_blocked_indices = [(a,d),(a,e),(a,f),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f)]
print(nrs1_blocked_indices)
nrs1_mask = np.ones_like(nrs1_slice, dtype=bool)
for idx in nrs1_blocked_indices:
    nrs1_mask[idx] = False


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
    elif nrs2_avg[i]<10 and i != 0:
        nrs2_avg[i] = nrs2_avg[i-1]

# Mask the max value and the 8 values around it
nrs2_blocked_indices = [(a,d),(a,e),(a,f),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f)]
nrs2_mask = np.ones_like(nrs2_slice, dtype=bool)
for idx in nrs2_blocked_indices:
    nrs2_mask[idx] = False

# Calculating the second brightest spot
nrs1_secmaxindex = np.unravel_index(np.nanargmax(nrs1_slice*nrs1_mask), nrs1_slice.shape)
nrs2_secmaxindex = np.unravel_index(np.nanargmax(nrs2_slice*nrs2_mask),nrs2_slice.shape)

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


nrs2_secavg = np.zeros(nrs2_wave.shape[0])
a = nrs2_secmaxindex[0]-1
b = nrs2_secmaxindex[0]
c = nrs2_secmaxindex[0]+1
d = nrs2_secmaxindex[1]-1
e = nrs2_secmaxindex[1]
f = nrs2_secmaxindex[1]+1
for i in range(nrs2_wave.shape[0]):
    nrs2_secavg[i] = (nrs2_data[i,a,d]+nrs2_data[i,a,e]+nrs2_data[i,a,f]+nrs2_data[i,b,d]+nrs2_data[i,b,e]+nrs2_data[i,b,f]+nrs2_data[i,c,d]+nrs2_data[i,c,e]+nrs2_data[i,c,f])/9
    if abs(nrs2_secavg[i]) > 1e5 and i != 0:
        nrs2_secavg[i] = nrs2_secavg[i-1]


# Average all pixels 
nrs1_allavg = np.nanmean(nrs1_data,axis=(1,2))
nrs2_allavg = np.nanmean(nrs2_data,axis=(1,2))

# Remove the outliers
for i in range(nrs1_allavg.shape[0]):
    if i!= 0 and abs(nrs1_allavg[i])> 10:
        nrs1_allavg[i] = nrs1_allavg[i-1]

for i in range(nrs2_allavg.shape[0]):
    if i!= 0 and abs(nrs2_allavg[i])>10:
        nrs2_allavg[i] = nrs2_allavg[i-1]


# Calculating the Planck curve
nrs1_Planck = np.zeros(nrs1_wave.shape[0])
Temp = 700
h = 6.62607015e-34
c = 3e8
kb = 1.380649e-23

for i in range(nrs1_wave.shape[0]):
    nrs1_Planck[i] = 6e-7*(2*h*c**2)/(nrs1_wave[i]**5)*1/(math.exp((h*c)/(nrs1_wave[i]*kb*Temp))-1)
   # nrs1_Planck[i] = (2*h*c**2)/(nrs1_wave[i]**5)*(1/(math.exp((h*c)/(nrs1_wave[i]*kb*Temp))-1))

nrs2_Planck = np.zeros(nrs2_wave.shape[0])

for i in range(nrs2_wave.shape[0]):
    nrs2_Planck[i] = 5e-7*(2*h*c**2)/(nrs2_wave[i]**5)*1/(math.exp((h*c)/(nrs2_wave[i]*kb*Temp))-1)


# Looking at a specific interval
    
low_limit = 3.0e-06
up_limit = 3.5e-06
'''low_limit = 3.983e-06
up_limit = 3.997e-06'''
'''
low_limit = 3.92e-06
up_limit = 4.02e-06
'''
nrs1_lengd = 0
for i in range(nrs1_wave.shape[0]):
    if(nrs1_wave[i]>=low_limit and nrs1_wave[i]<= up_limit):
        nrs1_lengd = nrs1_lengd +1


nrs2_lengd = 0
for i in range(nrs2_wave.shape[0]):
    if(nrs2_wave[i]>=low_limit and nrs2_wave[i]<= up_limit):
        nrs2_lengd = nrs2_lengd +1

nrs1_close_wave = np.zeros(nrs1_lengd,dtype='float32')
nrs1_close_value = np.zeros(nrs1_lengd,dtype='float32')
count = 0
for i in range(nrs1_wave.shape[0]):
    if nrs1_wave[i] >= low_limit and nrs1_wave[i] <= up_limit:
        nrs1_close_wave[count] = nrs1_wave[i]
        #nrs1_close_value[count] = nrs1_data[i,nrs1_maxindex[0],nrs1_maxindex[1]]
        nrs1_close_value[count] = nrs1_avg[i]
        count = count+1


nrs2_close_wave = np.zeros(nrs2_lengd,dtype='float32')
nrs2_close_value = np.zeros(nrs2_lengd,dtype='float32')
count = 0
for i in range(nrs2_wave.shape[0]):
    if nrs2_wave[i] >= low_limit and nrs2_wave[i] <= up_limit:
        nrs2_close_wave[count] = nrs2_wave[i]
        #nrs2_close_value[count] = nrs2_data[i,nrs2_maxindex[0],nrs2_maxindex[1]]
        nrs2_close_value[count] = nrs2_avg[i]
        count = count+1

# Looking at a specific interval
'''
low_limit = 3.92e-06
up_limit = 4.02e-06

low_limit = 3.983e-06
up_limit = 3.997e-06
'''
nrs1_lengd = 0
for i in range(nrs1_wave.shape[0]):
    if(nrs1_wave[i]>=low_limit and nrs1_wave[i]<= up_limit):
        nrs1_lengd = nrs1_lengd +1


nrs2_lengd = 0
for i in range(nrs2_wave.shape[0]):
    if(nrs2_wave[i]>=low_limit and nrs2_wave[i]<= up_limit):
        nrs2_lengd = nrs2_lengd +1

nrs1_close_secwave = np.zeros(nrs1_lengd,dtype='float32')
nrs1_close_secvalue = np.zeros(nrs1_lengd,dtype='float32')
count = 0
for i in range(nrs1_wave.shape[0]):
    if nrs1_wave[i] >= low_limit and nrs1_wave[i] <= up_limit:
        nrs1_close_secwave[count] = nrs1_wave[i]
        #nrs1_close_value[count] = nrs1_data[i,nrs1_maxindex[0],nrs1_maxindex[1]]
        nrs1_close_secvalue[count] = nrs1_secavg[i]
        count = count+1


nrs2_close_secwave = np.zeros(nrs2_lengd,dtype='float32')
nrs2_close_secvalue = np.zeros(nrs2_lengd,dtype='float32')
count = 0
for i in range(nrs2_wave.shape[0]):
    if nrs2_wave[i] >= low_limit and nrs2_wave[i] <= up_limit:
        nrs2_close_secwave[count] = nrs2_wave[i]
        #nrs2_close_value[count] = nrs2_data[i,nrs2_maxindex[0],nrs2_maxindex[1]]
        nrs2_close_secvalue[count] = nrs2_secavg[i]
        count = count+1


# Smoothing the data

nrs1_smooth_avg = signal.savgol_filter(nrs1_avg, window_length=150, polyorder=3, mode="nearest")
nrs2_smooth_avg = signal.savgol_filter(nrs2_avg, window_length=250, polyorder=3, mode="nearest")
nrs1_smooth_close = signal.savgol_filter(nrs1_close_value,window_length=40, polyorder=3, mode="nearest")
nrs2_smooth_close = signal.savgol_filter(nrs2_close_value,window_length=40, polyorder=3,mode="nearest")
nrs1_smooth_secavg = signal.savgol_filter(nrs1_secavg, window_length=150, polyorder=3, mode="nearest")
nrs2_smooth_secavg = signal.savgol_filter(nrs2_secavg, window_length=250, polyorder=3, mode="nearest")
nrs1_smooth_allavg = signal.savgol_filter(nrs1_allavg, window_length=150, polyorder=3, mode="nearest")
nrs2_smooth_allavg = signal.savgol_filter(nrs2_allavg, window_length=250, polyorder=3, mode="nearest")

#divisors = [i for i in range(1, nrs1_avg.shape[0] + 1) if nrs1_avg.shape[0] % i == 0]

#nrs1_flux_error = nrs1_error[nrs1_maxindex[0],nrs1_maxindex[1],:]

#nrs1_wave_average, nrs1_average, nrs1_error_average = average_errors(nrs1_wave,nrs1_avg,nrs1_flux_error,divisors[2])
#nrs1_short_wave, nrs1_short_flux = short_interval(nrs1_avg,low_limit,up_limit,nrs1_wave)

wavenumber, absorption_intensity1, vacuum_wavelength, column_density, stuff = model_data('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST-Jupiter/Jupiter/Data/h3_generated1.txt')
#scaled_model = scale_model(absorption_intensity1,nrs1_avg, nrs2_avg,vacuum_wavelength,nrs2_wave)
scaled_model = scale_model(column_density,nrs1_avg, nrs2_avg,vacuum_wavelength,nrs2_wave)
model_wave_short,model_flux_short = model_shortening(vacuum_wavelength, low_limit, up_limit, scaled_model)

total_absorption = scale_correctly(absorption_intensity1,vacuum_wavelength)



plt.figure(1)
plt.imshow(nrs1_data[1500,:,:])
plt.colorbar()

plt.figure(2)
plt.imshow(nrs2_data[2500,:,:])
plt.colorbar()

plt.figure(3)
plt.plot(nrs1_wave,nrs1_avg)
plt.plot(nrs2_wave,nrs2_avg)
#plt.plot(nrs1_wave,nrs1_smooth_avg)
#plt.plot(nrs2_wave,nrs2_smooth_avg)
#plt.plot(vacuum_wavelength,scaled_model,color='red', linestyle='--', alpha=0.5)
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(4)
plt.plot(nrs1_close_wave,nrs1_close_value)
plt.plot(nrs2_close_wave,nrs2_close_value)
#plt.plot(model_wave_short,model_flux_short,color='red', linestyle='--', alpha=0.5)
#plt.axvline(x=3985.5e-9, color='r', linestyle='--', label='Vertical Line')
#plt.axvline(x=3994.8e-9, color='r', linestyle='--', label='Vertical Line')
#plt.axvline(x=3985.5e-9, color='r', linestyle='--', label='Vertical Line')
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(5)
plt.plot(nrs1_close_wave,nrs1_close_secvalue)
plt.plot(nrs2_close_wave,nrs2_close_secvalue)
#plt.axvline(x=3985.5e-9, color='r', linestyle='--', label='Vertical Line')
#plt.axvline(x=3994.8e-9, color='r', linestyle='--', label='Vertical Line')
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(6)
plt.plot(nrs1_wave,nrs1_secavg)
plt.plot(nrs2_wave,nrs2_secavg)
plt.plot(nrs1_wave,nrs1_smooth_secavg)
plt.plot(nrs2_wave,nrs2_smooth_secavg)
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')



plt.figure(7)
plt.plot(nrs1_wave,nrs1_allavg)
plt.plot(nrs2_wave,nrs2_allavg)
plt.plot(nrs1_wave,nrs1_smooth_allavg)
plt.plot(nrs2_wave,nrs2_smooth_allavg)
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(8)

plt.plot(nrs1_wave,nrs1_smooth_avg)
plt.plot(nrs2_wave,nrs2_smooth_avg)
plt.plot(nrs1_wave,nrs1_smooth_secavg)
plt.plot(nrs2_wave,nrs2_smooth_secavg)

fig, axs = plt.subplots(2, 1)
axs[0].plot(vacuum_wavelength, absorption_intensity1, color='red')
axs[1].plot(vacuum_wavelength, column_density, color='green')
axs[0].set_title(f"Absorption Intensity")
axs[1].set_title(f"Column Density")
axs[0].set_xlabel("Wavelength m")
axs[0].set_ylabel("[cm-1/(cm atm)]")
axs[1].set_xlabel("Wavelength m ")
axs[1].set_ylabel("[W (mA)/(10^14 cm-2]")
plt.tight_layout()



plt.show()

