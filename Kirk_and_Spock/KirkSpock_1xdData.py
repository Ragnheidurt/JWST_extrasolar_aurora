from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys
sys.path.append('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Functions')
from average_errors import average_errors
from short_interval import short_interval


# Import data files
nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs1_x1d.fits')
nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs2_x1d.fits')

# Get the data
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)

# Get the header of the data
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)

# Extract wavelength and flux from the data
nrs1_wavelength = nrs1_data['WAVELENGTH']
nrs1_flux = nrs1_data['FLUX']
nrs1_flux_error = nrs1_data['FLUX_ERROR']

nrs2_wavelength = nrs2_data['WAVELENGTH']
nrs2_flux = nrs2_data['FLUX']
nrs2_flux_error = nrs2_data['FLUX_ERROR']


# Remove outliers
for i in range(nrs1_flux.shape[0]):
    if abs(nrs1_flux[i]) > 10 and i != 0:
        nrs1_flux[i] = nrs1_flux[i-1]
    elif nrs1_flux[i] < 0 and i != 0:
        nrs1_flux[i] = nrs1_flux[i-1]

for i in range(nrs2_flux.shape[0]):
    if abs(nrs2_flux[i]) > 10 and i != 0:
        nrs2_flux[i] = nrs2_flux[i-1]
    elif nrs2_flux[i] < 0 and i != 0:
        nrs2_flux[i] = nrs2_flux[i-1]

# Focus on a specific interval
low_limit = 3.993
up_limit = 3.997
nrs1_lengd = 0
for i in range(nrs1_wavelength.shape[0]):
    if(nrs1_wavelength[i]>=low_limit and nrs1_wavelength[i]<= up_limit):
        nrs1_lengd = nrs1_lengd +1


nrs2_lengd = 0
for i in range(nrs2_wavelength.shape[0]):
    if(nrs2_wavelength[i]>=low_limit and nrs2_wavelength[i]<= up_limit):
        nrs2_lengd = nrs2_lengd +1

nrs1_close_wave = np.zeros(nrs1_lengd,dtype='float32')
nrs1_close_value = np.zeros(nrs1_lengd,dtype='float32')
count = 0
for i in range(nrs1_wavelength.shape[0]):
    if nrs1_wavelength[i] >= low_limit and nrs1_wavelength[i] <= up_limit:
        nrs1_close_wave[count] = nrs1_wavelength[i]
        nrs1_close_value[count] = nrs1_flux[i]
        count = count+1


nrs2_close_wave = np.zeros(nrs2_lengd,dtype='float32')
nrs2_close_value = np.zeros(nrs2_lengd,dtype='float32')
count = 0
for i in range(nrs2_wavelength.shape[0]):
    if nrs2_wavelength[i] >= low_limit and nrs2_wavelength[i] <= up_limit:
        nrs2_close_wave[count] = nrs2_wavelength[i]
        nrs2_close_value[count] = nrs2_flux[i]
        count = count+1

# Smoothing the data
nrs1_window_all = int(nrs1_flux.shape[0]*0.2)
nrs1_window_all = int(nrs2_flux.shape[0]*0.2)
nrs1_window_close = int(nrs2_close_value.shape[0]*0.2)

nrs1_smooth = signal.savgol_filter(nrs1_flux, window_length=150, polyorder=3, mode="nearest")
nrs2_smooth = signal.savgol_filter(nrs2_flux, window_length=250, polyorder=3, mode="nearest")
nrs1_smooth_close = signal.savgol_filter(nrs1_close_value,window_length=40, polyorder=3, mode="nearest")

divisors1 = [i for i in range(1, nrs1_flux.shape[0] + 1) if nrs1_flux.shape[0] % i == 0]
divisors2 = [i for i in range(1,nrs2_flux.shape[0]+1) if nrs2_flux.shape[0]%i==0]


nrs1_wave_average, nrs1_average, nrs1_error_average = average_errors(nrs1_wavelength,nrs1_flux,nrs1_flux_error,divisors1[4])
nrs2_wave_average, nrs2_average, nrs2_error_average = average_errors(nrs2_wavelength,nrs2_flux,nrs2_flux_error,divisors2[5])
nrs1_short_wave, nrs1_short_flux = short_interval(nrs1_flux,low_limit,up_limit,nrs1_wavelength)

plt.figure(1)
plt.plot(nrs1_wavelength,nrs1_flux)
plt.plot(nrs2_wavelength,nrs2_flux)
plt.plot(nrs1_wavelength,nrs1_smooth)
plt.plot(nrs2_wavelength,nrs2_smooth)
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(2)
plt.plot(nrs1_close_wave,nrs1_close_value)
plt.plot(nrs1_close_wave,nrs1_smooth_close)
#plt.axvline(x=3.953, color='r', linestyle='--', label='Vertical Line')
#plt.axvline(x=3.9855, color='r', linestyle='--', label='Vertical Line')
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(3)
plt.plot(nrs1_wave_average,nrs1_average)
plt.fill_between(nrs1_wave_average,nrs1_average-nrs1_error_average, nrs1_average+nrs1_error_average,alpha = 0.5, color='green',label='error')
plt.plot(nrs2_wave_average,nrs2_average)
plt.fill_between(nrs2_wave_average,nrs2_average-nrs2_error_average, nrs2_average+nrs2_error_average,alpha = 0.5, color='green',label='error')


plt.figure(4)
plt.plot(nrs1_short_wave,nrs1_short_flux)


plt.show()



