from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import specutils


# Import data files
#nrs1 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_00_01.250Z\MAST_2024-03-03T17_00_01.250Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_x1d.fits')
#nrs2 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_00_01.250Z\MAST_2024-03-03T17_00_01.250Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_x1d.fits')
nrs1 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_06_05.404Z\MAST_2024-03-03T17_06_05.404Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs1_x1d.fits')
nrs2 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_06_05.404Z\MAST_2024-03-03T17_06_05.404Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs2_x1d.fits')
#nrs1 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_37_11.649Z\MAST_2024-03-03T17_37_11.649Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs1_x1d.fits')
#nrs2 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_37_11.649Z\MAST_2024-03-03T17_37_11.649Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs2_x1d.fits')

# Get the data
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)

# Get the header of the data
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)

# Extract wavelength and flux from the data
nrs1_wavelength = nrs1_data['WAVELENGTH']
nrs1_flux = nrs1_data['FLUX']

nrs2_wavelength = nrs2_data['WAVELENGTH']
nrs2_flux = nrs2_data['FLUX']

# Focus on a specific interval
'''
low_limit = 3.992
up_limit = 3.996
'''
low_limit = 4.010
up_limit = 4.014

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
nrs2_window_close = int(nrs2_close_value.shape[0]*0.2)

nrs1_smooth = signal.savgol_filter(nrs1_flux, window_length=150, polyorder=3, mode="nearest")
nrs2_smooth = signal.savgol_filter(nrs2_flux, window_length=250, polyorder=3, mode="nearest")
nrs2_smooth_close = signal.savgol_filter(nrs2_close_value,window_length=40, polyorder=3, mode="nearest")


plt.figure(1)
plt.plot(nrs1_wavelength,nrs1_flux)
plt.plot(nrs2_wavelength,nrs2_flux)
plt.plot(nrs1_wavelength,nrs1_smooth)
plt.plot(nrs2_wavelength,nrs2_smooth)
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(2)
plt.plot(nrs2_close_wave,nrs2_close_value)
plt.plot(nrs2_close_wave,nrs2_smooth_close)
plt.axvline(x=4.0132, color='r', linestyle='--', label='Vertical Line')
plt.axvline(x=4.0119, color='r', linestyle='--', label='Vertical Line')
#plt.axvline(x=3.9945, color='r', linestyle='--', label='Vertical Line')
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.show()



