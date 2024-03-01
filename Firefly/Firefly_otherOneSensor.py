from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# Import data files
nrs1 = get_pkg_data_filename('MAST_2024-02-14T09_20_36.412Z\MAST_2024-02-14T09_20_36.412Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs1_x1d.fits')

# Get the data
nrs1_data = fits.getdata(nrs1,ext=1)

# Get the header of the data
nrs1_header = fits.getheader(nrs1,ext=1)

# Extract wavelength and flux from the data
nrs1_wavelength = nrs1_data['WAVELENGTH']
nrs1_flux = nrs1_data['FLUX']

# Focus on a specific interval
low_limit = 3.92
up_limit = 4.02
nrs1_lengd = 0
for i in range(nrs1_wavelength.shape[0]):
    if(nrs1_wavelength[i]>=low_limit and nrs1_wavelength[i]<= up_limit):
        nrs1_lengd = nrs1_lengd +1


nrs1_close_wave = np.zeros(nrs1_lengd,dtype='float32')
nrs1_close_value = np.zeros(nrs1_lengd,dtype='float32')
count = 0
for i in range(nrs1_wavelength.shape[0]):
    if nrs1_wavelength[i] >= 3.92 and nrs1_wavelength[i] <= 4.02:
        nrs1_close_wave[count] = nrs1_wavelength[i]
        nrs1_close_value[count] = nrs1_flux[i]
        count = count+1


# Smoothing the data

nrs1_smooth = signal.savgol_filter(nrs1_flux, window_length=150, polyorder=3, mode="nearest")
nrs1_smooth_close = signal.savgol_filter(nrs1_close_value,window_length=40, polyorder=3, mode="nearest")


plt.figure(1)
plt.plot(nrs1_wavelength,nrs1_flux)
plt.plot(nrs1_wavelength,nrs1_smooth)

plt.figure(2)
plt.plot(nrs1_close_wave,nrs1_close_value)
plt.plot(nrs1_close_wave,nrs1_smooth_close)
plt.axvline(x=3.953, color='r', linestyle='--', label='Vertical Line')
plt.axvline(x=3.9855, color='r', linestyle='--', label='Vertical Line')

plt.show()



