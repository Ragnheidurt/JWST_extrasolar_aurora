from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import math
import sys
sys.path.append('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Functions')
from average_errors import average_errors
from short_interval import short_interval
from model_data import model_data
from model_data import model_shortening
from scale_model import scale_model


# Import data files
#nrs1 = get_pkg_data_filename('MAST_2024-02-07T09_10_43.556Z\MAST_2024-02-07T09_10_43.556Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_x1d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-02-07T09_10_43.556Z\MAST_2024-02-07T09_10_43.556Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_x1d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-14T09_20_36.412Z\MAST_2024-02-14T09_20_36.412Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs1_x1d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-02-14T09_20_36.412Z\MAST_2024-02-14T09_20_36.412Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs2_x1d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-29T12_33_30.811Z\MAST_2024-02-29T12_33_30.811Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_x1d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-02-29T12_33_30.811Z\MAST_2024-02-29T12_33_30.811Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_x1d.fits')

#
#nrs1 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_00_01.250Z\MAST_2024-03-03T17_00_01.250Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_x1d.fits')
#nrs2 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_00_01.250Z\MAST_2024-03-03T17_00_01.250Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_x1d.fits')
#nrs1 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_06_05.404Z\MAST_2024-03-03T17_06_05.404Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs1_x1d.fits')
#nrs2 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_06_05.404Z\MAST_2024-03-03T17_06_05.404Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs2_x1d.fits')
nrs1 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_37_11.649Z\MAST_2024-03-03T17_37_11.649Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs1_x1d.fits')
nrs2 = get_pkg_data_filename('Data\private_data\MAST_2024-03-03T17_37_11.649Z\MAST_2024-03-03T17_37_11.649Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs2_x1d.fits')

# Get the data
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)

# Get the header of the data
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)
print(nrs1_header)

#print(nrs1_data.dtype)

# Extract wavelength and flux from the data
nrs1_wavelength = nrs1_data['WAVELENGTH']
nrs1_flux = nrs1_data['FLUX']
nrs1_sb = nrs1_data['SURF_BRIGHT']
nrs1_flux_error = nrs1_data['FLUX_ERROR']
nrs1_flux_poisson = nrs1_data['FLUX_VAR_POISSON']

nrs2_wavelength = nrs2_data['WAVELENGTH']
nrs2_flux = nrs2_data['FLUX']
nrs2_sb = nrs2_data['SURF_BRIGHT']
nrs2_flux_error = nrs2_data['FLUX_ERROR']

# Focus on a specific interval
'''
low_limit = 3.992
up_limit = 3.996
'''
low_limit = 3.0
up_limit = 3.5

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

# Averaging every 5 points together
size = 12
nrs1_reshape = nrs1_flux.reshape(-1,size)
nrs1_wave_reshape = nrs1_wavelength.reshape(-1,size)
nrs1_average = np.mean(nrs1_reshape,axis=1)
nrs1_wave_average = np.mean(nrs1_wave_reshape,axis=1)
nrs1_average = nrs1_average.flatten()
nrs1_wave_average = nrs1_wave_average.flatten()
staerd = 32
nrs2_reshape = nrs2_flux.reshape(-1,staerd)
nrs2_wave_reshape = nrs2_wavelength.reshape(-1,staerd)
nrs2_average = np.mean(nrs2_reshape,axis=1)
nrs2_wave_average = np.mean(nrs2_wave_reshape,axis=1)
nrs2_average = nrs2_average.flatten()
nrs2_wave_average = nrs2_wave_average.flatten()

count = 0
nrs1_error_average = np.zeros(nrs1_average.shape[0])

for i in range(nrs1_flux_error.shape[0]):
    if i != 0 and math.isnan(nrs1_flux_error[i]):
        nrs1_flux_error[i] = nrs1_flux_error[i-1]
    elif math.isnan(nrs1_flux_error[i]):
        non_nan_indices = np.where(~np.isnan(nrs1_flux_error))[0]
        nrs1_flux_error[i] = nrs1_flux_error[non_nan_indices[0]]



for i in range(nrs1_average.shape[0]):
    sum = 0
    for j in range(size):
        sum = sum+nrs1_flux_error[count+j]**2
    nrs1_error_average[i] = math.sqrt(sum)
    count = count + size

count = 0
nrs2_error_average = np.zeros(nrs2_average.shape[0])

for i in range(nrs2_flux_error.shape[0]):
    if i != 0 and math.isnan(nrs2_flux_error[i]):
        nrs2_flux_error[i] = nrs2_flux_error[i-1]
    elif math.isnan(nrs2_flux_error[i]):
        non_nan_indices = np.where(~np.isnan(nrs2_flux_error))[0]
        nrs2_flux_error[i] = nrs2_flux_error[non_nan_indices[0]]



for i in range(nrs2_average.shape[0]):
    sum = 0
    for j in range(staerd):
        sum = sum+nrs2_flux_error[count+j]**2
    nrs2_error_average[i] = math.sqrt(sum)
    count = count + size

wavenumber, absorption_intensity1, vacuum_wavelength, column_density, stuff = model_data('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST-Jupiter/Jupiter/Data/h3_generated1.txt')
vacuum_wavelength = vacuum_wavelength*1e6
#scaled_model = scale_model(absorption_intensity1,nrs1_flux, nrs2_flux,vacuum_wavelength,nrs2_wavelength)
print(nrs1_wavelength)
nrs1_dist = nrs2_wavelength[-1]-nrs1_wavelength[0]
nrs1_interval = nrs1_dist/(nrs1_data.shape[0]+nrs2_data.shape[0])


# Create array with wavelength
nrs1_wave = np.zeros((nrs1_data.shape[0]+nrs2_data.shape[0]))

for i in range(nrs1_data.shape[0]+nrs2_data.shape[0]):
    if i != 0: 
        nrs1_wave[i] = nrs1_wave[i-1] + nrs1_interval
    else:
        nrs1_wave[i] = nrs1_wavelength[0]

print(nrs1_wave.shape[0])
print(type(nrs2_flux))
null = np.zeros(nrs1_flux.shape[0])
flux2_for_scale = np.concatenate((null,nrs2_flux))
scaled_model = scale_model(absorption_intensity1,nrs1_flux, flux2_for_scale,vacuum_wavelength,nrs1_wave)
model_wave_short,model_flux_short = model_shortening(vacuum_wavelength, low_limit, up_limit, scaled_model)


plt.figure(1)
plt.plot(nrs1_wavelength,nrs1_flux)
plt.plot(nrs2_wavelength,nrs2_flux)
plt.plot(nrs1_wavelength,nrs1_smooth)
plt.plot(nrs2_wavelength,nrs2_smooth)
#plt.plot(vacuum_wavelength,scaled_model,alpha=0.5, linestyle='--')
#plt.fill_between(nrs1_wavelength,nrs1_flux-nrs1_flux_error, nrs1_flux+nrs1_flux_error,alpha = 0.5, label='error')
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(2)
plt.plot(nrs1_close_wave,nrs1_close_value)
plt.plot(nrs2_close_wave,nrs2_close_value)
plt.plot(nrs2_close_wave,nrs2_smooth_close)
#plt.plot(model_wave_short,model_flux_short, alpha=0.5, linestyle='--')
#plt.axvline(x=4.0132, color='r', linestyle='--', label='Vertical Line')
#plt.axvline(x=4.0119, color='r', linestyle='--', label='Vertical Line')
#plt.axvline(x=3.9945, color='r', linestyle='--', label='Vertical Line')
plt.xlabel('Wavelength [$\mu$m]')
plt.ylabel('Brightness')

plt.figure(3)
plt.plot(nrs1_wavelength,nrs1_sb)
plt.plot(nrs2_wavelength,nrs2_sb)

plt.figure(4)
plt.plot(nrs1_wave_average,nrs1_error_average)
#plt.plot(nrs2_wavelength,nrs2_flux_error)

plt.figure(5)
#plt.plot(nrs1_wavelength,nrs1_flux)
#plt.plot(nrs1_wavelength,nrs1_smooth)
#plt.fill_between(nrs1_wavelength,nrs1_flux-nrs1_flux_poisson, nrs1_flux+nrs1_flux_poisson,alpha = 0.5, color='green',label='error')
#plt.errorbar(nrs1_wavelength,nrs1_flux,yerr=nrs1_flux_error,fmt='--', label='flux with error')
plt.plot(nrs1_wave_average,nrs1_average)
plt.fill_between(nrs1_wave_average,nrs1_average-nrs1_error_average, nrs1_average+nrs1_error_average,alpha = 0.5, color='green',label='error')
plt.plot(nrs2_wave_average,nrs2_average)
plt.fill_between(nrs2_wave_average,nrs2_average-nrs2_error_average, nrs2_average+nrs2_error_average,alpha = 0.5, color='green',label='error')



plt.show()



