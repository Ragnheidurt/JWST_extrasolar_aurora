from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy import signal
import sys
import os
from specutils import Spectrum1D
from specutils.analysis import correlation
from astropy import units as u
sys.path.append('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Functions')
from average_errors import average_errors
from short_interval import short_interval
from brightest_spot import brightest_spot
from model_data import model_data
from model_data import model_shortening
from scale_model import scale_model

print("Current working directory:", os.getcwd())

nrs1_ks = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs1_s3d.fits')
nrs2_ks = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs2_s3d.fits')
print("Current working directory: ", os.getcwd())
#nrs1_ff = get_pkg_data_filename('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Firefly/Data/private_data/MAST_2024-03-03T17_00_01.250Z/MAST_2024-03-03T17_00_01.250Z/JWST/jw01189011001_05101_00001/jw01189011001_05101_00001_nrs1_x1d.fits')
#nrs2_ff = get_pkg_data_filename('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Firefly/Data/private_data/MAST_2024-03-03T17_00_01.250Z/MAST_2024-03-03T17_00_01.250Z/JWST/jw01189011001_05101_00001/jw01189011001_05101_00001_nrs2_x1d.fits')

nrs1_ff_data = fits.getdata('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Firefly/Data/private_data/MAST_2024-03-03T17_00_01.250Z/MAST_2024-03-03T17_00_01.250Z/JWST/jw01189011001_05101_00001/jw01189011001_05101_00001_nrs1_x1d.fits',ext=1)
nrs2_ff_data = fits.getdata('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Firefly/Data/private_data/MAST_2024-03-03T17_00_01.250Z/MAST_2024-03-03T17_00_01.250Z/JWST/jw01189011001_05101_00001/jw01189011001_05101_00001_nrs2_x1d.fits',ext=1)

# Get the header of the data
nrs1_ff_header = fits.getheader('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Firefly/Data/private_data/MAST_2024-03-03T17_00_01.250Z/MAST_2024-03-03T17_00_01.250Z/JWST/jw01189011001_05101_00001/jw01189011001_05101_00001_nrs1_x1d.fits',ext=1)
nrs2_ff_header = fits.getheader('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Firefly/Data/private_data/MAST_2024-03-03T17_00_01.250Z/MAST_2024-03-03T17_00_01.250Z/JWST/jw01189011001_05101_00001/jw01189011001_05101_00001_nrs2_x1d.fits',ext=1)

#print(nrs1_data.dtype)

# Extract wavelength and flux from the data
nrs1_ff_wavelength = nrs1_ff_data['WAVELENGTH']
nrs1_ff_flux = nrs1_ff_data['FLUX']
nrs1_ff_sb = nrs1_ff_data['SURF_BRIGHT']
nrs1_ff_flux_error = nrs1_ff_data['FLUX_ERROR']
nrs1_ff_flux_poisson = nrs1_ff_data['FLUX_VAR_POISSON']

nrs2_ff_wavelength = nrs2_ff_data['WAVELENGTH']
nrs2_ff_flux = nrs2_ff_data['FLUX']
nrs2_ff_sb = nrs2_ff_data['SURF_BRIGHT']
nrs2_ff_flux_error = nrs2_ff_data['FLUX_ERROR']


# Extract the data from the files
nrs1_ks_data = fits.getdata(nrs1_ks,ext=1)
nrs2_ks_data = fits.getdata(nrs2_ks,ext=1)
nrs1_ks_error = fits.getdata(nrs1_ks,ext=2)
nrs2_ks_error = fits.getdata(nrs2_ks,ext=2)
nrs1_ks_wavelenght = fits.getdata(nrs1_ks,ext=4)
nrs2_ks_wavelenght = fits.getdata(nrs2_ks,ext=4)

# Extract header for both data sets
nrs1_ks_header = fits.getheader(nrs1_ks,ext=1)
nrs2_ks_header = fits.getheader(nrs2_ks,ext=1)
error_ks_header = fits.getheader(nrs1_ks,ext=2)

# Get the start and end wavelengths for both sensors
nrs1_ks_wavestart = nrs1_ks_header['WAVSTART']
nrs1_ks_wavend = nrs1_ks_header['WAVEND']
nrs2_ks_wavestart = nrs2_ks_header['WAVSTART']
nrs2_ks_wavend = nrs2_ks_header['WAVEND']

# Calculate the interval and size of wavelength data
nrs1_ks_dist = nrs1_ks_wavend-nrs1_ks_wavestart
nrs1_ks_interval = nrs1_ks_dist/nrs1_ks_data.shape[0]
nrs2_ks_dist = nrs2_ks_wavend-nrs2_ks_wavestart
nrs2_ks_interval = nrs2_ks_dist/nrs2_ks_data.shape[0]

# Create array with wavelength
nrs1_ks_wave = np.zeros(nrs1_ks_data.shape[0])
nrs2_ks_wave = np.zeros(nrs2_ks_data.shape[0])


for i in range(nrs1_ks_data.shape[0]):
    if i != 0: 
        nrs1_ks_wave[i] = nrs1_ks_wave[i-1] + nrs1_ks_interval
    else:
        nrs1_ks_wave[i] = nrs1_ks_wavestart

for i in range(nrs2_ks_data.shape[0]):
    if i != 0:
        nrs2_ks_wave[i] = nrs2_ks_wave[i-1] + nrs2_ks_interval
    else:
        nrs2_ks_wave[i] = nrs2_ks_wavestart

nrs1_ks_avg, nrs1_ks_errors, nrs1_ks_maxvalue = brightest_spot(nrs1_ks_data,0,nrs1_ks_error,1500)
nrs2_ks_avg, nrs2_ks_errors, nrs2_ks_maxvalue = brightest_spot(nrs2_ks_data,0,nrs2_ks_error,2500)
nrs1_ks_secavg, nrs1_ks_secerror, nrs1_ks_sec_maxvalue = brightest_spot(nrs1_ks_data,nrs1_ks_maxvalue,nrs1_ks_error,1500)
nrs2_ks_secavg, nrs2_ks_secerror, nrs2_ks_sec_maxvalue = brightest_spot(nrs2_ks_data,nrs2_ks_maxvalue,nrs2_ks_error,2500)

nrs1_ks_diff = nrs1_ks_avg-nrs1_ks_secavg
nrs2_ks_diff = nrs2_ks_avg-nrs2_ks_secavg

low_limit = 3.5e-06
up_limit = 4.5e-06

wavenumber, absorption_intensity1, vacuum_wavelength, column_density, stuff = model_data('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST-Jupiter/Jupiter/Data/h3_generated1.txt')
scaled_model = scale_model(column_density,nrs1_ks_avg,nrs2_ks_avg,vacuum_wavelength,nrs2_ks_wave)
model_wave_short,model_flux_short = model_shortening(vacuum_wavelength, low_limit, up_limit, scaled_model)

wave_ks_short1, flux_ks_short1 = short_interval(nrs1_ks_diff, low_limit, up_limit, nrs1_ks_wave)
wave_ks_short2, flux_ks_short2 = short_interval(nrs2_ks_diff, low_limit, up_limit, nrs2_ks_wave)

print(nrs1_ks_maxvalue[0])
nrs1_ks_brightest = nrs1_ks_data[:,nrs1_ks_maxvalue[0],nrs1_ks_maxvalue[1]]
nrs2_ks_brightest = nrs2_ks_data[:,nrs2_ks_maxvalue[0],nrs2_ks_maxvalue[1]]
nrs1_ks_nextBrightest = nrs1_ks_data[:,nrs1_ks_maxvalue[0]-1,nrs1_ks_maxvalue[1]-1]
nrs2_ks_nextBrightest = nrs2_ks_data[:,nrs2_ks_maxvalue[0]-1,nrs2_ks_maxvalue[1]-1]

nrs1_ks_brightest_unit = nrs1_ks_brightest*u.Jy
nrs2_ks_brightest_unit = nrs2_ks_brightest*u.Jy
nrs1_ks_nextBrightest_unit = nrs1_ks_nextBrightest*u.Jy
nrs2_ks_nextBrightest_unit = nrs2_ks_nextBrightest*u.Jy

nrs1_ks_avg_unit = nrs1_ks_avg*u.Jy
nrs1_ks_secavg_unit = nrs1_ks_secavg*u.Jy
nrs1_ks_wave_unit = nrs1_ks_wave*u.m
nrs1_ks_spectrum1 = Spectrum1D(spectral_axis=nrs1_ks_wave_unit, flux=nrs1_ks_avg_unit)
nrs1_ks_spectrum2 = Spectrum1D(spectral_axis=nrs1_ks_wave_unit, flux=nrs1_ks_secavg_unit)

nrs1_ks_spectrum_brightest = Spectrum1D(spectral_axis=nrs1_ks_wave_unit, flux=nrs1_ks_brightest_unit)
nrs1_ks_spectrum_nextBrightest = Spectrum1D(spectral_axis=nrs1_ks_wave_unit, flux=nrs1_ks_nextBrightest_unit)

nrs1_ks_brightest_corr, nrs1_ks_brightest_lag = correlation.template_correlate(nrs1_ks_spectrum_brightest,nrs1_ks_spectrum_nextBrightest)


nrs1_ff_flux_unit = nrs1_ff_flux*u.Jy
nrs1_ff_wavelength_unit = nrs1_ff_wavelength*u.um
nrs1_ff_spectrum1 = Spectrum1D(spectral_axis=nrs1_ff_wavelength_unit, flux=nrs1_ff_flux_unit)

#resampled_ff_spectrum1 = nrs1_ff_spectrum1.resample(spectral_axis=nrs1_ks_spectrum1.spectral_axis, flux = nrs1_ks_spectrum1.flux)


nrs1_ks_corr, nrs1_ks_lag = correlation.template_correlate(nrs1_ks_spectrum1,nrs1_ks_spectrum2)

nrs1_ks_scaled_spectrum2 = nrs1_ks_secavg*np.nanmax(nrs1_ks_avg)/np.nanmax(nrs1_ks_secavg)
nrs1_ks_difference_spectrum = nrs1_ks_avg-nrs1_ks_scaled_spectrum2

nrs2_ks_avg_unit = nrs2_ks_avg*u.Jy
nrs2_ks_secavg_unit = nrs2_ks_secavg*u.Jy
nrs2_ks_wave_unit = nrs2_ks_wave*u.um
nrs2_ks_wave_unit_moved = (nrs2_ks_wave)*u.um
nrs2_ks_avg_unit_moved = (nrs2_ks_avg)*u.Jy

nrs2_ks_spectrum1 = Spectrum1D(spectral_axis=nrs2_ks_wave_unit, flux=nrs2_ks_avg_unit)
nrs2_ks_spectrum2 = Spectrum1D(spectral_axis=nrs2_ks_wave_unit, flux=nrs2_ks_secavg_unit)

nrs2_ks_spectrum_brightest = Spectrum1D(spectral_axis=nrs2_ks_wave_unit, flux=nrs2_ks_brightest_unit)
nrs2_ks_spectrum_nextBrightest = Spectrum1D(spectral_axis=nrs2_ks_wave_unit, flux=nrs2_ks_nextBrightest_unit)

nrs2_ks_brightest_corr, nrs2_ks_brightest_lag = correlation.template_correlate(nrs2_ks_spectrum_brightest,nrs2_ks_spectrum_nextBrightest)


nrs2_ks_control_spectrum = Spectrum1D(spectral_axis=nrs2_ks_wave_unit_moved, flux=nrs2_ks_avg_unit_moved)

nrs2_ks_corr, nrs2_ks_lag = correlation.template_correlate(nrs2_ks_spectrum1,nrs2_ks_control_spectrum)

nrs2_ks_scaled_spectrum2 = nrs2_ks_secavg*np.nanmax(nrs2_ks_avg)/np.nanmax(nrs2_ks_secavg)
nrs2_ks_difference_spectrum = nrs2_ks_avg-nrs2_ks_scaled_spectrum2

plt.figure(1)
plt.plot(nrs1_ks_wave,nrs1_ks_brightest)
plt.plot(nrs2_ks_wave,nrs2_ks_brightest)

plt.figure(2)
plt.plot(nrs1_ks_wave,nrs1_ks_diff)
plt.plot(nrs2_ks_wave,nrs2_ks_diff)

plt.figure(3)
plt.plot(wave_ks_short1,flux_ks_short1)
plt.plot(wave_ks_short2,flux_ks_short2)
plt.plot(model_wave_short,model_flux_short,color='red', linestyle='--', alpha=0.5)

plt.figure(4)
plt.plot(nrs1_ks_wave,nrs1_ks_difference_spectrum)
plt.plot(nrs2_ks_wave,nrs2_ks_difference_spectrum)

plt.figure(5)
plt.plot(nrs1_ks_wave,nrs1_ks_avg)
plt.plot(nrs2_ks_wave,nrs2_ks_avg)

plt.figure(6)
plt.plot(nrs1_ks_lag, nrs1_ks_corr)

plt.figure(7)
plt.plot(nrs2_ks_lag, nrs2_ks_corr)

plt.figure(8)
plt.plot(nrs1_ff_wavelength,nrs1_ff_flux)
plt.plot(nrs2_ff_wavelength,nrs2_ff_flux)

fig, axs = plt.subplots(2, 1)
axs[0].plot(nrs1_ks_brightest_lag,nrs1_ks_brightest_corr, color='red')
axs[1].plot(nrs2_ks_brightest_lag, nrs2_ks_brightest_corr, color='green')
axs[0].set_title(f"Brightest pixel and the one next to it nrs1")
axs[1].set_title(f"Brightest pixel and the one next to it nrs2")
axs[0].set_xlabel("lag")
axs[0].set_ylabel("corr")
axs[1].set_xlabel("lag")
axs[1].set_ylabel("corr")
plt.tight_layout()


plt.show()






