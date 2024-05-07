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
from specutils.manipulation import FluxConservingResampler
from astropy import units as u
sys.path.append('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Functions')
from average_errors import average_errors
from short_interval import short_interval
from brightest_spot import brightest_spot
from model_data import model_data
from model_data import model_shortening
from scale_model import scale_model

# Firefly files
#nrs1_ff = get_pkg_data_filename('Firefly\Data\private_data\MAST_2024-03-03T17_00_01.250Z\MAST_2024-03-03T17_00_01.250Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_x1d.fits')
#nrs2_ff = get_pkg_data_filename('Firefly\Data\private_data\MAST_2024-03-03T17_00_01.250Z\MAST_2024-03-03T17_00_01.250Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_x1d.fits')
nrs1_ff = get_pkg_data_filename('Firefly\Data\private_data\MAST_2024-03-03T17_06_05.404Z\MAST_2024-03-03T17_06_05.404Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs1_x1d.fits')
nrs2_ff = get_pkg_data_filename('Firefly\Data\private_data\MAST_2024-03-03T17_06_05.404Z\MAST_2024-03-03T17_06_05.404Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs2_x1d.fits')
#nrs1_ff = get_pkg_data_filename('Firefly\Data\private_data\MAST_2024-03-03T17_37_11.649Z\MAST_2024-03-03T17_37_11.649Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs1_x1d.fits')
#nrs2_ff = get_pkg_data_filename('Firefly\Data\private_data\MAST_2024-03-03T17_37_11.649Z\MAST_2024-03-03T17_37_11.649Z\JWST\jw01189011001_05101_00003\jw01189011001_05101_00003_nrs2_x1d.fits')

# Kirk & Spock files
nrs1_ks = get_pkg_data_filename('Kirk_and_Spock\Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs1_x1d.fits')
nrs2_ks = get_pkg_data_filename('Kirk_and_Spock\Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs2_x1d.fits')

# Firefly get data
# Get the data
nrs1_ff_data = fits.getdata(nrs1_ff,ext=1)
nrs2_ff_data = fits.getdata(nrs2_ff,ext=1)

# Get the header of the data
nrs1_ff_header = fits.getheader(nrs1_ff,ext=1)
nrs2_ff_header = fits.getheader(nrs2_ff,ext=1)

# Extract wavelength and flux from the data
nrs1_ff_wavelength = nrs1_ff_data['WAVELENGTH']
nrs1_ff_flux = nrs1_ff_data['FLUX']

nrs2_ff_wavelength = nrs2_ff_data['WAVELENGTH']
nrs2_ff_flux = nrs2_ff_data['FLUX']

# Kirk & Spock get data
# Get the data
nrs1_ks_data = fits.getdata(nrs1_ks,ext=1)
nrs2_ks_data = fits.getdata(nrs2_ks,ext=1)

# Get the header of the data
nrs1_ks_header = fits.getheader(nrs1_ks,ext=1)
nrs2_ks_header = fits.getheader(nrs2_ks,ext=1)

# Extract wavelength and flux from the data
nrs1_ks_wavelength = nrs1_ks_data['WAVELENGTH']
nrs1_ks_flux = nrs1_ks_data['FLUX']

nrs2_ks_wavelength = nrs2_ks_data['WAVELENGTH']
nrs2_ks_flux = nrs2_ks_data['FLUX']

# Remove outliers for Firefly
for i in range(nrs1_ff_flux.shape[0]):
    if abs(nrs1_ff_flux[i]) > 10 and i != 0:
        nrs1_ff_flux[i] = nrs1_ff_flux[i-1]
    elif nrs1_ff_flux[i] < 0 and i != 0:
        nrs1_ff_flux[i] = nrs1_ff_flux[i-1]

for i in range(nrs2_ff_flux.shape[0]):
    if abs(nrs2_ff_flux[i]) > 10 and i != 0:
        nrs2_ff_flux[i] = nrs2_ff_flux[i-1]
    elif nrs2_ff_flux[i] < 0 and i != 0:
        nrs2_ff_flux[i] = nrs2_ff_flux[i-1]

# Remove outliers Kirk & Spock
for i in range(nrs1_ks_flux.shape[0]):
    if abs(nrs1_ks_flux[i]) > 10 and i != 0:
        nrs1_ks_flux[i] = nrs1_ks_flux[i-1]
    elif nrs1_ks_flux[i] < 0 and i != 0:
        nrs1_ks_flux[i] = nrs1_ks_flux[i-1]

for i in range(nrs2_ks_flux.shape[0]):
    if abs(nrs2_ks_flux[i]) > 10 and i != 0:
        nrs2_ks_flux[i] = nrs2_ks_flux[i-1]
    elif nrs2_ks_flux[i] < 0 and i != 0:
        nrs2_ks_flux[i] = nrs2_ks_flux[i-1]



ff_flux = np.concatenate((nrs1_ff_flux,nrs2_ff_flux))
ff_wave = np.concatenate((nrs1_ff_wavelength,nrs2_ff_wavelength))
ks_flux = np.concatenate((nrs1_ks_flux,nrs2_ks_flux))
ks_wave = np.concatenate((nrs1_ks_wavelength,nrs2_ks_wavelength))
ks_flux_new = []
ks_wave_new = []
for i in range(ks_flux.shape[0]):
    if ks_flux[i] != 0: 
        ks_flux_new.append(ks_flux[i])
        ks_wave_new.append(ks_wave[i])


print(ff_flux.shape)
print(ff_wave.shape)
print(len(ks_flux_new))
print(ks_wave.shape)
ks_flux_new = np.array(ks_flux_new)
ks_wave_new = np.array(ks_wave_new)
ff_flux_unit = ff_flux*u.Jy
ks_flux_unit = ks_flux_new*u.Jy
ff_wave_unit = ff_wave*u.m
ks_wave_unit = ks_wave_new*u.m

ff_flux_spectra = Spectrum1D(spectral_axis=ff_wave_unit, flux=ff_flux_unit)
ks_flux_spectra = Spectrum1D(spectral_axis=ks_wave_unit, flux=ks_flux_unit)

output_dispersion = ff_flux_spectra.spectral_axis
resampler = FluxConservingResampler()
resampled_ks_flux = resampler(ks_flux_spectra,output_dispersion)

subtractedflux = resampled_ks_flux-ff_flux_spectra
#nrs2_subtractedflux = nrs2_ff_flux-nrs2_ks_flux
subtractedflux2 = ff_flux_spectra-resampled_ks_flux

#nrs1_subtractedOtherflux = nrs1_ks_flux-nrs1_ff_flux
#nrs2_subtractedOtherflux = nrs2_ks_flux-nrs2_ff_flux


plt.figure(1)
plt.plot(nrs1_ff_wavelength,nrs1_ff_flux)
plt.plot(nrs2_ff_wavelength,nrs2_ff_flux)

plt.figure(2)
plt.plot(nrs1_ks_wavelength,nrs1_ks_flux)
plt.plot(nrs2_ks_wavelength,nrs2_ks_flux)

plt.figure(3)
plt.plot(nrs1_ff_wavelength,nrs1_ff_flux, color='red')
plt.plot(nrs2_ff_wavelength,nrs2_ff_flux, color='red')
plt.plot(nrs1_ks_wavelength,nrs1_ks_flux, color='green')
plt.plot(nrs2_ks_wavelength,nrs2_ks_flux, color='green')

plt.figure(4)
plt.plot(ff_flux)
plt.plot(ks_flux_new)

plt.figure(5)
plt.plot(subtractedflux.spectral_axis,subtractedflux.flux)
#plt.plot(nrs2_ff_wavelength,nrs2_subtractedOtherflux)

plt.figure(6)
plt.plot(subtractedflux2.spectral_axis,subtractedflux2.flux)

plt.show()

