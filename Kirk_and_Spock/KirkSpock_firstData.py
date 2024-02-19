from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math


# Read in files from both sensors
nrs1 = get_pkg_data_filename('MAST_2024-01-31T15_21_00.167Z\MAST_2024-01-31T15_21_00.167Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs1_s3d.fits')
nrs2 = get_pkg_data_filename('MAST_2024-01-31T15_21_00.167Z\MAST_2024-01-31T15_21_00.167Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs2_s3d.fits')


# Extract the data from the files
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)

# Extract header for both data sets
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)

# Get the start and end wavelengths for both sensors
nrs1_wavestart = nrs1_header['WAVESTART']
nrs1_wavend = nrs1_header['WAVEND']
nrs2_wavestart = nrs2_header['WAVESTART']
nrs2_wavend = nrs2_header['WAVEND']

# Calculate the interval and size of wavelength data
nrs1_dist = nrs1_wavestart-nrs1_wavend
nrs1_interval = nrs1_dist/nrs1_data.shape[0]
nrs2_dist = nrs2_wavestart-nrs2_wavend
nrs2_interval = nrs2_dist/nrs2_data.shape[0]

# Create array with wavelength
nrs1_wave = np.zeros(nrs1_data.shape[0])
nrs2_wave = np.zeros(nrs2_data.shape[0])

for i in nrs1_data.shape[0]:
    if i != 0: 
        nrs1_wave[i] = nrs1_wave[i-1] + nrs1_interval
    else:
        nrs1_wave[i] = nrs1_wavestart

for i in nrs2_data.shape[0]:
    if i != 0:
        nrs2_wave[i] = nrs2_wave[i-1] + nrs2_interval
    else:
        nrs2_wave[i] = nrs2_wavestart
        

