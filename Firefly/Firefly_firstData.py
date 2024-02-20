from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt

nrs1 = get_pkg_data_filename('MAST_2024-02-07T09_10_43.556Z\MAST_2024-02-07T09_10_43.556Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_s2d.fits')
nrs2 = get_pkg_data_filename('MAST_2024-02-07T09_10_43.556Z\MAST_2024-02-07T09_10_43.556Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_s2d.fits')

nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)

nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)

print(nrs1_header)
print(nrs2_header)

nrs1_wavestart = nrs1_header['WAVSTART']
nrs1_wavend = nrs1_header['WAVEND']
nrs2_wavestart = nrs2_header['WAVSTART']
nrs2_wavend = nrs2_header['WAVEND']

# Calculate the interval and size of wavelength data
nrs1_dist = nrs1_wavend-nrs1_wavestart
nrs1_interval = nrs1_dist/nrs1_data.shape[0]
nrs2_dist = nrs2_wavend-nrs2_wavestart
nrs2_interval = nrs2_dist/nrs2_data.shape[0]


print(nrs1_data.shape)
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

plt.figure(1)
plt.imshow(nrs1_data)

plt.figure(2)
plt.imshow(nrs2_data)

plt.figure(3)
plt.plot(nrs1_data)

plt.show()