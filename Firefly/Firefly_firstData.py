from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt

nrs1 = get_pkg_data_filename('MAST_2024-02-07T09_10_43.556Z\MAST_2024-02-07T09_10_43.556Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_x1d.fits')
nrs2 = get_pkg_data_filename('MAST_2024-02-07T09_10_43.556Z\MAST_2024-02-07T09_10_43.556Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_x1d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-29T12_33_30.811Z\MAST_2024-02-29T12_33_30.811Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs1_s2d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-02-29T12_33_30.811Z\MAST_2024-02-29T12_33_30.811Z\JWST\jw01189011001_05101_00001\jw01189011001_05101_00001_nrs2_s2d.fits')

nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)

nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)

# The Wavestart and Wavend come from the header, but the gap is given for the slit S200A1 and G395H F290LP

nrs1_wavestart = nrs1_header['WAVSTART']
nrs1_wavend = 3.69e-6
nrs2_wavestart = 3.79e-6
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


# Average all lines
nrs1_allavg = np.nanmean(nrs1_data,axis=(1))
nrs2_allavg = np.nanmean(nrs2_data,axis=(1))

# Looking at a close interval

low_limit = 3.92e-06
up_limit = 4.02e-06
nrs1_lengd = 0
for i in range(nrs1_wave.shape[0]):
    if(nrs1_wave[i]>=low_limit and nrs1_wave[i]<= up_limit):
        nrs1_lengd = nrs1_lengd +1


nrs2_lengd = 0
for i in range(nrs2_wave.shape[0]):
    if(nrs2_wave[i]>=low_limit and nrs2_wave[i]<= up_limit):
        nrs2_lengd = nrs2_lengd +1

nrs1_close_wave = np.zeros(nrs1_lengd,dtype='float32')
nrs1_close_value = np.zeros((nrs1_data.shape[0],nrs1_lengd))
count = 0
for i in range(nrs1_wave.shape[0]):
    if nrs1_wave[i] >= 3.92e-06 and nrs1_wave[i] <= 4.02e-06:
        nrs1_close_wave[count] = nrs1_wave[i]
        nrs1_close_value[:,count] = nrs1_data[:,i]
        count = count+1


nrs2_close_wave = np.zeros(nrs2_lengd,dtype='float32')
nrs2_close_value = np.zeros((nrs2_data.shape[0],nrs2_lengd))
count = 0
for i in range(nrs2_wave.shape[0]):
    if nrs2_wave[i] >= 3.92e-06 and nrs2_wave[i] <= 4.02e-06:
        nrs2_close_wave[count] = nrs2_wave[i]
        nrs2_close_value[:,count] = nrs2_data[:,i]
        count = count+1




plt.figure(1)
plt.imshow(nrs1_data)

plt.figure(2)
plt.imshow(nrs2_data)

plt.figure(3)
plt.plot(nrs1_wave,nrs1_data)
plt.plot(nrs2_wave,nrs2_data)

plt.figure(4)
plt.plot(nrs1_wave,nrs1_allavg)
plt.plot(nrs2_wave,nrs2_allavg)

plt.figure(5)
plt.plot(nrs1_close_wave,nrs1_close_value[0])
plt.plot(nrs2_close_wave,nrs2_close_value[0])

plt.show()
