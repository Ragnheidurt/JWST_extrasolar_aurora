from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt

#nrs1 = get_pkg_data_filename('MAST_2024-02-14T09_20_36.412Z\MAST_2024-02-14T09_20_36.412Z\JWST\jw01189011001_05101_00003\jw01189-o011_s00015_nirspec_f290lp-g395h-s200b1_cal.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-14T09_20_36.412Z\MAST_2024-02-14T09_20_36.412Z\JWST\jw01189011001_05101_00003\jw01189-o011_s00012_nirspec_clear-prism-s200a2_cal.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-14T09_20_36.412Z\MAST_2024-02-14T09_20_36.412Z\JWST\jw01189011001_05101_00003\jw01189-o011_s00012_nirspec_f290lp-g395h-s200a2_cal.fits')
nrs1 = get_pkg_data_filename('MAST_2024-02-14T09_20_36.412Z\MAST_2024-02-14T09_20_36.412Z\JWST\jw01189011001_05101_00002\jw01189011001_05101_00002_nrs1_s2d.fits')

nrs1_data = fits.getdata(nrs1,ext=1)

nrs1_header = fits.getheader(nrs1,ext=1)

# The Wavestart and Wavend come from the header, but the gap is given for the slit S200A1 and G395H F290LP
nrs1_wavestart = nrs1_header['WAVSTART']
nrs1_wavend = nrs1_header['WAVEND']


# Calculate the interval and size of wavelength data
nrs1_dist = nrs1_wavend-nrs1_wavestart
nrs1_interval = nrs1_dist/nrs1_data.shape[0]


print(nrs1_data.shape)
# Create array with wavelength
nrs1_wave = np.zeros(nrs1_data.shape[0])

for i in range(nrs1_data.shape[0]):
    if i != 0: 
        nrs1_wave[i] = nrs1_wave[i-1] + nrs1_interval
    else:
        nrs1_wave[i] = nrs1_wavestart




# Average all lines
nrs1_allavg = np.nanmean(nrs1_data,axis=(1))

# Looking at a close interval

low_limit = 3.92e-06
up_limit = 4.02e-06
nrs1_lengd = 0
for i in range(nrs1_wave.shape[0]):
    if(nrs1_wave[i]>=low_limit and nrs1_wave[i]<= up_limit):
        nrs1_lengd = nrs1_lengd +1




nrs1_close_wave = np.zeros(nrs1_lengd,dtype='float32')
nrs1_close_value = np.zeros((nrs1_data.shape[0],nrs1_lengd))
count = 0
for i in range(nrs1_wave.shape[0]):
    if nrs1_wave[i] >= 3.92e-06 and nrs1_wave[i] <= 4.02e-06:
        nrs1_close_wave[count] = nrs1_wave[i]
        nrs1_close_value[:,count] = nrs1_data[:,i]
        count = count+1




plt.figure(1)
plt.imshow(nrs1_data)


plt.figure(2)
plt.plot(nrs1_wave,nrs1_data)

plt.figure(4)
plt.plot(nrs1_wave,nrs1_allavg)

plt.figure(5)
plt.plot(nrs1_close_wave,nrs1_close_value[0])

plt.show()
