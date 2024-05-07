from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy import signal
import sys
sys.path.append('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Functions')
from average_errors import average_errors
from short_interval import short_interval
from model_data import model_data
from model_data import model_shortening
from scale_model import scale_model
from scale_model import scale_correctly
from brightest_spot import brightest_spot
from integrate_Spectra import integrate_Spectra


# Read in files from both sensors
#nrs1 = get_pkg_data_filename('MAST_2024-01-31T15_21_00.167Z\MAST_2024-01-31T15_21_00.167Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-01-31T15_21_00.167Z\MAST_2024-01-31T15_21_00.167Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs2_s3d.fits')
#nrs1 = get_pkg_data_filename('MAST_2024-02-20T10_10_20.787Z\MAST_2024-02-20T10_10_20.787Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('MAST_2024-02-20T10_10_20.787Z\MAST_2024-02-20T10_10_20.787Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs2_s3d.fits')

#
#nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_08_49.109Z\MAST_2024-03-03T22_08_49.109Z\JWST\jw01189004001_03106_00001\jw01189004001_03106_00001_nrs2_s3d.fits')
#nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_21_56.858Z\MAST_2024-03-03T22_21_56.858Z\JWST\jw01189004001_03106_00002\jw01189004001_03106_00002_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_21_56.858Z\MAST_2024-03-03T22_21_56.858Z\JWST\jw01189004001_03106_00002\jw01189004001_03106_00002_nrs2_s3d.fits')
#nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs1_s3d.fits')
#nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs2_s3d.fits')
nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_40_03.977Z\MAST_2024-03-03T22_40_03.977Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs1_s3d.fits')
nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_40_03.977Z\MAST_2024-03-03T22_40_03.977Z\JWST\jw01189004001_03106_00004\jw01189004001_03106_00004_nrs2_s3d.fits')

#nrs1_jupiter = get_pkg_data_filename('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST-Jupiter/Jupiter/Sure/MAST_2024-03-04T13_01_54.243Z/MAST_2024-03-04T13_01_54.243Z/JWST/jw01373003001_03105_00001/jw01373003001_03105_00001_nrs1_s3d.fits')
#nrs2_jupiter = get_pkg_data_filename('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST-Jupiter/Jupiter/Sure/MAST_2024-03-04T13_01_54.243Z/MAST_2024-03-04T13_01_54.243Z/JWST/jw01373003001_03105_00001/jw01373003001_03105_00001_nrs2_s3d.fits')


nrs1_jupiter_data = fits.getdata('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST-Jupiter/Jupiter/Sure/MAST_2024-03-04T13_01_54.243Z/MAST_2024-03-04T13_01_54.243Z/JWST/jw01373003001_03105_00001/jw01373003001_03105_00001_nrs1_s3d.fits', ext=1)
nrs2_jupiter_data = fits.getdata('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST-Jupiter/Jupiter/Sure/MAST_2024-03-04T13_01_54.243Z/MAST_2024-03-04T13_01_54.243Z/JWST/jw01373003001_03105_00001/jw01373003001_03105_00001_nrs1_s3d.fits',ext=1)
# Extract the data from the files
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)
nrs1_error = fits.getdata(nrs1,ext=2)

# Extract header for both data sets
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)
error_header = fits.getheader(nrs1,ext=2)

print(nrs1_header)
# Get the start and end wavelengths for both sensors
nrs1_wavestart = nrs1_header['WAVSTART']
nrs1_wavend = nrs1_header['WAVEND']
nrs2_wavestart = nrs2_header['WAVSTART']
nrs2_wavend = nrs2_header['WAVEND']

# Calculate the interval and size of wavelength data
nrs1_dist = nrs1_wavend-nrs1_wavestart
nrs1_interval = nrs1_dist/nrs1_data.shape[0]
nrs2_dist = nrs2_wavend-nrs2_wavestart
nrs2_interval = nrs2_dist/nrs2_data.shape[0]

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


nrs1_slice = nrs1_data[1500,:,:]
nrs1_maxindex = np.unravel_index(np.nanargmax(nrs1_slice), nrs1_slice.shape)
nrs1_minindex = np.unravel_index(np.nanargmin(nrs1_slice),nrs1_slice.shape)
nrs2_slice = nrs2_data[2500,:,:]
nrs2_maxindex = np.unravel_index(np.nanargmax(nrs2_slice),nrs2_slice.shape)
nrs2_minindex = np.unravel_index(np.nanargmin(nrs2_slice),nrs2_slice.shape)

nrs1_plot = nrs1_data[:,nrs1_maxindex[0]-1,nrs1_maxindex[1]+1]
nrs2_plot = nrs2_data[:,nrs2_maxindex[0]-1,nrs2_maxindex[1]+1]

for i in range(nrs1_plot.shape[0]):
    if i!= 0 and abs(nrs1_plot[i])> 700 and nrs1_plot[i]<0:
        nrs1_plot[i] = nrs1_plot[i-1]

for i in range(nrs2_plot.shape[0]):
    if i!= 0 and abs(nrs2_plot[i])>700 and nrs2_plot[i]<0:
        nrs2_plot[i] = nrs2_plot[i-1]

sr_conversionfactor = 0.1*0.1*(np.pi/(180*3600))**2
print(sr_conversionfactor)

fylki = np.array([[0, 0],
                [0, 1],
                [1, 0],
                [1, 1],
                [0, -1],
                [-1, 0],
                [-1, -1],
                [-1, 1],
                [1, -1]])

sum = 0

for i in range(9):
    pixel = nrs1_data[:,nrs1_maxindex[0]+fylki[i,0],nrs1_maxindex[1]+fylki[i,1]]
    plt.figure(i)
    plt.plot(nrs1_wave,pixel)
    pixel = pixel*sr_conversionfactor
    pixel = pixel*1e6
    median = np.nanmedian(pixel)
    lim = median*100
    #print('median: ' + str(i) + ': '+ str(median))
    for j in range(pixel.shape[0]):
        if j!= 0 and abs(pixel[j])> lim or pixel[j]<0:
            pixel[j] = pixel[j-1]

   
    value = integrate_Spectra(nrs1_wave,pixel)
    print('Value: ' + str(value))
    sum = sum + value
    #print('Sum first: ' + str(sum))
    pixel2 = nrs2_data[:,nrs2_maxindex[0]+fylki[i,0],nrs2_maxindex[1]+fylki[i,1]]
    pixel2 = pixel2*sr_conversionfactor
    pixel2 = pixel2*1e6
    median2 = np.nanmedian(pixel2)
    #print('median2 ' + str(i) + ': '+ str(median2))
    lim2 = median2*100
    for j in range(pixel2.shape[0]):
        if j!= 0 and abs(pixel2[j])>lim2 or pixel2[j]<0:
            pixel2[j] = pixel2[j-1]
    value2 = integrate_Spectra(nrs2_wave,pixel2)
    print('Value2: '+ str(value2))
    sum = sum+value2
    #print('Sum second: ' + str(sum))
    plt.figure(i)
    plt.plot(nrs1_wave,pixel)
    plt.plot(nrs2_wave,pixel2)

plt.show()

d = 4.4e17

total_power = sum*d**2*4*np.pi
print('The total power output of Kirk is: ' + str(total_power))


aurora1 = nrs1_jupiter_data[:,32,23]
aurora2 = nrs2_jupiter_data[:,32,23]
for i in range(aurora1.shape[0]):
    if i!= 0 and abs(aurora1[i])> 1000000 or aurora1[i]<0:
        aurora1[i] = aurora1[i-1]

for i in range(aurora2.shape[0]):
    if i!= 0 and abs(aurora2[i])>100000 or aurora2[i]<0:
        aurora2[i] = aurora2[i-1]


background1 = nrs1_jupiter_data[:,8,18]
background2 = nrs2_jupiter_data[:,8,18]
for i in range(background1.shape[0]):
    if i!= 0 and abs(background1[i])> 100000 or background1[i]<0:
        background1[i] = background1[i-1]

for i in range(background2.shape[0]):
    if i!= 0 and abs(background2[i])>100000 or background2[i]<0:
        background2[i] = background2[i-1]

jupiter1 = nrs1_jupiter_data[:,44,30]
jupiter2 = nrs2_jupiter_data[:,44,30]
for i in range(jupiter1.shape[0]):
    if i!= 0 and abs(jupiter1[i])> 100000 or jupiter1[i]<0:
        jupiter1[i] = jupiter1[i-1]

for i in range(jupiter2.shape[0]):
    if i!= 0 and abs(jupiter2[i])>100000 or jupiter2[i]<0:
        jupiter2[i] = jupiter2[i-1]

print(nrs1_maxindex[0])
print(nrs1_maxindex[1])


plt.figure(1)
plt.imshow(nrs1_data[1500,:,:])
plt.colorbar()
plt.scatter(nrs1_minindex[1], nrs1_minindex[0], color='red', s=50)


plt.figure(2)
plt.plot(nrs1_wave,nrs1_plot)
plt.plot(nrs2_wave,nrs2_plot)

plt.figure(3)
plt.imshow(nrs1_jupiter_data[1500,:,:])
plt.colorbar()

plt.figure(4)
plt.imshow(nrs2_jupiter_data[2500,:,:])
plt.colorbar()

plt.figure(5)
plt.plot(nrs1_wave,aurora1, color = 'red')
plt.plot(nrs2_wave,aurora2, color='red')
plt.plot(nrs1_wave,background1, color='green')
plt.plot(nrs2_wave,background2, color='green')
plt.plot(nrs1_wave,jupiter1, color='blue')
plt.plot(nrs2_wave,jupiter2, color='blue')


plt.show()

