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
sys.path.append('C:/Users/dansf/OneDrive/Documents/KTH/Thesis/Code/JWST_extrasolar_aurora/Functions')
from average_errors import average_errors
from short_interval import short_interval
from brightest_spot import brightest_spot

print("Current working directory:", os.getcwd())

nrs1 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs1_s3d.fits')
nrs2 = get_pkg_data_filename('Data\Private_data\MAST_2024-03-03T22_33_25.304Z\MAST_2024-03-03T22_33_25.304Z\JWST\jw01189004001_03106_00003\jw01189004001_03106_00003_nrs2_s3d.fits')

# Extract the data from the files
nrs1_data = fits.getdata(nrs1,ext=1)
nrs2_data = fits.getdata(nrs2,ext=1)
nrs1_error = fits.getdata(nrs1,ext=2)
nrs2_error = fits.getdata(nrs2,ext=2)
nrs1_wavelenght = fits.getdata(nrs1,ext=4)
nrs2_wavelenght = fits.getdata(nrs2,ext=4)

# Extract header for both data sets
nrs1_header = fits.getheader(nrs1,ext=1)
nrs2_header = fits.getheader(nrs2,ext=1)
error_header = fits.getheader(nrs1,ext=2)

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

nrs1_avg, nrs1_errors, nrs1_maxvalue = brightest_spot(nrs1_data,0,nrs1_error,1500)
nrs2_avg, nrs2_errors, nrs2_maxvalue = brightest_spot(nrs2_data,0,nrs2_error,2500)
nrs1_secavg, nrs1_secerror, nrs1_sec_maxvalue = brightest_spot(nrs1_data,nrs1_maxvalue,nrs1_error,1500)
nrs2_secavg, nrs2_secerror, nrs2_sec_maxvalue = brightest_spot(nrs2_data,nrs2_maxvalue,nrs2_error,2500)

plt.figure(1)
plt.plot(nrs1_wave,nrs1_avg)
plt.plot(nrs2_wave,nrs2_avg)

plt.show()


# Define metadata
collection_date = "2022-11-15"
data_name = "Data from JWST NIRSpec IFU of WISE 0458+6434"
project_name = "Looking for Auroral emissions from brown dwarfs"

# Define the headers and data
headers = ["Flux from brighter star", "Wavelength", "Flux from other star"]

# Combine the wavelength arrays
combined_wave = np.concatenate((nrs1_wave, nrs2_wave))

# Combine the flux and error data
combined_avg = np.concatenate((nrs1_avg, nrs2_avg))
combined_errors = np.concatenate((nrs1_errors, nrs2_errors))
combined_secavg = np.concatenate((nrs1_secavg, nrs2_secavg))
combined_secerror = np.concatenate((nrs1_secerror, nrs2_secerror))


plt.figure(2)
plt.plot(combined_wave,combined_avg)
plt.show()

rows = zip(combined_avg, combined_wave, combined_secavg)



try:
# Open a file in write mode ('w')
    with open("BD_aurora_data_ragnheidur.txt", "w") as file:
        print("File opened successfully.")
        # Write metadata to the file as comments
        file.write("# Collection Date: {}\n".format(collection_date))
        file.write("# Data Name: {}\n".format(data_name))
        file.write("# Project Name: {}\n".format(project_name))
        file.write("\n")  # Add a blank line
        
        # Write the headers to the file
        file.write(",".join(headers) + "\n")
        
        # Write the data to the file
        for row in rows:
            file.write(",".join(map(str, row)) + "\n")

    print("Data has been written to output.txt")
except Exception as e:
    print("An error occurred:", e)

