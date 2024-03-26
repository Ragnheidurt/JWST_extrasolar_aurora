from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy import signal
import sys

def model_data(filepath):
    with open(filepath,'r') as file:
        lines = file.readlines()

    wavenumber = []
    absorption_intensity1 = []
    vacuum_wavelength = []
    column_density = []
    stuff = []

    for line in lines:
        data = line.split()

        wavenumber.append(float(data[0]))
        absorption_intensity1.append(float(data[1]))
        vacuum_wavelength.append(float(data[2]))
        column_density.append(float(data[3]))
        stuff.append(data[4])

    wavenumber = np.array(wavenumber)
    absorption_intensity1 = np.array(absorption_intensity1)
    vacuum_wavelength = np.array(vacuum_wavelength)
    column_density = np.array(column_density)
    stuff = np.array(stuff)
    vacuum_wavelength = vacuum_wavelength*1e-10

    return wavenumber, absorption_intensity1, vacuum_wavelength, column_density, stuff

def model_shortening(vacuum_wavelength, low_limit, up_limit, scaled_model):
    model_wave_short = []
    model_flux_short = []
    for k in range(len(vacuum_wavelength)):
        if vacuum_wavelength[k]>= low_limit and vacuum_wavelength[k]<= up_limit:
            model_wave_short.append(vacuum_wavelength[k])
            model_flux_short.append(scaled_model[k])
    return model_wave_short,model_flux_short
