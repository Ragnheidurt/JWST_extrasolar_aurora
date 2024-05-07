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

def integrate_Spectra(wave, flux):
    c = 299792458
    wave_freq = c/wave
    diff = abs(np.diff(wave_freq))
    print(diff)
    integral = np.nansum(flux[:-1]*diff)
    return integral





