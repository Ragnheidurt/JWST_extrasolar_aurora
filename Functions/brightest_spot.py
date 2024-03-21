from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.signal import savgol_filter
from scipy import signal

def brightest_spot(matrix, shadow, error, index):
    if shadow == 0:
        num_non_nan = 0
        suitable_wavelength_index = None
        for i in range(matrix.shape[0]):
            if not np.any(np.isnan(matrix[i,:,:])):
                suitable_wavelength_index = i
                break
            if np.count_nonzero(~np.isnan(matrix[i,:,:]))>num_non_nan:
                num_non_nan = np.count_nonzero(~np.isnan(matrix[i,:,:]))
        if suitable_wavelength_index is None:
            suitable_wavelength_index = num_non_nan
            #print(num_non_nan)
        nrs1_slice = matrix[index,:,:]
        #print(nrs1_slice)
        nrs1_maxindex = np.unravel_index(np.nanargmax(nrs1_slice), nrs1_slice.shape)
        nrs1_avg = np.zeros(matrix.shape[0])
        nrs1_error = np.zeros(matrix.shape[0])
        a = nrs1_maxindex[0]-1
        b = nrs1_maxindex[0]
        c = nrs1_maxindex[0]+1
        d = nrs1_maxindex[1]-1
        e = nrs1_maxindex[1]
        f = nrs1_maxindex[1]+1
        for i in range(nrs1_avg.shape[0]):
            nrs1_avg[i] = (matrix[i,a,d]+matrix[i,a,e]+matrix[i,a,f]+matrix[i,b,d]+matrix[i,b,e]+matrix[i,b,f]+matrix[i,c,d]+matrix[i,c,e]+matrix[i,c,f])/9
            if abs(nrs1_avg[i]) > 1e5 and i != 0:
                nrs1_avg[i] = nrs1_avg[i-1]
            elif nrs1_avg[i]<10 and i != 0:
                nrs1_avg[i] = nrs1_avg[i-1]
        for i in range(nrs1_avg.shape[0]):
            nrs1_error[i] = 1/9*math.sqrt(error[i,a,d]**2+error[i,a,e]**2+error[i,a,f]**2+error[i,b,d]**2+error[i,b,e]**2+error[i,b,f]**2+error[i,c,d]**2+error[i,c,e]**2+error[i,c,f]**2)
            if abs(nrs1_error[i]) > 1e5 and i != 0:
                nrs1_error[i] = nrs1_error[i-1]
            elif nrs1_error[i]<10 and i != 0:
                nrs1_error[i] = nrs1_error[i-1]
    elif shadow != 0:
        num_non_nan = 0
        suitable_wavelength_index = None
        for i in range(matrix.shape[0]):
            if not np.any(np.isnan(matrix[i,:,:])):
                suitable_wavelength_index = i
                break
            if np.count_nonzero(~np.isnan(matrix[i,:,:]))>num_non_nan:
                num_non_nan = np.count_nonzero(~np.isnan(matrix[i,:,:]))
        if suitable_wavelength_index is None:
            suitable_wavelength_index = num_non_nan
        nrs1_slice = matrix[index,:,:]
        b = shadow[0]
        a = b-1
        c = b+1
        e = shadow[1]
        d = e-1
        f = e+1
        nrs1_blocked_indices = [(a,d),(a,e),(a,f),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f)]
        nrs1_mask = np.ones_like(nrs1_slice, dtype=bool)
        for idx in nrs1_blocked_indices:
            nrs1_mask[idx] = False
        nrs1_maxindex = np.unravel_index(np.nanargmax(nrs1_slice*nrs1_mask), nrs1_slice.shape)
        nrs1_avg = np.zeros(matrix.shape[0])
        nrs1_error = np.zeros(matrix.shape[0])
        a = nrs1_maxindex[0]-1
        b = nrs1_maxindex[0]
        c = nrs1_maxindex[0]+1
        d = nrs1_maxindex[1]-1
        e = nrs1_maxindex[1]
        f = nrs1_maxindex[1]+1
        for i in range(nrs1_avg.shape[0]):
            nrs1_avg[i] = (matrix[i,a,d]+matrix[i,a,e]+matrix[i,a,f]+matrix[i,b,d]+matrix[i,b,e]+matrix[i,b,f]+matrix[i,c,d]+matrix[i,c,e]+matrix[i,c,f])/9
            if abs(nrs1_avg[i]) > 1e5 and i != 0:
                nrs1_avg[i] = nrs1_avg[i-1]
            elif nrs1_avg[i]<10 and i != 0:
                nrs1_avg[i] = nrs1_avg[i-1]
        for i in range(nrs1_avg.shape[0]):
            nrs1_error[i] = math.sqrt(1/9*(error[i,a,d]**2+error[i,a,e]**2+error[i,a,f]**2+error[i,b,d]**2+error[i,b,e]**2+error[i,b,f]**2+error[i,c,d]**2+error[i,c,e]**2+error[i,c,f]**2))
            if abs(nrs1_error[i]) > 1e5 and i != 0:
                nrs1_error[i] = nrs1_error[i-1]
            elif nrs1_error[i]<10 and i != 0:
                nrs1_error[i] = nrs1_error[i-1]
    return nrs1_avg, nrs1_error, nrs1_maxindex


