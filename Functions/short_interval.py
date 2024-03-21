import numpy as np

def short_interval(flux, low_limit, up_limit, wave):
    start_index = np.searchsorted(wave,low_limit,side='left')
    end_index = np.searchsorted(wave,up_limit,side='right')
    wave = wave[start_index:end_index+1]
    flux = flux[start_index:end_index+1]
    return wave, flux