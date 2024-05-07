import numpy as np
import bisect

def scale_model(model, spectra1, spectra2, model_wave, spectra_wave):
    index = np.argmax(model)
    wave = model_wave[index]
    spectra_index_low = bisect.bisect_left(spectra_wave,model_wave[index-1])
    spectra_index_high = bisect.bisect_right(spectra_wave,model_wave[index+1])
    if not np.all(np.isnan(spectra1[spectra_index_high:spectra_index_low])):
        sub_spectra = spectra1[spectra_index_high:spectra_index_low]
        spectra = spectra1
    else:
        sub_spectra = spectra2[spectra_index_high:spectra_index_low]
        spectra = spectra2
    spectra_index = np.nanargmax(sub_spectra)
    spectra_index = spectra_index+spectra_index_high
    #if spectra_index != len(spectra_wave) and abs(spectra_wave[spectra_index]-wave)>abs(spectra_wave[spectra_index+1]-wave):
     #   spectra_index = spectra_index+1
    scaling_factor = spectra[spectra_index]/model[index]
    scaled_model = model*scaling_factor
    return scaled_model


def scale_correctly(model, model_wave):
    sorted_indices = np.argsort(model_wave)
    sorted_wave = model_wave[sorted_indices]
    sorted_model = model[sorted_indices]

    wavelength_step = sorted_wave[1]-sorted_wave[0]
    total_absorption = np.trapz(sorted_model,dx=wavelength_step)
    scaling_factor = 10e16/total_absorption
    scaled_model = sorted_model*scaling_factor
    return scaled_model
