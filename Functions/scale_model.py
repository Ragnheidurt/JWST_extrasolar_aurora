import numpy as np
import bisect

def scale_model(model, spectra, model_wave, spectra_wave):
    index = np.argmax(model)
    wave = model_wave[index]
    spectra_index_low = bisect.bisect_left(spectra_wave,model_wave[index-1])
    spectra_index_high = bisect.bisect_right(spectra_wave,model_wave[index+1])
    sub_spectra = spectra[spectra_index_high:spectra_index_low]
    spectra_index = np.nanargmax(sub_spectra)
    spectra_index = spectra_index+spectra_index_high
    #if spectra_index != len(spectra_wave) and abs(spectra_wave[spectra_index]-wave)>abs(spectra_wave[spectra_index+1]-wave):
     #   spectra_index = spectra_index+1
    scaling_factor = spectra[spectra_index]/model[index]
    scaled_model = model*scaling_factor
    return scaled_model


