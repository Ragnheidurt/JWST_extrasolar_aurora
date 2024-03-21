import math
import numpy as np

def average_errors(wave, flux, error, size):
    reshape = flux.reshape(-1,size)
    wave_reshape = wave.reshape(-1,size)
    average = np.mean(reshape,axis=1)
    wave_average = np.mean(wave_reshape,axis=1)
    average = average.flatten()
    wave_average = wave_average.flatten()
    count = 0
    error_average = np.zeros(average.shape[0])

    for i in range(error.shape[0]):
        if i != 0 and error[i]> 1:
            error[i] = error[i-1]
        elif error[i]>1:
            non_nan_indices = np.where(error<1000)[0]
            error[i] = error[non_nan_indices[0]]

    for i in range(average.shape[0]):
        sum = 0
        for j in range(size):
            sum = sum+error[count+j]**2
        error_average[i] = math.sqrt(sum)
        count = count + size

    return wave_average, average, error_average

    