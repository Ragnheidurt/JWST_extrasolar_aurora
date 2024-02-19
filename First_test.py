from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt
import math

testfile = get_pkg_data_filename('jw01189004001_03106_00003_nrs1_s3d.fits')
fits.info(testfile)


#Table.read.list_formats()
#event = Table.read(testfile,format='ascii.ecsv')
#print(event.columns)

Data = fits.getdata(testfile,ext=1)
header = fits.getheader(testfile,ext=1)
#print(Data.shape)
#print(header)
wavstart = header['WAVSTART']
wavend = header['WAVEND']
dist = wavend-wavstart
bil = dist/Data.shape[0]
array = np.zeros(Data.shape[0])

for i in range(Data.shape[0]):
    if i != 0:
        array[i] = array[i-1]+bil
    else:
        array[i] = wavstart

print(array[Data.shape[0]-1])

low_limit = 3.92e-06
up_limit = 4.02e-06
lengd = 0
for i in range(array.shape[0]):
    if(array[i]>=low_limit and array[i]<= up_limit):
        lengd = lengd +1
    

data_slice = Data[1500, :, :]

max_index = np.unravel_index(np.nanargmax(data_slice), data_slice.shape)

interval = 4.02e-06-3.92e-06
staerd = interval*bil
close_wave = np.zeros(lengd,dtype='float32')
close_value = np.zeros(lengd,dtype='float32')
count = 0
for i in range(array.shape[0]):
    if array[i] >= 3.92e-06 and array[i] <= 4.02e-06:
        close_wave[count] = array[i]
        close_value[count] = Data[i,max_index[0],max_index[1]]
        count = count+1

Planck = np.zeros(array.shape[0])
Temp = 600
h = 6.62607015e-34
c = 3e8
kb = 1.380649e-23

for i in range(array.shape[0]):
    Planck[i] = 5e-6*(2*h*c**2)/(array[i]**5)*1/(math.exp((h*c)/(array[i]*kb*Temp))-1)




    '''
    if math.exp((h*array[i])/(kb*Temp)) == 1:
        #print("How the fuck")
        #print(i)
        hi = 0
    else:
        print("yay")
        Planck[i] = ((2*h*array[i]**3)/(c**2))*(1/(math.exp(h*array[i]/(kb*Temp))-1))
    '''

    



#plt.figure()
#plt.imshow(Data,cmap='gray')
#plt.colorbar()
plt.figure(1)
plt.imshow(Data[1500,:,:])
plt.plot(max_index[1],max_index[0],'og',markersize=10)
plt.colorbar()
#plt.show()

plt.figure(2)
plt.plot(array[:],Data[:,max_index[0],max_index[1]])
plt.plot(array[:],Planck[:])
#plt.show()

plt.figure(3)
plt.plot(array,Data[:,20,13])

plt.figure(4)
plt.plot(close_wave,close_value)

plt.show()




