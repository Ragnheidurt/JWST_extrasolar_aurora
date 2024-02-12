from astropy.io import fits
from astropy.table import Table
from astropy.utils.data import get_pkg_data_filename
import numpy as np
import matplotlib.pyplot as plt

testfile = get_pkg_data_filename('MAST_2024-02-07T09_10_43.556Z/MAST_2024-02-07T09_10_43.556Z/JWST/jw01189011001_05101_00001/jw01189011001_05101_00001_nrs1_s2d.fits')
fits.info(testfile)

'''
Table.read.list_formats()
event = Table.read(testfile,hdu=1)
print(event.columns)
'''
Data = fits.getdata(testfile,ext=1)
header = fits.getheader(testfile,ext=1)
#print(Data)
#print(header)


plt.imshow(Data)
plt.show()
