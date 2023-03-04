import pandas as pd
from os import listdir
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import sys
import zipfile

obj = ['Cigno', 'Andromeda', 'Cassiopea']
data_path = '../Data/' + obj[0] + '/'
#files_list = []
#for file in listdir(data_path):
#    if file.endswith('.txt'):
#    #if file.endswith('_URSP.txt'):
#        files_list.append(file)

# read csv inside zip file
#ziptest = zipfile.ZipFile(data_path + 'test.zip')
#print(ziptest.namelist())
#files_list = ziptest.namelist()
#ziptest.close()

files_list = listdir(data_path)

def keys(var): # oppure usare libreria python natsort
    return [int(num) for num in re.findall("([0-9]{2})", var)]
    # lambda var: [int(num) for num in re.findall("([0-9]{2})", var)]

files_list.sort(key=keys)
print(files_list)
#print(files_list[0][0:2])

def linear(arr):
    return np.power(10, arr/10)

if len(sys.argv) < 2:
    print("Inserire numero file.")
    sys.exit(0)
file_num = int(sys.argv[1]) #range(2, 7+2)
df = pd.read_csv(data_path + files_list[file_num], usecols = list(range(3,2**13+3)), sep = ';', decimal = ',', header = None, low_memory = True)
#df = pd.read_csv(ziptest.open(files_list[file_num]), usecols = list(range(3,2**13+3)), sep = ';', decimal = ',', header = None, low_memory = True)
#ziptest.close()
#print(df.iloc[:,0])
print(files_list[file_num])
arr_avg = np.average(np.array(df), axis=0)
#arr_avg = np.average(np.array(df.iloc[:,:]), axis=0)
print(np.argmax(arr_avg))
base_freq = 1.3e9
bandwidth = 19531.25
HIline = [1420405751.768, 0.021106114054160]
print(np.argmax(arr_avg)*bandwidth+base_freq)
print(np.floor((HIline[0]-base_freq)/bandwidth)+1) # 6165
z_andromeda = -0.001004
freq_obs = HIline[0]/(1+z_andromeda)
print(np.floor((freq_obs-base_freq)/bandwidth)+1) # 6238

#plt.plot(arr_avg)
#plt.show()
#plt.plot(linear(arr_avg))
#plt.show()

#signal = df.iloc[:,6166]
#plt.plot(signal)
#singal_avg = signal.rolling(7).mean()
#plt.plot(singal_avg)
#plt.show()

#filter = uniform_filter1d(signal, 7, mode='mirror')
#plt.plot(signal)
#plt.plot(filter)
#plt.show()

#plt.plot(np.arange(1, 8193, 1)[:10], np.average(df.iloc[84-10:84+11,:10], axis=0))
bch = 6100  # begin channel
ech = 6300  # end channel
xdata = np.arange(base_freq, base_freq+8192*bandwidth, bandwidth)[bch-1:ech]
#ydata = np.average(np.power(10, df.iloc[:,6100:6300]/10), axis=0)
ydata = np.array(linear(df.iloc[:,bch-1:ech]))
#print(xdata[6164-6100])
#yfilt = uniform_filter1d(np.average(ydata, axis=0), 5, mode='mirror')
plt.plot(np.average(ydata, axis=1))
plt.show()
yfilt = uniform_filter1d(ydata[124,:], 5, mode='mirror')
#plt.plot(xdata, np.average(ydata, axis=0))
plt.plot(xdata, ydata[124,:])
#plt.plot(xdata, yfilt)
plt.show()

yavg = np.average(ydata, axis=1)
mask = yavg < 1.25e-7
ymasked = ydata[mask,:]
plt.plot(xdata, ymasked.T)
plt.show()
plt.plot(xdata, np.average(ydata, axis=0))
#plt.plot(xdata, ydata[60,:])
plt.show()

#find time of max of hydrogen line, requires narrow range of values in xdata
test = ydata-yavg.reshape(-1,1)
indexmax = np.unravel_index(test.argmax(), test.shape)
print(indexmax)
#print(ydata[indexmax])
#plt.plot(xdata, ydata[indexmax[0]])
#plt.show()

# plot equally spaced
#aaa = np.linspace(0, 5, 6)
#mesh = np.meshgrid(aaa, aaa, indexing="ij")
#print(mesh[0])
#plt.imshow(mesh[0].T, origin='lower')
#plt.show()