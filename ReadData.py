import pandas as pd
from os import listdir
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import argrelextrema
import sys
import zipfile
from pprint import pprint

obj = ['Cigno', 'Andromeda', 'Cassiopea']
data_path = '../Data/' + obj[1] + '/'

begin_channel = 6000  # begin channel
end_channel = 6400  # end channel
base_freq = 1.3e9
bandwidth = 19531.25
HIline = [1420405751.768, 0.021106114054160]
#print(np.floor((HIline[0]-base_freq)/bandwidth)+1) # channel 6165, array index 6164
z_andromeda = -0.001004
freq_obs = HIline[0]/(1+z_andromeda)
#print(np.floor((freq_obs-base_freq)/bandwidth)+1) # channel 6238, array index 6237

def read_csv_zip(path, lst):
    # read csv inside zip file and return list
    ziptest = zipfile.ZipFile(path + 'test.zip')
    print(ziptest.namelist())
    lst = ziptest.namelist()
    ziptest.close()
    return lst

def keys(var): # oppure usare libreria python natsort
    return [int(num) for num in re.findall("([0-9]{2})", var)]
    # lambda var: [int(num) for num in re.findall("([0-9]{2})", var)]

def generate_list(path) -> list:
    lst = listdir(path)
    lst.sort(key=keys)
    return lst

def generate_list_with_filter(path) -> list:
    lst = []
    for file in listdir(path):
        if file.endswith('.txt'):
        #if file.endswith('_URSP.txt'):
            lst.append(file)
    return lst

def linear(arr):
    return np.power(10, arr/10)

files_list = generate_list(data_path)
pprint(dict(enumerate(files_list)))

if len(sys.argv) < 2:
    print("Inserire numero file.")
    sys.exit(1)

def read_database(file_num):
    df = pd.read_csv(data_path + files_list[file_num], usecols = list(range(3,2**13+3)), sep = ';', decimal = ',', header = None, low_memory = True, dtype=np.float64)
    #df = pd.read_csv(ziptest.open(files_list[file_num]), usecols = list(range(3,2**13+3)), sep = ';', decimal = ',', header = None, low_memory = True)
    #ziptest.close()
    #print(df.iloc[:,0])
    print(f"File selezionato: {files_list[file_num]}")
    #arr_avg = np.average(np.array(df), axis=0) #arr_avg = np.average(np.array(df.iloc[:,:]), axis=0)

    xdata = np.arange(base_freq, base_freq+8192*bandwidth, bandwidth)[begin_channel-1:end_channel]
    ydata = np.array(linear(df.iloc[:,begin_channel-1:end_channel])) # linear
    return xdata, ydata

def plot_temporal_rolling_mean(arr, size = 7, channel_index = 6165):
    if channel_index is None:
        signal = np.average(arr, axis=1)
    else:
        signal = arr[:, channel_index - begin_channel]
    #singal_avg = signal.rolling(7).mean() # pandas way
    #plt.plot(singal_avg)
    filter = uniform_filter1d(signal, size, mode='mirror') # scipy way
    plt.plot(signal)
    plt.plot(filter)
    plt.show()

def plot_frequency_rolling_mean(arr, size = 5, time_index = None):
    if time_index is None:
        signal = np.average(arr, axis=0)
    else:
        signal = arr[time_index,:]
    filter = uniform_filter1d(signal, size, mode='mirror')
    plt.plot(signal)
    plt.plot(filter)
    plt.show()

def plot_without_greater_signals(xarr, yarr):
    yavg = np.average(yarr, axis=1)
    mask = yavg < 1.25e-7
    ymasked = yarr[mask,:]
    plt.plot(xarr, ymasked.T)
    plt.show()

index = 0

def plot_file(xarr, yarr):
    yarr = uniform_filter1d(yarr, 3, mode='mirror')
    global index
    plt.plot(xarr, yarr, label='Signal ' + str(index))
    index += 1

#file_num = int(sys.argv[1]) 
#xdata, ydata = read_database(file_num)
#tempavg = np.average(ydata, axis=0)

#extrema = argrelextrema(tempavg[55:85], np.greater)
#print(extrema)
#test = xdata[55:85]
#print(test[extrema[0][1]]-test[extrema[0][0]])
#print(xdata[55:75])

def plot_files(indexes):
    lst = []
    for elem in indexes[1:3]:
        lst.append(int(elem))
    ran = range(lst[0], lst[0]+1)
    if len(indexes) > 2:
        ran = range(lst[0], lst[1]+1)
    for arg in ran:
        xdata, ydata = read_database(arg)
        tempavg = np.average(ydata, axis=0)
        print(tempavg[6165-begin_channel])
        plot_file(xdata, tempavg)

if __name__ == '__main__':
    plot_files(sys.argv)
    plt.axvline(HIline[0], color='red', label='HI Line')
    plt.axvline(freq_obs, color='green')
    plt.legend()
    plt.show()