import re
import os
import zipfile
import pathlib

import numpy as np
import pandas as pd
import rampy as rp
import matplotlib.pyplot as plt
import astropy.constants as cn

from numpy.polynomial.polynomial import polyval

from scipy.signal import find_peaks
from scipy.interpolate import griddata, interp1d

from pathos.multiprocessing import ProcessingPool

from datetime import datetime, timedelta
from datetime import time as Time

from functools import cache

from astropy import units as u
from astropy.time import Time as ATime
from astropy.time import TimezoneInfo
from astropy.coordinates import SpectralCoord, EarthLocation, SkyCoord

metodi = {
    "1": "leastsq",
    "2": "least_squares",
    "3": "nelder",
    "4": "lbfgsb",
    "5": "basinhopping",
    "6": "ampgo",
    "7": "powell",
    "8": "cg",
    "9": "slsqp",
    "10": "differential_evolution",
}

def linear(arr):
    return np.power(10, arr/10)

def to_neper(arr):
    return arr/20*np.log(10)

def keys(var): # oppure usare libreria python natsort
    return [int(num) for num in re.findall("([0-9]{2})", var)]
    # lambda var: [int(num) for num in re.findall("([0-9]{2})", var)]

def generate_list(path) -> list:
    lst = os.listdir(path)
    lst.sort(key=keys)
    return lst

def generate_list_with_filter(path: str, extension: str, keys_func=keys) -> list[str]:
    lst = []
    for file in os.listdir(path):
        if file.endswith(extension):
            lst.append(file)
    lst.sort(key=keys_func)
    return lst

def read_csv_zip(path) -> tuple[list[str], zipfile.ZipFile]:
    # read csv inside zip file and return list
    zip_file = zipfile.ZipFile(path)
    lst = zip_file.namelist()
    return lst, zip_file

def load_VNA_df(path: str) -> pd.DataFrame:
    list_col = [0, 3]
    names = ['Frequenza', 'S21']
    return pd.read_csv(path, usecols = list_col, header = 8, nrows = 201, names = names, delim_whitespace = True, low_memory = True)

def load_VNA_parabola_df(path: str) -> pd.DataFrame:
    list_col = [0, 5]
    names = ['Frequenza', 'S21']
    df = pd.read_csv(path, usecols = list_col, header = 5, nrows = 201, names = names, low_memory = True)
    df.loc[:, 'S21'] = df.loc[:, 'S21'].apply(to_neper)
    df['Frequenza'] = df['Frequenza']*1e-9 # GHz
    return df

def get_alpha(arr: np.ndarray | pd.DataFrame) -> (np.ndarray | pd.DataFrame):
    lun_cavo_VNA = 2030 # mm
    return -2*arr/lun_cavo_VNA

def load_temps(path, names: list | np.ndarray, crio = False) -> pd.DataFrame:
    temps_amb = list(range(45, 48+1)) # da colonna 46 a colonna 49
    temps_crio = list(range(37, 40+1)) # da colonna 38 a colonna 41
    temps = temps_amb
    if crio:
        temps_crio.extend(temps_amb)
        temps = temps_crio.copy()
    return pd.read_csv(path, usecols = temps, header = None, names = names, sep = ';', low_memory = True)

def load_signal_df(path, bch: int, ech: int, times = False) -> pd.DataFrame:
    channels_list = range(bch, ech+1)
    cols = np.asarray(channels_list)+2
    if times:
        cols = np.insert(cols, 0, 0)
        channels_list = ['Tempo', *channels_list]
    df = pd.read_csv(path, usecols = cols, names = channels_list, sep = ';', decimal = ',', header = None, low_memory=True, nrows=150, dtype=np.float64)
    df.loc[:, bch:] = df.loc[:, bch:].apply(linear)
    if times:
        df.loc[:, 'Tempo'] = df.loc[:, 'Tempo']*1e-3
    return df

def plot_map(xdata, ydata, zdata, xpts = None, ypts = 1, save = None):
    import matplotlib as mp
    #mp.use("pgf")
    #mp.rcParams.update({"pgf.texsystem": "lualatex", "font.family": "serif", "text.usetex": True, "pgf.rcfonts": False})
    xdata = np.concatenate(xdata)
    ydata = np.concatenate(ydata)
    zdata = np.concatenate(zdata)
    if xpts is None:
        #x = np.linspace(np.amin(xdata), np.amax(xdata))
        x = np.r_[np.amin(xdata):np.amax(xdata)]
    else:
        #x = np.linspace(np.amin(xdata), np.amax(xdata), (np.amax(xdata)-np.amin(xdata)+1)*xpts)
        x = np.r_[np.amin(xdata):np.amax(xdata):(np.amax(xdata)-np.amin(xdata)+1)*xpts*1j]
    y = np.r_[np.amin(ydata):np.amax(ydata):(np.amax(ydata)-np.amin(ydata)+1)*ypts*1j]
    gridx, gridy = np.meshgrid(x, y)
    #gridx, gridy = np.mgrid[np.amin(xdata):np.amax(xdata), np.amin(ydata):np.amax(ydata):(np.amax(ydata)-np.amin(ydata)+1)*ypts*1j] # se si usa mgrid bisogna mettere grid.T in imshow
    grid = griddata(list(zip(xdata.flatten(), ydata.flatten())), zdata.flatten(), (gridx, gridy), method='cubic')
    plt.figure(figsize=(1920/96, 1080/96), dpi=96)
    #plt.figure(figsize=(3.25, 2.75))
    plt.imshow(grid, extent=(np.amin(xdata), np.amax(xdata), np.amin(ydata)-.5, np.amax(ydata)+.5), origin='lower', aspect='auto', vmin=0, cmap='plasma')
    # nipy_spectral - gist_heat - plasma
    plt.gca().invert_xaxis()
    plt.xlabel('Ra/degree')
    plt.ylabel('Dec/degree')
    cbar = plt.colorbar()
    cbar.set_label('Temperatura di brillanza/K')
    plt.tight_layout()
    if save is not None:
        plt.savefig(save, dpi=96, format="png")
    plt.show()
    plt.close()
    #plt.savefig('../Relazione/Figures/ContCignoEst.pgf', dpi=300)

    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface(gridx, gridy, grid, cmap=plt.cm.plasma)
    #fig.colorbar(surf)
    #plt.show()


@cache
def load_puntamenti_df(obj: str) -> pd.DataFrame:
    path_database = "../Data/" + obj + "/" + obj + ".csv"
    columns_list = ["Orario Milano"]
    df = pd.read_csv(path_database, usecols = columns_list, sep = ',', decimal = '.', low_memory = True)
    return df
    
def get_puntamento_time(obj: str, index: int) -> Time:
    df = load_puntamenti_df(obj)
    time = datetime.strptime(df.loc[index, 'Orario Milano'], '%H:%M').time()
    return time




class SignalFile():
    r"""
        Classe base che rappresenta un file dal ricevitore digitale.

        ...

        Parameters
        ---------
        file_path : str or ZipExtFile
            Percorso al file del ricevitore digitale oppure bytes da un file zip.
            
        begin_channel : int, optional
            Canale di inizio da cui leggere i dati del ricevitore.
            
        end_channel : int, optional
            Canale fino a cui leggere i dati del ricevitore.

        Attributes
        ----------
        HI_ch : int
            Canale della riga HI dell'idrogeno.
            
        base_freq: int
            Frequenza minima del ricevitore in GHz.
            
        bandwidth: float
            Banda del ricevitore in GHz.
            
        par_guad : ndarray
            Parametri di fit cubico del guadagno.
            
        par_temp : ndarray
            Parametri di fit lineare della temperatura di rumore.
            
        attenuazione_parabola : float
            Valore di attenuazione della parabola, e^-tau.
            
        beg_ch : int
            Canale di inizio.
            
        end_ch : int
            Canale di fine.
            
        path : str or ZipExtFile
            Percorso o bytes del file del ricevitore.
            
        filename: str
            Nome del file del ricevitore.
            
        date : datetime
            Data e ora presi dal nome del file.
            
        xdata : ndarray
            Array di frequenze dei canali del ricevitore, in GHz.
            
        half_time : datetime
            Orario a metà del file del ricevitore calcolato come metà tra l'inizio e la fine. Accessibile solo dopo aver chiamato `temperatura_cavo`.

        Methods
        -------
        get_filename: str
            Nome del file

        get_raw_signal: dataframe
            Carica segnale grezzo da file in un dataframe con colonne 'Tempo' e 'Canali'. 

        temperatura_cavo: float
            Trova la temperatura media del cavo della parabola durante la presa dati del file. Inoltre, setta l'attributo `half_time`.

        get_signal : dataframe
            Fornisce la temperatura del cielo del segnale.
    """
    HI_ch = 6165
    base_freq = 1.3 # GHz
    bandwidth = 19531.25e-9 # GHz
    #par_guad = np.asarray([-1.05477184e-04, 2.22774501e-04, -1.56835249e-04, 3.68038932e-05]) # da Calibrazione.py
    #par_temp = np.asarray([-239.47312245, 232.861347]) # da Calibrazione.py
    #par_guad = np.asarray([-1.075294697827863e-04, 2.271074533834585e-04, -1.598846074056052e-04, 3.751923145411951e-05])
    #par_temp = np.asarray([-245.85121069721578,  237.33899569546531])
    par_guad = np.asarray([-1.071507369683567e-04, 2.263078238482651e-04, -1.593218475158005e-04, 3.738721262891431e-05])
    par_temp = np.asarray([-308.06665563031, 281.03169023601333])
    attenuazione_parabola = 0.6154712625338543 # da Calibrazione.py

    def __init__(self, file_path, begin_channel = None, end_channel = None) -> None:
        self.beg_ch = self.HI_ch-200 if begin_channel is None else begin_channel
        self.end_ch = self.HI_ch+200 if end_channel is None else end_channel
        self.path = file_path
        self.filename = self.get_filename()
        self.date = datetime.strptime(self.filename, '%y%m%d_%H%M%S_USRP.txt')
        num_channels = self.end_ch-self.beg_ch+1
        self.xdata = (np.arange(num_channels) + self.beg_ch-0.5)*self.bandwidth + self.base_freq # centrato a metà canale
    
    def get_filename(self) -> str:
        r"""
            Ritorna il nome del file.

            Returns
            -------
            name : str
                Nome del file.
        """
        name = ''
        if isinstance(self.path, zipfile.ZipExtFile):
            name = self.path.name
        else:
            name = os.path.basename(self.path)
        return name
    
    @cache
    def get_raw_signal(self) -> pd.DataFrame:
        r"""
            Carica il segnale del file nell'intervallo di canali specificato in un dataframe.

            Returns
            -------
            df : dataframe
                Dataframe con il segnale.
        """
    #def get_raw_signal(self, times = False) -> pd.DataFrame:
        if isinstance(self.path, zipfile.ZipExtFile):
            self.path.seek(0)
        return load_signal_df(self.path, self.beg_ch, self.end_ch, True)
    
    def temperatura_cavo(self, df_temps_cavo: pd.DataFrame) -> np.float64:
        r"""
            Temperatura media del cavo durante l'osservazione del file. Inoltre, setta l'attributo `half_time`.

            Parameters
            ----------
            df_temps_cavo : dataframe
                Dataframe con le sole temperature del cavo.
            
            Returns
            -------
            temps_avg : float
                Media delle temperature del cavo in kelvin.
        """
        time = self.date.time()
        seconds = (time.hour-2)*60**2+time.minute*60+time.second
        first_index = seconds // 3
        signal = self.get_raw_signal()
        last_index = (seconds + int(signal.loc[signal.index[-1], 'Tempo'])) // 3
        temps = df_temps_cavo.iloc[first_index:last_index, :].agg('mean')#.agg(['mean', 'std', 'sem'])

        self.half_time = (last_index - first_index) * 1.5 + seconds + 2*60**2 # il numero 1.5 sarebbe 3/2 cioè 3 che deriva dalla divisione dei secondi fatta sopra e 2 dalla distanza dei due indici
        self.half_time = timedelta(seconds=self.half_time) + datetime.min

        return np.average(temps)+273.15
    
    def get_signal(self, df_temps_cavo: pd.DataFrame) -> pd.DataFrame:
        r"""
            Temperatura del cielo ricavata dal segnale.

            Parameters
            ----------
            df_temps_cavo : dataframe
                Dataframe con le sole temperature del cavo.

            Returns
            -------
            Tcielo : dataframe
                Dataframe contenente il file del ricevitore, ma convertito in temperatura del cielo in kelvin.
        """
        signal = self.get_raw_signal().iloc[:, 1:]
        # T_antenna = S/G-T_R
        Tantenna = signal/polyval(self.xdata, self.par_guad)-polyval(self.xdata, self.par_temp)
        temps = self.temperatura_cavo(df_temps_cavo)
        # T_antenna = T_cielo * e^-tau + (1-e^-tau)T_C
        Tcielo = (Tantenna-(1-self.attenuazione_parabola)*temps)/self.attenuazione_parabola
        return Tcielo

class HISignalFile(SignalFile):
    r"""
        Classe che descrive il segnale dell'idrogeno atomico per ogni file del ricevitore digitale. Sottoclasse di `SignalFile`.

        ...

        Parameters
        ----------
        file_path : str or ZipExtFile
            Percorso al file del ricevitore digitale oppure bytes da un file zip.

        df_temps_cavo : dataframe
            Dataframe con le sole temperature del cavo.

        begin_channel : int, optional
            Canale di inizio da cui leggere i dati del ricevitore.

        end_channel : int, optional
            Canale fino a cui leggere i dati del ricevitore.

        Attributes
        ----------
        HI_ch : int
            Canale della riga HI dell'idrogeno.

        base_freq: int
            Frequenza minima del ricevitore in GHz.

        bandwidth: float
            Banda del ricevitore in GHz.

        par_guad : ndarray
            Parametri di fit cubico del guadagno.

        par_temp : ndarray
            Parametri di fit lineare della temperatura di rumore.

        attenuazione_parabola : float
            Valore di attenuazione della parabola, e^-tau.

        beg_ch : int
            Canale di inizio.

        end_ch : int
            Canale di fine.

        path : str or ZipExtFile
            Percorso o bytes del file del ricevitore.

        filename: str
            Nome del file del ricevitore.

        date : datetime
            Data e ora presi dal nome del file.

        xdata : ndarray
            Array di frequenze dei canali del ricevitore, in GHz.

        half_time : datetime
            Orario a metà del file del ricevitore calcolato come metà tra l'inizio e la fine. Accessibile solo dopo aver chiamato `temperatura_cavo`.

        df_temps_cavo : dataframe
            Dataframe con le sole temperature del cavo.

        Methods
        -------
        get_filename: str
            Nome del file

        get_raw_signal: dataframe
            Carica segnale grezzo da file in un dataframe con colonne 'Tempo' e 'Canali'. 

        temperatura_cavo: float
            Trova la temperatura media del cavo della parabola durante la presa dati del file. Inoltre, setta l'attributo `half_time`.

        get_signal : dataframe
            Fornisce la temperatura del cielo del segnale.

        final_signal : ndarray
            Segnale dello spettro mediato nel tempo con baseline rimossa.
            
        get_peaks : int
            Altezza dei picchi nello spettro attorno al canale specificato, 20 canali da una parte e dall'altra.
    """
    def __init__(self, file_path, df_temps_cavo, begin_channel=None, end_channel=None) -> None:
        self.df_temps_cavo = df_temps_cavo
        super().__init__(file_path, begin_channel, end_channel)
    
    def final_signal(self) -> np.ndarray:
        r"""
            Segnale dello spettro mediato nel tempo con baseline rimossa. La funzione utilizza la doubly reweighted penalized least squares (drPLS) da `Rampy`.

            Returns
            -------
            Tbril : ndarray
                Temperatura di brillanza.
        """
        #pads = [20, 40]
        #indexes = np.r_[self.beg_ch:self.HI_ch-pads[0], self.HI_ch+pads[1]:self.end_ch]-self.beg_ch
        #fmodel = lm.models.PolynomialModel(3)
        #params = fmodel.make_params(c0 = dict(value = 7500), c1 = dict(value = -6600), c2 = dict(value = -1800), c3 = dict(value = 2000, min = 0))
        ydata = self.get_signal(self.df_temps_cavo).agg('mean', 0).to_numpy()
        #result = fmodel.fit(ydata.iloc[indexes], params, x=self.xdata[indexes], method=metodi['1'])
        #return ydata - result.eval(x=self.xdata)

        ycalc, base = rp.baseline(self.xdata, ydata, np.asarray([[self.xdata[180], self.xdata[250]]]), method='drPLS')
        return ycalc.flatten()
    
    def get_peaks(self, channel: int, pads = 20) -> np.ndarray:
        r"""
            Trova i picchi in un intorno del canale specificato. I picchi hanno altezza tra 8 e 70.

            Parameters
            ----------
            channel : int
                Canale attorno al quale cercare i picchi.
            pads : int, optional
                Numero di canali a destra e sinistra entro cui cercare i picchi.

            Returns
            -------
            peak_heights : ndarray
                Altezze dei picchi.
        """
        data = self.final_signal()
        beg = channel-self.beg_ch
        #peak = find_peaks(data.loc[channel-pads:channel+pads], [8,70])
        peak = find_peaks(data[beg-pads:beg+pads], [8,70])
        #peak_pos = peak[0] + channel - pads
        #return [peak_pos, peak[1]['peak_heights']]
        return peak[1]['peak_heights']

class SignalRow():
    r"""
        Classe base che rappresenta una osservazione a declinazione costante.

        ...

        Parameters
        ----------
        file_path : str
            Percorso a file zip con osservazioni a stessa declinazione. Esempio: `'../Data/Cigno/Est/42.zip'`.

        Attributes
        ----------
        temps_path : str
            Percorso al file zip dove sono contenute le temperature del cavo della parabola.

        path : str
            Percorso a file zip con osservazioni a stessa declinazione.

        times : list
            Lista contenente l'ora in cui sono avvenute le osservazioni.

        Methods
        -------
        get_df_temps : dataframe
            Prende il dataframe contenente le sole temperature del cavo della parabola.
        
        hour_to_deg : int
            Trasforma un angolo da ore a gradi.
    """
    temps_path = '../Data/Parabola/Parabola.zip'
    
    def __init__(self, file_path: str) -> None:
        self.path = file_path
        self.times = []
    
    def get_df_temps(self, file: str) -> pd.DataFrame:
        r"""
            Ritorna il dataframe contenente le sole temperature del cavo della parabola.

            Parameters
            ----------
            file : str
                Nome del file della parabola.

            Returns
            -------
            df_temps_cavo : dataframe
                Dataframe contenente le sole temperature del cavo della parabola.
        """
        date = datetime.strptime(file, '%y%m%d_%H%M%S_USRP.txt').date()
        names = [f'Amb{i}' for i in np.arange(4)+1]
        _, zip_temps = read_csv_zip(self.temps_path)
        df_temps_cavo = load_temps(zip_temps.open(date.strftime('TDA%Y_%m_%d.txt')), names)
        zip_temps.close()
        return df_temps_cavo

    def hour_to_deg(self, time: Time) -> int:
        r"""
            Trasforma un angolo da ore a gradi.

            Parameters
            ----------
            time : datetime.time
                Angolo in ore e minuti in formato `datetime.time`.

            Returns
            -------
            angle : float
                Angolo trasformato.
        """
        return (time.hour+time.minute/60)*15 # 15 = 360/24 cioè ore/24*360 gradi

class HISignalRow(SignalRow):
    r"""
        Classe che descrive una osservazione a declinazione costante per l'idrogeno atomico. Sottoclasse di `SignalRow`.

        ...

        Parameters
        ----------
        file_path : str
            Percorso a file zip con osservazioni a stessa declinazione. Esempio: `'../Data/Cigno/Est/42.zip'`.

        Attributes
        ----------
        temps_path : str
            Percorso al file zip dove sono contenute le temperature del cavo della parabola.

        path : str
            Percorso a file zip con osservazioni a stessa declinazione.

        times : list
            Lista contenente l'ora in cui sono avvenute le osservazioni.

        Methods
        -------
        get_df_temps : dataframe
            Prende il dataframe contenente le sole temperature del cavo della parabola.
        
        hour_to_deg : int
            Trasforma un angolo da ore a gradi.
        
        get_row_zdata : ndarray
            Ritorna i valori dei picchi massimi delle temperature di brillanza dei file osservati. Inoltre, setta `times`.
        
        get_row_coordinates : list of two ndarrays
            Ritorna l'ascensione retta in gradi e la temperatura di brillanza corrispondente.
    """
    def __init__(self, file_path: str) -> None:
        super().__init__(file_path)

    def get_row_zdata(self) -> np.ndarray:
        r"""
            Ritorna i valori dei picchi massimi delle temperature di brillanza dei file osservati. La funzione setta l'attributo `times`.

            Returns
            -------
            peaks : ndarray
                Valori dei picchi delle temperature di brillanza dei file osservati.
        """
        files_list, zip_obs = read_csv_zip(self.path)
        df_temps_cavo = self.get_df_temps(files_list[0])

        #arr = []
        #for file in files_list:
        #    file = HISignalFile(zip_obs.open(file), df_temps_cavo)
        #    arr.append(max(file.get_peak(file.HI_ch)[1]))
        #    self.times.append(file.half_time)

        #def func_map(filename) -> HISignalFile:
        #    return HISignalFile(zip_obs.open(filename), df_temps_cavo)

        #with mp.Pool(processes=4) as pool:
        #with ThreadPool(processes=4) as pool:
        #    pool_result = pool.map_async(func_map, files_list).get()
        #    result = pool.map_async(lambda x: max(x.get_peak(x.HI_ch)), pool_result).get()
        #    self.times = pool.map_async(lambda x: x.half_time, pool_result).get()
        #    pool.close()
        
        def func_vector(filename) -> tuple[np.float64, datetime]:
            file = HISignalFile(zip_obs.open(filename), df_temps_cavo)
            peak = max(file.get_peaks(file.HI_ch))
            times = file.half_time
            return peak, times

        vec = np.vectorize(func_vector)
        peaks, self.times = vec(files_list)
        zip_obs.close()
        
        return peaks
        #return res

    def get_row_coordinates(self, time: Time, asc_obj = 300) -> list[np.ndarray]:
        r"""
            Ritorna l'ascensione retta in gradi e la temperatura di brillanza corrispondente ai file osservati. La funzione calcola la differenza temporale tra l'ora del transito e l'ora del file. Poi la converte in gradi e calcola la differenza con l'ascensione retta dell'oggetto.

            Parameters
            ----------
            time : datetime.time
                Ora in cui viene fatto il puntamento.

            asc_obj : int or float, optional
                Ascensione retta dell'oggetto celeste considerato in gradi.

            Returns
            -------
            coordinates : list of two ndarrays
                Ascensione retta e temperatura di brillanza delle osservazioni.
        """
        zdata = self.get_row_zdata()
        #tim = timedelta(hours=7, minutes=30)
        #asc_obj = 300 # deg
        #asc_obj = timedelta(hours=20)
        asc_puntamento = self.hour_to_deg(time)
        vec_h2d = np.vectorize(self.hour_to_deg)
        asc_obs = vec_h2d(np.asarray(self.times))
        xdata = asc_obj - (asc_puntamento - asc_obs) 
        #diff = [i.time() for i in diff]
        return [xdata, zdata]

class ContinuumSignalFile(SignalFile):
    r"""
        Classe che descrive il segnale nel per ogni file del ricevitore digitale. Sottoclasse di `SignalFile`.

        ...

        Parameters
        ---------
        file_path : str or ZipExtFile
            Percorso al file del ricevitore digitale oppure bytes da un file zip.
            
        begin_channel : int, optional
            Canale di inizio da cui leggere i dati del ricevitore.
            
        end_channel : int, optional
            Canale fino a cui leggere i dati del ricevitore.

        Attributes
        ----------
        HI_ch : int
            Canale della riga HI dell'idrogeno.
            
        base_freq: int
            Frequenza minima del ricevitore in GHz.
            
        bandwidth: float
            Banda del ricevitore in GHz.
            
        par_guad : ndarray
            Parametri di fit cubico del guadagno.
            
        par_temp : ndarray
            Parametri di fit lineare della temperatura di rumore.
            
        attenuazione_parabola : float
            Valore di attenuazione della parabola, e^-tau.
            
        beg_ch : int
            Canale di inizio.
            
        end_ch : int
            Canale di fine.
            
        path : str or ZipExtFile
            Percorso o bytes del file del ricevitore.
            
        filename: str
            Nome del file del ricevitore.
            
        date : datetime
            Data e ora presi dal nome del file.
            
        xdata : ndarray
            Array di frequenze dei canali del ricevitore, in GHz.
            
        half_time : datetime
            Orario a metà del file del ricevitore calcolato come metà tra l'inizio e la fine. Accessibile solo dopo aver chiamato `temperatura_cavo`.

        df_temps_cavo : dataframe
            Dataframe con le sole temperature del cavo.

        Methods
        -------
        get_filename: str
            Nome del file

        get_raw_signal: dataframe
            Carica segnale grezzo da file in un dataframe con colonne 'Tempo' e 'Canali'. 

        temperatura_cavo: float
            Trova la temperatura media del cavo della parabola durante la presa dati del file. Inoltre, setta l'attributo `half_time`.

        get_signal : dataframe
            Fornisce la temperatura del cielo del segnale.

        final_signal : ndarray
            Segnale dello spettro mediato nel tempo con baseline rimossa.
    """
    def __init__(self, file_path, df_temps_cavo, begin_channel=None, end_channel=None) -> None:
        self.df_temps_cavo = df_temps_cavo
        super().__init__(file_path, begin_channel, end_channel)
    
    def _final_signal1(self) -> np.ndarray:
        r"""

            Parameters
            ----------

            Returns
            -------
        """
        #indexes = np.r_[5990:6050, 6080:6155, 6250:6350] # intervalli da includere
        indexes = np.r_[6080:self.HI_ch-20, 6250:6350] # continuo senza HI
        tdata = np.arange(150)
        ydata = self.get_signal(self.df_temps_cavo).loc[:, indexes].agg('mean', 1).to_numpy()
        ycalc, base = rp.baseline(tdata, ydata, np.asarray([[tdata[0], tdata[-1]]]), method='drPLS')
        ycalc = ycalc.flatten()
        return ycalc
        # return np.average(ycalc)

    def final_signal(self) -> np.ndarray:
        r"""
            Segnale dello spettro mediato nel tempo con baseline rimossa. La funzione utilizza la doubly reweighted penalized least squares (drPLS) da `Rampy`.

            Returns
            -------
            Tbril : ndarray
                Temperatura di brillanza.
        """
        #indexes = np.r_[5990:6050, 6080:6155, 6250:6350] # intervalli da includere
        #indexes = np.r_[6080:6155, 6165-20:6165+20, 6250:6350]
        #indexes = np.r_[self.beg_ch:6165-20, 6165+20:self.end_ch]
        #indexes = np.r_[6165+30:self.end_ch]

        #indexes = np.r_[6080:6205, 6250:6350] # continuo con HI
        #indexes = np.r_[6080:self.HI_ch-20, 6250:6350] # continuo senza HI

        #indexes = np.r_[self.beg_ch:6000, 6050:6060, 6070:self.end_ch]
        #indexes = np.r_[self.beg_ch:5975, 5982:6000, 6050:6060, 6070:self.HI_ch-10, 6235:self.end_ch]

        #indexes = np.r_[self.beg_ch:5975, 5982:6000, 6050:6060, 6070:6205, 6235:self.end_ch] # con HI senza interferenze
        #indexes = np.r_[self.beg_ch:5966, 5973:5975, 5982:6000, 6050:6060, 6070:self.HI_ch-10, 6235:self.end_ch] # senza HI senza interferenze  # 246 canali

        #indexes = np.r_[self.beg_ch:5965, 5973:5975, 5982:6000, 6050:6060, 6070:6205, 6235:6411, 6428:6465, 6477:self.end_ch] # con HI senza interferenze, più largo
        indexes = np.r_[self.beg_ch:5965, 5973:5975, 5982:6000, 6050:6060, 6070:self.HI_ch-10, 6235:6411, 6428:6465, 6477:self.end_ch] # senza HI senza interferenze, più largo # Questo # 416 canali
        # rimuovere 6080:6105 ?
        # mega picco 6000:6050

        #tdata = np.arange(150)
        xdata = self.xdata.copy()#[indexes-self.beg_ch]
        signal = self.get_signal(self.df_temps_cavo)#.loc[:, indexes]
        ydata = signal.agg('mean', 0).to_numpy()
        ycalc, base = rp.baseline(xdata, ydata, np.asarray([[xdata[0], xdata[-1]]]), method='drPLS')
        ycalc = ycalc.flatten()[indexes-self.beg_ch]
        return ycalc
        # return np.average(ycalc)

    def _final_signal3(self) -> pd.DataFrame: # metodo interpolazione baseline - non funziona
        r"""

            Parameters
            ----------

            Returns
            -------
        """
        ydata = self.get_signal(self.df_temps_cavo).agg('mean', 0)
        return ydata

class ContinuumSignalRow(SignalRow):
    r"""
        Classe che descrive una osservazione a declinazione costante per il continuo. Sottoclasse di `SignalRow`.

        ...

        Parameters
        ----------
        file_path : str
            Percorso a file zip con osservazioni a stessa declinazione. Esempio: `'../Data/Cigno/Est/42.zip'`.

        Attributes
        ----------
        temps_path : str
            Percorso al file zip dove sono contenute le temperature del cavo della parabola.

        path : str
            Percorso a file zip con osservazioni a stessa declinazione.

        times : list
            Lista contenente l'ora in cui sono avvenute le osservazioni.

        Methods
        -------
        get_df_temps : dataframe
            Prende il dataframe contenente le sole temperature del cavo della parabola.
        
        hour_to_deg : int
            Trasforma un angolo da ore a gradi.
        
        get_row_zdata : ndarray
            Ritorna i valori dei picchi massimi delle temperature di brillanza dei file osservati. Inoltre, setta `times`.
        
        get_row_coordinates : list of two ndarrays
            Ritorna l'ascensione retta in gradi e la temperatura di brillanza corrispondente.
    """
    temps_path = '../Data/Parabola/Parabola.zip'

    def __init__(self, file_path: str) -> None:
        self.path = file_path # Percorso a file zip con osservazioni a stessa declinazione
        self.times = []

    def get_row_zdata(self) -> np.ndarray:
        r"""
            Ritorna la temperatura di brillanza mediata in frequenza dei file osservati. La funzione setta l'attributo `times`.

            Returns
            -------
            peaks : ndarray
                Temperatura di brillanza mediata in frequenza.
        """
        files_list, zip_obs = read_csv_zip(self.path)
        df_temps_cavo = self.get_df_temps(files_list[0])
        bch = 5900
        ech = 6500

        def func_vector(filename) -> tuple[np.float64, datetime]:
            file = ContinuumSignalFile(zip_obs.open(filename), df_temps_cavo, bch, ech)
            avg = np.average(file.final_signal())
            times = file.half_time
            return avg, times

        ################# signal3 - metodo interpolazione baseline - non funziona
        #def get_base(filename):
        #    obs = ContinuumSignalFile(zip_obs.open(filename), df_temps_cavo)
        #    ydata = obs.final_signal().to_numpy()
        #    xdata = np.arange(obs.beg_ch, obs.end_ch+1)
        #    ycalc, base = rp.baseline(xdata, ydata, np.asarray([[xdata[180], xdata[250]]]), method='drPLS')
        #    return base.flatten()

        #base_first = get_base(files_list[0])
        #base_last = get_base(files_list[-1])
        #linfit = interp1d([0, len(files_list)], np.vstack([base_first, base_last]), axis=0)
        #indexes = np.r_[6100:6165-20, 6280:6350]#, 6250:6350]
        #def func_vector1(filename) -> tuple[np.float64, datetime]:
        #    file = ContinuumSignalFile(zip_obs.open(filename), df_temps_cavo)
        #    base = linfit(files_list.index(filename))
        #    ydata = file.final_signal().add(-base).loc[indexes]
        #    times = file.half_time
        #    return np.average(ydata), times
        #################

        vec = np.vectorize(func_vector)
        result, self.times = vec(files_list)
        zip_obs.close()
        
        return result

    def get_row_coordinates(self, time: Time, asc_obj = 300) -> list[np.ndarray]:
        r"""
            Ritorna l'ascensione retta in gradi e la temperatura di brillanza corrispondente ai file osservati. La funzione calcola la differenza temporale tra l'ora del transito e l'ora del file. Poi la converte in gradi e calcola la differenza con l'ascensione retta dell'oggetto.

            Parameters
            ----------
            time : datetime.time
                Ora in cui viene fatto il puntamento.

            asc_obj : int or float, optional
                Ascensione retta dell'oggetto celeste considerato in gradi.

            Returns
            -------
            coordinates : list of two ndarrays
                Ascensione retta e temperatura di brillanza delle osservazioni.
        """
        zdata = self.get_row_zdata()
        asc_puntamento = self.hour_to_deg(time)
        vec_h2d = np.vectorize(self.hour_to_deg)
        asc_obs = vec_h2d(np.asarray(self.times))
        xdata = asc_obj - (asc_puntamento - asc_obs) 
        return [xdata, zdata]

class Map():
    r"""
        Classe che rappresenta la mappa delle osservazioni in HI o continuo.

        ...

        Parameters
        ----------
        data_path : str
            Percorso alla cartella con i file zip delle osservazioni a declinazioni costanti. Esempio: `'../Data/Cigno/Est/'`.
        
        map_type : str, optional
            Può essere `HI` oppure `Cont` e descrive se fare la mappa in HI o nel continuo.

        Attributes
        ----------
        data_path : str
            Percorso alla cartella con i file zip delle osservazioni a declinazioni costanti.

        class_name : class type
            Classe con cui fare la mappa, HI o continuo.

        Methods
        -------
        load_puntamenti_df : dataframe
            Ritorna il dataframe degli orari delle osservazioni.

        get_puntamento_time : datetime.time
            Ritorna l'ora di un puntamento.
        
        get_raw_map : tuple of three ndarrays
            Ritorna la mappa del cielo in una ennupla di tre array contenenti l'ascensione retta, la declinazione e il valore calcolato da una sottoclasse di `SignalRow`.
        
        save_map
            Salva gli array di `get_raw_map` in un file compresso.
        
        load_map : tuple of three ndarrays
            Ennupla di tre array caricati da un file compresso.
        
        Notes
        -----
        Per ogni riga si possono anche caricare numeri diversi di file. Per come è scritta la classe, caricare la mappa da file permette arbitrary code execution (ACE).
    """
    def __init__(self, data_path: str, map_type = 'HI') -> None:
        self.path = data_path # path a cartella con files zip con osservazioni
        self.class_name = HISignalRow if map_type != 'Cont' else ContinuumSignalRow
    
    @cache
    def load_puntamenti_df(self) -> pd.DataFrame:
        r"""
            Ritorna il dataframe degli orari delle osservazioni.

            Returns
            -------
            df : dataframe
                Dataframe con gli orari delle osservazioni.
        """
        path_database = "../Data/Cigno/Cigno.csv"
        columns_list = ["Orario Milano"]
        df = pd.read_csv(path_database, usecols = columns_list, sep = ',', decimal = '.', low_memory = True)
        return df
    
    def get_puntamento_time(self, index: int) -> Time:
        r"""
            Ritorna l'ora di un puntamento.

            Parameters
            ----------
            index : int
                Indice dell'osservazione corrispondente alla sua posizione nel dataframe degli orari.

            Returns
            -------
            time : datetime.time
                Orario di transito dell'oggetto puntato.
        """
        df = self.load_puntamenti_df()
        time = datetime.strptime(df.loc[index, 'Orario Milano'], '%H:%M').time()#.replace(tzinfo=zone)
        #return timedelta(hours=time.hour, minutes=time.minute)
        return time
    
    @cache
    def get_raw_map(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
            Ritorna la mappa del cielo in una ennupla di tre array contenenti l'ascensione retta, la declinazione e il valore calcolato da una sottoclasse di `SignalRow`.

            Returns
            -------
            ra, dec, zvalue : tuple of three ndarrays
                Tre array con entrate anche non omogenee. Essi sono ascensione retta, declinazione e valore in z, temperatura o velocità.

            Notes
            -----
            L'asse 0 dell'array definisce la riga della mappa, l'asse 1 dell'array contiene i valori citati. Quest'ultimo asse potrebbe avere lunghezze diverse in base a quanti file si caricano.
        """
        files_zip = generate_list_with_filter(self.path, '.zip')

        index_ovest = 13

        def func_map(file, index):
            row = self.class_name(self.path + file)
            xdata, zdata = row.get_row_coordinates(self.get_puntamento_time(index))
            ydata = np.full(len(xdata), (index % index_ovest)+35)
            return [xdata, ydata, zdata]

        i = np.arange(len(files_zip))

        path = pathlib.Path(self.path)
        if path.name == 'Ovest':
            i += index_ovest

        with ProcessingPool(nodes = 6) as pool:
            pool.restart()
            pool_result = pool.amap(func_map, files_zip, i).get()
            pool_result = np.asarray(pool_result, dtype=object)
            pool.close()
            pool.join()
            pool.clear()
        #xd = pool_result[:, ::2, :].squeeze()
        #zd = pool_result[:, 1::2, :].squeeze()
        #xd, zd = np.transpose(pool_result, (1,0,2)) # ix2x21 -> 2xix21
        #xd, zd = np.concatenate(pool_result, axis=1)
        xd, yd, zd = np.swapaxes(pool_result, 0, 1)

        #ylst = np.tile(num_decs+35, (21, 1)).T # correggere numero di dec
        return xd, yd, zd

    def save_map(self, path: str, overwrite = False) -> None:
        r"""
            Salva la mappa in tre array in un file compresso. Può essere chiamata direttamente.

            Parameters
            ----------
            path : str
                Percorso e nome del file in cui salvare la mappa. Vedere le note.
            overwrite : bool, optional
                Sovrascrivere il file salvato se già presente.

            Notes
            -----
            Il percorso che si inserisce deve terminare in '.npz' altrimenti usando lo stesso percorso non si può caricare il file.

            La funzione controlla se sia già presente il file. Se `overwrite = False` allora non viene sovrascritto. 
        """
        if not os.path.exists(path) or overwrite:
            xdata, ydata, zdata = self.get_raw_map()
            np.savez_compressed(path, ra=xdata, dec=ydata, temp=zdata)
        return
    
    def load_map(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
            Carica la mappa da un file zip in una ennupla di tre array.

            Parameters
            ----------
            path : str
                Percorso e nome del file da cui caricare la mappa. Vedere le note.

            Returns
            -------
            ra, dec, zvalue : tuple of three ndarrays
                Tre array con entrate anche non omogenee. Essi sono ascensione retta, declinazione e zvalue.

            Notes
            -----
            Il percorso che si inserisce deve terminare in '.npz' altrimenti usando lo stesso percorso non si può caricare il file.

            Poiché la classe è scritta per accomodare righe con numeri di file diversi, questa funzione può essere usata per arbitrary code execution (ACE).
        """
        arrays = np.load(path, allow_pickle=True)
        return arrays['ra'], arrays['dec'], arrays['temp']

class AndromedaFile(SignalRow):
    r"""
        Classe base che rappresenta una osservazione di Andromeda. Sottoclasse di `SignalRow`.

        ...

        Parameters
        ----------
        file_path : str
            Percorso a file zip con osservazioni a stessa declinazione. Esempio: `'../Data/Andromeda/Est/obs1.zip'`.

        Attributes
        ----------
        temps_path : str
            Percorso al file zip dove sono contenute le temperature del cavo della parabola.

        path : str
            Percorso a file zip con osservazioni a stessa declinazione.

        times : list
            Lista contenente l'ora in cui sono avvenute le osservazioni.
        
        HI_line : float
            Frequenza della riga dell'idrogeno a riposo in GHz.
        
        bch : int
            Canale di inizio.

        ech : int
            Canale di fine.

        date : datetime
            Data di osservazione di Andromeda. L'ora riguarda il primo file. Non c'è fuso orario.

        Methods
        -------
        get_df_temps : dataframe
            Prende il dataframe contenente le sole temperature del cavo della parabola.
        
        hour_to_deg : int
            Trasforma un angolo da ore a gradi.

        channel_to_frequency : float
            Trasforma un canale alla sua frequenza corrispondente. Si calcola la frequenza a metà della banda.

        get_peaks : tuple of two ndarrays
            Ritorna il segnale e la posizione dei picchi caratteristici di Andromeda.
        
        get_measured_velocity : tuple of float and ndarray
            Ritorna la velocità sistemica misurata e un array in cui ci sono le velocità corrispondenti ai due picchi.
    """
    HI_line = 1.420405751768 # GHz
    bch = 6100
    ech = 6165+200
    def __init__(self, file_path: str) -> None:
        self.date = None
        super().__init__(file_path)

    def channel_to_frequency(self, channel: int) -> float: # GHz
        base_freq = 1.3 # GHz
        bandwidth = 19531.25e-9 # GHz
        return base_freq+(channel-.5)*bandwidth

    #def to_std(self, interval): # interval = bandwidth 
    #    return interval/(2*np.sqrt(3))

    #def to_velocity(self, freq): # GHz -> km/s
    #    light_speed = 299792.458 # km/s
    #    return (self.HI_line/freq-1)*light_speed

    def get_peaks(self) -> tuple[np.ndarray, np.ndarray]:
        r"""
            Ritorna il segnale e la posizione dei picchi caratteristici di Andromeda.

            Returns
            -------
            peaks, indexes : tuples of two ndarrays
                Segnale dei picchi la cui posizione è data da `indexes`. Questa variabile rappresenta i canali, non la posizione nell'array.
        """
        files_list, zip_obs = read_csv_zip(self.path)
        df_temps_cavo = self.get_df_temps(files_list[0])
        self.date = datetime.strptime(files_list[0], '%y%m%d_%H%M%S_USRP.txt') # No timezone

        def func_vector(filename) -> tuple[np.float64, datetime]:
            file = HISignalFile(zip_obs.open(filename), df_temps_cavo, self.bch, self.ech)
            signal = file.final_signal()
            return signal

        vec = np.vectorize(func_vector, signature='()->(n)', otypes=[np.float64])
        result = vec(files_list[5:14])
        avg_signal = np.average(result, axis = 0)
        zip_obs.close()

        indexes = np.r_[6158:6190, 6245:6285]
        peaks = find_peaks(avg_signal[indexes-self.bch], height = 5, distance = 20)
        
        return peaks, indexes

    def get_measured_velocity(self) -> tuple[np.float64, np.ndarray]:
        r"""
            Ritorna la velocità sistemica misurata e un array in cui ci sono le velocità corrispondenti ai due picchi.

            Returns
            -------
            systemic, rotational : tuple of float and ndarray
                Velocità sistemica e rotazionale. La velocità rotazionale presenta prima la velocità (con segno) maggiore e poi quella minore, rispettivamente in allontanamento ed in avvicinamento.
        """
        xdata = np.arange(self.bch, self.ech+1) # channels
        xdata_freq = self.channel_to_frequency(xdata)
        xdata_vel = SpectralCoord(xdata_freq * u.GHz, doppler_convention='optical', doppler_rest = self.HI_line * u.GHz).to(u.km/u.s).value
        peaks_indexes, indexes = self.get_peaks()
        peaks_vel = (xdata_vel[indexes-self.bch])[peaks_indexes[0]]
        systemic_velocity = np.average(peaks_vel, weights = peaks_indexes[1]['peak_heights'])
        rotational_velocity = np.abs(peaks_vel-systemic_velocity)
        return systemic_velocity, rotational_velocity

class Andromeda():
    r"""
        Classe che rappresenta le osservazioni di Andromeda.

        ...

        Parameters
        ----------
        data_path : str
            Percorso a cartella con file zip con le osservazioni. Esempio: `'../Data/Andromeda/Est/'`.

        Attributes
        ----------
        path : str
            Percorso a cartella con file zip con le osservazioni.

        Methods
        -------
        load_puntamenti_df : dataframe
            Ritorna il dataframe degli orari delle osservazioni e il loro fuso orario.

        get_puntamento_time_and_zone : tuple of datetime.time and datetime.TimezoneInfo
            Ritorna l'ora ed il fuso orario di un puntamento.
        
        get_true_radial : float
            Trasforma la velocità misurata nel sistema baricentrico.
        
        get_velocity : ndarray
            Ritorna la velocità sistemica e le velocità rotazionali mediate su tutti i file con le loro incertezze.
    """
    def __init__(self, data_path) -> None:
        self.path = data_path # path a cartella con files zip con osservazioni

    @cache
    def load_puntamenti_df(self) -> pd.DataFrame:
        r"""
            Ritorna il dataframe degli orari delle osservazioni e il loro fuso orario.

            Returns
            -------
            df : dataframe
                Dataframe con gli orari e i fusi orari (in UTC) delle osservazioni.
        """
        path_database = "../Data/Andromeda/Andromeda.csv"
        columns_list = ["UTC", "Orario Milano"]
        df = pd.read_csv(path_database, usecols = columns_list, sep = ',', decimal = '.', low_memory = True)
        return df
    
    def get_puntamento_time_and_zone(self, index: int) -> tuple[Time, TimezoneInfo]:
        r"""
            Ritorna l'ora ed il fuso orario di un puntamento.

            Parameters
            ----------
            index : int
                Indice dell'osservazione corrispondente alla sua posizione nel dataframe degli orari.

            Returns
            -------
            time, zone : tuple of datetime.time and datetime.TimezoneInfo
                Orario di transito dell'oggetto puntato e fuso orario associato in UTC.
        """
        df = self.load_puntamenti_df()
        time = datetime.strptime(df.loc[index, 'Orario Milano'], '%H:%M').time()#.replace(tzinfo=zone)
        zone = TimezoneInfo(df.loc[index, 'UTC']*u.hour)
        return time, zone

    def get_true_radial(self, meas: float, corr: float) -> float:
        r"""
            Trasforma la velocità misurata nel sistema baricentrico.

            Parameters
            ----------
            meas : float
                Velocità misurata in km/s.

            corr : float
                Correzione alla velocità in km/s.

            Returns
            -------
            bary : float
                Velocità trasformata nel sistema baricentrico in km/s.
            
        """
        return meas+corr+meas*corr/cn.c.to(u.km/u.s).value

    def get_velocity(self) -> np.ndarray:
        r"""
            Ritorna la velocità sistemica e le velocità rotazionali mediate su tutti i file con le loro incertezze.

            Returns
            -------
            velocities : ndarray
                Array bidimensionale delle velocità sistemica e di rotazione. La prima entrata sono le velocità, la seconda le loro deviazioni standard.
        """
        ParabolaCoords = EarthLocation.from_geodetic(lat=45.513128910006685, lon=9.211034210864991, height=160) # Parabola Bicocca
        AndromedaCoords = SkyCoord(ra='00h42m44.3503s', dec='41d16m08.634s', distance=765*u.kpc, frame='icrs') # Andromeda, J2000, NASA/IPAC Extragalactic Database (NED)

        files_zip = generate_list_with_filter(self.path, '.zip', lambda var: [int(num) for num in re.findall("([0-9]+)", var)])

        def func_map(file, index):
            obs = AndromedaFile(self.path + file)
            systemic_velocity, rotational_velocity = obs.get_measured_velocity()

            time, zone = self.get_puntamento_time_and_zone(index)
            date = obs.date.replace(hour = time.hour, minute = time.minute, second = 0, tzinfo=zone)
            time_obs = ATime(date, location=ParabolaCoords)

            correction = AndromedaCoords.radial_velocity_correction(obstime=time_obs).to(u.km/u.s).value
            barycentric_velocity = self.get_true_radial(systemic_velocity, correction)
            return [barycentric_velocity, rotational_velocity]
        
        i = np.arange(len(files_zip))

        with ProcessingPool(nodes = 6) as pool:
            pool.restart()
            pool_result = pool.amap(func_map, files_zip, i).get()
            pool.close()
            pool.join()
            pool.clear()
        
        res = np.asarray(pool_result, dtype=object).T
        bar_vel = np.average(res[0])
        bar_std = np.std(res[0], ddof=1)
        rot_vel = np.average(res[1], axis=0)
        rot_std = np.std(res[1], axis=0, ddof=1)

        return np.asarray([[bar_vel, *rot_vel], [bar_std, *rot_std]])

class Cassiopea():
    r"""
        Classe che descrive le osservazioni di Cassiopea.

        ...

        Parameters
        ----------
        data_path : str
            Percorso a cartella con file zip con le osservazioni. Esempio: `'../Data/Cassiopea/Est/'`.

        Attributes
        ----------
        path : str
            Percorso a cartella con file zip con le osservazioni.

        Methods
        -------
        get_coordinates : tuple of two lists
            Ritorna l'ascensione retta e il zvalue (temperatura o velocità) calcolato da una sottoclasse di `SignalRow`.
    """
    def __init__(self, data_path) -> None:
        self.path = data_path # path a cartella con files zip con osservazioni

    def get_coordinates(self) -> tuple[list, list]:
        r"""
            Ritorna l'ascensione retta e il zvalue (temperatura o velocità) calcolato da una sottoclasse di `SignalRow`. Ogni lista contiene i valori raggruppati per osservazione.

            Returns
            -------
            ra, zvalue : tuple of two lists
                Ascensione retta e zvalue ritornate come liste in cui i dati sono raggruppati per osservazione, i.e. `ra[0]` è l'ascensione retta della prima osservazione.
        """
        files_zip = generate_list_with_filter(self.path, '.zip', lambda var: [int(num) for num in re.findall("([0-9]+)", var)])
        xdata = []
        zdata = []

        index = 0
        index_ovest = 2
        path = pathlib.Path(self.path)
        if path.name == 'Ovest':
            index += index_ovest

        for file in files_zip:
            obs = HISignalRow(self.path + file)
            xd, zd = obs.get_row_coordinates(get_puntamento_time('Cassiopea', index), 350)
            xdata.append(xd)
            zdata.append(zd)
            index += 1

        return xdata, zdata