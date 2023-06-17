import pandas as pd
from datetime import datetime, timezone, timedelta
from astropy.coordinates import Angle, SkyCoord, EarthLocation, AltAz
from astropy.time import Time, TimezoneInfo
import astropy.units as u
import zipfile
from pprint import pprint
import numpy as np

from utils import read_csv_zip

path_dati = "../Data/Parabola/"
path_database = "../Data/Cassiopea/Puntamenti.csv"

columns_list = ["UTC", "Data", "Orario Milano", "Azimut parabola", "Elevazione", "Declinazione", "Direzione", "Azimut misurato", "Elevazione misurata", "Inizio", "Fine", "Path"]

df = pd.read_csv(path_database, usecols = columns_list, sep = ',', decimal = '.', low_memory = True)
# ordina est-ovest, poi in declinazione
df = df.sort_values(['Direzione', 'Declinazione'], na_position='last').reset_index(drop=True)

files_list, file_zip = read_csv_zip(path_dati + 'Parabola.zip')
#pprint(dict(enumerate(files_list)))

list_col = [0,1,2,3,4,5,9,10]
names = ['Giorno', 'Mese', 'Anno', 'Ora', 'Minuto', 'Secondo', 'Azimut', 'Elevazione']

def get_ora(id):
    limit = lambda x: min(x, 23)
    ora = str(limit(df_target.loc[id, 'Ora']+2)) + ":" + str(df_target.loc[id, 'Minuto']) + ":" + str(df_target.loc[id, 'Secondo'])
    return pd.to_datetime(ora, format='%H:%M:%S').time()

for index in df.index:
    zone = timezone(timedelta(hours=int(df.loc[index, 'UTC'])))
    date = datetime.strptime(df.loc[index, 'Data'] + ' ' + df.loc[index, 'Orario Milano'], '%d/%m/%y %H:%M').replace(tzinfo=zone)
    parabola_filename = date.strftime('TDA%Y_%m_%d.txt')
    #df_parabola = pd.read_csv(file_zip.open(files_list[14]), usecols = list_col, header=None, names=names, sep = ';', decimal = ',', low_memory = True)
    if parabola_filename not in files_list:
        continue
    df_parabola = pd.read_csv(file_zip.open(parabola_filename), usecols = list_col, header=None, names=names, sep = ';', decimal = ',', low_memory = True)
    df_parabola[['Azimut', 'Elevazione']] = df_parabola[['Azimut', 'Elevazione']].astype(float)
    az = df.loc[index, 'Azimut parabola']
    el = df.loc[index, 'Elevazione']
    #print(df_parabola.loc[(df_parabola['Azimut'] >= (az-0.1)) & (df_parabola['Azimut'] <= (az+0.1)) & (df_parabola['Elevazione'] >= (el-0.1)) & (df_parabola['Elevazione'] <= (el+0.1))])
    df_target = df_parabola.loc[(np.round(np.abs(df_parabola['Azimut']-az), 2) <= 0.2) & (np.round(np.abs(df_parabola['Elevazione']-el), 2) <= 0.2)].reset_index(drop=True)
    df.loc[index, ['Azimut misurato', 'Elevazione misurata']] = df_target.loc[100, ['Azimut', 'Elevazione']].values

    df.loc[index, 'Inizio'] = get_ora(0)
    df.loc[index, 'Fine'] = get_ora(df_target.index[-1])
    #print(df.loc[index, 'Fine'])
file_zip.close()
df.to_csv('../Data/Cassiopea/Cassiopea.csv', index=False)