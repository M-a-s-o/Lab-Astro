import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import sys

from scipy import stats
from scipy.odr import RealData, Model, ODR
from scipy.ndimage import uniform_filter1d
from scipy.special import expit, logit
import lmfit as lm
import statsmodels.api as sm

from os import listdir
from pprint import pprint

from utils import *

from uncertainties import unumpy as unp

import matplotlib as mp
#mp.use("pgf")
#mp.rcParams.update({"pgf.texsystem": "lualatex", "font.family": "serif", "text.usetex": True, "pgf.rcfonts": False})

#####           Attenuazione per unità di lunghezza VNA             #####
data_path = '../Data/' + 'VNA/'
file_names = dict(enumerate(generate_list_with_filter(data_path, '.s2p')))
#pprint(file_names)

def get_final_VNA_df(path_all: str, path_adapters: str) -> pd.DataFrame:
    df_all = load_VNA_df(path_all)
    df_adap = load_VNA_df(path_adapters)
    df = df_all.subtract(df_adap)
    df['Frequenza'] = df_all['Frequenza']*1e-9
    df.iloc[:,1:] = df.iloc[:,1:].apply(to_neper)
    return df

df_room = get_final_VNA_df(data_path + file_names[5], data_path + file_names[6])
df_nitr = get_final_VNA_df(data_path + file_names[7], data_path + file_names[8])

# errore VNA tramite fit e residui da fare in dB, ordine dei centesimi di dB
# calcolare l'errore quando si è in scala lineare
# TRASFORMARE RESIDUI IN LINEARE E POI CALCOLARE ERRORE PER POI RICOVENTIRLO IN ERRORE DI ALPHA
def incertezza_VNA(df):
    fmodel = lm.models.PolynomialModel(3)
    params = fmodel.make_params(c0 = dict(value = -0.034), c1 = dict(value = -0.0569), c2 = dict(value = 6.2e-3), c3 = dict(value = -3.41e-4))
    result = fmodel.fit(df['S21'], params, x=df['Frequenza'], method=metodi['3'])
    #print(result.fit_report())

    def plot():
        sm.qqplot(result.residual, fit=True, line='s')
        plt.grid()
        plt.show()
        plt.hist(result.residual)
        plt.show()

    
    def plot1():
        plt.plot(df['Frequenza'], df['S21'])
        plt.plot(df['Frequenza'], result.best_fit)
        plt.show()
    
    #print(stats.shapiro(result.residual)) # p-value = 13% per Room e 1.8% per Nitr quando calcolato in dB, ma shapiro non va bene per tanti dati? ~200
    #print(stats.normaltest(result.residual)) # p-value = 15\% Room e 0.45\% Nitr, calcolato in Neper
    return np.std(result.residual, ddof=1)#, result.residual
    #return stats.sem(result.residual)

std_room = incertezza_VNA(df_room) # ricontrollare p value shapiro e testare se std è stessa che fare fit con gaussiana
std_nitr = incertezza_VNA(df_nitr)

# se calcolato errore VNA per file all e adapt separatamente, si ha correlazione dei residui pari a 1 quindi l'incertezza sulla differenza all-adapt è nulla, pertanto si ha solo incertezza sistematica cioè 0.0068 dB (~0.000782878 Neper) come dal manuale e circa quanto viene dai fit diretti della differenza
# questo però vuol dire che incertezza sull'azoto è troppo grande e compatibile con zero
#stdtest1 = incertezza_VNA(load_VNA_df(data_path + file_names[7])*1e-9)
#stdtest2 = incertezza_VNA(load_VNA_df(data_path + file_names[8])*1e-9)
#print(stats.pearsonr(tst1, tst2)) # tst* sono i residui
#stdtot = np.sqrt(stdtest1**2+stdtest2**2-2*stdtest1*stdtest2)/np.sqrt(6)
#print(to_neper(stdtot))
#print(to_neper(np.sqrt(stdtot**2+0.0068**2)))

# index 150 per 2.5GHz
# index 42 per 1.42GHz

index = 42
pads = 3

alpha_room = np.average(get_alpha(df_room['S21'])[index-pads:index+pads])
alpha_nitr = np.average(get_alpha(df_nitr['S21'])[index-pads:index+pads])
std_room = -get_alpha(std_room)/np.sqrt(2*pads)
std_nitr = -get_alpha(std_nitr)/np.sqrt(2*pads)

#plt.figure(figsize=(6, 3))
#plt.plot(df_room['Frequenza'], get_alpha(df_room['S21']), label='Ambiente')
#plt.plot(df_nitr['Frequenza'], get_alpha(df_nitr['S21']), label='Criogenico')
#plt.xlabel("Frequenza/GHz")
#plt.ylabel("Coefficiente attenuazione/(Np/mm)")
#plt.ticklabel_format(axis="y", style="sci", scilimits=(-4,5))
#plt.grid()
#plt.legend()
#plt.tight_layout()
#plt.savefig('../Relazione/Figures/CoeffAttFreq.pgf')
#plt.show()

Tamb = 294 # K
Tnitr = 77 # K 
KCelsius = 273.15 # K

print(f"Alpha room 1.4 GHz: {alpha_room} +- {std_room} Neper/mm")
print(f"Alpha nitrogen 1.4 GHz: {alpha_nitr} +- {std_nitr} Neper/mm")

# File 2: Una volta ottenuta l’attenuazione del cavo a T ambiente e a T criogenica, alla frequenza utile per le proprie misure, si ricava il valore dell'attenuazione per unità di lunghezza e quindi si effettua un fit lineare per ottenere tale valore in funzione della temperatura.
# Fare propagazione di incertezze

def fit_lineare_temp_elio():
    xdata = np.asarray([0.6, 0.82, 2.5]) # frequenza in GHz
    ydata = np.asarray([1.6923e-5, 1.9833e-5, 4.2e-5]) # Valori di alpha in Neper/mm # 2.5/6.5*0.5+0.5  # 2-0.2/6*0.5
    fmodel = lm.models.PolynomialModel(1)
    params = fmodel.make_params(c0 = dict(value = 0), c1 = dict(value = 1))
    result = fmodel.fit(ydata, params, x=xdata, method=metodi['1'])

    xval = df_room['Frequenza'][index]
    val_freq = result.eval(x=xval)
    coeff = np.asarray([1, xval]) # derivate rispetto ai valori con incertezza, cioè c0 e c1
    var_freq = np.linalg.multi_dot([coeff, result.covar, coeff])

    def plot():
        xs = np.linspace(0.5, 3, 100)
        plt.plot(xs, result.eval(x=xs), label='Fit')
        plt.plot(xdata, ydata, 'o', label='Punti')
        plt.xlabel("Frequenza/GHz")
        plt.ylabel("Coefficiente attenuazione/(Np/mm)")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-4,5))
        plt.grid()
        plt.legend()
        plt.show()
    
    #plot()

    return val_freq, np.sqrt(var_freq)

def fit_lineare_attenuazione():
    alpha_elio, std_elio = fit_lineare_temp_elio()
    xdata = np.asarray([4.21, Tnitr, Tamb]) # temperatura in K
    ydata = np.asarray([alpha_elio, alpha_nitr, alpha_room]) # alpha in Neper/mm
    yerr = np.asarray([std_elio, std_nitr, std_room])

    fmodel = lm.models.PolynomialModel(1)
    params = fmodel.make_params(c0 = dict(value = 0), c1 = dict(value = 1))
    result = fmodel.fit(ydata, params, x=xdata, weights=1./yerr, method=metodi['1'])

    #print(result.fit_report())

    best_params = np.asarray(list(dict.values(result.best_values)))
    #print(best_params)
    #for i, j, k in zip([item[0] for item in list(dict.items(params))], best_params, np.sqrt(np.diag(result.covar))):
    #    print(f"{i}:\t{j}\t +/- {k}")

    #print(f"p-value = {stats.chi2.sf(result.chisqr,len(xdata)-len(params))}")

    def plot():
        plt.figure(figsize=(6, 3))
        plt.errorbar(xdata, ydata, yerr, marker='*', ls='none', label='Punti', color='C1')
        xs = np.linspace(1, 300, 500)
        plt.plot(xs, result.eval(x=xs), label='Retta', color='C0')
        plt.xlabel("Temperatura/K")
        plt.ylabel("Coefficiente attenuazione/(Np/mm)")
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-4,5))
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig('../Relazione/Figures/CoeffAttTemp.pgf')
        #plt.show()
    
    #plot()

    return best_params

    #for name, param in result.params.items():
    #    print(f"{name}: {param.value} {param.stderr}")
    #test = list(result.params.items())
    #print(test[0][1].value)

    #print(f"Degrees of freedom = {len(xdata)-len(params)}")
    #print(f"p-value = {1-stats.chi2.cdf(result.chisqr,len(xdata)-len(params))}")
    #print(f"Matrice di covarianza:\n{result.covar}")
    # Plot normalised residuals
    def plot1():
        plt.figure(figsize=(30,15))
        plt.axhline(y=0)
        plt.plot(xdata, result.residual, 'bo')
        plt.show()
        plt.close()

    # Normality tests for residuals
    def plot2():
        sm.qqplot(result.residual, fit=True, line='s')
        plt.show()
        plt.close()
        print(stats.shapiro(result.residual))

    #### plot ####
    def plot3():
        xdum=np.linspace(np.amin(xdata),np.amax(xdata), 1000)
        plt.figure(figsize=(30,15))
        plt.plot(xdata, ydata, '--b*', label='data', alpha=.3)
        #plt.errorbar(xdata, ydata, yerr=yerr, fmt='bo', label='data')
        #plt.plot(xdata, result.best_fit, 'r-', label='best fit')
        plt.plot(xdum, retta_alpha(xdum, *best_params), 'g--', label='best fit')
        plt.xlabel('xdata')
        plt.ylabel('ydata')
        plt.legend()
        #plt.show()
        plt.close()

param_retta_alpha = fit_lineare_attenuazione()
def alpha_di_T(temp):
    return np.polynomial.polynomial.polyval(temp, param_retta_alpha)





#####       Profilo temperatura calibrazione e temperatura al ricevitore         #####
# Il profilo di temperatura del cavo coassiale si stima attraverso i sensori di temperatura che sono a contatto con esso.
# Quattro sensori criogenici (parte verticale) in kelvin +/- 1% (min. 1 K) -- otto colonne, ma 4 indipendenti
# Due sensori (parte orizzontale) a temperatura ambiente in celsius +/- 0.5 °C -- ci sono altri due sensori
# File 4
# Temperatura ambiente
# Prima colonna file TDA è su carico a temperatura ambiente (warm load)
# seconda colonna è sul tratto orizzontale del cavo cold load in prossimità della piegatura
# terza colonna è su tratto orizzontale vicino al connettore del ricevitore
# quarta colonna è misura della temperatura dell'aria
# i sensori criogenici sono nell'ordine di altezza
# sensore crio 1 è a colonna 38 di file TDA
# sensore ambiente 1 è a colonna 46

alt_nitr = 29.6 # cm
lun_cavo = 114*10 # mm
pos_sensors = np.asarray([alt_nitr, 38, 48, 60, 88, 95+2, 95+16])*10 # mm # il primo è l'altezza dell'azoto liquido     # ultimo è lunghezza del cavo coldload = 95+19 = 114 cm
incert_sensors = np.asarray([2, 1, 1, 2, 1, np.sqrt(1+1), np.sqrt(1+1)])/np.sqrt(3)*10 # mm

#files_list, file_zip = read_csv_zip(data_path)
#pprint(dict(enumerate(files_list)))

# ricavare tempertura da file mediando, ricordando incertezza
names = ['Crio1', 'Crio2', 'Crio3', 'Crio4', 'WarmLoad', 'Amb1', 'Amb2', 'Aria']
file_zip = zipfile.ZipFile(data_path + 'Temperature.zip')
df_temps = load_temps(file_zip.open('TDA2023_04_04.txt'), names, True)
file_zip.close()
pads = 100 # rimuove le prime entrate: il cavo si deve termalizzare
indexes = [df_temps.loc[:, 'Crio1'].first_valid_index()+pads, df_temps.loc[:, 'Crio1'].last_valid_index()]
df_temps = df_temps.loc[indexes[0]:indexes[1], :].agg(['mean', 'std', 'sem'])#.reset_index(drop=True)      # sem = standard error of mean
df_temps.insert(len(df_temps.columns)-2, 'WarmLoad', df_temps.pop('WarmLoad')) # riordina le colonne
df_temps.iloc[0, 4:] = df_temps.iloc[0, 4:].add(KCelsius) # converte temperatura ambiente da °C a K
# incertezza dominata dall'errore sistematico, di misura
# https://physics.stackexchange.com/questions/23441/how-to-combine-measurement-error-with-statistic-error

def get_incert_temps(df: pd.DataFrame):
    incert_crio = lambda temp: max(temp*0.01, 1) # +/- 1% (min. 1 K)
    crio = np.sqrt(df.loc['mean', :'Crio4'].apply(incert_crio)**2+df.loc['sem', :'Crio4']**2)
    amb = np.sqrt(0.5**2+df.loc['sem', 'Amb1':]**2) # 0.5 °C
    return pd.concat([crio, amb])

df_temps.loc['sem', :] = get_incert_temps(df_temps)
#print(df_temps)


# fittare sigmoide con tutti i punti, su tutto il cavo
def fit_sigmoide_temperatura(xdata, ydata, xerr, yerr):
    # sigmoide
    def sigmoid(pars, x):
        a, k, x0, d = pars
        #return a/(1+np.exp(-k*(x-x0)))+d
        return a*expit(k*(x-x0))+d

    # wrapper per lmfit
    def sigm(x, a, k, x0, d):
        return sigmoid([a, k, x0, d], x)

    # Lmfit
    fmodel = lm.Model(sigm)
    params = fmodel.make_params(a = dict(value = 242, min = 0), k = dict(value = 1.44e-2, min = 0), x0 = dict(value = 45, min = 0), d = dict(value = 50, min = 0))
    result = fmodel.fit(ydata, params, x=xdata, weights=1./yerr, method=metodi['1'])

    #print(f"\nLmfit: {result.fit_report()}")

    best_params = np.asarray(list(dict.values(result.best_values)))

    # ODR - orthogonal distance regression
    data = RealData(xdata, ydata, sx = xerr, sy = yerr)
    fmodel = Model(sigmoid)
    odr = ODR(data, fmodel, beta0 = best_params)
    out = odr.run()
    #print("\nODR sigmoide, profilo di temperatura:")
    #out.pprint() # residual variance è il chi-quadro ridotto
    #dof = len(xdata)-len(params)
    #print(f"p-value = {stats.chi2.sf(out.res_var*dof, dof)}")

    #### plot ####
    def plot():
        xdum = np.linspace(np.amin(xdata), np.amax(xdata), 1000)
        #plt.figure(figsize = (30, 15))
        #plt.errorbar(xdata, ydata, xerr = xerr, yerr = yerr, fmt = 'b*', label = 'data')
        #plt.plot(xdum, sigmoid(out.beta, xdum), 'g--', label = 'best fit')
        #plt.xlabel('Posizione/mm')
        #plt.ylabel('Temperatura/K')
        #plt.legend()
        #plt.show()
        fig, ax1 = plt.subplots()
        fig.set_size_inches(w=6, h=3)
        ax1.errorbar(xdata, ydata, xerr = xerr, yerr = yerr, marker = '*', color='C1', ls='none', label = 'Punti')
        ax1.plot(xdum, sigmoid(out.beta, xdum), color='C0', label = 'Sigmoide')
        ax1.set_xlabel('Posizione/mm')
        ax1.set_ylabel('Temperatura/K')
        y1, y2 = ax1.get_ylim()
        #ax2 = ax1.twinx()
        #ax2.set_ylim(y1-KCelsius, y2-KCelsius)
        #ax2.set_ylabel('Temperatura/°C')
        ax1.grid()
        ax1.legend()
        fig.tight_layout()
        plt.savefig('../Relazione/Figures/ProfiloTemp.pgf')
        #plt.show()
        plt.close()
    
    #plot()

    #print(out.beta)
    return out.beta # parametri migliori dal fit

Tdata = np.insert(np.asarray(df_temps.loc['mean', :'Amb2']), 0, Tnitr)
Terr = np.insert(np.asarray(df_temps.loc['sem', :'Amb2']), 0, 1/np.sqrt(3))
param_profilo_temp = fit_sigmoide_temperatura(pos_sensors, Tdata, incert_sensors, Terr)


# File 3
# usare sigmoide per ricavare T_c(x) temperatura media del tratto

def temperatura_ricevitore(params): # propaga la temperatura del carico nell'azoto fino al ricevitore
    # La funzione sigmoide è iniettiva, pertanto invertibile. Valutando l'inversa in punti distanti 1 K, si ricavano i tratti di cavo desiderati
    def sigmoid_inverse(T, a, k, x0, d):
        #return x0-1./k*np.log(a/(T-d)-1)
        return x0+1./k*logit((T-d)/a) 

    temp_points = np.arange(Tnitr, params[0]+params[3]) # 77, 78, 79, ... K
    x_points = sigmoid_inverse(temp_points, *params) # primo valore ~ 299 mm al posto di 296 mm, ma incertezza è comunque maggiore
    # x_points[0] = alt_nitr*10 # rimpiazza primo valore con 296 mm

    Delta_x = np.diff(x_points, append = lun_cavo) # prepend = alt_nitr
    esponen = np.exp(-alpha_di_T(temp_points+.5)*Delta_x)
    addend = (temp_points+.5)*(1.-esponen)

    # Si propaga la temperatura del coldload lungo il cavo così da sapere che temperatura vede il ricevitore
    # formula presa da https://en.wikipedia.org/wiki/Recurrence_relation paragrafo "Solving first-order non-homogeneous recurrence relations with variable coefficients"
    def get_temp():
        fatt1 = esponen.prod()
        prod_a = esponen.cumprod()
        fatt2 = Tnitr + np.sum(addend/prod_a)
        return fatt1*fatt2

    # controllare se la funzione sopra fa quello che presumo: sì
    def get_temp2():
        #fatt1 = 1
        #for i in esponen:
        #    fatt1 *= i
        somma = 0
        for m in range(len(addend)):
            produc = 1
            k = 0
            while k <= m:
                produc *= esponen[k]
                k += 1
            somma += addend[m]/produc
        return somma

    # in modo ricorsivo per davvero
    def get_temp3():
        #Temp = temp_points[0]
        #for temps in range(len(temp_points)):
        #    Temp = Temp*esponen[temps]+addend[temps]
        Temp = Tnitr
        for esp, addd in zip(esponen, addend):
            Temp = Temp*esp+addd
        return Temp

    return get_temp()

temp_ricev = temperatura_ricevitore(param_profilo_temp)
print(f"Temperatura al ricevitore: {temp_ricev} K")
print(f"Temperatura warm-load: {df_temps.loc['mean', 'WarmLoad']} +- {df_temps.loc['sem', 'WarmLoad']} K")


#####               Monte Carlo per stima di incertezza su temperatura carico                 #####
## monte carlo temperature e posizioni, 100-1000 realizzazione, casuale con gaussiana intorno a valore medio, per ogni estrazione fare profilo di temperatura, poi propagare
from pathos.multiprocessing import ProcessingPool
num_samples = 1000
def incert_coldload():
    rng = np.random.default_rng()
    temps_MC = rng.normal(df_temps.loc['mean', :'Amb2'], df_temps.loc['sem', :'Amb2'], (num_samples, len(df_temps.loc[:, :'Amb2'].columns)))
    arr_temp_nitr = np.full((num_samples, 1), Tnitr)
    temps_MC = np.concatenate([arr_temp_nitr, temps_MC], axis = 1)
    pos_MC = rng.normal(pos_sensors, incert_sensors, (num_samples, len(pos_sensors)))
    #params_MC = np.asarray([fit_sigmoide_temperatura(pos, temps, incert_sensors, Terr) for pos,temps in zip(pos_MC, temps_MC)])
    #temps_ricev = np.asarray([temperatura_ricevitore(pars) for pars in params_MC])
    #print(np.average(temps_ricev))
    #print(stats.sem(temps_ricev))

    def sigmoide_async(pos, tmp):
        return fit_sigmoide_temperatura(pos, tmp, incert_sensors, Terr)

    #with mp.Pool(processes=4) as pool:
    with ProcessingPool(nodes = 3) as pool:
        #pool_result = pool.starmap_async(sigmoide_async, zip(pos_MC, temps_MC)).get()
        #pool_result = pool.map_async(temperatura_ricevitore, pool_result).get()
        pool_result = pool.amap(sigmoide_async, pos_MC, temps_MC).get()
        pool_result = pool.amap(temperatura_ricevitore, pool_result).get()

        #pool_result = pool.starmap_async(fit_sigmoide_temperatura, zip(repeat(posdata, len(temps_MC)), temps_MC, repeat(poserr, len(temps_MC)), repeat(Terr, len(temps_MC)))).get()
        #pool_result = [pool.apply_async(fit_sigmoide_temperatura, [posdata, temps, poserr, Terr]) for temps in temps_MC]
        #pool_result = [result.get() for result in pool_result]
        #pool_result = [pool.apply_async(temperatura_ricevitore, [pars]) for pars in pool_result]
        #pool_result = [result.get() for result in pool_result]
        pool.close()
    return pool_result

#pool_result = incert_coldload()
#plt.figure(figsize=(6, 3))
#plt.hist(pool_result, 'auto', density=True)
#plt.xlabel("Temperatura/K")
#plt.ylabel("Conteggi normalizzati")
##plt.grid()
#plt.tight_layout()
#plt.savefig('../Relazione/Figures/DistTempCryo.pgf')
##plt.show()
#std_temp_ricev = np.nanstd(pool_result, ddof=1)
std_temp_ricev = 0.14 # modificare per essere salvato in npz
print(f"Incertezza stimata con {num_samples} realizzazioni: {std_temp_ricev} K")




#####       Gain e Temperatura di rumore         #####
channel = 6165
begin_channel = channel-200 #5700 #6000  # begin channel
end_channel = channel+200 #6600 #6400  # end channel
num_channels = end_channel-begin_channel+1
channes_list = list(range(begin_channel, end_channel+1))
base_freq = 1.3 # GHz
bandwidth = 19531.25e-9 # GHz

def gain_e_temp_rumore():
    signal_names, signal_zip = read_csv_zip(data_path + 'Segnale.zip')
    #pprint(dict(enumerate(signal_names)))

    def load_signals(signal_name):
        #return pd.read_csv(signal_zip.open(signal_name), usecols = cols, names=channes_list, sep = ';', decimal = ',', header = None, low_memory=True, engine='c', nrows=150, dtype=np.float64)
        return load_signal_df(signal_zip.open(signal_name), begin_channel, end_channel)

    df_signal = pd.concat(map(load_signals, signal_names), ignore_index=True, copy=False)
    signal_zip.close()
    xdata = (np.arange(num_channels) + begin_channel-1)*bandwidth + base_freq
    #print(f"Intervallo frequenza: {xdata[channel-begin_channel]} - {xdata[channel+1-begin_channel]} GHz")
    ydata = df_signal.iloc[120:-10, :].reset_index(drop=True)

    # trova la separazione tra le misure con warm load e cold load
    #times_split = np.asarray([10, 258, 367, 534, 804, 1008, 1214, 1379, 1565, 1728, 1945, 2120])

    # File 3 Calibrazione e Diapositiva 23 Lab astro 3
    # S1 = G (T1 + T_R) dove S è il segnale in volt, G è il guadagno, T1 è la temperatura misurata e T_R è la temperatura di rumore
    # stesso per S2 e si ricava G e T_R:
    # G = (S2-S1)/(T2-T1)
    # T_R = S1/G - T1 = S2/G - T2
    highs = np.asarray([[10, 258], [367, 534], [804, 1008], [1214, 1379], [1565, 1728]])#, [1945, 2120]])
    lows = np.asarray([[258, 367], [652, 804], [1062, 1214], [1379, 1565], [1786, 1945]])
    # valori fatti a mano /\
    pads = 5

    # prendere una coppia caldo-freddo, calcolare guadagno e rumore per ciascuna coppia e poi mediare
    ids_highs = [np.arange(i+pads, j-pads+1) for i,j in highs]
    ids_lows = [np.arange(i+pads, j-pads+1) for i,j in lows]
    mean_high = np.asarray([np.average(ydata.iloc[ids, :], axis=0) for ids in ids_highs])
    mean_low = np.asarray([np.average(ydata.iloc[ids, :], axis=0) for ids in ids_lows])

    std_high = np.asarray([stats.sem(ydata.iloc[ids, :], axis=0, ddof=1) for ids in ids_highs]) # usare stats.sem?
    std_low = np.asarray([stats.sem(ydata.iloc[ids, :], axis=0, ddof=1) for ids in ids_lows]) # stesso /\
    un_highs = unp.uarray(mean_high, std_high)
    un_lows = unp.uarray(mean_low, std_low)

    un_temps = unp.uarray(temp_ricev, std_temp_ricev)
    un_guadagno = (un_highs-un_lows)/(unp.uarray(df_temps.loc['mean', 'WarmLoad'], df_temps.loc['sem', 'WarmLoad']) - un_temps)
    un_temp_rumore = un_lows/un_guadagno - un_temps

    un_guadagno = np.average(un_guadagno, axis=0)
    un_temp_rumore = np.average(un_temp_rumore, axis=0)
    #guadagno = unp.nominal_values(un_guadagno)
    #temp_rumore = unp.nominal_values(un_temp_rumore)

    #guadagno = (mean_high - mean_low)/(df_temps.loc['mean', 'WarmLoad'] - temp_ricev) 
    #guadagno = np.average(guadagno, axis=0)
    #temp_rumore = np.average(mean_low, axis=0)/guadagno - temp_ricev

    def plot():
        plt.figure(figsize=(6, 3))
        #plt.plot(ydata.iloc[ids_highs[0], 42], color='C0', label='Ambiente')
        for ids in ids_highs[:]:
            plt.plot(ydata.iloc[ids, 42], color='C0')
        #plt.plot(ydata.iloc[ids_lows[0], 42], color='C1', label='Criogenico')
        for ids in ids_lows[:]:
            plt.plot(ydata.iloc[ids, 42], color='C1')
        #plt.plot(ydata.iloc[np.concatenate(ids_highs).ravel(), 0], label='Amb')
        #plt.plot(ydata.iloc[np.concatenate(ids_lows).ravel(), 0], label='Cryo')
        plt.xlabel("Tempo/(arb. unit)")
        plt.ylabel("Segnale del ricevitore/(arb. unit)")
        plt.grid()
        #plt.legend()
        plt.tight_layout()
        plt.savefig('../Relazione/Figures/SegnaliRicevitore.pgf')
        #plt.show() # specificare che è solo per 1.42 GHz
    
    #plot()

    def media_lorentziana():
        # media del segnale crio e ambiente con multi-lorentziana
        hist, edges = np.histogram(ydata, 100)
        #plt.hist(ydata, 100)
        #plt.stairs(hist, edges)
        #plt.show()
        xdata = (edges[:-1]+edges[1:])/2.0

        def add_term(x, add=0.0):
            return add

        fmodel = (lm.models.SplitLorentzianModel(prefix='g1_')+lm.models.SplitLorentzianModel(prefix='g2_')+lm.Model(add_term, prefix='g3_'))
        #fmodel = lm.Model(split_lorentzian, prefix='g1_')+lm.Model(split_lorentzian, prefix='g2_')
        #params = fmodel.make_params(g1_amplitude = 120, g1_center = 3e-8, g1_sigma = 5e-9, g2_amplitude = 60, g2_center = 6e-8, g2_sigma = 5e-9)
        fmodel.set_param_hint('g1_center', value=3e-8, min=0, max=5e-8)
        fmodel.set_param_hint('g2_center', value=6e-8, min=4e-8, max=7e-8)
        fmodel.set_param_hint('g1_amplitude', value=1.4e-6, min=0, max=1)
        fmodel.set_param_hint('g2_amplitude', value=7.5e-7, min=0, max=1)
        fmodel.set_param_hint('g1_sigma', value=5e-9, min=0, max=1e-6)
        fmodel.set_param_hint('g2_sigma', value=1e-8, min=0, max=1e-6)
        fmodel.set_param_hint('g1_sigma_r', value=5e-9, min=0, max=1e-6)
        fmodel.set_param_hint('g2_sigma_r', value=1e-8, min=0, max=1e-6)
        fmodel.set_param_hint('g3_add', value=-1, min=-20, max=0)
        params = fmodel.make_params()
        result = fmodel.fit(hist, params, x=xdata, method=metodi['3'])
        #print(result.fit_report())
        #print(f"p-value = {stats.chi2.sf(result.chisqr, len(xdata)-len(params))}")
        comps = result.eval_components(result.params, x=xdata)
        plt.plot(xdata, hist)
        plt.plot(xdata, result.best_fit, label='model')
        #plt.plot(xdata, comps['g1_'], label='1')
        #plt.plot(xdata, comps['g2_'], label='2')
        plt.legend()
        #plt.show()
        plt.close()
        mean_low = result.params['g1_center'].value # Watt
        mean_high = result.params['g2_center'].value # Watt
        #print(mean_high)
        #print(mean_low)

    #return guadagno, temp_rumore
    return un_guadagno, un_temp_rumore

# Salva gain e temperatura di rumore in un file
# inserire anche le funzioni sopra
import os
guad_rumore_file_name = data_path + 'Guadagno_TempRumore'
overwrite = False
if os.path.exists(guad_rumore_file_name + '.npz') and not overwrite:
    arrays = np.load(guad_rumore_file_name + '.npz', allow_pickle=True)
    guadagno, temp_rumore = arrays['guadagno'], arrays['temp_rumore']
else:
    guadagno, temp_rumore = gain_e_temp_rumore() # uno per ogni canale
    np.savez_compressed(guad_rumore_file_name, guadagno = guadagno, temp_rumore = temp_rumore)
print(f"Guadagno HI a 1.42 GHz: {guadagno[channel-begin_channel]} K^-1")
print(f"Temperatura rumore HI 1.42 GHz: {temp_rumore[channel-begin_channel]} K")

xdata = (np.arange(num_channels) + begin_channel-1)*bandwidth + base_freq
plt.errorbar(xdata, unp.nominal_values(temp_rumore), unp.std_devs(temp_rumore))
#plt.errorbar(xdata, unp.nominal_values(guadagno), unp.std_devs(guadagno))
#plt.show()

# fittare guadagno e temperatura di rumore con cubica poi trovare Tantenna e Tcielo
def fit_guadagno_temp_rumore():
    xdata = (np.arange(num_channels) + begin_channel-1)*bandwidth + base_freq #GHz
    ydata = unp.nominal_values(guadagno)
    yerr = unp.std_devs(guadagno)

    fmodel = lm.models.PolynomialModel(3)
    params = fmodel.make_params(c0 = dict(value = -1e-4), c1 = dict(value = 2e-4), c2 = dict(value = -1.5e-4), c3 = dict(value = 3e-5, min = 0))
    result = fmodel.fit(ydata, params, x=xdata, weights=1./yerr, method=metodi['1'])
    res = [np.asarray(list(dict.values(result.best_values)))] # c0, c1, c2, c3
    #print(f"p-value = {stats.chi2.sf(result.chisqr,len(xdata)-len(params))}") # 100%

    indexes = np.r_[0:2+1, 6:254+1, 258:len(xdata)]
    ydata = unp.nominal_values(temp_rumore)[indexes]
    yerr = unp.std_devs(temp_rumore)[indexes]
    xdata = xdata[indexes].copy()
    fmodel = lm.models.PolynomialModel(1)
    params = fmodel.make_params(c0 = dict(value = 90), c1 = dict(value = 200))
    #result = fmodel.fit(temp_rumore, params, x=xdata, method=metodi['1'])
    result = fmodel.fit(ydata, params, x=xdata, weights=1./yerr, method=metodi['1'])
    #print(f"p-value = {stats.chi2.sf(result.chisqr,len(xdata)-len(params))}") # 96.9%
    res.append(np.asarray(list(dict.values(result.best_values)))) # c0, c1
    #plt.plot(xdata, ydata)
    #plt.plot(xdata, result.best_fit)
    #plt.show()

    return res

par_guad, par_temp = fit_guadagno_temp_rumore()
np.set_printoptions(15)
print(par_guad)
print(par_temp)




####            Attenuazione cavo Parabola              ####
df_cavo_parabola = load_VNA_parabola_df(data_path + 'Cavo_1.4_GHz.csv')
df_cavo_parabola.loc[:, 'S21'] = df_cavo_parabola.loc[:, 'S21'].apply(lambda x: np.exp(2*x))
attenuazione_parabola = np.average(df_cavo_parabola.iloc[index-3:index+3, 1])
incert_cavo_parabola = incertezza_VNA(df_cavo_parabola.iloc[index-4:index+10, :])
#tau_cavo_parabola = -2*np.average(df_cavo_parabola.iloc[index-3:index+3, 1]) # tau = alpha*Delta x, Delta x = 12 m
#attenuazione_parabola = np.exp(-tau_cavo_parabola) # exp(-tau) # non si calcola alpha perché tanto bisogna rimoltiplicare per 12 m
print(f"Attenuazione parabola: {attenuazione_parabola} +- {incert_cavo_parabola}")






# per il continuo, osservare in funzione del tempo, selezionare banda in frequenza pulita (più meno piatti con andamento del guadagno), intorno alla riga HI, prendere più canali possibili senza esagerare, anche su più intervalli, studiare segnale in funzione del tempo, offset cambia con il tempo, si suppone che rimanga più o meno costante, una volta calibrato il segnale si ricava il segnale osservato quando si ha anche HI, prendere intervalli di tempo di singolo file, se rumore è basso, usare anche più intervalli, ma non più di tre punti per file; una volta integrato sull'intervallo scelto pulito, studiare in funzione del tempo partendo da una direzione in cui il segnale di HI si vede poco fino al massimo per poi calare fino a scomparire; offset scelto quando segnale da riga HI debole; da fare per ogni osservazione, nell'intervallo scelto, non mettere la riga HI

# per il continuo, prendere intervalli puliti non per forza attorno HI, trovare temperatura del cielo, mediare in frequenza e osservare nel tempo, fittare con polinomiale per rimuovere offset
# considerare anche media mobile dei dati


# Prendere offset primo e ultimo file, vedere se sono uguali, altrimenti interpolare linearmente. Rimuovere l'offset ai file in mezzo usano l'interpolazione --- NON FUNZIONA, questo metodo fornisce pessimi risultati

# Mappa HI con dati in z pari a velocità radiale, cioè posizione del picco di idrogeno

####        COSE DA FARE        ####
# Capire perché quando si chiama due volte la classe Map, alla seconda non va, problemi di multiprocessing? Se si chiama get_raw_map funziona, ma plottando tra una chiamata e l'altra si blocca. Provare a mettere la funzione di plot come metodo?

# Propagare per bene le incertezze. Per ora fatto: alpha, temperature sensori, temperatura coldload al ricevitore, guadagno, temperatura di rumore. Manca: fit VNA cavo parabola