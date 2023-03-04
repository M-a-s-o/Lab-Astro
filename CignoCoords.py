from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, SkyCoord, Angle
from astropy.time import Time, TimezoneInfo
import astropy.units as u
import numpy as np
from sys import argv
from Movimenti import movimenti

off_set = 21*u.deg # 1.4 GHz
TimezoneMilano = TimezoneInfo(1*u.hour)
ParabolaCoords = EarthLocation.from_geodetic(lat=45.513128910006685, lon=9.211034210864991, height=160) # Parabola Bicocca
#CygACoords = SkyCoord(ra='19h59m28.35656837s', dec='40d44m02.0972325s') # Cygnus A, J2000, SIMBAD
np.set_printoptions(precision=1)

def interpol(dec: Angle) -> Angle:
    # CygCoords (ra,dec)
    # From (20h, 35d) through (20h20m, 40d) to (20h40m, 45d)
    ra = Angle('20h')
    res = dec-Angle('35d')+ra
    return res.to_string(unit=u.hour)

def CoordsBatch():
    #coord = np.array([['20h00m00s']*11, [str(i)+'d00m00s' for i in range(35,46)]])
    #coord = np.array([[interpol(Angle(str(i) + 'd')), str(i)+'d00m00s'] for i in range(35, 46)]).T
    #CygCoords = SkyCoord(ra=coord[0], dec=coord[1]) # test coords for scan
    
    coord = np.array([str(i)+'d00m00s' for i in range(39,41+1)])
    CygCoords = SkyCoord(ra='20h00m00s', dec=coord) # test coords for scan

    print(CygCoords)
    step = 1
    beg = 2
    times = [datetime(2023, 3, i, 14, 0, 0, tzinfo=TimezoneMilano) for i in range(beg,beg+step*len(coord), step)]
    times = Time(times, location=ParabolaCoords)
    print(times)
    AltAzFrame = AltAz(location=ParabolaCoords, obstime=times)
    CygCoords = CygCoords.transform_to(AltAzFrame)
    #test = np.vectorize(movimenti)         controllo automatico dei puntamenti
    #if test(CygCoords.az.deg, CygCoords.alt.deg).all():
    print(f"Elevazione = {CygCoords.alt}")
    print(f"Azimuth corretto = {(CygCoords.az+off_set)}")

def CoordsSingle():
    CygCoords = SkyCoord(ra='20h00m00s', dec='36d00m00s') # test coords for scan
    time = datetime(2023, 3, 5, 6, 0, 0, tzinfo=TimezoneMilano)
    time = Time(time, location=ParabolaCoords)

    AltAzFrame = AltAz(location=ParabolaCoords, obstime=time)
    print(f"Passaggio Milano: {time.to_datetime(timezone=TimezoneMilano)}") # UTC+1
    print(f"Declinazione = {CygCoords.dec}")
    CygCoords = CygCoords.transform_to(AltAzFrame)
    #print(CygCoords)
    #print(f"Elevazione = {CygCoords.alt.to_string(unit=u.deg)}")
    print(f"Elevazione = {CygCoords.alt:.1f}")
    #print(f"Azimuth = {(CygCoords.az).to_string(unit=u.hour)}")
    #print(f"Azimuth corretto = {(CygCoords.az+off_set).to_string(unit=u.deg)}")
    print(f"Azimuth corretto = {(CygCoords.az+off_set):.1f}")

if __name__ == '__main__':
    if len(argv) >= 2:
        CoordsBatch()
    else:
        CoordsSingle()