from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimezoneInfo
import astropy.units as u
from tabulate import tabulate
from numpy import set_printoptions, round

ParabolaCoords = EarthLocation.from_geodetic(lat=45.513128910006685, lon=9.211034210864991, height=160) # Parabola Bicocca
#CygACoords = SkyCoord(ra='19h59m28.35656837s', dec='40d44m02.0972325s') # Cygnus A, J2000, SIMBAD
AndromedaCoords = SkyCoord(ra='00h42m44.3503s', dec='41d16m08.634s') # Andromeda, J2000, NASA/IPAC Extragalactic Database (NED)
CassiopeaCoords = SkyCoord(ra='350d00m00s', dec='60d00m00s')
TimezoneMilano = TimezoneInfo(2*u.hour)
off_set = 21*u.deg # 1.4 GHz

def Cigno():
    #time = Time(Time.now(), location=ParabolaCoords)
    #time = Time(time, location=ParabolaCoords)

    #time = Time('2022-12-08 18:10:00', location=ParabolaCoords, scale='utc', format='iso')
    #time.format = 'datetime'

    CygCoords = SkyCoord(ra='20h00m00s', dec='40d00m00s') # test coords for scan
    time = datetime(2023, 4, 16, 11, 30, 0, tzinfo=TimezoneMilano)
    time = Time(time, location=ParabolaCoords)

    AltAzFrame = AltAz(location=ParabolaCoords, obstime=time)
    TimezoneParabola = TimezoneInfo(-1*u.hour)
    #time.value.day
    #print(time.value) # UTC+0
    #print(time.to_datetime(timezone=TimezoneParabola)) # UTC-1
    #print(CygCoords)
    CygCoords = CygCoords.transform_to(AltAzFrame)
    #print("\nCigno:")
    #print(time)
    ##print(f"Elevazione = {CygCoords.alt.to_string(unit=u.deg)}")
    #print(f"Elevazione = {CygCoords.alt:.1f}")
    #print(f"Azimuth corretto = {(CygCoords.az+off_set):.1f}")
    return ['Cigno', time.to_datetime(timezone=TimezoneMilano), round(CygCoords.az+off_set,1), round(CygCoords.alt,1)]

def Andromeda():
    global AndromedaCoords
    time = datetime(2023, 5, 25, 7, 0, 0, tzinfo=TimezoneMilano)
    time = Time(time, location=ParabolaCoords)
    AltAzFrame2 = AltAz(location=ParabolaCoords, obstime=time)
    AndromedaCoords = AndromedaCoords.transform_to(AltAzFrame2)
    #print("\nAndromeda:")
    #print(time)
    #print(f"Elevazione = {AndromedaCoords.alt:.1f}")
    #print(f"Azimuth corretto = {(AndromedaCoords.az+off_set):.1f}")
    return ['Andromeda', time.to_datetime(timezone=TimezoneMilano), round(AndromedaCoords.az+off_set,1), round(AndromedaCoords.alt,1)]

def Cassiopea():
    global CassiopeaCoords
    time = datetime(2023, 5, 25, 19, 0, 0, tzinfo=TimezoneMilano)
    time = Time(time, location=ParabolaCoords)
    AltAzFrame3 = AltAz(location=ParabolaCoords, obstime=time)
    CassiopeaCoords = CassiopeaCoords.transform_to(AltAzFrame3)
    #print("\nCassiopea:")
    #print(time)
    #print(f"Elevazione = {CassiopeaCoords.alt:.1f}")
    #print(f"Azimuth corretto = {(CassiopeaCoords.az+off_set):.1f}")
    return ['Cassiopea', time.to_datetime(timezone=TimezoneMilano), round(CassiopeaCoords.az+off_set,1), round(CassiopeaCoords.alt,1)]

if __name__ == '__main__':
    lst = [Cigno(), Andromeda(), Cassiopea()]
    tabl = tabulate(lst, headers=['Oggetto', 'Data UTC', 'Azimut parabola', 'Elevazione'], tablefmt='simple_grid')
    print(tabl)