from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimezoneInfo
import astropy.units as u

def AzimuthElevazione(time: datetime):
    ParabolaCoords = EarthLocation.from_geodetic(lat=45.513128910006685, lon=9.211034210864991, height=160) # Parabola Bicocca
    #CygACoords = SkyCoord(ra='19h59m28.35656837s', dec='40d44m02.0972325s') # Cygnus A, J2000, SIMBAD
    CygACoords = SkyCoord(ra='20h00m00s', dec='38d00m00s') # test coords for scan
    AndromedaCoords = SkyCoord(ra='00h42m44.3503s', dec='41d16m08.634s') # Andromeda, J2000, NASA/IPAC Extragalactic Database (NED)
    off_set = 21*u.deg # 1.4 GHz

    #time = Time(Time.now(), location=ParabolaCoords)
    time = Time(time, location=ParabolaCoords)

    #time = Time('2022-12-08 18:10:00', location=ParabolaCoords, scale='utc', format='iso')
    #time.format = 'datetime'

    time = datetime(2023, 1, 22, 16, 30, 0, tzinfo=TimezoneInfo(1*u.hour))
    time = Time(time, location=ParabolaCoords)

    AltAzFrame = AltAz(location=ParabolaCoords, obstime=time)
    TimezoneParabola = TimezoneInfo(-1*u.hour)
    time.value.day
    print(time.value) # UTC+0
    print(time.to_datetime(timezone=TimezoneParabola)) # UTC-1
    print(CygACoords)
    CygACoords = CygACoords.transform_to(AltAzFrame)
    #print(f"Elevazione = {CygACoords.alt.to_string(unit=u.deg)}")
    print(f"Elevazione = {CygACoords.alt}")
    #print(f"Azimuth = {(CygACoords.az).to_string(unit=u.hour)}")
    #print(f"Azimuth corretto = {(CygACoords.az+off_set).to_string(unit=u.deg)}")
    print(f"Azimuth corretto = {(CygACoords.az+off_set)}")

    time = datetime(2023, 1, 30, 12, 0, 0, tzinfo=TimezoneInfo(1*u.hour))
    time = Time(time, location=ParabolaCoords)
    AltAzFrame2 = AltAz(location=ParabolaCoords, obstime=time)
    AndromedaCoords = AndromedaCoords.transform_to(AltAzFrame2)
    #print(f"Elevazione = {AndromedaCoords.alt.to_string(unit=u.deg)}")
    #print(f"Azimuth corretto = {(AndromedaCoords.az+off_set).to_string(unit=u.deg)}")
    print("Andromeda:")
    print(f"Elevazione = {AndromedaCoords.alt}")
    print(f"Azimuth corretto = {(AndromedaCoords.az+off_set)}")

if __name__ == '__main__':
    AzimuthElevazione(Time.now())