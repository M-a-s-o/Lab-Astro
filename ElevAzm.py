from datetime import datetime
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time, TimezoneInfo
import astropy.units as u

def AzimuthElevazione(time: datetime):
    ParabolaCoords = EarthLocation.from_geodetic(lat=45.513128910006685, lon=9.211034210864991, height=160) # Parabola Bicocca
    #CygACoords = SkyCoord(ra='19h59m28.35656837s', dec='40d44m02.0972325s') # Cygnus A, J2000, SIMBAD
    AndromedaCoords = SkyCoord(ra='00h42m44.3503s', dec='41d16m08.634s') # Andromeda, J2000, NASA/IPAC Extragalactic Database (NED)
    off_set = 21*u.deg # 1.4 GHz

    #time = Time(Time.now(), location=ParabolaCoords)
    time = Time(time, location=ParabolaCoords)

    #time = Time('2022-12-08 18:10:00', location=ParabolaCoords, scale='utc', format='iso')
    #time.format = 'datetime'

    CygCoords = SkyCoord(ra='20h00m00s', dec='45d00m00s') # test coords for scan
    time = datetime(2023, 3, 20, 17, 0, 0, tzinfo=TimezoneInfo(1*u.hour))
    time = Time(time, location=ParabolaCoords)

    AltAzFrame = AltAz(location=ParabolaCoords, obstime=time)
    TimezoneParabola = TimezoneInfo(-1*u.hour)
    #time.value.day
    #print(time.value) # UTC+0
    #print(time.to_datetime(timezone=TimezoneParabola)) # UTC-1
    #print(CygCoords)
    CygCoords = CygCoords.transform_to(AltAzFrame)
    print("Cigno:")
    #print(f"Elevazione = {CygCoords.alt.to_string(unit=u.deg)}")
    print(f"Elevazione = {CygCoords.alt:.1f}")
    #print(f"Azimuth = {(CygCoords.az).to_string(unit=u.hour)}")
    #print(f"Azimuth corretto = {(CygCoords.az+off_set).to_string(unit=u.deg)}")
    print(f"Azimuth corretto = {(CygCoords.az+off_set):.1f}")

    time = datetime(2023, 2, 28, 18, 0, 0, tzinfo=TimezoneInfo(1*u.hour))
    time = Time(time, location=ParabolaCoords)
    AltAzFrame2 = AltAz(location=ParabolaCoords, obstime=time)
    AndromedaCoords = AndromedaCoords.transform_to(AltAzFrame2)
    #print(f"Elevazione = {AndromedaCoords.alt.to_string(unit=u.deg)}")
    #print(f"Azimuth corretto = {(AndromedaCoords.az+off_set).to_string(unit=u.deg)}")
    print("Andromeda:")
    print(f"Elevazione = {AndromedaCoords.alt:.1f}")
    print(f"Azimuth corretto = {(AndromedaCoords.az+off_set):.1f}")

    CassiopeaCoords = SkyCoord(ra='350d00m00s', dec='60d00m00s')
    time = datetime(2023, 2, 28, 18, 0, 0, tzinfo=TimezoneInfo(1*u.hour))
    time = Time(time, location=ParabolaCoords)
    AltAzFrame3 = AltAz(location=ParabolaCoords, obstime=time)
    CassiopeaCoords = CassiopeaCoords.transform_to(AltAzFrame3)
    #print(f"Elevazione = {CassiopeaCoords.alt.to_string(unit=u.deg)}")
    #print(f"Azimuth corretto = {(CassiopeaCoords.az+off_set).to_string(unit=u.deg)}")
    print("Cassiopea:")
    print(f"Elevazione = {CassiopeaCoords.alt:.1f}")
    print(f"Azimuth corretto = {(CassiopeaCoords.az+off_set):.1f}")

if __name__ == '__main__':
    AzimuthElevazione(Time.now())