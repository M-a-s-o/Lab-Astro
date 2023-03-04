from sys import argv
from numpy import vectorize, linspace, meshgrid, array, hypot, cos
from matplotlib.pyplot import pcolormesh, show
from scipy.spatial import KDTree

def _old_warning_limiti(az: float, alt: float) -> None:
    xcoords = array([100, 158, 203, 223, 290, 310, 346]).reshape(-1, 1)
    ycoords = array([19.1, 30, 35, 60.8, 71, 76.2]).reshape(-1, 1)
    xdist = KDTree(xcoords).query(az)[0]
    ydist = KDTree(ycoords).query(alt)[0]
    dist = hypot(xdist*cos(ydist), ydist)
    #if ( < 2):
    #    print("Attenzione azimuth!")
    #if ( < .5):
    #    print("Attenzione elevazione!")
    if (dist < 3):
        print("Attenzione limiti parabola.")
    if (az <= 2 or az >= 358):
        print("Attenzione barriera 0-360.")

def warning_limiti(az: float, alt: float) -> None:
    xcoords = array([203, 223, 290, 310]).reshape(-1, 1)
    ycoords = array([20, 60.5, 70.5, 75.5]).reshape(-1, 1)
    xdist = KDTree(xcoords).query(az)[0]
    ydist = KDTree(ycoords).query(alt)[0]
    dist = hypot(xdist, ydist)
    #if (dist < 3):
    #    print("Attenzione limiti parabola.")
    print(xdist)
    print(ydist)
    print(dist)
    if xdist < 2 and dist <= 15:
        print("Attenzione azimuth.")
    if ydist < 1 and dist <= 15:
        print("Attenzione elevazione.")
    if (az <= 2 or az >= 358):
        print("Attenzione barriera 0-360.")

def top(az: float) -> float:
    if (az >= 203 and az <= 223):
        return 70.5
    if (az >= 290 and az <= 310):
        return 60.5
    #if (az <= 203 or (az >= 223 and az <= 290) or az >= 310):
    return 75.5

def _old_bottom(az: float) -> float:
    if (az > 158 and az < 346):
        return 19.1 # mettere 20 ?
    if (az > 100 and az <= 158):
        return 30
    return 35

def bottom(az: float) -> float:
    return 20

def movimenti(az: float, alt: float) -> bool:
    az = abs(az)
    if (az > 360):
        az = az % 360
    alt = abs(alt)
    if (alt > bottom(az) and alt < top(az)):
        warning_limiti(az, alt)
        return True
    return False

def movimenti_plot(az: float, alt: float) -> bool:
    if (alt > bottom(az) and alt < top(az)):
        return True
    return False

def plot_posizioni() -> None:
    vfunc = vectorize(movimenti_plot)
    xvec = linspace(0, 360, 361)
    yvec = linspace(0, 90, 91)
    xv, yv = meshgrid(xvec, yvec, indexing='ij')
    res = vfunc(xv, yv)
    pcolormesh(res.T)
    show()

if __name__ == "__main__":
    if len(argv) >= 3:
        lst = []
        for arg in argv[1:3]:
            lst.append(float(arg))
        print(movimenti(*lst))
    elif len(argv) == 1:
        plot_posizioni()
    else:
        print("Passare due argomenti: azimuth ed elevazione.")