# Import necessary packages
import numpy
from iapws import IAPWS97
from pyXSteam.XSteam import XSteam

# Testing XSteam
# To be added to environment set-up: `pip install pyXSteam`


steam_table = XSteam(XSteam.UNIT_SYSTEM_MKS)
pressure_vector = numpy.arange(220, 230)
print(pressure_vector)
print(steam_table.hL_p(pressure_vector))


# Testing iapws
# To be added to environment set-up: `pip install iapws`

sat_steam = IAPWS97(P=1, x=1)  # saturated steam with known P
sat_liquid = IAPWS97(T=370, x=0)  # saturated liquid with known T
steam = IAPWS97(P=2.5, T=500)  # steam with known P and T
print(sat_steam.h, sat_liquid.h, steam.h)  # calculated enthalpies
