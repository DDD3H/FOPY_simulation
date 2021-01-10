from FOPYparameters import *
from FOPYthrust import *
import math
import numpy as np

SIMULATION  = 2

# constant
g = 9.80665           # acceleration of gravity[m/s^2]
S = math.pi * d * d / 4
mp0 = m0 - mf
Lele = - LeleDeg * math.pi/180
Laz = LazDeg * math.pi / 180
Waz = WazDeg * math.pi / 180
tThrust = thrust.size * dt
Cmq = - Cnalpha / 2 * ( ( lcp - lcg0 ) / l ) ** 2
Ip0 = (lcgp - lcg0) ** 2 * (mm0)  # moment of inertia of fuel & N2O
WazH = WazDeg * math.pi / 180
# print(g,S, mp0, Lele,Laz,Waz, tThrust, Cmq, Ip0, WazH)
# initial values
the0 = Lele
psi0 = Laz
THE = math.acos(((math.cos(the0)*math.cos(psi0)+math.cos(psi0)+math.cos(the0)-1))/2)
vlambda = [(- math.sin(the0) * math.sin(psi0))/(2*math.sin(THE)), (math.sin(the0)*math.cos(psi0)+math.sin(the0))/(2*math.sin(THE)), (math.cos(the0) * math.sin(psi0) + math.sin(psi0))/(2 * math.sin(THE))]
# print(str(THE))
q = [vlambda[0] * math.sin(THE/2), vlambda[1] * math.sin(THE/2), vlambda[2] * math.sin(THE/2), math.cos(THE/2)]
Xe = [0, 0, 0]
Ve = [0, 0, 0]
omg = [0, 0, 0]

print(q)

# size
if SIMULATION == 1 or 2:
    log_t     = np.zeros((1,n))
    log_T     = np.zeros((1,n))
    log_m     = np.zeros((1,n))
    log_I     = np.zeros((1,n))
    log_lcg   = np.zeros((1,n))
    log_rho   = np.zeros((1,n))
    log_Vw    = np.zeros((3,n))
    log_Vab   = np.zeros((3,n))
    log_Va    = np.zeros((1,n))
    log_alpha = np.zeros((1,n))
    log_bet   = np.zeros((1,n))
    log_D     = np.zeros((1,n))
    log_Y     = np.zeros((1,n))
    log_N     = np.zeros((1,n))
    log_Fe    = np.zeros((3,n))
    log_Ae    = np.zeros((3,n))
    log_Ve    = np.zeros((3,n))
    log_Aeab  = np.zeros((1,n))
    log_Veab  = np.zeros((1,n))
    log_Xe    = np.zeros((3,n))
    log_Kj    = np.zeros((1,n))
    log_Ka    = np.zeros((1,n))
    log_omg   = np.zeros((3,n))
    log_q     = np.zeros((4,n))
    log_the   = np.zeros((1,n))
    log_psi   = np.zeros((1,n))


#output
Ve1  = Ve[0]
Ve2  = Ve[1]
Ve3  = Ve[2]
Xe1  = Xe[0]
Xe2  = Xe[1]
Xe3  = Xe[2]
omg2 = omg[1]
omg3 = omg[2]
q1   = q[0]
q2   = q[1]
q3   = q[2]
q4   = q[3]
