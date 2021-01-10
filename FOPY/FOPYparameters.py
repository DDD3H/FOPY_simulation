from FOPYparain import *
#シミュレーションを行うための変数を設定します
# length [m]
l        = PMT[0]       # total length
d        = PMT[1]       # outer diameter
lcg0     = PMT[2]       # center of gravity @ take-off
lcgf     = PMT[3]       # center of gravity @ engine-cut-off
lcgp     = PMT[4]       # center of gravity of fuel & N2O

# weight [kg]
m0       = PMT[5]       # weight @ take-off
mf       = PMT[6]       # weight @ engine-cut-off
#m0 = 6.0; mf = m0 - (7.197-6.790);

# moment of inetia [kgm^2]
I0       = PMT[7]       # moment of inertia @take-off
If       = PMT[8]       # moment of inertia @ engine-cut-off

# coefficient [-]
lcp      = PMT[9]      # center of pressure
Cd       = PMT[10]      # drag coefficient original 0.45 840m
Cnalpha  = PMT[11]      # normal force coefficient

# parachute
Vpara1   = PMT[12]      # falling velocity of 1st parachute [m/s]
Vpara2   = PMT[13]      # falling velocity of 2nd parachute[m/s]
Hpara    = 0          # 2nd parachute's deployment altitude  [m]
Dpara    = 1.5         # para delay [s]

# launcher
LeleDeg  = 70           # angle of elevation (vertical=90deg) [deg]
LazDeg   = 150          # azimuth (east=0deg / south=270deg) [deg]
lLnchr   = 5          # length [m]

# wind
WindModel = 3            # 風モデル選択
                         # 1:べき風/ 2:一様風 / 3:統計風
Cdv      = 6.0          # べき定数 [-](いじらない)
                         # in case of WindModel=1
WazDeg   = 0          # 風が吹く方位[deg]
                         # east=0deg / south=270deg
Vwaz     = 1            # 地上風速[m/s]
Zr       = 5            # attitude anemometer located  [m](いじらない)

HeightH  = 0           # minimum attitude of winddata[m](統計風移行高度)
                        # Zr以上の整数にすること

# simulation(いじらない)
dt       = 0.01         # simulation step [s]
                         # MUST BE 0.01s OR LESS!!!
n        = 50000        # maximum nunber of simulation steps [-]
                         # if the rocket doesn't reach the ground,
                         # change 'n' to bigger one.

# print(l + d + lc
# g0 + lcgf + m0 + WazDeg, type(LazDeg))
