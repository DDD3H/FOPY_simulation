import csv
import pandas as pd
import numpy as np
mm0 = 2.476    #点火時下段エンジン質量[kg]
Thrustdata = pd.read_csv('20201120_Noshiro0001.csv', header=None)
thrust = np.array(Thrustdata.iloc[:][1])