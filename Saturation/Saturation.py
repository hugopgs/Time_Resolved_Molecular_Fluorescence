
import numpy as np
import matplotlib.pyplot as plt

file="Saturation/Saturation.csv"
import pandas as pd

df = pd.read_csv(file, sep=";", decimal=",")
# data = np.loadtxt(file, delimiter=';', skiprows=1)
data = df.to_numpy()
P = data[:,0]
C = data[:,1]
print(C)
OD=1.5
coeff=1/(10**OD -1)

C_norm= C/2000
plt.figure()
plt.scatter(  C_norm ,P*coeff, label='Data Points')
plt.xlabel('Counts (normalized)')
plt.ylabel('Power Fluo in  microwatts')
plt.show()