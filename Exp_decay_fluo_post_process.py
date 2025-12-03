import numpy as np
import matplotlib.pyplot as plt

file="deux_TP_HugoEnzo_spads1_2025-12-02T16_32_23.csv"
import pandas as pd

df = pd.read_csv(file, sep=";", decimal=",")
# data = np.loadtxt(file, delimiter=';', skiprows=1)
data = df.to_numpy()
Histo = data[:,2]
bin_values = data[:,0]


  
# Exponential decay fitting
def exp_decay(x, A, tau, C):
    return A * np.exp(-x / tau) + C

from scipy.optimize import curve_fit
# Initial guess for the parameters A, tau, C
initial_guess = [np.max(Histo), 1000, np.min(Histo)]
# Fit the data
params, covariance = curve_fit(exp_decay, bin_values, Histo, p0=initial_guess)
A_fit, tau_fit, C_fit = params
uncertainties = np.sqrt(np.diag(covariance))
A_err, tau_err, C_err = uncertainties


print(
    f"Fitted parameters:\n"
    f"  A   = {A_fit:.5g} ± {A_err:.5g}\n"
    f"  tau = {tau_fit:.5g} ± {tau_err:.5g}\n"
    f"  C   = {C_fit:.5g} ± {C_err:.5g}"
)

Gamma=1/tau_fit
print(f"Gamma (1/tau): {Gamma:.3e} ps^-1")
print(f"tau: {tau_fit:.3e} ps")
plt.figure()
plt.plot(bin_values, Histo, label='Raw Data')
plt.plot(bin_values, exp_decay(bin_values, *params), label=f'Fitted Curve, \n parameters: A={A_fit:.3}, tau={tau_fit:.3}, C={C_fit:.3}', color='red')
plt.xlabel('Time Bins')
plt.ylabel('Counts')
plt.legend()
plt.show()

