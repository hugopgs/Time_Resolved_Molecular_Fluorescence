

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.optimize import curve_fit

# Assuming 'get_Histo_from_csv' is in Function.py
from Function import * # ==========================================
# 1. MODEL DEFINITIONS
# ==========================================

def model_mono(t, params):
    """ Mono-exponential: Offset + A * exp(-t/tau) """
    offset, A, tau = params
    return offset + A * np.exp(-t / tau)

def model_bi(t, params):
    """ Bi-exponential: Offset + A1 * exp(-t/tau1) + A2 * exp(-t/tau2) """
    offset, A1, tau1, A2, tau2 = params
    return offset + A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def residuals_weighted(params, t, data, weights):
    """ 
    Calculates the weighted residuals vector for least_squares.
    Input weights must be 1/sigma.
    Return: (Data - Model) / sigma
    """
    if len(params) == 3:
        model_val = model_mono(t, params)
    else:
        model_val = model_bi(t, params)
    return (data - model_val) * weights

def get_manual_errors(result):
    """
    Calculates parameter errors (standard deviation) manually 
    from the Jacobian matrix.
    """
    J = result.jac
    try:
        cov_matrix = np.linalg.inv(J.T @ J)
        perr = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        perr = np.zeros(len(result.x)) 
    return perr

def print_result(name, params, errors, param_names, chi2_raw, n_bins, is_weighted=False):
    """
    Calculates DoF and Reduced Chi-Squared, then prints report.
    """
    n_params = len(params)
    dof = n_bins - n_params
    chi2_red = chi2_raw / dof
    
    # Calculate Max Residual (we need to know if we are printing sigmas or counts)
    # Note: We can't calculate max resid here without the data, 
    # so we will rely on the chi2 passed in, but we print the interpretation.

    print(f"\n{'='*60}")
    print(f"--- {name} ---")
    print(f"{'='*60}")
    
    # Print Parameters
    for i, p_name in enumerate(param_names):
        print(f"{p_name:6s} = {params[i]:12.4f} +/- {errors[i]:.4f}")
    
    print(f"{'-'*30}")
    print(f"Data Points (Bins) : {n_bins}")
    print(f"Degrees of Freedom : {dof}")
    
    if is_weighted:
        print(f"Chi^2 (Weighted)   : {chi2_raw:.4f}")
        print(f"Reduced Chi^2      : {chi2_red:.4f}  <-- (Ideal is ~1.0)")
    else:
        print(f"Chi^2 (Sum Sq)     : {chi2_raw:.4f}")
        print(f"Reduced Chi^2      : {chi2_red:.4f}  (Unweighted units)")

def plot_result(t, data, model_data, title, weights=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- 1. Main Data Plot ---
    ax1.semilogy(t, data, 'k.', alpha=0.2, markersize=4, label='Data')
    ax1.semilogy(t, model_data, 'r-', linewidth=1.5, label='Fit')
    ax1.set_title(title)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Counts (log)')
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.3)
    
    # --- 2. Residuals Plot ---
    raw_residuals = data - model_data
    
    if weights is not None:
        # Weighted Residuals: (Data-Model)/Sigma
        res_to_plot = raw_residuals * weights
        ylab = "W. Res. (sigmas)"
        title_res = "Weighted Residuals"
    else:
        # Raw Residuals
        res_to_plot = raw_residuals
        ylab = "Raw Residuals (Counts)"
        title_res = "Raw Residuals"

    # Plot
    ax2.plot(t, res_to_plot, 'b-', alpha=0.7, linewidth=1)
    ax2.axhline(0, c='k', ls='-', lw=1)

    # --- Indicators ---
    max_val = np.max(res_to_plot)
    min_val = np.min(res_to_plot)
    dist = max_val - min_val
    
    ax2.axhline(max_val, color='green', linestyle='--', alpha=0.8)
    ax2.axhline(min_val, color='green', linestyle='--', alpha=0.8)
    
    x_pos = t[-1] 
    ax2.vlines(x=x_pos, ymin=min_val, ymax=max_val, colors='orange', linewidth=2)
    mid_point = (max_val + min_val) / 2
    ax2.text(x_pos, mid_point, f' Spread: {dist:.2f}', 
             color='orange', fontweight='bold', ha='right', va='center', backgroundcolor='white')

    ax2.set_title(title_res)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel(ylab)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. MAIN SCRIPT
# ==========================================

if __name__ == "__main__":

    # ---- LOAD DATA ----
    
    file_1="Saturation/deux_TP_HugoEnzo_saturation_spads1_2025-12-03T14_22_13.csv"

    file_2="Saturation/deux_TP_HugoEnzo_saturation10_spads1_2025-12-03T14_19_36.csv"

    Histo, bin_values = get_Histo_from_csv(file_1)
    Histo_2, bin_values_2 = get_Histo_from_csv(file_2)

    # ---- PREPROCESS ----
    max_index = np.argmax(Histo)
    histogram = Histo[max_index:]
    t = bin_values[:len(histogram)]
    
    n_bins = len(histogram) # Number of data points
    
    # Robust Initial Guess
    p0_mono = [np.min(histogram), np.max(histogram), 10.0]

    # ---------------------------------------------------------
    # FIT 3: MONO-EXP (Weighted 1/sqrt(N))
    # ---------------------------------------------------------
    print("\n\n--- Starting Weighted Fits ---")
    
    sigma = np.sqrt(histogram)
    sigma[sigma == 0] = 1.0  
    weights = 1.0 / sigma

    res_w = least_squares(
        residuals_weighted, 
        x0=p0_mono,
        args=(t, histogram, weights)
    )

    popt3 = res_w.x
    perr3 = get_manual_errors(res_w)
    
    # Calculate Chi2 (Sum of Squared Weighted Residuals)
    # res_w.fun IS the weighted residual vector
    chi2_3 = np.sum(res_w.fun**2)

    print_result("3. Mono-Exp (Weighted)", popt3, perr3, ['Off', 'A', 'tau'], chi2_3, n_bins, is_weighted=True)
    plot_result(t, histogram, model_mono(t, popt3), "Mono (Weighted)", weights=weights)

    # ---- PREPROCESS ----
    max_index = np.argmax(Histo)
    histogram = Histo[max_index:]
    t = bin_values[:len(histogram)]
    
    n_bins = len(histogram) # Number of data points
    
    # Robust Initial Guess
    p0_mono = [np.min(histogram), np.max(histogram), 10.0]
