import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# ==========================================
# 1. MODÈLES ET OUTILS
# ==========================================

def model_mono(t, A, tau, offset):
    """ Modèle mono-exponentielle : y = A * exp(-t/tau) + Offset """
    return A * np.exp(-t / tau) + offset

def perform_fit(t, y, label):
    """ Effectue le fit, calcule les résidus et les stats """
    sigma = np.sqrt(np.abs(y))
    sigma[sigma == 0] = 1.0
    
    p0 = [np.max(y), 1500.0, np.min(y)]
    try:
        popt, pcov = curve_fit(model_mono, t, y, p0=p0, sigma=sigma, absolute_sigma=True)
        perr = np.sqrt(np.diag(pcov))
        
        model_y = model_mono(t, *popt)
        w_res = (y - model_y) / sigma
        chi2_red = np.sum(w_res**2) / (len(y) - len(popt))
        
        return popt, perr, model_y, w_res, chi2_red
    except RuntimeError:
        print(f"Fit failed for {label}")
        return None

# ==========================================
# 2. CONFIGURATION ET CHARGEMENT
# ==========================================

if __name__ == "__main__":
    # Liste complète (les 3 fichiers)
    all_files = [
        "Pola/Polarisation_Data/TP_HugoEnzo_pola_184_3min_powerup_spads1_2025-12-04T14_58_23.csv",
        "Pola/Polarisation_Data/TP_HugoEnzo_pola_219_3min_powerup_spads1_2025-12-04T14_50_24.csv",
        "Pola/Polarisation_Data/TP_HugoEnzo_pola_129_3min_powerup_spads1_2025-12-04T14_54_11.csv"
    ]
    
    # Fichiers actifs pour l'anisotropie (ceux que tu n'avais pas commentés)
    active_files = [
        "Pola/Polarisation_Data/TP_HugoEnzo_pola_219_3min_powerup_spads1_2025-12-04T14_50_24.csv",
        "Pola/Polarisation_Data/TP_HugoEnzo_pola_129_3min_powerup_spads1_2025-12-04T14_54_11.csv"
    ]

    labels = ['184 deg', '219 deg', '129 deg']
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    results = {}

    # Initialisation Figure 1
    fig1 = plt.figure(figsize=(14, 8))
    gs = fig1.add_gridspec(3, 2, width_ratios=[1.5, 1])
    ax_main = fig1.add_subplot(gs[:, 0])
    ax_residuals = [fig1.add_subplot(gs[i, 1]) for i in range(3)]

    print(f"{'Label':<10} | {'Tau (ps)':<18} | {'Chi2_red':<10}")
    print("-" * 45)

    # ==========================================
    # 3. BOUCLE DE TRAITEMENT (3 FICHIERS)
    # ==========================================
    for i, file_path in enumerate(all_files):
        try:
            df = pd.read_csv(file_path, sep=";", decimal=",")
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            continue

        raw_data = df.to_numpy()
        full_time, full_histo = raw_data[:, 0], raw_data[:, 2]

        # Recadrage au pic max
        max_idx = np.argmax(full_histo)
        y_cut = full_histo[max_idx:]
        t_cut = full_time[:len(y_cut)] # Garde l'échelle de temps relative au début du déclin

        # Fit
        fit_res = perform_fit(t_cut, y_cut, labels[i])
        if fit_res:
            popt, perr, model_y, w_res, chi2_red = fit_res
            results[file_path] = {'t': t_cut, 'y': y_cut, 'model': model_y, 'tau': popt[1]}
            
            print(f"{labels[i]:<10} | {popt[1]:.2f} +/- {perr[1]:.2f} | {chi2_red:.3f}")

            # Plot Main
            ax_main.semilogy(t_cut, y_cut, '.', color=colors[i], alpha=0.3, markersize=3)
            ax_main.semilogy(t_cut, model_y, '-', color=colors[i], label=f"{labels[i]} (τ={popt[1]:.0f} ps)")

            # Plot Résidus
            ax_res = ax_residuals[i]
            ax_res.plot(t_cut, w_res, color=colors[i], alpha=0.8)
            ax_res.axhline(0, color='black', lw=1)
            ax_res.set_ylabel(f"Res ({labels[i]})")
            
            # Indicateur de dispersion
            spread = np.ptp(w_res)
            ax_res.text(0.95, 0.05, f"Δ {spread:.1f}", transform=ax_res.transAxes, ha='right', fontweight='bold')

    # Cosmétique Figure 1
    ax_main.set_title("Fluorescence Decay - All Files")
    ax_main.legend()
    ax_residuals[-1].set_xlabel("Time (ps)")

    # ==========================================
    # 4. CALCUL ANISOTROPIE (2 FICHIERS UNIQUEMENT)
    # ==========================================
    if len([f for f in active_files if f in results]) == 2:
        f1, f2 = active_files[1], active_files[0]
        
        # On s'assure qu'ils ont la même longueur pour la soustraction
        min_len = min(len(results[f1]['y']), len(results[f2]['y']))
        y_diff = results[f1]['y'][:min_len] - results[f2]['y'][:min_len]
        t_diff = results[f1]['t'][:min_len]

        # Fit de la différence
        # fit_diff = perform_fit(t_diff, y_diff, "Anisotropy")
        
        plt.figure(figsize=(8, 6))
        plt.scatter(t_diff, y_diff/np.max(y_diff), s=1, alpha=0.5, label="Diff: Parallel - Orthogonal")
        # if fit_diff:
        #     plt.plot(t_diff, fit_diff[2], 'r-', label=f"Fit (τ={fit_diff[0][1]:.0f} ps)")
        
        plt.xlabel("Time (ps)")
        plt.ylabel("Counts Difference")
        plt.title("Anisotropy Analysis (Based on Active Files)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

    plt.show()