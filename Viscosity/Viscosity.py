import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from scipy.optimize import curve_fit

# ==========================================
# 1. MODÈLE
# ==========================================
def model_mono(t, A, tau, offset):
    return A * np.exp(-t / tau) + offset

# ==========================================
# 2. SCRIPT PRINCIPAL
# ==========================================
if __name__ == "__main__":

    # ---- CONFIGURATION ----
    folder = "Viscosity/Glycerol_Data"
    
    # 1. RÉCUPÉRATION DES FICHIERS
    try:
        # On récupère tous les .csv du dossier
        all_files = sorted([
            os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if f.endswith(".csv")
        ])
    except FileNotFoundError:
        print(f"Erreur : Le dossier '{folder}' est introuvable.")
        exit()

    n_files = len(all_files)
    if n_files == 0:
        print("Aucun fichier trouvé.")
        exit()

    # 2. PRÉPARATION DE L'AFFICHAGE
    fig = plt.figure(figsize=(14, 2.5 * n_files)) 
    gs = fig.add_gridspec(n_files, 2, width_ratios=[1.5, 1])
    ax_main = fig.add_subplot(gs[:, 0]) 
    colors = plt.cm.plasma(np.linspace(0, 0.8, n_files))

    print(f"{'Angle':<15} | {'Tau (ps)':<15} | {'Chi2_red':<10}")
    print("-" * 50)

    # 3. BOUCLE DE TRAITEMENT
    for i, file_path in enumerate(all_files):
        filename = os.path.basename(file_path)
        
        # Extraction de l'angle (ex: 129, 184, 219) via Regex
        match = re.search(r"pola_(\d+)", filename)
        label_name = f"Angle {match.group(1)}° (Glycerol)" if match else filename[:20]

        try:
            df = pd.read_csv(file_path, sep=";", decimal=",")
            data = df.to_numpy()
            full_time, full_histo = data[:, 0], data[:, 2]

            # Recadrage au maximum
            max_idx = np.argmax(full_histo)
            y_raw = full_histo[max_idx:]
            t_data = full_time[:len(y_raw)]
            
            # Normalisation
            norm = np.max(y_raw)
            y_norm = y_raw / norm
            sigma_norm = np.sqrt(y_raw) / norm
            sigma_norm[sigma_norm == 0] = 1.0 / norm

            # Fit
            popt, pcov = curve_fit(model_mono, t_data, y_norm, p0=[1, 2500, 0], 
                                   sigma=sigma_norm, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            
            tau_fit = popt[1]
            model_y = model_mono(t_data, *popt)
            chi2_red = np.sum(((y_norm - model_y) / sigma_norm)**2) / (len(y_norm) - 3)

            print(f"{label_name:<15} | {tau_fit:.2f} +/- {perr[1]:.2f} | {chi2_red:.3f}")

            # Graphique principal (Log)
            ax_main.semilogy(t_data, y_norm, '.', color=colors[i], alpha=0.2, markersize=2)
            ax_main.semilogy(t_data, model_y, '-', color=colors[i], lw=1.5, 
                             label=f"{label_name} (τ={tau_fit:.0f} ps)")

            # Résidus
            ax_res = fig.add_subplot(gs[i, 1])
            ax_res.plot(t_data, (y_norm - model_y) / sigma_norm, color=colors[i], lw=0.7)
            ax_res.axhline(0, color='k', lw=0.5)
            ax_res.set_ylabel(f"Res {label_name}", fontsize=8)
            if i < n_files - 1: ax_res.set_xticklabels([])

        except Exception as e:
            print(f"Erreur sur {filename}: {e}")

    # Finalisation
    ax_main.set_xlabel("Time (ps)")
    ax_main.set_ylabel("Normalized Intensity")
    ax_main.set_title("Fluorescence Decay in Glycerol (Viscosity Analysis)")
    ax_main.legend(loc='upper right', fontsize='small')
    ax_main.grid(True, which="both", ls='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()