import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from scipy.optimize import curve_fit

# ==========================================
# 1. DÉFINITION DU MODÈLE
# ==========================================

def model_mono(t, A, tau, offset):
    """ Modèle mono-exponentielle : y = A * exp(-t/tau) + Offset """
    return A * np.exp(-t / tau) + offset

# ==========================================
# 2. SCRIPT DE TRAITEMENT
# ==========================================

if __name__ == "__main__":

    # ---- CONFIGURATION ----
    folder = "Quenching/Quenching_data_csv"
    concentrations_cibles = ["0", "10", "20", "30", "40"]
    
    # 1. RECHERCHE ET FILTRAGE DES FICHIERS
    try:
        files_found = []
        for f in os.listdir(folder):
            # On cherche le motif KI_X dans le nom du fichier
            match = re.search(r"KI_(\d+)", f)
            if match and match.group(1) in concentrations_cibles:
                files_found.append({
                    'path': os.path.join(folder, f),
                    'conc': int(match.group(1)),
                    'label': f"KI {match.group(1)} mM"
                })
        
        # Tri par concentration croissante
        files_found = sorted(files_found, key=lambda x: x['conc'])
        
    except FileNotFoundError:
        print(f"Erreur : Le dossier '{folder}' est introuvable.")
        exit()

    n_files = len(files_found)
    if n_files == 0:
        print("Aucun fichier correspondant aux concentrations cibles n'a été trouvé.")
        exit()

    # 2. PRÉPARATION DE LA FIGURE (Grille dynamique)
    fig = plt.figure(figsize=(14, 2.8 * n_files)) 
    gs = fig.add_gridspec(n_files, 2, width_ratios=[1.5, 1])
    ax_main = fig.add_subplot(gs[:, 0]) 
    
    # Palette de couleurs (Viridis pour une progression visuelle)
    colors = plt.cm.viridis(np.linspace(0, 0.8, n_files))

    print(f"Traitement de {n_files} fichiers...")
    print(f"{'Label':<15} | {'Tau (ps)':<18} | {'Chi2_red':<10}")
    print("-" * 50)

    # 3. BOUCLE DE TRAITEMENT
    for i, item in enumerate(files_found):
        try:
            # Chargement des données
            df = pd.read_csv(item['path'], sep=";", decimal=",")
            data = df.to_numpy()
            t_full, y_full = data[:, 0], data[:, 2]

            # Prétraitement : Coupure au pic maximum
            max_idx = np.argmax(y_full)
            y_raw = y_full[max_idx:]
            t_fit = t_full[:len(y_raw)]
            
            # Normalisation pour faciliter la comparaison visuelle
            norm = np.max(y_raw)
            y_norm = y_raw / norm
            
            # Poids (Statistique de Poisson)
            sigma_norm = np.sqrt(y_raw) / norm
            sigma_norm[sigma_norm == 0] = 1.0 / norm # Évite div par 0

            # Fit Mono-exponentiel
            # p0: [Amplitude=1, Tau=2000ps, Offset=0]
            popt, pcov = curve_fit(model_mono, t_fit, y_norm, p0=[1, 2000, 0], 
                                   sigma=sigma_norm, absolute_sigma=True)
            perr = np.sqrt(np.diag(pcov))
            
            tau_fit = popt[1]
            model_y = model_mono(t_fit, *popt)
            
            # Calcul du Chi2 réduit
            w_res = (y_norm - model_y) / sigma_norm
            chi2_red = np.sum(w_res**2) / (len(y_norm) - 3)

            print(f"{item['label']:<15} | {tau_fit:.2f} +/- {perr[1]:.2f} | {chi2_red:.3f}")

            # --- GRAPHIQUE PRINCIPAL (Echelle Log) ---
            ax_main.semilogy(t_fit, y_norm, '.', color=colors[i], alpha=0.15, markersize=2)
            ax_main.semilogy(t_fit, model_y, '-', color=colors[i], lw=1.5, 
                             label=f"{item['label']} ($\\tau$={tau_fit:.0f} ps)")

            # --- GRAPHIQUES DES RÉSIDUS (Droite) ---
            ax_res = fig.add_subplot(gs[i, 1])
            ax_res.plot(t_fit, w_res, color=colors[i], lw=0.8, alpha=0.8)
            ax_res.axhline(0, color='black', lw=0.6, ls='--')
            ax_res.set_ylabel(f"Res ({item['label']})", fontsize=8)
            ax_res.grid(True, alpha=0.2)
            
            if i < n_files - 1:
                ax_res.set_xticklabels([])
            else:
                ax_res.set_xlabel("Time (ps)")

        except Exception as e:
            print(f"Erreur sur {item['label']} : {e}")

    # --- FINALISATION DE LA FIGURE ---
    ax_main.set_xlabel("Time (ps)")
    ax_main.set_ylabel("Normalized Intensity (Log)")
    ax_main.set_title("Quenching Analysis - Fluorescence Decays")
    ax_main.legend(loc='upper right', fontsize='small', frameon=True)
    ax_main.grid(True, which="both", ls='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()