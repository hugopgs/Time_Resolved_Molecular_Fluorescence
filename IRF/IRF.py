import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def process_irf(file_path):
    """Charge le CSV et calcule les paramètres de la FWHM."""
    try:
        df = pd.read_csv(file_path, sep=";", decimal=",")
        data = df.to_numpy()
        t = data[:, 0]
        h = data[:, 2]
        
        # Calcul FWHM
        max_val = np.max(h)
        half_max = max_val / 2
        
        # Trouver les indices de croisement
        indices = np.where(np.diff(np.sign(h - half_max)))[0]
        
        x1, x2 = None, None
        if len(indices) >= 2:
            # Interpolation linéaire pour x1 et x2
            idx1, idx2 = indices[0], indices[1]
            x1 = t[idx1] + (t[idx1+1] - t[idx1]) * (half_max - h[idx1]) / (h[idx1+1] - h[idx1])
            x2 = t[idx2] + (t[idx2+1] - t[idx2]) * (half_max - h[idx2]) / (h[idx2+1] - h[idx2])
            fwhm_val = x2 - x1
        else:
            fwhm_val = 0
            
        return t, h, x1, x2, half_max, fwhm_val
    except Exception as e:
        print(f"Erreur sur le fichier {file_path}: {e}")
        return None

# --- Configuration ---
files = [
    "IRF/IRF_miroir_pos1.csv",
    "IRF/IRF_miroir_pos2.csv" # Vérifie l'extension .csv si besoin
]
labels = ["IRF ", "IRF position mirroir 2"]
colors = ["tab:blue", "tab:orange"]

plt.figure(figsize=(10, 6))

# --- Boucle de traitement et plot ---
for i, file in enumerate(files):
    result = process_irf(file)
    
    if result:
        t, h, x1, x2, hm, fwhm_val = result
        
        # Plot de la courbe principale
        line, = plt.plot(t, h, label=f"{labels[i]} (FWHM: {fwhm_val:.1f} ps)", color=colors[i])
        
        # Si la FWHM a été trouvée, on l'affiche
        if x1 and x2:
            plt.hlines(hm, xmin=x1, xmax=x2, colors=colors[i], linestyles="--", alpha=0.6)
            plt.plot([x1, x2], [hm, hm], "o", color=colors[i], markersize=4)
            # Optionnel : remplir sous la courbe
            plt.fill_between(t, h, hm, where=(t >= x1) & (t <= x2), color=colors[i], alpha=0.1)

# --- Mise en forme ---
plt.title("Comparaison des IRF avec FWHM")
plt.xlabel("Time (ps)")
plt.ylabel("Counts")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()