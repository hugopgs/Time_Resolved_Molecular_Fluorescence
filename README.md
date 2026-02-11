# Time_Resolved_Molecular_Fluorescence


Ce dépôt contient une suite de scripts Python dédiés à l'analyse de données de Fluoréscence, développés dans le cadre dd'un TP en **Master Quantum Technologies** à l'Université de Strasbourg et à **Télécom Physique Strasbourg**.

## Fonctionnalités

Le projet est divisé en six modules principaux pour le traitement des fichiers de comptage de photons (TCSPC) au format `.csv` :


### 1. Affichage de la fluoresnce et Fit 
Analyse automatique des déclins de fluorescence. 
**Ajustement (Fitting)** Définition des différents fit. 


### 2. Caractérisation de l'IRF (Instrument Response Function)
Mesure de la résolution temporelle du système de détection.
* **Calcul de la FWHM** : Détermination de la largeur à mi-hauteur par interpolation linéaire pour une précision accrue.
* **Comparaison** : Superposition de plusieurs IRF sur un même graphique pour vérifier la stabilité temporelle du setup.

### 3. Études de Polarisation et Anisotropie
Calcul de la dynamique de réorientation moléculaire.
* **Soustraction de signaux** : Traitement des composantes parallèles ($I_{//}$) et perpendiculaires ($I_{\perp}$).
* **Anisotropie** : Extraction du coefficient d'anisotropie et fit de la décroissance pour obtenir le temps de corrélation rotationnel.


### 4. Analyse du Quenching (Viscosité & Concentration)
Analyse automatique des déclins de fluorescence pour une série de concentrations (ex: KI 0 à 40 mM).
* **Extraction automatique** : Utilise des expressions régulières (Regex) pour identifier les concentrations dans les noms de fichiers.
* **Ajustement (Fitting)** : Fit mono-exponentiel pondéré par la statistique de Poisson.
* **Visualisation** : Génération d'un graphique principal en échelle logarithmique et d'une colonne de résidus pondérés pour chaque fit.

### 5. Test Saturation des SPADS
Affichage saturation des SPADS.

## 🛠️ Installation & Dépendances

Les scripts nécessitent Python 3.8+ et les bibliothèques scientifiques standards :

```bash
pip install numpy matplotlib pandas scipy
