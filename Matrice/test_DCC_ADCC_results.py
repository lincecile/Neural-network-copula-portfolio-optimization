import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Charger les résultats DCC et ADCC
def load_results(dcc_file='dcc_results_all_weekly.pkl', adcc_file='adcc_results_all_weekly.pkl'):
    """Charger les fichiers pickle DCC et ADCC"""
    
    # Charger DCC
    with open(dcc_file, 'rb') as f:
        dcc_results = pickle.load(f)
    
    # Charger ADCC  
    with open(adcc_file, 'rb') as f:
        adcc_results = pickle.load(f)
    
    return dcc_results, adcc_results

# Fonction pour compter le nombre de matrices par pickle
def count_matrices_per_pickle(dcc_results, adcc_results):
    """Afficher le nombre de matrices par fichier pickle"""
    
    print("\n=== NOMBRE DE MATRICES PAR PICKLE ===")
    
    # Compter les matrices DCC
    dcc_count = len(dcc_results['matrices'])
    print(f"Nombre de matrices dans DCC: {dcc_count}")
    
    # Première et dernière date DCC
    dcc_dates = sorted(list(dcc_results['matrices'].keys()))
    if dcc_dates:
        print(f"Première date DCC: {dcc_dates[0].strftime('%Y-%m-%d')}")
        print(f"Dernière date DCC: {dcc_dates[-1].strftime('%Y-%m-%d')}")
    
    # Compter les matrices ADCC
    adcc_count = len(adcc_results['matrices'])
    print(f"\nNombre de matrices dans ADCC: {adcc_count}")
    
    # Première et dernière date ADCC
    adcc_dates = sorted(list(adcc_results['matrices'].keys()))
    if adcc_dates:
        print(f"Première date ADCC: {adcc_dates[0].strftime('%Y-%m-%d')}")
        print(f"Dernière date ADCC: {adcc_dates[-1].strftime('%Y-%m-%d')}")
    
    # Comparaison
    common_dates = set(dcc_dates).intersection(set(adcc_dates))
    print(f"\nNombre de dates communes: {len(common_dates)}")

# Analyser la structure des données
def analyze_structure(dcc_results, adcc_results):
    """Analyser la structure des données chargées"""
    
    print("=== STRUCTURE DES DONNÉES ===")
    print("\nDCC Results:")
    print(f"- Keys: {dcc_results.keys()}")
    print(f"- Nombre de matrices hebdomadaires: {len(dcc_results['matrices'])}")
    print(f"- Première date: {min(dcc_results['weekly_dates'])}")
    print(f"- Dernière date: {max(dcc_results['weekly_dates'])}")
    print(f"- Paramètres DCC: alpha={dcc_results['alpha']:.4f}, beta={dcc_results['beta']:.4f}")
    
    print("\nADCC Results:")
    print(f"- Keys: {adcc_results.keys()}")
    print(f"- Nombre de matrices hebdomadaires: {len(adcc_results['matrices'])}")
    print(f"- Première date: {min(adcc_results['weekly_dates'])}")
    print(f"- Dernière date: {max(adcc_results['weekly_dates'])}")
    print(f"- Paramètres ADCC: alpha={adcc_results['alpha']:.4f}, beta={adcc_results['beta']:.4f}, gamma={adcc_results['gamma']:.4f}")

# Comparer des matrices spécifiques
def compare_matrices_at_date(dcc_results, adcc_results, target_date):
    """Comparer les matrices DCC et ADCC à une date spécifique"""
    
    # Trouver la date la plus proche disponible
    available_dcc_dates = list(dcc_results['matrices'].keys())
    available_adcc_dates = list(adcc_results['matrices'].keys())
    
    # Chercher la date exacte ou la plus proche
    closest_dcc_date = None
    closest_adcc_date = None
    
    for date in available_dcc_dates:
        if date == target_date or (closest_dcc_date is None and date >= target_date):
            closest_dcc_date = date
    
    for date in available_adcc_dates:
        if date == target_date or (closest_adcc_date is None and date >= target_date):
            closest_adcc_date = date
            
    print(f"\n=== COMPARAISON DES MATRICES POUR {target_date} ===")
    print(f"Date DCC utilisée: {closest_dcc_date}")
    print(f"Date ADCC utilisée: {closest_adcc_date}")
    
    # Extraire les matrices
    dcc_matrix = dcc_results['matrices'][closest_dcc_date]
    adcc_matrix = adcc_results['matrices'][closest_adcc_date]
    
    # Afficher les matrices
    print("\nMatrice DCC:")
    print(dcc_matrix.round(6))
    
    print("\nMatrice ADCC:")
    print(adcc_matrix.round(6))
    
    # Calculer la différence
    difference = adcc_matrix - dcc_matrix
    print("\nDifférence (ADCC - DCC):")
    print(difference.round(6))
    
    return dcc_matrix, adcc_matrix, difference

# Visualiser les différences
def visualize_differences(dcc_results, adcc_results, sample_dates=None):
    """Visualiser les différences entre DCC et ADCC pour plusieurs dates"""
    
    if sample_dates is None:
        # Sélectionner 5 dates espacées
        all_dates = sorted(list(dcc_results['matrices'].keys()))
        sample_dates = [all_dates[i] for i in range(0, len(all_dates), len(all_dates)//5)][:5]
    
    fig, axes = plt.subplots(1, len(sample_dates), figsize=(20, 4))
    if len(sample_dates) == 1:
        axes = [axes]
    
    for i, date in enumerate(sample_dates):
        dcc_matrix = dcc_results['matrices'][date]
        adcc_matrix = adcc_results['matrices'][date]
        difference = adcc_matrix - dcc_matrix
        
        # Créer un heatmap pour la différence
        sns.heatmap(difference, 
                   annot=True, 
                   fmt='.4f', 
                   center=0, 
                   cmap='RdBu_r',
                   ax=axes[i],
                   cbar_kws={'shrink': 0.8})
        axes[i].set_title(f'ADCC - DCC\n{date.date()}')
    
    plt.tight_layout()
    plt.show()

# Analyser l'évolution des corrélations
def plot_correlation_evolution(dcc_results, adcc_results, pair='SPY-QQQ'):
    """Tracer l'évolution des corrélations pour une paire d'actifs"""
    
    # Extraire les noms d'actifs
    first_matrix = next(iter(dcc_results['matrices'].values()))
    tickers = list(first_matrix.columns)
    
    # Trouver les indices pour la paire demandée
    parts = pair.split('-')
    idx1 = tickers.index(f"{parts[0]} US Equity")
    idx2 = tickers.index(f"{parts[1]} US Equity")
    
    # Extraire les corrélations pour toutes les dates
    dates = sorted(list(dcc_results['matrices'].keys()))
    dcc_corrs = []
    adcc_corrs = []
    
    for date in dates:
        # DCC
        dcc_matrix = dcc_results['matrices'][date]
        dcc_std = np.sqrt(np.diag(dcc_matrix))
        dcc_corr_matrix = dcc_matrix / np.outer(dcc_std, dcc_std)
        dcc_corrs.append(dcc_corr_matrix.iloc[idx1, idx2])
        
        # ADCC
        adcc_matrix = adcc_results['matrices'][date]
        adcc_std = np.sqrt(np.diag(adcc_matrix))
        adcc_corr_matrix = adcc_matrix / np.outer(adcc_std, adcc_std)
        adcc_corrs.append(adcc_corr_matrix.iloc[idx1, idx2])
    
    # Tracer les résultats
    plt.figure(figsize=(12, 6))
    plt.plot(dates, dcc_corrs, label='DCC', linewidth=2)
    plt.plot(dates, adcc_corrs, label='ADCC', linewidth=2)
    plt.title(f'Corrélation dynamique {pair}')
    plt.xlabel('Date')
    plt.ylabel('Corrélation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Statistiques comparatives
def comparative_statistics(dcc_results, adcc_results):
    """Comparer les statistiques entre DCC et ADCC"""
    
    all_dates = sorted(list(dcc_results['matrices'].keys()))
    differences = []
    
    for date in all_dates:
        dcc_matrix = dcc_results['matrices'][date]
        adcc_matrix = adcc_results['matrices'][date]
        diff = adcc_matrix - dcc_matrix
        differences.append(diff)
    
    # Statistiques globales sur les différences
    all_diff_values = [diff.values.flatten() for diff in differences]
    all_diff_flat = np.concatenate(all_diff_values)
    
    print("\n=== STATISTIQUES COMPARATIVES ===")
    print(f"Moyenne des différences (ADCC - DCC): {np.mean(all_diff_flat):.6f}")
    print(f"Écart-type des différences: {np.std(all_diff_flat):.6f}")
    print(f"Min des différences: {np.min(all_diff_flat):.6f}")
    print(f"Max des différences: {np.max(all_diff_flat):.6f}")
    
    # Distribution des différences
    plt.figure(figsize=(10, 6))
    plt.hist(all_diff_flat, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution des différences (ADCC - DCC)')
    plt.xlabel('Différence')
    plt.ylabel('Fréquence')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.grid(True, alpha=0.3)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger les résultats
    dcc_results, adcc_results = load_results()
    
    # Compter les matrices par pickle
    count_matrices_per_pickle(dcc_results, adcc_results)
    
    # Analyser la structure
    analyze_structure(dcc_results, adcc_results)
    
    # Comparer des matrices spécifiques
    # Prendre la première date disponible comme exemple
    first_date = min(dcc_results['weekly_dates'])
    compare_matrices_at_date(dcc_results, adcc_results, first_date)
    
    # Visualiser les différences
    visualize_differences(dcc_results, adcc_results)
    
    # Evolution des corrélations
    plot_correlation_evolution(dcc_results, adcc_results, 'SPY-QQQ')
    plot_correlation_evolution(dcc_results, adcc_results, 'SPY-DIA')
    plot_correlation_evolution(dcc_results, adcc_results, 'DIA-QQQ')
    
    # Statistiques comparatives
    comparative_statistics(dcc_results, adcc_results)