import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import pickle
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt


from Matrice.DCC import DCC_Portfolio
from clean_df_paper import df_total_set_daily, df_in_sample_set_daily, df_out_sample_set_daily
from clean_df_paper import df_training_set_weekly, df_test_set_weekly, df_out_sample_set_weekly

class SkewedTCopulaModel:
    def __init__(self, correl_results_file):
        """
        Modèle de copule t asymétrique utilisant les matrices de corrélation DCC
        
        Args:
            correl_results_file: Fichier contenant les matrices de corrélation DCC
        """
        # Charger les matrices DCC
        with open(correl_results_file, 'rb') as f:
            self.correl_results = pickle.load(f)
            
        self.weekly_matrices = self.correl_results['matrices']
        self.weekly_dates = self.correl_results['weekly_dates']
        
        # Paramètres de la copule t asymétrique
        self.nu = None  # Degrés de liberté
        self.gamma = None  # Paramètre d'asymétrie (vecteur pour chaque actif)
        
        # Stockage des résultats
        self.copula_params = {}  # Paramètres estimés pour chaque semaine
    
    @staticmethod
    def skewed_t_logpdf(x, df, gamma):
        """
        Densité log de la distribution t de Student asymétrique univariée
        
        Args:
            x: Points où évaluer la densité
            df: Degrés de liberté
            gamma: Paramètre d'asymétrie
        """
        const_term = np.log(2) + np.log(gamma + 1/gamma) + stats.t.logpdf(x * gamma * np.sign(x), df)
        return const_term
    
    @staticmethod
    def skewed_t_cdf(x, df, gamma):
        """
        Fonction de répartition de la distribution t de Student asymétrique univariée
        
        Args:
            x: Points où évaluer la CDF
            df: Degrés de liberté
            gamma: Paramètre d'asymétrie
        """
        return np.where(
            x < 0,
            2 * gamma / (1 + gamma**2) * stats.t.cdf(x * gamma, df),
            1 - 2 / (1 + gamma**2) * stats.t.cdf(-x / gamma, df)
        )
    
    @staticmethod
    def skewed_t_ppf(q, df, gamma):
        """
        Fonction quantile (inverse de la CDF) de la distribution t asymétrique
        
        Args:
            q: Probabilités (entre 0 et 1)
            df: Degrés de liberté
            gamma: Paramètre d'asymétrie
        """
        threshold = 2 * gamma / (1 + gamma**2)
        
        return np.where(
            q < threshold,
            gamma * stats.t.ppf(q * (1 + gamma**2) / (2 * gamma), df),
            -1 / gamma * stats.t.ppf((1 - q) * (1 + gamma**2) / 2, df)
        )
    
    def multivariate_skewed_t_logpdf(self, X, R, df, gamma):
        """
        Densité log de la copule t de Student asymétrique multivariée
        
        Args:
            X: Matrice d'observations (n_obs x n_dim)
            R: Matrice de corrélation (n_dim x n_dim)
            df: Degrés de liberté
            gamma: Vecteur de paramètres d'asymétrie (n_dim)
        """
        n_dim = X.shape[1]
        
        # Transformer les observations en quantiles uniformes
        U = np.zeros_like(X)
        for j in range(n_dim):
            U[:, j] = SkewedTCopulaModel.skewed_t_cdf(X[:, j], df, gamma[j])
        
        # Transformer les quantiles uniformes en quantiles t standards
        Z = stats.t.ppf(U, df)
        
        # Calculer la densité de la copule t
        constant = stats.t.logpdf(Z, df).sum(axis=1)
        mahalanobis = np.zeros(X.shape[0])
        
        R_inv = np.linalg.inv(R)
        for i in range(X.shape[0]):
            z = Z[i]
            mahalanobis[i] = z @ R_inv @ z - z @ z
            
        log_det = np.linalg.slogdet(R)[1]
        
        # Densité de la copule t
        copula_density = -0.5 * log_det - (df + n_dim) / 2 * np.log(1 + mahalanobis / df) + 0.5 * mahalanobis
        
        # Ajouter les termes marginaux
        for j in range(n_dim):
            constant -= SkewedTCopulaModel.skewed_t_logpdf(X[:, j], df, gamma[j])
        
        return constant + copula_density
    
    def estimate_skewed_t_copula(self, returns, R, initial_params=None):
        """
        Estimer les paramètres de la copule t asymétrique par maximum de vraisemblance
        
        Args:
            returns: Rendements des actifs
            R: Matrice de corrélation DCC
            initial_params: Paramètres initiaux [df, gamma1, gamma2, ...]
        """
        n_dim = returns.shape[1]
        
        if initial_params is None:
            initial_params = np.ones(n_dim + 1)
            initial_params[0] = 5.0  # df initial
        
        def neg_loglik(params):
            df = params[0]
            gamma = params[1:]
            
            if df <= 2 or np.any(gamma <= 0.2) or np.any(gamma >= 5):
                return 1e10
            
            return -np.sum(self.multivariate_skewed_t_logpdf(returns, R, df, gamma))
        
        bounds = [(2.1, 50)] + [(0.2, 5)] * n_dim
        
        result = minimize(
            neg_loglik,
            initial_params,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            return result.x
        else:
            return initial_params
    
    def estimate_weekly_copula_parameters(self, weekly_returns, window_size=250):
        """
        Estimer les paramètres de la copule t asymétrique pour chaque semaine
        en utilisant une fenêtre glissante
        
        Args:
            weekly_returns: DataFrame des rendements hebdomadaires
            window_size: Taille de la fenêtre glissante (en jours de trading)
        """
        print("\n=== Estimation des paramètres de la copule t asymétrique ===")
        
        tickers = weekly_returns.columns
        
        for i, date in enumerate(self.weekly_dates):
            print(f"\nTraitement de la date: {date}")

            # Obtenir la matrice de corrélation DCC pour cette semaine
            if date not in self.weekly_matrices:
                print(f"Attention: Pas de matrice DCC pour {date}")
                continue
                
            R = self.weekly_matrices[date].values
            
            # Définir la fenêtre de données
            window_end = date
            window_start = window_end - timedelta(weeks=52)  # Plus large pour s'assurer d'avoir assez de données
            
            # Extraire les rendements dans la fenêtre
            window_returns = weekly_returns.loc[window_start:window_end]
            
            # S'assurer d'avoir au moins 52 semaines (environ 1 an)
            if len(window_returns) < 52:
                print(f"Attention: Seulement {len(window_returns)} semaines disponibles pour {date}")
                continue
            
            # Estimer les paramètres
            params = self.estimate_skewed_t_copula(window_returns.values, R)
            
            self.copula_params[date] = {
                'df': params[0],
                'gamma': {ticker: gamma for ticker, gamma in zip(tickers, params[1:])}
            }
            print(f"Paramètres estimés pour {date}: df={params[0]}, gamma={params[1:]}")

    def export_results(self, base_filename='skewed_t_copula_results'):
        """Exporter les résultats"""
        with open(f'{base_filename}.pkl', 'wb') as f:
            pickle.dump({
                'copula_params': self.copula_params,
                'weekly_dates': self.weekly_dates
            }, f)
        
        print(f"\nRésultats exportés: {base_filename}.pkl")
    
    def run_full_pipeline(self, weekly_returns):
        """
        Exécuter la pipeline complète pour la copule t asymétrique
        
        Args:
            weekly_returns: DataFrame des rendements hebdomadaires
        """
        # 1. Estimer les paramètres pour chaque semaine
        self.estimate_weekly_copula_parameters(weekly_returns)

        # 2. Exporter les résultats
        self.export_results()
        
        print("\n=== Pipeline de la copule t asymétrique complète ===")
        print(f"Paramètres estimés pour {len(self.copula_params)} semaines")
    
    def simulate_skewed_t_copula(self, date, n_samples=1000):
        """
        Simuler des échantillons à partir de la copule t asymétrique
        
        Args:
            date: Date pour laquelle utiliser les paramètres
            n_samples: Nombre d'échantillons à générer
        
        Returns:
            DataFrame des échantillons simulés
        """
        if date not in self.copula_params:
            raise ValueError(f"Pas de paramètres disponibles pour {date}")
        
        if date not in self.weekly_matrices:
            raise ValueError(f"Pas de matrice DCC disponible pour {date}")
        
        params = self.copula_params[date]
        R = self.weekly_matrices[date].values
        df = params['df']
        tickers = list(params['gamma'].keys())
        gamma_vec = np.array([params['gamma'][ticker] for ticker in tickers])
        
        # Générer des échantillons t multivariés
        n_dim = len(tickers)
        Z = np.random.standard_t(df, size=(n_samples, n_dim))
        
        # Appliquer la matrice de corrélation (décomposition de Cholesky)
        L = np.linalg.cholesky(R)
        X = Z @ L.T
        
        # Transformer en CDF de la t standard
        U = stats.t.cdf(X, df)
        
        # Transformer en valeurs de la t asymétrique
        Y = np.zeros_like(U)
        for j in range(n_dim):
            Y[:, j] = SkewedTCopulaModel.skewed_t_ppf(U[:, j], df, gamma_vec[j])
        
        return pd.DataFrame(Y, columns=tickers)
    
    def visualize_copula(self, date, weekly_returns=None):
        """
        Visualiser les résultats de la copule t asymétrique
        
        Args:
            date: Date pour laquelle visualiser les résultats
            weekly_returns: DataFrame des rendements hebdomadaires pour comparaison
        """
        if date not in self.copula_params:
            print(f"Pas de paramètres disponibles pour {date}")
            return
        
        params = self.copula_params[date]
        tickers = list(params['gamma'].keys())
        
        # Récupérer les paramètres
        df = params['df']
        gamma_vec = [params['gamma'][ticker] for ticker in tickers]
        R = self.weekly_matrices[date].values
        
        print("\n===== Visualisation de la Copule t Asymétrique =====")
        print(f"Date: {date}")
        print(f"Degrés de liberté (df): {df:.2f}")
        print("Paramètres d'asymétrie (gamma):")
        for ticker, gamma in zip(tickers, gamma_vec):
            print(f"  {ticker}: {gamma:.4f}")
        
        print("\nMatrice de corrélation DCC:")
        pd.set_option('display.precision', 4)
        corr_df = pd.DataFrame(R, index=tickers, columns=tickers)
        print(corr_df)
        
        # Simuler des données
        n_samples = 2000
        simulated_data = self.simulate_skewed_t_copula(date, n_samples=n_samples)
        
        # Créer la figure
        fig = plt.figure(figsize=(15, 10))
        
        # Limiter à 3 actifs pour la visualisation
        if len(tickers) > 3:
            viz_tickers = tickers[:3]
            print(f"\nNote: Visualisation limitée aux 3 premiers actifs: {viz_tickers}")
        else:
            viz_tickers = tickers
        
        # Créer une grille de scatter plots pour les paires d'actifs
        n_viz = len(viz_tickers)
        plot_idx = 1
        
        for i in range(n_viz):
            for j in range(i+1, n_viz):
                ax = fig.add_subplot(n_viz-1, n_viz-1, plot_idx)
                ax.scatter(
                    simulated_data[viz_tickers[i]], 
                    simulated_data[viz_tickers[j]], 
                    alpha=0.5, 
                    s=5, 
                    c='blue', 
                    label='Simulated'
                )
                
                # Ajouter les données réelles si disponibles
                if weekly_returns is not None:
                    # Trouver les rendements autour de cette date
                    window_start = date - timedelta(weeks=26)
                    window_end = date + timedelta(weeks=26)
                    window_returns = weekly_returns.loc[window_start:window_end]
                    
                    ax.scatter(
                        window_returns[viz_tickers[i]], 
                        window_returns[viz_tickers[j]], 
                        alpha=0.7, 
                        s=10, 
                        c='red', 
                        label='Actual'
                    )
                
                ax.set_xlabel(viz_tickers[i])
                ax.set_ylabel(viz_tickers[j])
                
                if i == 0 and j == 1:
                    ax.legend()
                
                plot_idx += 1
        
        plt.tight_layout()
        plt.suptitle(f"Copule t asymétrique pour {date} (df={df:.2f})", y=1.02)
        
        # Enregistrer la figure
        fig_file = f"skewed_t_copula_visualization_{date.strftime('%Y%m%d')}.png"
        plt.savefig(fig_file)
        plt.close()
        
        print(f"\nVisualisation enregistrée dans: {fig_file}")
        
        # Afficher aussi une comparaison des statistiques
        print("\nStatistiques des rendements simulés:")
        stats_df = simulated_data.describe().T
        stats_df['skew'] = simulated_data.skew()
        stats_df['kurtosis'] = simulated_data.kurtosis()
        print(stats_df)
        
        # Afficher la matrice de corrélation des données simulées
        print("\nMatrice de corrélation des rendements simulés:")
        sim_corr = simulated_data.corr()
        print(sim_corr)
        
        # Comparer avec la matrice de corrélation DCC
        print("\nDifférence entre la corrélation simulée et la matrice DCC:")
        diff = sim_corr - corr_df
        print(diff)

def main():
    # Créer l'instance unique
    dcc_portfolio = DCC_Portfolio(
        all_data=df_total_set_daily,
        in_sample_data=df_in_sample_set_daily,
        out_sample_data=df_out_sample_set_daily,
        out_sample_weekly_data= df_out_sample_set_weekly
    )
    
    # Exécuter toute la pipeline
    dcc_portfolio.run_full_pipeline()
    
    # Afficher des exemples
    dcc_portfolio.display_example_matrices()
    
    # Exporter les résultats
    dcc_portfolio.export_results()
    
    # Créer et exécuter la copule
    skewed_t_copula = SkewedTCopulaModel('dcc_correlation_results_all_weekly.pkl')
    skewed_t_copula.run_full_pipeline(df_out_sample_set_weekly)
    
    # Visualiser la copule pour une date spécifique
    # Utiliser la première date disponible après l'estimation des paramètres
    if len(skewed_t_copula.copula_params) > 0:
        selected_date = list(skewed_t_copula.copula_params.keys())[0]
        print(f"\n=== Visualisation de la copule pour {selected_date} ===")
        skewed_t_copula.visualize_copula(selected_date, df_out_sample_set_weekly)
    else:
        print("Aucune date disponible pour visualiser la copule")


if __name__ == "__main__":
    main()