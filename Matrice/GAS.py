import pandas as pd
import numpy as np
from arch import arch_model
from scipy.optimize import minimize
import warnings
import pickle
from datetime import timedelta
from clean_df_all import df_total_set_daily, df_out_sample_set_daily, df_in_sample_set_daily, df_out_sample_set_weekly
warnings.filterwarnings("ignore")

class GAS_Portfolio:
    def __init__(self, all_data, in_sample_data, out_sample_data, out_sample_weekly_data):
        """
        Une classe pour gérer la pipeline GAS - implémentation manuelle
        
        Args:
            all_data: Toutes les données (in-sample + out-of-sample)
            in_sample_data: Données in-sample pour estimer les paramètres GAS
            out_sample_data: Données out-of-sample journalières
            out_sample_weekly_data: Dates hebdomadaires pour le rééquilibrage
        """
        self.all_data = all_data
        self.in_sample_data = in_sample_data
        self.out_sample_data = out_sample_data
        self.out_sample_weekly_data = out_sample_weekly_data
        
        self.tickers = list(self.all_data.columns)
        self.window_size = 250  # 1 an comme dans le papier
        
        # Paramètres ARMA spécifiés dans le papier
        self.arma_params = {
            'SPY': (8, 8),
            'DIA': (10, 10), 
            'QQQ': (7, 7)
        }
        
        # Paramètres GAS
        self.omega = None  # Intercept
        self.A = None      # Matrice A (paramètre de score)
        self.B = None      # Matrice B (paramètre autorégressif)
        
        # Stockage des résultats
        self.in_sample_results = None
        
        # Matrices de corrélation
        self.out_sample_correlation_matrices = {}
        self.weekly_correlation_matrices = {}
        
        # Matrices de covariance
        self.out_sample_covariance_matrices = {}
        self.weekly_covariance_matrices = {}
        
    def get_arma_order(self, ticker):
        """Obtenir l'ordre ARMA pour un ETF"""
        for key in self.arma_params:
            if key in ticker.upper():
                return self.arma_params[key]
        raise ValueError(f"ARMA order not found for ticker: {ticker}")
    
    def fit_arma_gjr_garch_models(self, returns_data):
        """Ajuster ARMA-GJR-GARCH sur un ensemble de données"""
        residuals_dict = {}
        variances_dict = {}
        
        for ticker in self.tickers:
            returns = returns_data[ticker].values
            p, q = self.get_arma_order(ticker)
            
            try:
                model = arch_model(
                    returns,
                    mean='AR',
                    vol='GARCH',
                    lags=p,
                    p=1, o=1, q=1
                )
                
                result = model.fit(disp='off', show_warning=False)
                residuals = result.resid
                volatility = result.conditional_volatility
                
                # Gestion des valeurs problématiques
                mask = ~(np.isnan(residuals) | np.isnan(volatility) | (volatility <= 0))
                if not mask.all():
                    residuals = pd.Series(residuals).fillna(method='ffill').fillna(method='bfill').values
                    volatility = pd.Series(volatility).fillna(method='ffill').fillna(method='bfill').values
                    volatility = np.maximum(volatility, 1e-8)
                
                residuals_dict[ticker] = residuals / volatility
                variances_dict[ticker] = volatility ** 2
                
            except:
                # Fallback simple
                residuals = returns - np.mean(returns)
                sigma = np.std(returns)
                residuals_dict[ticker] = residuals / sigma
                variances_dict[ticker] = np.full_like(returns, sigma**2)
        
        standardized_residuals = pd.DataFrame(residuals_dict, index=returns_data.index)
        conditional_variances = pd.DataFrame(variances_dict, index=returns_data.index)
        
        # Nettoyer
        standardized_residuals = standardized_residuals.fillna(0)
        conditional_variances = conditional_variances.fillna(returns_data.var())
        
        return standardized_residuals, conditional_variances
    
    def ensure_positive_definite(self, matrix, min_eigenvalue=1e-8):
        """S'assurer qu'une matrice est définie positive"""
        # Vérifier si la matrice est symétrique
        matrix = (matrix + matrix.T) / 2
        
        # Décomposition spectrale
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Corriger les valeurs propres négatives
        eigvals = np.maximum(eigvals, min_eigenvalue)
        
        # Reconstruire la matrice
        matrix_corrected = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        # S'assurer que la diagonale est à 1 pour une matrice de corrélation
        if np.allclose(np.diag(matrix), 1.0, atol=1e-6):
            diag_sqrt = np.sqrt(np.diag(matrix_corrected))
            matrix_corrected = matrix_corrected / np.outer(diag_sqrt, diag_sqrt)
            np.fill_diagonal(matrix_corrected, 1.0)
        
        return matrix_corrected
    
    def calculate_score(self, epsilon_t, R_t):
        """
        Calculer le score pour le modèle GAS.
        Le score est la dérivée de la log-vraisemblance par rapport aux paramètres.
        """
        # Pour une distribution normale multivariée, le score est:
        # S_t = 0.5 * (epsilon_t * epsilon_t' - R_t)
        n = R_t.shape[0]
        
        # Calculer le produit extérieur des résidus standardisés
        outer_prod = np.outer(epsilon_t, epsilon_t)
        
        # Le score est simplement la différence entre ce produit et la matrice de corrélation
        score = 0.5 * (outer_prod - R_t)
        
        return score
    
    def estimate_gas_parameters(self, standardized_residuals):
        """Estimer les paramètres du modèle GAS par maximum de vraisemblance"""
        epsilon = standardized_residuals.values
        T, n = epsilon.shape
        
        # Fonction de log-vraisemblance pour le modèle GAS
        def gas_log_likelihood(params):
            try:
                # Extraire les paramètres
                omega = params[0]
                a = params[1]  # paramètre A (scalaire)
                b = params[2]  # paramètre B (scalaire)
                
                # Vérification des contraintes
                if a <= 0 or b <= 0 or a + b >= 0.999:
                    return 1e6
                
                # Matrice de corrélation inconditionnelle
                R_bar = np.corrcoef(epsilon.T)
                R_bar = self.ensure_positive_definite(R_bar)
                
                # Initialisation des matrices
                R_t = np.zeros((T, n, n))
                R_t[0] = R_bar.copy()
                
                log_lik = 0
                
                for t in range(1, T):
                    # Calcul du score pour t-1
                    score_tm1 = self.calculate_score(epsilon[t-1], R_t[t-1])
                    
                    # Mise à jour GAS
                    # R_t = (1-a-b)*R_bar + a*score_t-1 + b*R_t-1
                    R_t[t] = (1 - a - b) * R_bar + a * score_tm1 + b * R_t[t-1]
                    
                    # S'assurer que R_t est une matrice de corrélation valide
                    R_t[t] = self.ensure_positive_definite(R_t[t])
                    
                    # Calculer la log-vraisemblance
                    sign, logdet = np.linalg.slogdet(R_t[t])
                    if sign <= 0:
                        return 1e6
                    
                    # Terme quadratique de la densité normale multivariée
                    R_inv = np.linalg.inv(R_t[t])
                    quad_form = epsilon[t] @ R_inv @ epsilon[t]
                    
                    # Contribution à la log-vraisemblance (sans les constantes)
                    log_lik += -0.5 * (logdet + quad_form)
                
                return -log_lik
            except Exception as e:
                return 1e6
        
        # Optimisation pour trouver les paramètres optimaux
        initial_guess = [0.01, 0.05, 0.90]  # omega, a, b
        bounds = [(0.001, 0.1), (0.001, 0.3), (0.65, 0.998)]
        
        result = minimize(
            gas_log_likelihood,
            x0=initial_guess,
            method='L-BFGS-B',
            bounds=bounds
        )
        
        if result.success:
            omega, a, b = result.x
            
            # Convertir les paramètres scalaires en matrices
            A = a  # Garder comme scalaire pour simplifier
            B = b  # Garder comme scalaire pour simplifier
            
            return omega, A, B
        else:
            print("L'estimation des paramètres a échoué. Utilisation des valeurs par défaut.")
            return 0.01, 0.05, 0.90
    
    def calculate_gas_matrices(self, standardized_residuals, conditional_variances, omega=None, A=None, B=None):
        """Calculer les matrices de corrélation et covariance GAS"""
        if omega is None:
            omega = self.omega
        if A is None:
            A = self.A
        if B is None:
            B = self.B
        
        epsilon = standardized_residuals.values
        T, n = epsilon.shape
        
        # Matrice de corrélation inconditionnelle
        R_bar = np.corrcoef(epsilon.T)
        R_bar = self.ensure_positive_definite(R_bar)
        
        # Initialisation des matrices
        R_t = np.zeros((T, n, n))
        H_t = np.zeros((T, n, n))
        
        # Initialiser R_0
        R_t[0] = R_bar.copy()
        
        for t in range(T):
            # Calculer la matrice de covariance H_t
            D_t = np.diag(np.sqrt(conditional_variances.iloc[t].values))
            H_t[t] = D_t @ R_t[t] @ D_t
            
            # Mettre à jour R_t pour la prochaine période
            if t < T - 1:
                try:
                    # Calcul du score
                    score_t = self.calculate_score(epsilon[t], R_t[t])
                    
                    # Mise à jour GAS
                    # R_t+1 = (1-A-B)*R_bar + A*score_t + B*R_t
                    R_t[t+1] = (1 - A - B) * R_bar + A * score_t + B * R_t[t]
                    
                    # S'assurer que R_t+1 est une matrice de corrélation valide
                    R_t[t+1] = self.ensure_positive_definite(R_t[t+1])
                    
                except Exception as e:
                    # En cas d'erreur, utiliser une mise à jour simple
                    R_t[t+1] = (1 - B) * R_bar + B * R_t[t]
                    R_t[t+1] = self.ensure_positive_definite(R_t[t+1])
        
        return R_t, H_t
    
    def fit_in_sample_gas(self):
        """Ajuster le modèle GAS sur les données in-sample et estimer les paramètres"""
        print("=== Ajustement GAS sur in-sample ===")
        
        # Ajuster ARMA-GJR-GARCH
        residuals, variances = self.fit_arma_gjr_garch_models(self.in_sample_data)
        
        # Estimer les paramètres GAS
        self.omega, self.A, self.B = self.estimate_gas_parameters(residuals)
        print(f"Paramètres GAS estimés: omega={self.omega}, A={self.A}, B={self.B}")
        
        # Calculer les matrices de corrélation et covariance
        R_t, H_t = self.calculate_gas_matrices(residuals, variances)
        
        # Stocker les résultats
        self.in_sample_results = {
            'R_t': R_t,
            'H_t': H_t,
            'standardized_residuals': residuals,
            'conditional_variances': variances,
            'omega': self.omega,
            'A': self.A,
            'B': self.B
        }
        
        return self.omega, self.A, self.B
    
    def calculate_out_sample_matrices(self):
        """Calculer les matrices de corrélation et covariance out-of-sample avec fenêtre glissante"""
        print("\n=== Calcul des matrices out-of-sample ===")
        
        if self.omega is None or self.A is None or self.B is None:
            raise ValueError("Paramètres GAS doivent être estimés d'abord avec fit_in_sample_gas()")
        
        out_sample_dates = self.out_sample_data.index
        
        for i, target_date in enumerate(out_sample_dates):
            if i % 20 == 0:
                print(f"Progrès: {i}/{len(out_sample_dates)} dates traitées")
            
            # Définir la fenêtre
            window_start = target_date - timedelta(days=self.window_size * 2)
            window_end = target_date - timedelta(days=1)
            
            # Extraire les données de la fenêtre
            window_data = self.all_data.loc[window_start:window_end]
            
            # S'assurer qu'on a assez de jours de trading
            if len(window_data) > self.window_size:
                window_data = window_data.iloc[-self.window_size:]
            elif len(window_data) < 50:  # Minimum raisonnable de jours
                print(f"Attention: seulement {len(window_data)} jours disponibles pour {target_date}")
                if len(window_data) == 0:
                    continue
            
            # Ajuster ARMA-GJR-GARCH sur cette fenêtre
            residuals, variances = self.fit_arma_gjr_garch_models(window_data)
            
            # Calculer les matrices avec les paramètres GAS fixes
            R_t, H_t = self.calculate_gas_matrices(
                residuals, 
                variances, 
                self.omega, 
                self.A, 
                self.B
            )
            
            # Stocker la dernière matrice (prévision pour target_date)
            last_corr_matrix = pd.DataFrame(
                R_t[-1], 
                columns=self.tickers, 
                index=self.tickers
            )
            last_cov_matrix = pd.DataFrame(
                H_t[-1], 
                columns=self.tickers, 
                index=self.tickers
            )
            
            self.out_sample_correlation_matrices[target_date] = last_corr_matrix
            self.out_sample_covariance_matrices[target_date] = last_cov_matrix
        
        print(f"Calcul terminé: {len(self.out_sample_correlation_matrices)} matrices calculées")
    
    def extract_weekly_matrices(self):
        """Extraire les matrices pour les dates hebdomadaires"""
        print("\n=== Extraction des matrices hebdomadaires ===")
        
        weekly_dates = self.out_sample_weekly_data.index
        
        for weekly_date in weekly_dates:
            # Trouver la date de trading précédente (normalement le vendredi)
            available_dates = [d for d in self.out_sample_correlation_matrices.keys() if d < weekly_date]
            
            if available_dates:
                closest_date = max(available_dates)
                self.weekly_correlation_matrices[weekly_date] = self.out_sample_correlation_matrices[closest_date]
                self.weekly_covariance_matrices[weekly_date] = self.out_sample_covariance_matrices[closest_date]
                print(f"Date hebdo {weekly_date}: utilisation de la matrice du {closest_date}")
            else:
                print(f"Attention: Pas de matrice disponible avant {weekly_date}")
        
        print(f"Matrices hebdomadaires extraites: {len(self.weekly_correlation_matrices)}")
    
    def run_full_pipeline(self):
        """Exécuter toute la pipeline GAS"""
        # 1. Ajuster sur in-sample et estimer les paramètres
        self.fit_in_sample_gas()
        
        # 2. Calculer les matrices out-of-sample
        self.calculate_out_sample_matrices()
        
        # 3. Extraire les matrices hebdomadaires
        self.extract_weekly_matrices()
        
        print("\n=== Pipeline GAS complète ===")
        print(f"Paramètres GAS: omega={self.omega}, A={self.A}, B={self.B}")
        print(f"Matrices out-of-sample: {len(self.out_sample_correlation_matrices)}")
        print(f"Matrices hebdomadaires: {len(self.weekly_correlation_matrices)}")
    
    def export_results(self, base_filename='gas_manual'):
        """Exporter tous les résultats avec les matrices de corrélation et covariance hebdomadaires"""
        # Export des matrices de corrélation
        with open(f'{base_filename}_correlation_all_weekly.pkl', 'wb') as f:
            pickle.dump({
                'matrices': self.weekly_correlation_matrices,
                'weekly_dates': list(self.weekly_correlation_matrices.keys()),
                'omega': self.omega,
                'A': self.A,
                'B': self.B
            }, f)
        
        # Export des matrices de covariance
        with open(f'{base_filename}_covariance_all_weekly.pkl', 'wb') as f:
            pickle.dump({
                'matrices': self.weekly_covariance_matrices,
                'weekly_dates': list(self.weekly_covariance_matrices.keys()),
                'omega': self.omega,
                'A': self.A,
                'B': self.B
            }, f)
        
        print(f"\nRésultats exportés:")
        print(f"- Matrices de corrélation hebdomadaires: {base_filename}_correlation_all_weekly.pkl")
        print(f"- Matrices de covariance hebdomadaires: {base_filename}_covariance_all_weekly.pkl")
    
    def get_weekly_correlation_matrix(self, date):
        """Obtenir une matrice de corrélation pour une date hebdomadaire"""
        if date not in self.weekly_correlation_matrices:
            raise ValueError(f"Matrice non disponible pour {date}")
        return self.weekly_correlation_matrices[date]
    
    def get_weekly_covariance_matrix(self, date):
        """Obtenir une matrice de covariance pour une date hebdomadaire"""
        if date not in self.weekly_covariance_matrices:
            raise ValueError(f"Matrice non disponible pour {date}")
        return self.weekly_covariance_matrices[date]
    
    def display_example_matrices(self):
        """Afficher quelques exemples de matrices"""
        print("\n=== Exemples de matrices ===")
        
        # In-sample
        if self.in_sample_results:
            last_in_sample_corr = pd.DataFrame(
                self.in_sample_results['R_t'][-1],
                columns=self.tickers,
                index=self.tickers
            )
            last_in_sample_cov = pd.DataFrame(
                self.in_sample_results['H_t'][-1],
                columns=self.tickers,
                index=self.tickers
            )
            print(f"\nDernière matrice de corrélation in-sample:")
            print(last_in_sample_corr.round(6))
            print(f"\nDernière matrice de covariance in-sample:")
            print(last_in_sample_cov.round(6))
        
        # Out-of-sample
        if self.out_sample_correlation_matrices:
            first_date = min(self.out_sample_correlation_matrices.keys())
            print(f"\nPremière matrice de corrélation out-of-sample ({first_date}):")
            print(self.out_sample_correlation_matrices[first_date].round(6))
            print(f"\nPremière matrice de covariance out-of-sample ({first_date}):")
            print(self.out_sample_covariance_matrices[first_date].round(6))
        
        # Hebdomadaires
        if self.weekly_correlation_matrices:
            first_weekly_date = min(self.weekly_correlation_matrices.keys())
            print(f"\nPremière matrice de corrélation hebdomadaire ({first_weekly_date}):")
            print(self.weekly_correlation_matrices[first_weekly_date].round(6))
            print(f"\nPremière matrice de covariance hebdomadaire ({first_weekly_date}):")
            print(self.weekly_covariance_matrices[first_weekly_date].round(6))

def main():
    """Fonction principale pour exécuter la pipeline GAS"""
    # Créer l'instance unique
    gas_portfolio = GAS_Portfolio(
        all_data=df_total_set_daily,
        in_sample_data=df_in_sample_set_daily,
        out_sample_data=df_out_sample_set_daily,
        out_sample_weekly_data=df_out_sample_set_weekly
    )
    
    # Exécuter toute la pipeline
    gas_portfolio.run_full_pipeline()
    
    # Afficher des exemples
    gas_portfolio.display_example_matrices()
    
    # Exporter les résultats
    gas_portfolio.export_results()

if __name__ == "__main__":
    main()