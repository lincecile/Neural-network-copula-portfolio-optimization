import pandas as pd
import numpy as np
from arch import arch_model
from scipy.optimize import minimize
import warnings
import pickle
from datetime import timedelta
from clean_df_paper import df_total_set_daily, df_out_sample_set_daily, df_in_sample_set_daily, df_out_sample_set_weekly
warnings.filterwarnings("ignore")

class DCC_Portfolio:
    def __init__(self, all_data, in_sample_data, out_sample_data, out_sample_weekly_data):
        """
        Une classe unique pour gérer toute la pipeline DCC du papier
        
        Args:
            all_data: Toutes les données (in-sample + out-of-sample)
            in_sample_data: Données in-sample pour estimer les paramètres DCC
            out_sample_data: Données out-of-sample journalières
            out_sample_weekly_data: Dates hebdomadaires pour le rééquilibrage
        """
        self.all_data = all_data
        self.in_sample_data = in_sample_data
        self.out_sample_data = out_sample_data
        self.out_sample_weekly_data = out_sample_weekly_data
        
        self.tickers = list(self.all_data.columns)
        self.window_size = 250  # 1 an
        
        # Paramètres ARMA spécifiés dans le papier
        self.arma_params = {
            'SPY': (8, 8),
            'DIA': (10, 10), 
            'QQQ': (7, 7)
        }
        
        # Paramètres DCC estimés sur in-sample
        self.alpha = None
        self.beta = None
        
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
    
    def estimate_dcc_parameters(self, standardized_residuals):
        """Estimer alpha et beta par maximum de vraisemblance"""
        epsilon = standardized_residuals.values
        T, n = epsilon.shape
        
        def dcc_log_likelihood(params):
            alpha, beta = params
            
            if alpha <= 0 or beta <= 0 or alpha + beta >= 1:
                return 1e6
                
            Q_bar = np.corrcoef(epsilon.T)
            Q_t = np.zeros((T, n, n))
            Q_t[0] = Q_bar
            
            log_lik = 0
            
            for t in range(1, T):
                Q_t[t] = (1 - alpha - beta) * Q_bar + alpha * np.outer(epsilon[t-1], epsilon[t-1]) + beta * Q_t[t-1]
                
                Q_diag_inv = 1.0 / np.sqrt(np.diag(Q_t[t]))
                R_t = Q_t[t] * np.outer(Q_diag_inv, Q_diag_inv)
                np.fill_diagonal(R_t, 1.0)
                
                sign, logdet = np.linalg.slogdet(R_t)
                if sign <= 0:
                    return 1e6
                    
                log_lik += -0.5 * (logdet + epsilon[t] @ np.linalg.inv(R_t) @ epsilon[t] - epsilon[t] @ epsilon[t])
            
            return -log_lik
        
        bounds = [(0.001, 0.999), (0.001, 0.999)]
        constraints = [{'type': 'ineq', 'fun': lambda x: 0.999 - x[0] - x[1]}]
        
        result = minimize(
            dcc_log_likelihood,
            x0=[0.01, 0.95],
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            return [0.01, 0.95]
    
    def calculate_dcc_matrices(self, standardized_residuals, conditional_variances, alpha=None, beta=None):
        """Calculer les matrices de corrélation et covariance DCC pour un ensemble de données"""
        if alpha is None:
            alpha = self.alpha
        if beta is None:
            beta = self.beta
            
        epsilon = standardized_residuals.values
        T, n = epsilon.shape
        
        Q_bar = np.corrcoef(epsilon.T)
        Q_t = np.zeros((T, n, n))
        R_t = np.zeros((T, n, n))
        H_t = np.zeros((T, n, n))
        
        Q_t[0] = Q_bar.copy()
        
        for t in range(T):
            if t > 0:
                Q_t[t] = (1 - alpha - beta) * Q_bar + alpha * np.outer(epsilon[t-1], epsilon[t-1]) + beta * Q_t[t-1]
            
            # Calcul de la matrice de corrélation
            Q_diag = np.sqrt(np.diag(Q_t[t]))
            R_t[t] = Q_t[t] / np.outer(Q_diag, Q_diag)
            np.fill_diagonal(R_t[t], 1.0)
            
            # Calcul de la matrice de covariance
            D_t = np.diag(np.sqrt(conditional_variances.iloc[t].values))
            H_t[t] = D_t @ R_t[t] @ D_t
            
        return R_t, H_t
    
    def fit_in_sample_dcc(self):
        """Ajuster le modèle DCC sur les données in-sample et estimer les paramètres"""
        print("=== Ajustement DCC sur in-sample ===")
        
        # Ajuster ARMA-GJR-GARCH
        residuals, variances = self.fit_arma_gjr_garch_models(self.in_sample_data)
        
        # Estimer les paramètres DCC
        self.alpha, self.beta = self.estimate_dcc_parameters(residuals)
        print(f"Paramètres DCC estimés: alpha={self.alpha}, beta={self.beta}")
        
        # Calculer les matrices de corrélation et covariance
        R_t, H_t = self.calculate_dcc_matrices(residuals, variances)
        
        # Stocker les résultats
        self.in_sample_results = {
            'R_t': R_t,
            'H_t': H_t,
            'standardized_residuals': residuals,
            'conditional_variances': variances,
            'alpha': self.alpha,
            'beta': self.beta
        }
        
        return self.alpha, self.beta
    
    def calculate_out_sample_matrices(self):
        """Calculer les matrices de corrélation et covariance out-of-sample avec fenêtre glissante"""
        print("\n=== Calcul des matrices out-of-sample ===")
        
        if self.alpha is None or self.beta is None:
            raise ValueError("Paramètres DCC doivent être estimés d'abord avec fit_in_sample_dcc()")
        
        out_sample_dates = self.out_sample_data.index
        
        for i, target_date in enumerate(out_sample_dates):
            if i % 20 == 0:
                print(f"Progrès: {i}/{len(out_sample_dates)} dates traitées")
            
            # Définir la fenêtre
            window_start = target_date - timedelta(days=self.window_size * 2)
            window_end = target_date - timedelta(days=1)
            
            # Extraire les données de la fenêtre
            window_data = self.all_data.loc[window_start:window_end]
            
            # S'assurer qu'on a exactement 250 jours de trading
            if len(window_data) > self.window_size:
                window_data = window_data.iloc[-self.window_size:]
            elif len(window_data) < self.window_size:
                print(f"Attention: seulement {len(window_data)} jours disponibles pour {target_date}")
            
            # Ajuster ARMA-GJR-GARCH sur cette fenêtre
            residuals, variances = self.fit_arma_gjr_garch_models(window_data)
            
            # Calculer les matrices avec les paramètres DCC fixes
            R_t, H_t = self.calculate_dcc_matrices(residuals, variances, self.alpha, self.beta)
            
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
        """Exécuter toute la pipeline DCC"""
        # 1. Ajuster sur in-sample et estimer les paramètres
        self.fit_in_sample_dcc()
        
        # 2. Calculer les matrices out-of-sample
        self.calculate_out_sample_matrices()
        
        # 3. Extraire les matrices hebdomadaires
        self.extract_weekly_matrices()
        
        print("\n=== Pipeline DCC complète ===")
        print(f"Paramètres DCC: alpha={self.alpha}, beta={self.beta}")
        print(f"Matrices out-of-sample: {len(self.out_sample_correlation_matrices)}")
        print(f"Matrices hebdomadaires: {len(self.weekly_correlation_matrices)}")
    
    def export_results(self, base_filename='dcc'):
        """Exporter les résultats des matrices de corrélation et covariance"""
        # Export des matrices de corrélation
        with open(f'{base_filename}_correlation_all_weekly.pkl', 'wb') as f:
            pickle.dump({
                'matrices': self.weekly_correlation_matrices,
                'weekly_dates': list(self.weekly_correlation_matrices.keys()),
                'alpha': self.alpha,
                'beta': self.beta
            }, f)
        
        # Export des matrices de covariance
        with open(f'{base_filename}_covariance_all_weekly.pkl', 'wb') as f:
            pickle.dump({
                'matrices': self.weekly_covariance_matrices,
                'weekly_dates': list(self.weekly_covariance_matrices.keys()),
                'alpha': self.alpha,
                'beta': self.beta
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
    """Fonction principale pour exécuter la pipeline DCC"""
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

if __name__ == "__main__":
    main()