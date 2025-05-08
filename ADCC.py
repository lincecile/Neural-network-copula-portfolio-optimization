import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from arch.univariate import GARCH
from scipy.optimize import minimize

from clean_df_paper import df_training_set_weekly, df_test_set_weekly, df_out_sample_set_weekly

class ADCCModel:
    def __init__(self, returns_df):
        """
        Initialise le modèle ADCC pour plusieurs séries de rendements
        
        Parameters:
        -----------
        returns_df : pandas DataFrame
            DataFrame contenant les rendements des actifs
        """
        self.returns = returns_df
        self.tickers = list(returns_df.columns)
        self.T, self.n = returns_df.shape
        
        if self.n < 2:
            raise ValueError("Le modèle ADCC nécessite au moins deux actifs")
        
        self.univariate_models = {}
        self.std_residuals = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        self.conditional_variances = pd.DataFrame(index=returns_df.index, columns=returns_df.columns)
        self.Q_t = np.zeros((self.T+1, self.n, self.n))
        self.R_t = np.zeros((self.T, self.n, self.n))
        self.H_t = np.zeros((self.T, self.n, self.n))
    
    def fit_univariate_garch(self, p=1, q=1):
        """
        Ajuste des modèles GARCH univariés pour chaque série
        """
        for ticker in self.tickers:
            print(f"Ajustement du modèle GARCH pour {ticker}")
            returns = self.returns[ticker].values
            
            model = arch_model(returns, vol='GARCH', p=p, q=q, mean='Zero')
            result = model.fit(disp='off')
            
            self.univariate_models[ticker] = result
            self.std_residuals[ticker] = result.resid / result.conditional_volatility
            self.conditional_variances[ticker] = result.conditional_volatility**2
        
        print("Modèles GARCH univariés ajustés avec succès")
    
    def _adcc_likelihood(self, params):
        """
        Fonction de vraisemblance pour l'estimation ADCC
        
        Parameters:
        -----------
        params : list ou array
            [a, b, gamma] où a, b, gamma sont les paramètres du modèle ADCC
        """
        a, b, gamma = params
        
        if a < 0 or b < 0 or gamma < -1 or gamma > 1 or a + b >= 0.999:
            return 1e10
        
        epsilon_t = self.std_residuals.values
        n = self.n
        T = self.T
        
        # Calculer la matrice de corrélation non conditionnelle
        Q_bar = np.corrcoef(epsilon_t.T)
        
        self.Q_t[0] = Q_bar.copy()
        
        log_likelihood = 0
        
        for t in range(T):
            eps_tm1 = epsilon_t[t-1] if t > 0 else np.zeros(n)
            
            # Séparer les résidus en parties positives et négatives
            epsilon_pos = np.maximum(eps_tm1, 0)
            epsilon_neg = np.maximum(-eps_tm1, 0)
            
            # Mettre à jour Q_t selon l'équation ADCC
            self.Q_t[t] = (1 - a - b) * Q_bar + a * np.outer(epsilon_pos, epsilon_pos) + b * np.outer(epsilon_neg, epsilon_neg) + gamma * (np.outer(epsilon_pos, epsilon_neg) + np.outer(epsilon_neg, epsilon_pos))
            
            # Calculer la matrice de corrélation conditionnelle
            Q_t_diag = np.diag(np.sqrt(np.diag(self.Q_t[t])))
            Q_t_diag_inv = np.linalg.inv(Q_t_diag)
            self.R_t[t] = Q_t_diag_inv @ self.Q_t[t] @ Q_t_diag_inv
            np.fill_diagonal(self.R_t[t], 1.0)
            
            # Calculer la matrice de covariance conditionnelle H_t
            D_t = np.diag(np.sqrt([self.conditional_variances.iloc[t, i] for i in range(n)]))
            self.H_t[t] = D_t @ self.R_t[t] @ D_t
            
            # Calculer la log-vraisemblance
            eps_t = epsilon_t[t]
            if not np.all(np.isnan(eps_t)):
                try:
                    H_t_det = np.linalg.det(self.H_t[t])
                    if H_t_det <= 0:
                        return 1e10
                    
                    H_t_inv = np.linalg.inv(self.H_t[t])
                    log_likelihood -= 0.5 * (np.log(H_t_det) + eps_t.T @ H_t_inv @ eps_t)
                except:
                    return 1e10
        
        return -log_likelihood
    
    def fit_adcc(self, a_start=0.1, b_start=0.8, gamma_start=0.0):
        """
        Ajuste le modèle ADCC
        """
        print("Ajustement du modèle ADCC...")
        
        initial_params = [a_start, b_start, gamma_start]
        bounds = [(0.001, 0.3), (0.5, 0.999), (-1, 1)]
        
        result = minimize(
            self._adcc_likelihood,
            initial_params,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        self.adcc_params = result.x
        self.adcc_loglikelihood = -result.fun
        
        print(f"Modèle ADCC ajusté avec succès : a = {self.adcc_params[0]:.4f}, b = {self.adcc_params[1]:.4f}, gamma = {self.adcc_params[2]:.4f}")
        
        # Calculer une dernière fois les matrices avec les paramètres optimaux
        _ = self._adcc_likelihood(self.adcc_params)
        
        return result
    
    def get_dynamic_correlations(self):
        """
        Renvoie les corrélations dynamiques entre les paires de tickers
        """
        pairs = []
        correlations = {}
        
        for i in range(self.n):
            for j in range(i+1, self.n):
                pair = f"{self.tickers[i]}-{self.tickers[j]}"
                pairs.append(pair)
                correlations[pair] = [self.R_t[t, i, j] for t in range(self.T)]
        
        return pd.DataFrame(correlations, index=self.returns.index)

    def plot_dynamic_correlations(self, figsize=(12, 8)):
        """
        Trace l'évolution des corrélations dynamiques
        """
        corr_df = self.get_dynamic_correlations()
        
        plt.figure(figsize=figsize)
        for column in corr_df.columns:
            plt.plot(corr_df.index, corr_df[column], label=column)
        
        plt.title("Corrélations Dynamiques Asymétriques")
        plt.xlabel("Date")
        plt.ylabel("Corrélation")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_conditional_volatilities(self, figsize=(12, 8)):
        """
        Trace l'évolution des volatilités conditionnelles
        """
        vol_df = np.sqrt(self.conditional_variances)
        
        plt.figure(figsize=figsize)
        for column in vol_df.columns:
            plt.plot(vol_df.index, vol_df[column], label=column)
        
        plt.title("Volatilités Conditionnelles")
        plt.xlabel("Date")
        plt.ylabel("Volatilité")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return plt.gcf()

# Exemple d'utilisation avec des données fictives
if __name__ == "__main__":

    # Exemple d'utilisation
    # df_training_set_daily = DataFrame de rendements pour plusieurs actifs (à définir)
    
    adcc_model = ADCCModel(df_training_set_weekly)
    adcc_model.fit_univariate_garch(p=1, q=1)
    adcc_model.fit_adcc()
    
    # Tracer les corrélations dynamiques
    adcc_model.plot_dynamic_correlations()
    plt.show()
    
    # Tracer les volatilités conditionnelles
    adcc_model.plot_conditional_volatilities()
    plt.show()
