from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import itertools
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from clean_df_paper import df_test_set_daily, df_training_set_daily


class ARMAOptimizer:
    """
    Classe pour optimiser les paramètres ARMA en utilisant différents critères (AIC, BIC, RMSE)
    """
    def __init__(self, p_max=10, q_max=10):
        self.p_max = p_max
        self.q_max = q_max
        self.paper_params = {
            'SPY US Equity': (8, 8),
            'DIA US Equity': (10, 10),
            'QQQ US Equity': (7, 7)
        }

    def grid_search(self, series, criterion='aic'):
        """
        Réalise une recherche en grille des meilleurs paramètres ARMA sur les données d'entraînement

        Args:
            series: Série temporelle d'entraînement à modéliser
            criterion: Critère d'optimisation ('aic', 'bic', ou 'rmse')

        Returns:
            tuple: (meilleurs paramètres, meilleur modèle)
        """
        best_value = float("inf")
        best_params = None
        best_model = None

        # Création de toutes les combinaisons de p et q
        p_range = range(0, self.p_max + 1)
        q_range = range(0, self.q_max + 1)

        for p, q in itertools.product(p_range, q_range):
            if p == 0 and q == 0:
                continue

            try:
                model = ARIMA(series, order=(p, 0, q))
                results = model.fit()

                # Sélectionner la valeur à optimiser selon le critère
                if criterion.lower() == 'aic':
                    value = results.aic
                elif criterion.lower() == 'bic':
                    value = results.bic
                elif criterion.lower() == 'rmse':
                    # Calculer le RMSE sur les données d'entraînement (fitted values)
                    predictions = results.fittedvalues
                    value = np.sqrt(mean_squared_error(series, predictions))
                else:
                    raise ValueError(f"Critère non reconnu: {criterion}")

                # Enregistrer le modèle avec la meilleure valeur
                if value < best_value:
                    best_value = value
                    best_params = (p, q)
                    best_model = results

                print(f"ARMA({p},{q}) - {criterion.upper()}: {value:.4f}")

            except Exception as e:
                continue

        print(f"\nMeilleurs paramètres {criterion.upper()} (sur l'entraînement): ARMA{best_params} avec {criterion.upper()} = {best_value:.4f}")
        return best_params, best_model

    def evaluate_model(self, train_data, test_data, p, q, etf_name):
        """
        Évalue un modèle ARMA sur les données de test avec des paramètres donnés
        """
        # Entraîner le modèle sur les données d'entraînement
        model = ARIMA(train_data, order=(p, 0, q))
        model_fit = model.fit()

        # Prédire sur les données de test
        predictions = model_fit.forecast(steps=len(test_data))

        # Calculer les métriques d'erreur
        mse = mean_squared_error(test_data, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(test_data, predictions)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100

        print(f"Performance du modèle ARMA({p},{q}) pour {etf_name} sur les données de test:")
        print(f"MAE: {mae:.6f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"RMSE: {rmse:.6f}")

        return model_fit, predictions, {'rmse': rmse, 'mae': mae, 'mape': mape}

    def find_best_models(self, etfs, criteria=['aic', 'bic', 'rmse']):
        """
        Trouve les meilleurs modèles pour chaque ETF selon différents critères en utilisant les données d'entraînement
        """
        results = {}

        for etf in etfs:
            results[etf] = {}
            series = df_training_set_daily[etf] # Utiliser les données d'entraînement

            # Recherche pour chaque critère
            for criterion in criteria:
                print(f"\nRecherche des meilleurs paramètres ARMA pour {etf} ({criterion.upper()} sur l'entraînement):")
                best_params, best_model = self.grid_search(series, criterion)
                results[etf][criterion] = {
                    'params': best_params,
                    'model': best_model
                }

            # Comparer avec les paramètres de l'article
            if etf in self.paper_params:
                print(f"\n{etf}: Paramètres de l'article = ARMA{self.paper_params[etf]}")

            # Afficher un résumé
            print(f"\nRésumé pour {etf} (recherche sur l'entraînement):")
            for criterion in criteria:
                print(f"  {criterion.upper()}: Meilleurs paramètres = ARMA{results[etf][criterion]['params']}")

        return results

    def evaluate_all_models(self, etfs, best_models):
        """
        Évalue les modèles de l'article et les meilleurs modèles trouvés sur les données de test
        """
        evaluation_results = {}

        for etf in etfs:
            evaluation_results[etf] = {}
            train_data = df_training_set_daily[etf]
            test_data = df_test_set_daily[etf]

            # Évaluer le modèle de l'article
            if etf in self.paper_params:
                p, q = self.paper_params[etf]
                print(f"\nÉvaluation du modèle de l'article pour {etf} (ARMA{p, q}) sur les données de test:")
                _, _, metrics = self.evaluate_model(train_data, test_data, p, q, etf)
                evaluation_results[etf]['paper'] = {
                    'params': (p, q),
                    'metrics': metrics
                }

            # Évaluer le meilleur modèle trouvé par RMSE sur l'entraînement
            if 'rmse' in best_models[etf]:
                best_params_rmse_train = best_models[etf]['rmse']['params']
                p_best_rmse, q_best_rmse = best_params_rmse_train
                print(f"\nÉvaluation du meilleur modèle RMSE (trouvé sur l'entraînement) pour {etf} (ARMA{p_best_rmse, q_best_rmse}) sur les données de test:")
                _, _, metrics = self.evaluate_model(train_data, test_data, p_best_rmse, q_best_rmse, etf)
                evaluation_results[etf]['best_rmse_train'] = {
                    'params': (p_best_rmse, q_best_rmse),
                    'metrics': metrics
                }

        return evaluation_results


def main():
    """Fonction principale pour exécuter l'optimisation ARMA"""
    # Initialiser l'optimiseur
    optimizer = ARMAOptimizer(p_max=10, q_max=10)

    # Liste des ETFs
    etfs = df_test_set_daily.columns

    # 1. Trouver les meilleurs modèles selon différents critères en utilisant les données d'entraînement
    print("=" * 50)
    print("RECHERCHE DES MEILLEURS PARAMÈTRES (SUR LES DONNÉES D'ENTRAÎNEMENT)")
    print("=" * 50)
    best_models = optimizer.find_best_models(etfs, criteria=['aic', 'bic', 'rmse'])

    # 2. Évaluer les modèles sur les données de test
    print("\n" + "=" * 50)
    print("ÉVALUATION DES MODÈLES (SUR LES DONNÉES DE TEST)")
    print("=" * 50)
    evaluation_results = optimizer.evaluate_all_models(etfs, best_models)

    # 3. Afficher un résumé des résultats
    print("\n" + "=" * 50)
    print("RÉSUMÉ DES RÉSULTATS")
    print("=" * 50)

    # Liste unique des ETFs pour éviter les duplications
    unique_etfs = list(dict.fromkeys(etfs))

    for etf in unique_etfs:
        print(f"\nRésultats pour {etf}:")
        
        # Afficher les paramètres de l'article s'ils existent
        if etf in optimizer.paper_params:
            print(f"  Paramètres de l'article: {optimizer.paper_params[etf]}")
        
        # Afficher les meilleurs paramètres pour chaque critère
        if etf in best_models:
            for criterion in ['aic', 'bic', 'rmse']:
                if criterion in best_models[etf]:
                    print(f"  Meilleurs paramètres {criterion.upper()} (sur l'entraînement): {best_models[etf][criterion]['params']}")
        
        # Afficher la performance RMSE
        if etf in evaluation_results and 'best_rmse_train' in evaluation_results[etf]:
            rmse = evaluation_results[etf]['best_rmse_train']['metrics']['rmse']
            print(f"  Performance du meilleur modèle RMSE (entraînement) sur le test: {rmse:.6f}")
        
        print()  # Ligne vide entre chaque ETF

if __name__ == "__main__":
    main()