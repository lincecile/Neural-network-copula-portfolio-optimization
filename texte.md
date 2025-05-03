
# Intro

Presentation blabla

# Data

Prez data



# Benchmark


## ARMA

$$\hat{Y}_t = \hat{\varphi}_0 + \hat{\varphi}_1 Y_{t-1} + \hat{\varphi}_2 Y_{t-2} + \cdots + \hat{\varphi}_p Y_{t-p} + \hat{\varepsilon}_t - \hat{w}_1 \hat{\varepsilon}_{t-1} - \hat{w}_2 \hat{\varepsilon}_{t-2} - \cdots - \hat{w}_q \hat{\varepsilon}_{t-q}$$

- $Y_t$ : la variable expliquée au temps $t$
- $Y_{t-1}, Y_{t-2}, \cdots, Y_{t-p}$ : la variable expliquée retardée
- $\hat{\varphi}_0, \hat{\varphi}_1, \ldots, \hat{\varphi}_p$ : les coefficients à estimer
-  $\hat{\varepsilon}_t$ : le résidu
-  $\hat{\varepsilon}_{t-1}, \hat{\varepsilon}_{t-2}, \ldots, \hat{\varepsilon}_{t-q}$ : le résidu retardé
-  $\hat{w}_1, \hat{w}_2, \ldots, \hat{w}_q$ : les coefficients des résidus

Le benchmark donne des résultats peu concluant.

## STAR



# Modèles implémentés NN pour le forcast

## Multi-Layer Perceptron (MLP)

## Recurrent Neural Network (RNN) 

## Psi Sigma Network (PSN)



# Statistique

## MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} \left| y_t - \hat{y}_t \right|$$

- $y_t$ : valeur réelle à l'instant $t$
- $\hat{y}_t$ : valeur prédite à l'instant $t$
- $n$ : nombre total d'observations
- MAE mesure l'erreur moyenne absolue entre les valeurs prédites et réelles

## MAPE (Mean Absolute Percentage Error)

$$\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right|$$

- Exprimé en pourcentage
- Non défini quand $y_t = 0$
- Utile pour comparer l’erreur entre séries de différentes échelles

## RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{ \frac{1}{n} \sum_{t=1}^{n} \left( y_t - \hat{y}_t \right)^2 }
$$

- Plus sensible aux grandes erreurs que le MAE (car les erreurs sont élevées au carré)
- Pratique quand les erreurs importantes doivent être pénalisées davantage

## THEIL-U (Theil’s Inequality Coefficient)

$$U = \frac{ \sqrt{ \frac{1}{n} \sum_{t=1}^{n} \left( y_t - \hat{y}_t \right)^2 } }
{ \sqrt{ \frac{1}{n} \sum_{t=1}^{n} y_t^2 } + \sqrt{ \frac{1}{n} \sum_{t=1}^{n} \hat{y}_t^2 } }$$

- $U \in [0, 1]$ : plus c’est proche de 0, meilleure est la prévision
- $U = 0$ : prévision parfaite
- Compare la performance du modèle à une prévision naïve

# Test

## PT Pesaran et Timmermann (1992)

Le test de Pesaran et Timmermann (1992), noté PT, permet d’évaluer la capacité directionnelle d’un modèle de prévision, c’est-à-dire sa capacité à prédire correctement le sens des variations (hausse ou baisse) d’une série temporelle, comme les rendements d’un ETF.

- Hypothèse nulle ($H_0$) : le modèle n’a aucun pouvoir prédictif directionnel ; il ne prédit pas mieux que le hasard.
- Hypothèse alternative ($H_1$) : le modèle possède un pouvoir de prédiction directionnel significatif.
- Interprétation : si la statistique du test est significative, on peut conclure que le modèle est capable de prédire correctement la direction des rendements.

## DM Diebold et Mariano (1995)

Le test de Diebold et Mariano (1995), noté DM, est utilisé pour comparer la précision des prévisions entre deux modèles, en se basant sur une fonction de perte, généralement l’erreur quadratique moyenne (MSE).


- Hypothèse nulle ($H_0$) : les deux modèles ont une précision de prévision égale.
- Hypothèse alternative ($H_1$) : l’un des deux modèles fournit des prévisions significativement plus précises.
- Interprétation :
    - Une valeur négative de la statistique DM indique que le premier modèle (par exemple, PSN) est plus précis que le second.
    - Une statistique significative permet de conclure que cette différence de précision est statistiquement valable.
    



# Modèles implémentés Copule/CVaR pour la dépendance entre asset

## Copule

## CVaR

## DCC

## ADCC

## GAS


# Resultat
## Annualized return

$$R_{\text{annual}} = \left( \prod_{t=1}^{n} (1 + r_t) \right)^{\frac{1}{T}} - 1$$

- $r_t$ : rendement périodique (ex. quotidien, mensuel)
- $n$ : nombre total de périodes
- $T$ : nombre de périodes par an (ex: 252 pour les jours de bourse, 12 pour les mois)
- Mesure le rendement moyen annuel d'un investissement

## Sharpe ratio

$$\text{Sharpe} = \frac{\mathbb{E}[R_p - R_f]}{\sigma_p}$$

- $R_p$ : rendement du portefeuille
- $R_f$ : taux sans risque
- $\mathbb{E}[R_p - R_f]$ : rendement excédentaire moyen
- $\sigma_p$ : volatilité (écart-type) du rendement du portefeuille
- Mesure le rendement ajusté au risque
- Plus le ratio est élevé, meilleure est la performance ajustée au risque


## Maximum drawdown

$$\text{Max Drawdown} = \max_{t \in [0, T]} \left( \frac{\text{Peak}_t - \text{Trough}_t}{\text{Peak}_t} \right)$$

- $\text{Peak}_t$ : valeur maximale du portefeuille atteinte jusqu’à l’instant $t$
- $\text{Trough}_t$ : valeur minimale suivante après le pic
- Mesure la pire perte relative en capital à partir d’un sommet
- Indicateur clé du risque de baisse (downside risk)









# Résultat





