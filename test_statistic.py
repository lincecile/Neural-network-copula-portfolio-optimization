from statsmodels.stats.proportion import proportions_ztest
import numpy as np

def pesaran_timmermann_test(y_true, y_pred):
    actual_direction = np.sign(y_true)
    predicted_direction = np.sign(y_pred)

    correct_directions = (actual_direction == predicted_direction).astype(int)
    n_correct = correct_directions.sum()
    n = len(correct_directions)

    # H0: Le modèle ne prédit pas mieux que le hasard (p = 0.5)
    stat, p_value = proportions_ztest(count=n_correct, nobs=n, value=0.5)
    return stat, p_value


def diebold_mariano_test(y_true, y_pred_model, y_pred_benchmark, h=1):
    d = np.abs(y_true - y_pred_model) - np.abs(y_true - y_pred_benchmark)
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1)

    dm_stat = mean_d / np.sqrt(var_d / len(d))
    return dm_stat

