import pandas as pd
import statsmodels.api as sm
import numpy as np

class LinearRegressionSM:
    def __init__(self, left_hand_side: pd.DataFrame, right_hand_side: pd.DataFrame):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        X = sm.add_constant(self.right_hand_side)
        self._model = sm.OLS(self.left_hand_side, X).fit()

    def get_params(self):
        beta_coefficients = pd.Series(self._model.params, name='Beta coefficients')
        return beta_coefficients

    def get_pvalues(self):
        p_values = pd.Series(self._model.pvalues, name='P-values for the corresponding coefficients')
        return p_values

    def get_wald_test_result(self, restriction_matrix):
        wald_test = self._model.wald_test(restriction_matrix)
        f_value = wald_test.statistic
        p_value = wald_test.pvalue

        result = f"F-value: {f_value:.3f}, p-value: {p_value:.3f}"
        return result

    def get_model_goodness_values(self):
        adjusted_r_squared = self._model.rsquared_adj
        aic = self._model.aic
        bic = self._model.bic

        result = f"Adjusted R-squared: {adjusted_r_squared:.3f}, Akaike IC: {aic:.3f}, Bayes IC: {bic:.3f}"
        return result
class LinearRegressionNP():
    pass

import pandas as pd
import pathlib
from typing import List, Union, Any
import numpy as np
import numpy.linalg as la
from scipy.stats import t, f

df_sp500 = pd.read_parquet('Macintosh HD/Felhasználók/csujadaniel/Dokumentumok/GitHub/ECOPY_23241/data/sp500.parquet', engine='fastparquet')
df_factors = pd.read_parquet('Macintosh HD/Felhasználók/csujadaniel/Dokumentumok/GitHub/ECOPY_23241/data/ff_factors.parquet', engine='fastparquet')

merged_df = pd.merge(df_sp500, df_factors, on='Date', how='left')

merged_df['Excess Return'] = merged_df['Monthly Returns'] - merged_df['RF']

merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)

merged_df.dropna(subset=['ex_ret_1', 'HML'], inplace=True)


class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None
        self.coefficients = None
        self.values = None
        self.standard_errors = None

    import numpy.linalg as la

    def fgls_regression(left_hand_side, right_hand_side):

        X = right_hand_side
        y = left_hand_side
        X_with_const = np.hstack([np.ones((X.shape[0], 1)), X.values])
        ols_coefficients = la.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y.values

        residuals = y.values - X_with_const @ ols_coefficients

        squared_residuals = residuals ** 2

        y_new = np.log(squared_residuals)
        new_model_coefficients = la.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y_new

        predicted_errors = np.exp(X_with_const @ new_model_coefficients)

        V_inv = np.diag(1 / predicted_errors)

        gls_coefficients = la.inv(X_with_const.T @ V_inv @ X_with_const) @ X_with_const.T @ V_inv @ y.values

        return pd.Series(gls_coefficients, index=["Intercept"] + right_hand_side.columns.tolist())

    def get_params(self):
            return pd.Series(self.coefficients, name="Beta coefficients")

    def get_pvalues(self):

        t_statistics = self.coefficients / self.standard_errors

        p_values = [min(t.cdf(t_stat, df=len(self.left_hand_side) - len(self.coefficients)),
                        1 - t.cdf(t_stat, df=len(self.left_hand_side) - len(self.coefficients))) * 2
                    for t_stat in t_statistics]

        return pd.Series(p_values, index=["P-values for the corresponding coefficients"])

    def get_wald_test_result(self, R):
        R = np.array(R)
        beta = self.coefficients
        X_with_const = np.hstack([np.ones((self.right_hand_side.shape[0], 1)), self.right_hand_side.values])
        V_inv = np.linalg.inv(X_with_const.T @ X_with_const)
        wald_stat = (R @ beta).T @ np.linalg.inv(R @ V_inv @ R.T) @ (R @ beta)

        k = R.shape[0]
        F_stat = wald_stat / k
        df1 = k
        df2 = self.right_hand_side.shape[0] - np.linalg.matrix_rank(X_with_const)
        p_value = f.sf(F_stat, df1, df2)

        return f"Wald: {wald_stat:.3f}, p-value: {p_value:.3f}"

    def get_model_goodness_values(self):
        y = self.left_hand_side.values
        X = self.right_hand_side.values
        X_with_const = np.hstack([np.ones((X.shape[0], 1)), X])
        y_hat = X_with_const @ self.coefficients

        TSS = np.sum((y - np.mean(y)) ** 2)

        SSE = np.sum((y - y_hat) ** 2)

        R_squared = 1 - (SSE / TSS)

        n = len(y)
        p = X_with_const.shape[1]
        Adjusted_R_squared = 1 - ((SSE / (n - p)) / (TSS / (n - 1)))

        return f"Centered R-squared: {R_squared:.3f}, Adjusted R-squared: {Adjusted_R_squared:.3f}"












