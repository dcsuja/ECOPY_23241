import math

import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.tools import add_constant


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
import statsmodels.api as sm
import pandas as pd
import numpy as np
from scipy import stats


class LinearRegressionSM:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None

    def fit(self):
        alfa = sm.add_constant(self.right_hand_side)
        beta_0 = self.left_hand_side
        self._model = sm.OLS(beta_0, alfa).fit()

    def get_params(self):
        return self._model.params.rename('Beta coefficients')

    def get_pvalues(self):
        return self._model.pvalues.rename('P-values for the corresponding coefficients')

    def get_wald_test_result(self, restrictions):
        wald_test = self._model.wald_test(restrictions)
        fvalue = float(wald_test.statistic)
        pvalue = float(wald_test.pvalue)
        return f'F-value: {fvalue:.3}, p-value: {pvalue:.3}'

    def get_model_goodness_values(self):
        ars = self._model.rsquared_adj
        ak = self._model.aic
        by = self._model.bic
        return f'Adjusted R-squared: {ars:.3}, Akaike IC: {ak:.3}, Bayes IC: {by:.3}'


class LinearRegressionNP:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side

    def fit(self):
        self.right_hand_side = pd.concat([pd.Series(1, index=self.right_hand_side.index, name='Constant'),
                                         self.right_hand_side], axis=1)
        self.X = self.right_hand_side.to_numpy()
        self.y = self.left_hand_side.to_numpy()
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
        self.beta_coefficients = pd.Series(self.beta, index=self.right_hand_side.columns, name='Beta coefficients')

    def get_params(self):
        return self.beta_coefficients

    def get_pvalues(self):
        n, K = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        SSE = ((self.y - self.X @ self.beta) ** 2).sum() / (n - K)
        XTX_inv = np.linalg.inv(self.X.T @ self.X)
        t_stats = self.beta / np.sqrt(SSE * np.diag(XTX_inv))
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - K))
        return pd.Series(p_values, index=self.right_hand_side.columns,
                         name='P-values for the corresponding coefficients')

    def get_wald_test_result(self, R):
        RES = self.y - self.X @ self.beta
        R_M = np.array(R)
        R = R_M @ self.beta
        n = len(self.left_hand_side)
        M, K = R_M.shape
        Sigma2 = np.sum(RES ** 2) / (n-K)
        H = R_M @ np.linalg.inv(self.X.T @ self.X) @ R_M.T
        wald_value = (R.T @ np.linalg.inv(H) @ R) / (M*Sigma2)
        p_value = 1 - stats.f.cdf(wald_value, dfn=M, dfd=n-K)
        return f"Wald: {wald_value:.3f}, p-value: {p_value:.3f}"

    def get_model_goodness_values(self) -> str:
        n, K = self.right_hand_side.shape[0], self.right_hand_side.shape[1]
        SSE = ((self.y - self.X @ self.beta) ** 2).sum()
        SST = ((self.y - self.y.mean()) ** 2).sum()
        SSR = SST - SSE
        crs = SSR / SST
        ars = 1 - (1 - crs) * ((n - 1) / (n - K))
        return f"Centered R-squared: {crs:.3f}, Adjusted R-squared: {ars:.3f}"

import pandas as pd
import pathlib
from typing import List, Union, Any
import numpy as np
import numpy.linalg as la
from scipy.stats import t, f


sp500_df = pd.read_parquet('../../data/sp500.parquet', engine='fastparquet')
ff_factors_df = pd.read_parquet('../../data/ff_factors.parquet', engine='fastparquet')
merged_df = pd.merge(sp500_df, ff_factors_df, on='Date', how='left')
merged_df['Excess Return'] = merged_df['Monthly Returns'] - merged_df['RF']
merged_df = merged_df.sort_values(by='Date')
merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)
merged_df = merged_df.dropna(subset=['ex_ret_1'])
merged_df = merged_df.dropna(subset=['HML'])
amazon_df = merged_df[merged_df['Symbol'] == 'AMZN']
amazon_df = amazon_df.drop(columns=['Symbol'])

class LinearRegressionGLS:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self._model = None
        self.coefficients = None
        self.values = None
        self.standard_errors = None

    import numpy.linalg as la

    def fit(self):
        X = self.right_hand_side
        X_with_const = np.hstack([np.ones((X.shape[0], 1)), X.values])

        y = self.left_hand_side.values

        beta_ols = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y

        residuals = y - X_with_const @ beta_ols
        sigma_squared = np.var(residuals, ddof=X_with_const.shape[1])

        W_inv = np.diag([1 / sigma_squared] * len(y))

        beta_fgls = np.linalg.inv(X_with_const.T @ W_inv @ X_with_const) @ X_with_const.T @ W_inv @ y

        self.coefficients = beta_fgls
        return self.coefficients

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


sp500_df = pd.read_parquet('../../data/sp500.parquet', engine='fastparquet')
ff_factors_df = pd.read_parquet('../../data/ff_factors.parquet', engine='fastparquet')
merged_df = pd.merge(sp500_df, ff_factors_df, on='Date', how='left')
merged_df['Excess Return'] = merged_df['Monthly Returns'] - merged_df['RF']
merged_df = merged_df.sort_values(by='Date')
merged_df['ex_ret_1'] = merged_df.groupby('Symbol')['Excess Return'].shift(-1)
merged_df = merged_df.dropna(subset=['ex_ret_1'])
merged_df = merged_df.dropna(subset=['ex_ret_1', 'HML'])
amazon_df = merged_df[merged_df['Symbol'] == 'AMZN']
amazon_df = amazon_df.drop(columns=['Symbol'])

from pathlib import Path
import pandas as pd
import typing
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.optimize import minimize
from statsmodels.tools.tools import add_constant

class LinearRegressionML:
    def __init__(self, left_hand_side, right_hand_side):
        self.left_hand_side = left_hand_side
        self.right_hand_side = right_hand_side
        self.right_hand_side = add_constant(self.right_hand_side)
        self._params = None

    def fit(self):
        def neg_log_likelihood(params, X, y):
            predicted = np.dot(X, params)
            log_likelihood = -0.5 * (np.log(2 * np.pi * np.var(y)) + np.sum((y - predicted) ** 2) / np.var(y))
            return -log_likelihood

        initial_params = np.zeros(self.right_hand_side.shape[1]) + 0.1
        try:
            result = minimize(neg_log_likelihood, initial_params, args=(self.right_hand_side, self.left_hand_side),
                              method='L-BFGS-B')
            self._params = result.x
        except Exception as e:
            raise ValueError(f"MLE fit failed: {e}")

    def get_params(self):
            return pd.Series(self._params, index=self.right_hand_side.columns, name='Beta coefficients')

    def get_pvalues(self):
        def neg_log_likelihood(params, X, y):
            predicted = np.dot(X, params)
            log_likelihood = -0.5 * (np.log(2 * np.pi * np.var(y)) + np.sum((y - predicted) ** 2) / np.var(y))
            return -log_likelihood

        full_model_ll = neg_log_likelihood(self._params, self.right_hand_side, self.left_hand_side)

        p_values = []
        for i in range(len(self._params)):
            params_restricted = self._params.copy()
            params_restricted[i] = 0

            restricted_model_ll = neg_log_likelihood(params_restricted, self.right_hand_side, self.left_hand_side)

            lr_stat = 2 * (full_model_ll - restricted_model_ll)
            p_value = 1 - chi2.cdf(lr_stat, df=1)
            p_values.append(p_value)

        return pd.Series(p_values, index=self.right_hand_side.columns,
                         name='P-values for the corresponding coefficients')

    def get_model_goodness_values(self):
            residuals = self.left_hand_side - np.dot(self.right_hand_side, self._params)
            SSE = np.sum(residuals ** 2)
            SST = np.sum((self.left_hand_side - np.mean(self.left_hand_side)) ** 2)
            r_squared = 1 - (SSE / SST)
            adj_r_squared = 1 - (1 - r_squared) * (len(self.left_hand_side) - 1) / (len(self.left_hand_side) - len(self.right_hand_side.columns))
            return f'Centered R-squared: {r_squared:.3f}, Adjusted R-squared: {adj_r_squared:.3f}'










