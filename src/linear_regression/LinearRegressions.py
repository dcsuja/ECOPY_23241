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



