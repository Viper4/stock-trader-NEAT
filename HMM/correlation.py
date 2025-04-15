import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif
from scipy.stats import chi2_contingency
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


class CorrelationAnalysis(object):
    def __init__(self, features):
        self.features = features

    def cramers_v(self, x, y):
        """
        Computes Cramér's V statistic for categorical correlation.
        """
        confusion_matrix = pd.crosstab(x, y)
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        r, k = confusion_matrix.shape
        return np.sqrt(chi2 / (n * (min(r, k) - 1)))

    def compute_feature_correlation(self, bars):
        """
        Computes correlation scores between features and the categorical 'regime'.
        Uses ANOVA F-test for numerical features, Mutual Information, and Cramér's V.
        """
        results = {}

        # Encode categorical regime as integers (Bear = 0, Choppy = 1, Bull = 2)
        bars["regime_encoded"] = bars["regime"].astype("category").cat.codes

        for feature in self.features:
            if bars[feature].dtype in [np.float64, np.int64]:  # Numerical Features
                # ANOVA F-test
                f_stat, _ = f_classif(bars[[feature]], bars["regime_encoded"])
                mi = mutual_info_classif(bars[[feature]], bars["regime_encoded"])[0]
                results[feature] = {"ANOVA F-Score": f_stat[0], "Mutual Info": mi}
            else:
                # Categorical Features
                cramers_v_score = self.cramers_v(bars[feature], bars["regime"])
                results[feature] = {"Cramér’s V": cramers_v_score}

        return pd.DataFrame(results).T

    def plot_correlation(self, correlation_matrix):
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", robust=True, yticklabels=True, xticklabels=True)
        plt.title("Feature Correlation with Market Regime")
        plt.show()
