import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro, probplot

class LinearRegressionAssumptions:
    def __init__(self, X, y, model, save_path="plots"):
        """
        Initialize with features (X), target (y), trained regression model, and save path.
        """
        self.X = X
        self.y = y
        self.model = model
        self.residuals = self.model.resid  # Get residuals from model
        self.save_path = save_path

        # Create directory for plots if it does not exist
        os.makedirs(self.save_path, exist_ok=True)

    def check_linearity(self):
        """Check linearity by plotting predicted vs actual values."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.model.fittedvalues, self.y, color="blue", alpha=0.6)
        plt.xlabel("Predicted Values")
        plt.ylabel("Actual Values")
        plt.title("Linearity Check: Predicted vs Actual")
        plt.axline((0, 0), slope=1, color='red', linestyle="dashed")  # y=x line
        plt.savefig(f"{self.save_path}/linearity_check.png")  # Save figure
        plt.show()

    def check_normality(self):
        """Check if residuals are normally distributed using histogram & QQ plot."""
        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(self.residuals, bins=30, kde=True)
        plt.title("Histogram of Residuals")
        plt.xlabel("Residuals")

        # QQ Plot
        plt.subplot(1, 2, 2)
        probplot(self.residuals, dist="norm", plot=plt)
        plt.title("QQ Plot for Normality Check")

        plt.savefig(f"{self.save_path}/normality_check.png")  # Save figure
        plt.show()

        # Shapiro-Wilk Test
        stat, p_value = shapiro(self.residuals)
        print(f"Shapiro-Wilk Test p-value: {p_value}")
        if p_value > 0.05:
            print("Residuals are normally distributed (Fail to reject H0).")
        else:
            print("Residuals are NOT normally distributed (Reject H0).")

    def check_multicollinearity(self):
        """Check multicollinearity using Variance Inflation Factor (VIF)."""
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X_with_const = sm.add_constant(self.X)

        vif_data = pd.DataFrame()
        vif_data["Feature"] = X_with_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]

        print("\nVariance Inflation Factor (VIF) Table:")
        print(vif_data)

        # Save the VIF table as a CSV file
        vif_data.to_csv(f"{self.save_path}/vif_table.csv", index=False)

    def check_homoscedasticity(self):
        """Check homoscedasticity using a residuals vs predicted scatter plot."""
        plt.figure(figsize=(8, 6))
        plt.scatter(self.model.fittedvalues, self.residuals, color="blue", alpha=0.6)
        plt.axhline(y=0, color="red", linestyle="dashed")
        plt.xlabel("Fitted Values")
        plt.ylabel("Residuals")
        plt.title("Homoscedasticity Check: Residuals vs Fitted Values")
        plt.savefig(f"{self.save_path}/homoscedasticity_check.png")  # Save figure
        plt.show()

    def check_autocorrelation(self):
        """Check independence of residuals using the Durbin-Watson test."""
        from statsmodels.stats.stattools import durbin_watson

        dw_stat = durbin_watson(self.residuals)
        print(f"Durbin-Watson Statistic: {dw_stat}")
        if 1.5 < dw_stat < 2.5:
            print("No significant autocorrelation detected.")
        else:
            print("Autocorrelation may be present in residuals!")

    def run_all_tests(self):
        """Run all assumption tests for Linear Regression."""
        print("\nChecking Linearity:")
        self.check_linearity()

        print("\nChecking Normality:")
        self.check_normality()

        print("\nChecking Multicollinearity:")
        self.check_multicollinearity()

        print("\nChecking Homoscedasticity:")
        self.check_homoscedasticity()

        print("\nChecking Autocorrelation:")
        self.check_autocorrelation()
