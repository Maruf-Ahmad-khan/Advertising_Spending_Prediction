﻿# Advertising_Spending_Prediction

 # Report on Simple Linear Regression Model  

The SimpleLinearRegression class implements
Ordinary Least Squares (OLS) regression using statsmodels.
It takes an independent variable (`x`) and a dependent variable (`y`), adding a constant term for the intercept. The fit() method builds and trains the regression model, while the summary() method provides a detailed statistical report, including coefficients, p-values, R-squared, and residual diagnostics.
This class serves as a fundamental tool for analyzing relationships between variables and making predictions based on linear regression assumptions.


# Report on Linear Regression Assumptions Check 

The LinearRegressionAssumptions class automates the evaluation of key linear regression assumptions to ensure model reliability.
It assesses linearity by comparing predicted vs. actual values, and normality using histograms, QQ plots, and the Shapiro-Wilk test.
Multicollinearity is checked via the Variance Inflation Factor (VIF), while homoscedasticity is validated through residual plots. 
The Durbin-Watson test is used to detect autocorrelation in residuals.
Additionally, the class saves diagnostic plots and reports, aiding in model validation and interpretation.
