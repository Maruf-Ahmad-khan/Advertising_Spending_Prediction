import pandas as pd
import statsmodels.api as sm

class SimpleLinearRegression:
     
     def __init__(self, x, y):
          """_summary_

          Args:
              x (pd.Series): Independent variable
              y (pd.Series): dependent variable
              
          """
          
          self.x = x
          self.y = y
          self.x = sm.add_constant(self.x)
          
     def fit(self):
          
          """
            Fit the linear regression model using the 
            independent and dependent variables.
            
            Returns:
            model (sm.OLS) FIt the linear regression model.
            
          """
          model = sm.OLS(self.y, self.x).fit()
          return model
     
     
     def summary(self):
          
          """
           Print the summary of the LR
           Returns:
           None
           
          """
          model = self.fit()
          print(model.summary())
          return model
     
          