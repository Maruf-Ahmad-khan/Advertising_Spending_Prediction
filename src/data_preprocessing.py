import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DataProcessing:
     
     def __init__(self, df):
          self.df = df
          
     def identify_outliers(self, data:pd.DataFrame)->None:
          
          """
           This Function helps us to
           identify the outliers using 
           box plot visualization.
           
           parameters:
           data(pd.Series): The data to be visualized.
           
           Returns:
           None
           
          """
          
          fig, ax  = plt.subplots()
          ax.boxplot(data)
          ax.set_title("Box Plot of Data")
          ax.set_ylabel("Value")
          plt.show()
          
          
          
     def identify_outliers_zscore(self, data:pd.Series, threshold: float = 3):
          
          """
           parameters:
            data (pd.Series): The data to be analyzed.
            threshold (float) : The Z-Score threshold used
            to identify outliers.
            outliers are data points with a Z-Score greater
            than this threshold.
            Default value is 3.
            
            Returns:
            outliers (pd.Series): A Series of outliers in the data.
            
          """
          
          mean = np.mean(data)
          std = np.std(data)
          z_score = (data - mean) / std
          outliers =data[np.abs(z_score) > threshold]
          return outliers