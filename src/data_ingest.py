import pandas as pd

class DataIngestion:
     
     """
      Class for data 
      ingesting advertising spending
      and sales data
      from a CSV file.
      
     """
     
     def __init__(self, file_path):
          
          """
           Initialize the class
           with file path.
           
           : param file_path: str, path to the CSV file.
           
          """
          self.file_path = file_path
          
     def load_data(self):
          """
           Load the data from the CSV file.
           :return: pandas DataFrame.
           
          """
          
          data = pd.read_csv(self.file_path)
          return data
     
     
     
     def Get_X_Y(self):
          
          """
           Get the features (X) and target (y) variables from the data.
           :return: tuple, features (X) and target (y) variables
           
          """
          
          data = self.load_data()
          X = data["TV"]
          y = data["Sales"]
          
          # combine them int single df
          
          df = pd.concat([X, y], axis=1)
          return X, y, df