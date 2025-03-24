from assumptions_test import LinearRegressionAssumptions
from build_model import SimpleLinearRegression
from data_preprocessing import DataProcessing
from data_ingest import DataIngestion

if __name__ == "__main__":
     
     """
      Initialize Dataingetion class with file path
     """
     data_ingest = DataIngestion(r"C:\Users\mk744\OneDrive - Poornima University\Desktop\OLS Regression Challange\Data\advertising.csv")
     
     # Load the data and get the features and target variables
     
     X, y , df = data_ingest.Get_X_Y()
     df.to_csv("./Data/simple_df.csv", index=False)
     print(df)
     
     """
       Initialize the Dataprocessing class with the data
       
     """
     data_process = DataProcessing(df)
     data_process.identify_outliers(df['TV'])
     outliers = data_process.identify_outliers_zscore(df['TV'])
     print(outliers)
     
     # Train the model
     lr_model = SimpleLinearRegression(X, y)
     model = lr_model.summary()
     
     # Run Assumption Tests
     assumptions = LinearRegressionAssumptions(X, y, model, save_path="plots")
     assumptions.run_all_tests()
     