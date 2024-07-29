# Import necessary libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# Custom Scaler class to standardize selected columns
class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_ = None
        self.var_ = None

    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]

# Class to handle the absenteeism model
class absenteeism_model():
      
    def __init__(self, model_file, scaler_file):
        # Read the 'model' and 'scaler' files which were saved
        with open('model','rb') as model_file, open('scaler', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None
        
    # Method to load and clean the data
    def load_and_clean_data(self, data_file):
        # Import the data
        df = pd.read_csv(data_file, delimiter=',')
        # Store the data in a new variable for later use
        self.df_with_predictions = df.copy()
        # Drop the 'ID' column as it's not needed for prediction
        df = df.drop(['ID'], axis=1)
        # Add a placeholder column for 'Absenteeism Time in Hours'
        df['Absenteeism Time in Hours'] = 'NaN'

        # Create dummy variables for 'Reason for Absence'
        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first=True)
        
        # Split reason_columns into 4 types and convert to integer
        reason_type_1 = reason_columns.loc[:,1:14].max(axis=1).fillna(0).astype(int)
        reason_type_2 = reason_columns.loc[:,15:17].max(axis=1).fillna(0).astype(int)
        reason_type_3 = reason_columns.loc[:,18:21].max(axis=1).fillna(0).astype(int)
        reason_type_4 = reason_columns.loc[:,22:].max(axis=1).fillna(0).astype(int)
        
        # Drop the 'Reason for Absence' column from df to avoid multicollinearity
        df = df.drop(['Reason for Absence'], axis=1)
        
        # Concatenate df and the 4 types of reason for absence
        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)
        
        # Assign names to the 4 reason type columns
        column_names = ['Date', 'Transportation Expense', 'Distance to Work', 'Age',
                       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                       'Pets', 'Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
        df.columns = column_names

        # Re-order the columns in df
        column_names_reordered = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Date', 'Transportation Expense', 
                                  'Distance to Work', 'Age', 'Daily Work Load Average', 'Body Mass Index', 'Education', 
                                  'Children', 'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_reordered]
  
        # Convert the 'Date' column into datetime format
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        # Create a list with month values retrieved from the 'Date' column
        list_months = []
        for i in range(df.shape[0]):
            list_months.append(df['Date'][i].month)

        # Insert the month values in a new column in df, called 'Month Value'
        df['Month Value'] = list_months

        # Create a new feature called 'Day of the Week' based on the 'Date' column
        df['Day of the Week'] = df['Date'].apply(lambda x: x.weekday())

        # Drop the 'Date' column from df as it's no longer needed
        df = df.drop(['Date'], axis=1)

        # Re-order the columns in df
        column_names_upd = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Month Value', 'Day of the Week',
                            'Transportation Expense', 'Distance to Work', 'Age',
                            'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
                            'Pets', 'Absenteeism Time in Hours']
        df = df[column_names_upd]

        # Map 'Education' variables to binary values
        df['Education'] = df['Education'].map({1:0, 2:1, 3:1, 4:1})

        # Replace any NaN values with 0
        df = df.fillna(value=0)

        # Drop the original absenteeism time as it's not needed for prediction
        df = df.drop(['Absenteeism Time in Hours'], axis=1)
        
        # Drop variables deemed unnecessary
        df = df.drop(['Day of the Week', 'Daily Work Load Average', 'Distance to Work'], axis=1)
        
        # Save the preprocessed data for future use
        self.preprocessed_data = df.copy()
        
        # Standardize the data
        self.data = self.scaler.transform(df)
    
    # Method to output the probability of a data point being 1 (absenteeism)
    def predicted_probability(self):
        if self.data is not None:  
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred
    
    # Method to output the predicted category (0 or 1) based on the model
    def predicted_output_category(self):
        if self.data is not None:
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs
    
    # Method to predict outputs and probabilities and add these values to the preprocessed data
    def predicted_outputs(self):
        if self.data is not None:
            self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg.predict(self.data)
            return self.preprocessed_data

