import numpy as np
import pandas as pd
import streamlit as strl
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Define important constants derived from training
theta_min = [-1.48229680e-03, 1.14649633e-03,  4.95730192e-01, 7.60458522e-03, -3.34079900e-03,
       -3.79918705e-03,  1.47173698e-03,  2.72785870e-04, -3.76054851e-04,
        9.87797779e-04, 7.26770843e-04,  5.89161810e-03,
       -3.53857728e-03, -5.71078344e-04, -2.79111386e-03,  4.27502416e-03]

theta_max = [-5.27122756e-04, -1.79857585e-03,  4.99074714e-01, -1.23218404e-03, -2.30580098e-03,
        3.86608596e-03,  5.45755932e-03, -4.39610665e-04,  7.76019941e-04,
        2.77595458e-03,  1.09283280e-03, -6.00256800e-03,
        6.02234769e-05, -2.76599985e-03,  6.58596959e-04, -5.01810265e-03]

min_salary1 = 55000
max_salary1 = 65000
min_salary2 = 80000
max_salary2 = 130000

# Create dictionaries for LabelEncoders and MinMaxScaler
label_encoders = {}
min_max_scaler = MinMaxScaler()

# Load dataset
df = pd.read_csv(r'data.csv') #path to the data.csv file

def map_values(df):
    # Mapping for 'Paid Time Off (PTO)' and 'Bonuses and Incentive Programs'
    df['Paid Time Off (PTO)'] = df['Paid Time Off (PTO)'].map({0: False, 1: True})
    df['Bonuses and Incentive Programs'] = df['Bonuses and Incentive Programs'].map({0: False, 1: True})

    # Mapping for 'Health Insurance'
    df['Health Insurance'] = df['Health Insurance'].map({0: True, 1: False})

    return df

# Apply value mapping to the dataset
df = map_values(df)

# Initialize and fit LabelEncoders for categorical columns
categorical_columns = ['Job Title', 'Qualifications', 'Company HQ Country', 'Specialization', 'Company',
                       'Job Portal', 'Work Type', 'Paid Time Off (PTO)', 'Company Industry',
                       'Bonuses and Incentive Programs', 'Gender Preference', 'Health Insurance', 'Company HQ City']

for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(df[col].astype(str))

# Min-max scaling function
def min_max_scaling(feature, min_val, max_val):
    return (feature - min_val) / (max_val - min_val)

# Reverse min-max scaling
def unscaling(scaled_value, min_val, max_val):
    return scaled_value * (max_val - min_val) + min_val

def predict(theta_min, theta_max, X):
    pred1 = np.dot(X, theta_min)
    pred2 = np.dot(X, theta_max)
    return unscaling(pred1, min_salary1, max_salary1), unscaling(pred2, min_salary2, max_salary2)

def fetch_encoded_row(df, encoded_df, selections):
    # Create a mask for matching rows
    mask = np.ones(len(df), dtype=bool)
    for col, val in selections.items():
        if df[col].dtype == 'object':
            encoded_val = label_encoders[col].transform([str(val)])[0]
            mask &= encoded_df[col] == encoded_val
        elif pd.api.types.is_numeric_dtype(df[col]):
            val = float(val)
            mask &= df[col] == val
    
    # Filter rows in encoded_df using the mask
    filtered_encoded_df = encoded_df[mask]
    
    # Filter columns in encoded_df based on selections keys
    selected_columns = list(selections.keys())
    
    return filtered_encoded_df[selected_columns]

def main():
    col1, col2, col3, col4 = strl.columns(4)
    col1.metric("Accuracy for Min Salary ", "92.19%", "0.2%")
    col2.metric("Accuracy for Max Salary", "92.70%", "0.2%")
    col3.metric("Error for Min Salary", "7.30%", "0.2%")
    col4.metric("Error for Max Salary", "7.81%", "0.2%")

    strl.title('EMPLOYEE SALARY PREDICTION APP')

    # Numerical variables
    min_experience1 = int(df['Minimum Experience (years)'].min())
    max_experience1 = int(df['Minimum Experience (years)'].max())
    min_experience2 = int(df['Maximum Experience (years)'].min())
    max_experience2 = int(df['Maximum Experience (years)'].max())
    experience_range = strl.slider('Select your range of experience (years)', min_experience1, max_experience2, (min_experience1, max_experience2))
    MinimumExperience, MaximumExperience = experience_range

    # Categorical variables
    selections = {}
    for col in categorical_columns:
        selections[col] = strl.selectbox(f"Select {col}", df[col].unique().tolist(), index=None, placeholder=f"Select {col}..")
        strl.write(f"You selected: {selections[col]}")

    if strl.button('Predict Salary'):
        # Encode categorical variables
        encoded_categorical = []
        for col, value in selections.items():
            if value is not None:
                encoded_value = label_encoders[col].transform([str(value)])[0]
                encoded_categorical.append(encoded_value)
            else:
                strl.warning(f"Please select a value for {col}")
                return

        # Scale numerical variables
        scaled_min_exp = min_max_scaling(MinimumExperience, min_experience1, max_experience1)
        scaled_max_exp = min_max_scaling(MaximumExperience, min_experience2, max_experience2)

        # Prepare input data for prediction
        input_data = np.array([1, scaled_min_exp, scaled_max_exp] + encoded_categorical)

        # Make prediction
        predicted_min, predicted_max = predict(theta_min, theta_max, input_data)
        strl.success(f"Predicted Salary Range (in Dollars): {predicted_min:.2f} - {predicted_max:.2f}")

if __name__ == '__main__':
    main()
