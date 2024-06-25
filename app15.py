import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Specify the absolute paths to your CSV files
train_file_path = 'train.csv'
test_file_path = 'test.csv'

# Loading the data from CSV files
df_train = pd.read_csv(train_file_path)
df_test = pd.read_csv(test_file_path)

# Data Preprocessing
# Filling missing values in 'Item_Weight' with mean
df_train['Item_Weight'].fillna(df_train['Item_Weight'].mean(), inplace=True)
df_test['Item_Weight'].fillna(df_test['Item_Weight'].mean(), inplace=True)

# Filling missing values in 'Outlet_Size' with mode
mode_of_outlet_size = df_train.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=lambda x: x.mode()[0])
df_train['Outlet_Size'].fillna(df_train['Outlet_Type'].map(mode_of_outlet_size.T.iloc[0]), inplace=True)
df_test['Outlet_Size'].fillna(df_test['Outlet_Type'].map(mode_of_outlet_size.T.iloc[0]), inplace=True)

# Replacing categories in 'Item_Fat_Content'
df_train.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
df_test.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)

# Creating copies of the original data for plotting
df_train_original = df_train.copy()
df_test_original = df_test.copy()

# Label Encoding
encoder = LabelEncoder()
cols_to_encode = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

for col in cols_to_encode:
    df_train[col] = encoder.fit_transform(df_train[col])
    df_test[col] = encoder.transform(df_test[col])

# Splitting the data into features and target
X = df_train.drop(columns='Item_Outlet_Sales', axis=1)
Y = df_train['Item_Outlet_Sales']

# Splitting the data into training and testing sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=20)

# Training the XGBoost model
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Making predictions on training data
training_data_prediction = regressor.predict(X_train)

# R squared Value for training data
r2_train = metrics.r2_score(Y_train, training_data_prediction)
st.write('R Squared value (Training) = ', r2_train)

# Making predictions on validation data
val_data_prediction = regressor.predict(X_val)

# R squared Value for validation data
r2_val = metrics.r2_score(Y_val, val_data_prediction)
st.write('R Squared value (Validation) = ', r2_val)

# Making predictions on the actual test set
test_set_prediction = regressor.predict(df_test)

# Training the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, Y_train)

# Making predictions on validation data with Linear Regression
linear_val_predictions = linear_model.predict(X_val)

# R squared Value for validation data with Linear Regression
r2_val_linear = metrics.r2_score(Y_val, linear_val_predictions)
st.write('R Squared value (Validation) with Linear Regression = ', r2_val_linear)

# Training the Random Forest model
random_forest_model = RandomForestRegressor(random_state=22, n_estimators=150)
random_forest_model.fit(X_train, Y_train)

# Making predictions on validation data with Random Forest
forest_val_predictions = random_forest_model.predict(X_val)

# R squared Value for validation data with Random Forest
r2_val_forest = metrics.r2_score(Y_val, forest_val_predictions)
st.write('R Squared value (Validation) with Random Forest = ', r2_val_forest)

# Creating the submission file
submission_df = pd.DataFrame({
    'Item_Identifier': df_test['Item_Identifier'],
    'Outlet_Identifier': df_test['Outlet_Identifier'],
    'Item_Outlet_Sales': test_set_prediction
})

# Saving predictions to CSV
output_path = r'C:\Users\Dell\Desktop\big_mart_sales_predictions14.csv'
submission_df.to_csv(output_path, index=False)

# Layout the plots in columns
col1, col2, col3 = st.columns(3)

# Plot Item_Weight distribution
with col1:
    st.subheader("Item Weight Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_train_original['Item_Weight'], kde=True, ax=ax)
    plt.title('Item Weight Distribution')
    st.pyplot(fig)

# Plot Item_Visibility distribution
with col2:
    st.subheader("Item Visibility Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_train_original['Item_Visibility'], kde=True, ax=ax)
    plt.title('Item Visibility Distribution')
    st.pyplot(fig)

# Plot Item_Outlet_Sales distribution
with col3:
    st.subheader("Item Outlet Sales Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_train_original['Item_Outlet_Sales'], kde=True, ax=ax)
    plt.title('Item Outlet Sales Distribution')
    st.pyplot(fig)

# Second row of plots
col4, col5, col6 = st.columns(3)

# Plot Outlet_Establishment_Year column
with col4:
    st.subheader("Outlet Establishment Year Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Outlet_Establishment_Year', data=df_train_original, ax=ax)
    plt.title("Outlet Establishment Year Count")
    st.pyplot(fig)

# Plot Item_Fat_Content column
with col5:
    st.subheader("Item Fat Content Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Item_Fat_Content', data=df_train_original, ax=ax)
    plt.title("Item Fat Content Count")
    st.pyplot(fig)

# Plot Item_Type column
with col6:
    st.subheader("Item Type Count")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.countplot(x='Item_Type', data=df_train_original, ax=ax)
    plt.title("Item Type Count")
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Third row of plots
col7, col8 = st.columns(2)

# Plot Outlet_Size column
with col7:
    st.subheader("Outlet Size Count")
    fig, ax = plt.subplots()
    sns.countplot(x='Outlet_Size', data=df_train_original, ax=ax)
    plt.title("Outlet Size Count")
    st.pyplot(fig)

# Plot Predicted Sales for Each Item and Outlet
with col8:
    st.subheader("Predicted Sales for Each Item and Outlet")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.scatterplot(data=submission_df, x='Item_Identifier', y='Item_Outlet_Sales', hue='Outlet_Identifier', palette='viridis', ax=ax)
    plt.title('Predicted Sales for Each Item and Outlet')
    plt.xlabel('Item Identifier')
    plt.ylabel('Predicted Item Outlet Sales')
    plt.xticks(np.arange(0, 1601, 200))  # Set the x-axis range from 0 to 1600 with steps of 200
    plt.legend(title='Outlet Identifier', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)
