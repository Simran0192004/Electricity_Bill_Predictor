#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

#importing the dataset
df = pd.read_csv('elec_bill.csv')
df['bill_amt'] = df['KWH'] * 7

#cleaning the csv file
feat = ['REGIONC', 'TYPEHUQ', 'CDD65', 'HDD65', 'OA_LAT', 'GWT', 'DesignDBT1']
df_clean = df [feat + ['bill_amt']].dropna()

#selecting the features
X = df_clean[feat]
Y = df_clean['bill_amt']

#Splitting the data into splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

#Random_Forest_Regression
regressor = RandomForestRegressor(n_estimators = 100, random_state = 42)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

#plotting the graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
plt.xlabel('Actual Bill Amount (₹)')
plt.ylabel('Predicted Bill Amount (₹)')
plt.title('Random Forest Regression: Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()
