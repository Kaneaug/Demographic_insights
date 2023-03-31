import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv(r'C:\Users\kanem\Downloads\acs2017_census_tract_data.csv.zip')
# Replace all null values with NaN
df.replace('', np.nan, inplace=True)
# Drop all rows that contain NaN values
df.dropna(inplace=True)
df.shape
# Select the features we want to use for our linear regression model
X = df[['IncomePerCap']]
y = df['Poverty']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target variable for the test data using the trained model
y_pred = model.predict(X_test)

# Calculate the root mean squared error (RMSE) of the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE:', rmse)

# Plot the regression line along with the data points
sns.regplot(x='IncomePerCap', y='Poverty', data=df)

plt.scatter(X_train['IncomePerCap'], y_train, color='blue', label='Training Data', alpha=0.5)
plt.scatter(X_test['IncomePerCap'], y_test, color='green', label='Test Data', alpha=0.5)
plt.plot(X_test['IncomePerCap'], y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Income Per Capita')
plt.ylabel('Poverty')
plt.legend(loc='upper right')
plt.show()

# Plot the scatterplot with regression line and correlation coefficient
sns.set_style("whitegrid")
plt.figure(figsize=(10, 8))
sns.regplot(x='Income', y='Poverty', data=df, line_kws={'color': 'red'})
plt.title('Income vs Poverty', fontsize=14)
plt.xlabel('Income', fontsize=12)
plt.ylabel('Poverty', fontsize=12)

# Add correlation coefficient to the plot
corr_coeff = df[['Income', 'Poverty']].corr().loc['Income', 'Poverty'].round(2)
plt.text(0.15, 0.9, f'Correlation Coefficient: {corr_coeff}', transform=plt.gca().transAxes, fontsize=12)

plt.show()


