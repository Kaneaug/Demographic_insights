import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the data
df = pd.read_csv(r'C:\Users\kanem\Downloads\acs2017_census_tract_data.csv.zip')

# Drop all rows that contain NaN values
df.replace('', np.nan, inplace=True)
df.dropna(inplace=True)
df.shape
# Select the features we want to use for our linear regression model
X = df[['Walk']]
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
sns.regplot(x='Walk', y='Poverty', data=df)

plt.scatter(X_train['Walk'], y_train, color='blue', label='Training Data',alpha=0.5)
plt.scatter(X_test['Walk'], y_test, color='green', label='Test Data', alpha=0.5)
plt.plot(X_test['Walk'], y_pred, color='red', linewidth=2, label='Predicted')
plt.xlabel('Percentage of Population Who Walk to Work')
plt.ylabel('Poverty Rate')
plt.legend(loc='upper right')
plt.show()
