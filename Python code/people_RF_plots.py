import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv(r'C:\Users\kanem\Downloads\acs2017_census_tract_data.csv.zip')

# Drop all rows that contain NaN values
df.replace('', np.nan, inplace=True)
df.dropna(inplace=True)

X = df[['TotalPop', 'Men', 'Women', 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific', 'VotingAgeCitizen', 
        'IncomeErr', 'IncomePerCap', 'IncomePerCapErr', 'ChildPoverty', 'Professional', 'Service', 'Office', 
        'Construction', 'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome', 'MeanCommute', 
        'Employed', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']]
y = df['Income']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest model and fit it to the training data
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Predict the target variable for the test data using the trained model
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
mse = mean_squared_error(y_test, y_pred)
print('MSE:', mse)

plt.hist(df['Income'], bins=25, color='darkblue', alpha=0.7, edgecolor='aquamarine', linewidth=1.5)
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Count')
plt.grid(axis='y', alpha=0.5)
plt.show()

sns.kdeplot(df['Income'], shade=True, color='darkblue')
plt.title('Distribution of Income')
plt.xlabel('Income')
plt.ylabel('Density')
plt.show()

sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Correlation Heatmap')
plt.show()

fig, axs = plt.subplots(2, 4, figsize=(16, 8))
axs = axs.flatten()

for i, feature in enumerate(['Men', 'Women', 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific']):
    axs[i].hist(df[feature], bins=20, alpha=0.7, color='royalblue')
    axs[i].set_title(f'Distribution of {feature}')
    axs[i].set_xlabel(feature)
    axs[i].set_ylabel('Count')
    axs[i].grid(axis='y', alpha=0.5)

plt.tight_layout()
plt.show()
