import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'C:\Users\kanem\Downloads\acs2017_census_tract_data.csv.zip')

# Replace all null values with NaN
df.replace('', np.nan, inplace=True)
# Drop all rows that contain NaN values
df.dropna(inplace=True)
df.shape
df.columns
df

# Select the columns to use for clustering
X = df[['TotalPop', 'Men', 'Women', 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific', 'VotingAgeCitizen', 'Income', 'IncomePerCap', 'Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction', 'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp', 'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']]

# Standardize the dataframe
scaler = StandardScaler()
X_transform =scaler.fit_transform(X)
# Perform Kmeans clustering with 5 clusters
kmeans = KMeans(n_clusters=5).fit(X_transform)

# Add the cluster labels to the original dataset
X['Cluster'] = kmeans.labels_

X['Cluster']

X

# Plot the scatter plot
plt.scatter(X['TotalPop'], X['Income'], c=kmeans.labels_, cmap='rainbow', alpha=0.5)
plt.xlabel('Total Population')
plt.ylabel('Income')
plt.show()

# Plot the clusters using plotly
fig = px.scatter(df, x='Income', y='Poverty',color=X['Cluster'],
                 hover_data=['State', 'County'],
                 title='KMeans Clustering Results',
                 labels={'Income': 'Median Income', 'Poverty': 'Poverty Rate'})
fig.show()

# Plot the clusters using seaborn
sns.scatterplot(x='TotalPop', y='Employed', hue=X['Cluster'], data=df)

# Pivot the data to create a heatmap
heatmap_data = df.groupby(['Cluster']).mean().drop('TractId', axis=1)
heatmap_data = heatmap_data.T

# Plot the heatmap using seaborn
sns.heatmap(heatmap_data, cmap='YlGnBu')

# Create a parallel coordinates plot using plotly
fig = px.parallel_coordinates(df, color='Cluster')
fig.show()

# Plot parallel coordinates
parallel_coordinates(df.drop(['TractId', 'State', 'County'], axis=1), 'Cluster', colormap='viridis')

# Display the plot
plt.show()

