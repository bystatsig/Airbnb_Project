"""
Questions:
Create a regression model to predict the number of reviews, or availability in the next year. Assume lack of availability is a successful

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)

dfl = pd.read_csv('./data_files/listings.csv')
dfc = pd.read_csv('./data_files/calendar.csv')  # each ID has 365 days
dfr = pd.read_csv('./data_files/reviews.csv')

dfl.shape
dfl.describe()
print(dfl.dtypes.to_string())
dfl.describe()['reviews_per_month']

dfr.shape
dfr['date'].unique()
dfc['listing_id'] == '241032'.value_counts()
print(dfc['listing_id'].value_counts().to_string())

# Convert currency variables to float
dfl['price'] = dfl['price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['weekly_price'] = dfl['weekly_price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['monthly_price'] = dfl['monthly_price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['security_deposit'] = dfl['security_deposit'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['cleaning_fee'] = dfl['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype('float64')

# Reduce the continuous variable to those of interest (ie. remove id as they don't provide useful information)
dfl_cont = dfl.select_dtypes(include=['float64', 'int64'])  # Get dataframe of only Category Columns
dfl_cont_int = dfl_cont.drop(columns=['id', 'scrape_id', 'host_id', 'square_feet', 'license'])  # Drop square_feet due to lack of data

# Create correlation matrix to see if there are any insightful correlations between variables.
mask = np.triu(np.ones_like(dfl_cont_int.corr(), dtype=bool))  # only show half of corr plot
plt.figure(figsize=(13, 9))
sns.heatmap(dfl_cont_int.corr(), annot=True, fmt='.1f', vmin=-1, vmax=+1, center=0, annot_kws={"size": 8},
            mask=mask).figure.tight_layout()
plt.show()
