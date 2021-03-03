"""
Questions:
Create a regression model to predict the number of reviews, or availability in the next year. Assume lack of availability is a successful


Resources:
"""

import pandas as pd
pd.set_option('display.width',800)
pd.set_option('display.max_columns',30)

dfl = pd.read_csv('./data_files/listings.csv')
dfc = pd.read_csv('./data_files/calendar.csv')
dfr = pd.read_csv('./data_files/reviews.csv')

df.shape
df.describe()
print(df.dtypes.to_string())
df.describe()['reviews_per_month']

dfr.shape
dfr['date'].unique()
dfc['listing_id'] =='241032'.value_counts()
print(dfc['listing_id'].value_counts().to_string())

