"""

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
dfl

# Convert currency variables to float
dfl['price'] = dfl['price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['weekly_price'] = dfl['weekly_price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['monthly_price'] = dfl['monthly_price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['security_deposit'] = dfl['security_deposit'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['cleaning_fee'] = dfl['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype('float64')

# Reduce the continuous variable to those of interest (ie. remove id as they don't provide useful information)
dfl_cont = dfl.select_dtypes(include=['float64', 'int64'])  # Get dataframe of only Continuous Columns
dfl_cont_int = dfl_cont.drop(columns=['id', 'scrape_id', 'host_id', 'square_feet', 'license'])  # Drop square_feet due to lack of data

# Create correlation matrix to see if there are any insightful correlations between variables.
mask = np.triu(np.ones_like(dfl_cont_int.corr(), dtype=bool))  # only show half of corr plot
plt.figure(figsize=(13, 9))
sns.heatmap(dfl_cont_int.corr(), annot=True, fmt='.1f', vmin=-1, vmax=+1, center=0, annot_kws={"size": 8}, mask=mask).figure.tight_layout()
plt.show()

# Experimentation------------------------------------------
print(dfl[['review_scores_value','review_scores_rating']].to_string())
dfl['review_scores_value'].describe()
dfl.describe()[['review_scores_rating','availability_365','availability_30']]

# Create a dummy Response Variable of Interest for percentage of the time the listing is booked (30 and 365 days out)
dfl_cont_int['pct_booked_30'] = 1 - dfl['availability_30']/30
# Plot
plt.suptitle('Distribution of Listings Occupancy Rates', fontsize=16)
histo = sns.histplot(dfl_cont_int['pct_booked_30'], bins=30)
histo.set(ylabel='Count of Listings', xlabel='Occupancy Rate in Next 30 Days')
plt.show()

dfl_cont_int.describe()['pct_booked_365']    # Average occupancy rate of 33%
dfl_cont_int.describe()['pct_booked_30']     # Average occupancy rate of 44%
print(dfl.groupby(['neighbourhood']).describe()['pct_booked_30'].to_string())

# Category Columns
dfl_cat = dfl.select_dtypes(include= 'object')     # Get Category columns
dfl_cat_int = dfl_cat[['host_response_time' ,'host_is_superhost','host_identity_verified']]
dfl_cat.dtypes
dfl_cat.isnull().mean()
dfl_cat_int.value_counts('host_response_time')


#todo Create Linear Model to predict occupancy rate (next 30 days)
#todo Find github ReadMe to plagurize


