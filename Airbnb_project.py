"""

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

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
dfl_cont_int = dfl_cont.drop(columns=['id', 'scrape_id', 'host_id', 'square_feet', 'license', 'monthly_price'])  # Drop square_feet and monthly_price due to lack of data

# Plot correlation matrix to see if there are any insightful correlations between variables.
mask = np.triu(np.ones_like(dfl_cont_int.corr(), dtype=bool))  # only show half of corr plot
plt.figure(figsize=(13, 9))
sns.heatmap(dfl_cont_int.corr(), annot=True, fmt='.1f', vmin=-1, vmax=+1, center=0, annot_kws={"size": 8}, mask=mask).figure.tight_layout()
plt.show()

# Create a dummy Response Variable of Interest for percentage of the time the listing is booked (30 and 365 days out)
dfl_cont_int['pct_booked_30'] = 1 - dfl['availability_30']/30
# Plot
plt.suptitle('Distribution of Listings Occupancy Rates', fontsize=16)
histo = sns.histplot(dfl_cont_int['pct_booked_30'], bins=30)
histo.set(ylabel='Count of Listings', xlabel='Occupancy Rate in Next 30 Days')
plt.show()

dfl_cont_int.describe()['pct_booked_30']     # Average occupancy rate of 44%
print(dfl.groupby(['neighbourhood']).describe()['pct_booked_30'].to_string())

# Impute mean for null values
fill_mean = lambda col: col.fillna(col.mean())  # Create mean function
fill_dfl_cont_int = dfl_cont_int.apply(fill_mean, axis=0)  # Use function to Fill missing values with the mean of the column.

dfl_cont_int.price.describe()
fill_dfl_cont_int.price.describe()

# Split data in train/test
y = fill_dfl_cont_int['pct_booked_30']
x = fill_dfl_cont_int.drop('pct_booked_30', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)    # random_state (keep value the same to recreate results)
# Split data in train/test (PRICE)
y = fill_dfl_cont_int['price']
x = fill_dfl_cont_int.drop('price', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)    # random_state (keep value the same to recreate results)

# 1 - INSTANTIATE THE MODEL
lm_model = LinearRegression(normalize=True)
# 2 - FIT THE MODEL TO THE TRAINING DATA SET
lm_model.fit(x_train, y_train)
# 3 - PREDICT TEST DATA FROM THE MODEL
y_test_preds = lm_model.predict(x_test)
# 4 - SCORE THE MODEL (BASED ON THE Y_TEST DATASET)
r2_test = r2_score(y_test, y_test_preds)    # Rsquared metric on TEST data
"The r-squared score (TEST fit) for your model was {} on {} values.".format(r2_test, len(y_test))
# How good is fit to the Train data?
y_TRAIN_preds = lm_model.predict(x_train)
r2_TRAIN = r2_score(y_train, y_TRAIN_preds)
"The r-squared score (TRAIN fit) for your model was {} on {} values.".format(r2_TRAIN, len(y_train))

# Plot predictions to actual test predictions
scatter = sns.scatterplot(y_test,y_test_preds)
scatter.set(ylabel= 'Predicted Values')

# Evaluate coefficients
coefs_df = pd.DataFrame()
coefs_df['est_int'] = x_train.columns
coefs_df['coefs'] = lm_model.coef_
coefs_df['abs_coefs'] = np.abs(lm_model.coef_)
coefs_df = coefs_df.sort_values('abs_coefs', ascending=False)
coefs_df.head(20)


# TODO Category Columns
dfl_cat = dfl.select_dtypes(include= 'object')     # Get Category columns
dfl_cat_int = dfl_cat[['host_response_time' ,'host_is_superhost','host_identity_verified']]
dfl_cat.dtypes
dfl_cat.isnull().mean()
dfl_cat_int.value_counts('host_response_time')


#todo Create Linear Model to predict occupancy rate (next 30 days)
#todo Find github ReadMe to plagurize


