"""

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
pd.set_option('display.width', 800)
pd.set_option('display.max_columns', 30)

dfl = pd.read_csv('./data_files/listings.csv')

# Convert currency & rate variables to float
dfl['price'] = dfl['price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['weekly_price'] = dfl['weekly_price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['monthly_price'] = dfl['monthly_price'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['security_deposit'] = dfl['security_deposit'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['cleaning_fee'] = dfl['cleaning_fee'].str.replace('$', '').str.replace(',', '').astype('float64')
dfl['host_response_rate'] = dfl['host_response_rate'].str.replace('%', '').astype('float64')
#print(dfl.dtypes.to_string())

# Get continuous variables. Reduce to those of interest (for example: remove id as they don't provide useful information)
dfl_cont = dfl.select_dtypes(include=['float64', 'int64'])  # Get dataframe of only Continuous Columns
dfl_cont_int = dfl_cont.drop(columns=['id', 'scrape_id', 'host_id', 'square_feet', 'license', 'monthly_price'])  # Drop square_feet and monthly_price due to lack of data

# Impute means for continuous variables with null values
fill_mean = lambda col: col.fillna(col.mean())  # Create mean function
dfl_cont_int = dfl_cont_int.apply(fill_mean, axis=0)  # Use function to Fill missing values with the mean of the column.

###################### Plot corr matrix to see if there are any insightful correlations between continuous variables.
mask = np.triu(np.ones_like(dfl_cont_int.corr(), dtype=bool))  # only show half of corr plot
plt.figure(figsize=(13, 9))
sns.heatmap(dfl_cont_int.corr(), annot=True, fmt='.1f', vmin=-1, vmax=+1, center=0, annot_kws={"size": 8}, mask=mask).figure.tight_layout()
plt.show()

# Get categorical variables of interest
dfl_categories = dfl.select_dtypes(include=['object'])
# dfl_categories.nunique()    # Reduced variables to those with unique values less than x
# dfl_categories['host_response_time'].value_counts()
dfl_categories_int = dfl_categories[['property_type', 'cancellation_policy', 'neighbourhood_group_cleansed', 'host_response_time']]

# Convert categorical variables of interest to dummy variables (1's and 0's)
cols = dfl_categories_int.columns
dummy = pd.get_dummies(dfl_categories_int[cols], prefix=cols, prefix_sep='_', drop_first=True, dummy_na=True)
# dummy.sum()

# Combine continuous and categorical variables
dfl_combined = pd.merge(dfl_cont_int, dummy, left_index=True, right_index=True)


# Create a Response Variable or percentage of the time the listing is booked. Remove vars which are highly correlated to response.
dfl_combined['pct_booked_30'] = (1 - dfl['availability_30']/30)     # Add response variable (continuous)

###################### Plot histogram of dummy Response Variable
plt.suptitle('Distribution of Listings Occupancy Rates', fontsize=16)
histo = sns.histplot(dfl_combined['pct_booked_30'], bins=30)
histo.set(ylabel='Count of Listings', xlabel='Occupancy Rate in Next 30 Days')
plt.show()
######################
dfl_combined.describe()['pct_booked_30']     # Average occupancy rate of 44%

# Create new response variable with Binary Success (0 or 1)
dfl_combined['pct_booked_30_bnry'] = dfl_combined['pct_booked_30'].round()     # Add response variable (BINARY)
dfl_final = dfl_combined.drop(columns=['availability_30', 'availability_60', 'availability_90', 'availability_365', 'pct_booked_30'])    # Drop redundant features because they will cause over fitting



# Split data in train/test (occupancy rate)
# y = dfl_final['pct_booked_30']
# x = dfl_final.drop('pct_booked_30', axis=1)
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)    # random_state (keep value the same to recreate results)
# Split data in train/test (BINARY occupancy rate)
y = dfl_final['pct_booked_30_bnry']
x = dfl_final.drop('pct_booked_30_bnry', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=42)    # random_state (keep value the same to recreate results)


# 1 - INSTANTIATE THE MODEL
# lm_model = LinearRegression()
lm_model = LogisticRegression()
# 2 - FIT THE MODEL TO THE TRAINING DATA SET
lm_model.fit(x_train, y_train)
# 3 - PREDICT TEST DATA FROM THE MODEL
y_test_preds = lm_model.predict(x_test)
# 4 - SCORE THE MODEL (BASED ON THE Y_TEST DATASET)
#log_score = logisticRegr.score(x_test, y_test)
r2_test = r2_score(y_test, y_test_preds)    # Rsquared metric on TEST data
print("The r-squared score (TEST fit) for your model was {} on {} values.".format(r2_test, len(y_test)))

# How good is fit to the Train data?
y_TRAIN_preds = lm_model.predict(x_train)
r2_TRAIN = r2_score(y_train, y_TRAIN_preds)
"The r-squared score (TRAIN fit) for your model was {} on {} values.".format(r2_TRAIN, len(y_train))

# Plot predictions to actual test values
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


