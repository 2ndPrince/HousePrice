# v4
# House Prices
%pylab inline
%matplotlib inline

# [CSV to DataFrame]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale


df = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/train.csv')
df_test = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/test.csv')

# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# newdf = df.select_dtypes(include=numerics)
# df_num.corr()['SalePrice'].sort_values(ascending=False)

# 'Id' removed
num_attribs = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt'    
,'YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces'] # Manually select whose corr > 0.4
# df[num_attribs].corr()['SalePrice'].sort_values(ascending=False)  

df_num = df[num_attribs]
y = df['SalePrice']

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

imputer = Imputer(strategy="median")
imputer.fit(df_num)
df_num = df_num.fillna(df_num.mean()) # imputer, somehow, had no impact on NaN values.
# df_num.isnull().values.any() => False

scaler = StandardScaler()
scaler.fit(df_num)

# ----------------------------------------------------------------- #

# Id and SalePrice excluded, category whose corr >0.3
cat_attribs = ['HeatingQC','Foundation','GarageType','MasVnrType',
	'GarageFinish','BsmtQual','KitchenQual','ExterQual','BsmtFinType1',
	'Neighborhood','SaleType','SaleCondition','FireplaceQu','BsmtExposure','Exterior2nd','Exterior1st']

df_cat = df[cat_attribs]
df_cat_dummy = pd.get_dummies(df_cat)

df_cat_dummy.isnull().values.any() # False. No pre-processing required.

# df_num.shape -> (1460,13) df_cat_dummy.shape -> (1460,122)
df_combined = pd.concat([df_num, df_cat_dummy], axis=1) # df_combined.shape -> (1460,135)

df_combined.isnull().values.any() # Gives me False

# ----------------------------------------------------------------- #
def display_scores(scores):
	print("Scores:", scores)
	print("Mean:", scores.mean())
	print("Standard deviation:", scores.std())
		
# ----------------------------------------------------------------- #

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_combined, y, test_size = 0.3, random_state=42)

# Create the regressor:
lin_reg = LinearRegression()
forest_reg = RandomForestRegressor()

# Fit the regressor to the training data
lin_reg.fit(X_train, y_train)
forest_reg.fit(X_train, y_train)


# Predict on the test data: y_pred
y_pred_lin = lin_reg.predict(X_test)
y_pred_forest = forest_reg.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(lin_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lin))
print("Root Mean Squared Error: {}".format(rmse))

# Compute and print R^2 and RMSE
print("R^2: {}".format(forest_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_forest))
print("Root Mean Squared Error: {}".format(rmse))

# ------------------------------------------- #

# Import the necessary modules
from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(lin_reg, df_combined, y, cv=5)
print(cv_scores)
print("Average Lin_Reg 5-Fold CV Score: {}".format(np.mean(cv_scores)))


cv_scores = cross_val_score(forest_reg, df_combined, y, cv=5)
print(cv_scores)
print("Average Forest_reg 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# ------------------------------------------- #

df_test = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/test.csv')

df_num_submission = df_test[num_attribs]

imputer.fit(df_num_submission)
df_num_submission = df_num_submission.fillna(df_num.mean()) # Note that NA filled with df_num.
# df_num_submission.isnull().values.any() => False

scaler.fit(df_num_submission)



df_cat_submission = df_test[cat_attribs]
df_cat_dummy_submission = pd.get_dummies(df_cat_submission)

df_cat_dummy_submission.isnull().values.any() # False. No pre-processing required.

# https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data
# Get missing columns in the training test
missing_cols = set( df_cat_dummy.columns ) - set( df_cat_dummy_submission.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    df_cat_dummy_submission[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
df_cat_dummy_submission = df_cat_dummy_submission[df_cat_dummy.columns]
# ------------------------------------------- #


# df_num_submission.shape -> (1459,13) df_cat_dummy_submission.shape -> (1459,119)
df_combined_submission = pd.concat([df_num_submission, df_cat_dummy_submission], axis=1) # df_combined.shape -> (1460,135)


df_combined_submission.isnull().values.any() # Gives me False

# ------------------------------------------- #


# Outputting the result
# ids = df_combined_submission['Id']
ids = df_test['Id']
predictions_lin = lin_reg.predict(df_combined_submission)
predictions_forest = forest_reg.predict(df_combined_submission)
# predictions = reg_all.predict(data_test.drop('Id', axis=1))

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions_lin })
output.to_csv('house_youngseok_v4_lin.csv', index = False)
output.head()

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions_forest })
output.to_csv('house_youngseok_v4_forest.csv', index = False)
output.head()