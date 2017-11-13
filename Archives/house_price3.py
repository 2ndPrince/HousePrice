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

df_num.corr()['SalePrice'].sort_values(ascending=False)

# 'Id' removed
num_columns_important = ['SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt'    
,'YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces'] # Manually select whose corr > 0.4
# df[num_columns_important].corr()['SalePrice'].sort_values(ascending=False)  

X = df[num_columns_important].drop('SalePrice', axis=1)
y = df[num_columns_important]['SalePrice']


X = X.fillna(X.mean())
X_scaled = scale(X)
#X.isnull().values.any() # Gives me False

# ----------------------------------------------------------------- #

# Id and SalePrice excluded, category whose corr >0.3
category = ['HeatingQC','Foundation','GarageType','MasVnrType',
	'GarageFinish','BsmtQual','KitchenQual','ExterQual','BsmtFinType1',
	'Neighborhood','SaleType','SaleCondition','FireplaceQu','BsmtExposure','Exterior2nd','Exterior1st']

df_category = df[category]
df_category_dummy = pd.get_dummies(df_category)

df_category_dummy.isnull().values.any() # False. No pre-processing required.

# X.shape -> (1460,14) df_category_dummy.shape -> (1460,122)
df_combined = pd.concat([X, df_category_dummy], axis=1) # df_combined.shape -> (1460,136)

df_combined.isnull().values.any() # Gives me False

# ----------------------------------------------------------------- #

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(df_combined, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()
forest_reg = RandomForestRegressor()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)
forest_reg.fit(X_train, y_train)




reg_all = forest_reg




# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

# ------------------------------------------- #

# Import the necessary modules
from sklearn.model_selection import cross_val_score

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_all, df_combined, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# ------------------------------------------- #

df_test = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/test.csv')
num_columns_important_submission = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt'    
,'YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces']

df_test_num = df_test[num_columns_important_submission]
df_test_num = df_test_num.fillna(df_test_num.mean())

df_test_num.isnull().values.any()

# Id and SalePrice excluded
category = ['HeatingQC','Foundation','GarageType','MasVnrType',
	'GarageFinish','BsmtQual','KitchenQual','ExterQual','BsmtFinType1',
	'Neighborhood','SaleType','SaleCondition','FireplaceQu','BsmtExposure','Exterior2nd','Exterior1st']

df_category_submission = df_test[category]
df_category_dummy_submission = pd.get_dummies(df_category)

df_category_dummy_submission.isnull().values.any() # False. No pre-processing required.

df_combined_submission = pd.concat([df_test_num, df_category_dummy_submission], axis=1)
# df_combined_submission.shape -> (1460,136)
df_combined_submission = df_combined_submission.fillna(df_combined_submission.mean())
df_combined_submission.isnull().values.any()

# ------------------------------------------- #
df_test = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/test.csv')
# Outputting the result
# ids = df_combined_submission['Id']
ids = df_test['Id']
predictions = reg_all.predict(df_combined_submission)
# predictions = reg_all.predict(data_test.drop('Id', axis=1))

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('house_youngseok_v4.csv', index = False)
output.head()