# House Prices
%pylab inline
%matplotlib inline

# [CSV to DataFrame]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#df = pd.read_csv('C:/Users/User/Desktop/DataScience/Contest/House_Prices/train.csv')
# df_test = pd.read_csv('C:/Users/User/Desktop/DataScience/Contest/House_Prices/test.csv')
df = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/train.csv')
df_test = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/test.csv')


num_columns = ['Id', 'LotFrontage','MasVnrArea','GarageYrBlt','Id','MSSubClass','LotArea',
	'OverallQual','OverallCond','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF',
	'TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',
	'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars',
	'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea',
	'MiscVal','MoSold','YrSold','SalePrice']
	
df_num = df[num_columns]

# numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# newdf = df.select_dtypes(include=numerics)

df_num.corr()['SalePrice'].sort_values(ascending=False)


num_columns_important = ['Id', 'SalePrice','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt'    
,'YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces'] # Manually select whose corr > 0.4
df[num_columns_important].corr()['SalePrice'].sort_values(ascending=False)  

X = df[num_columns_important].drop('SalePrice', axis=1)
y = df[num_columns_important]['SalePrice']

# ------------------------------------------- #

X = X.fillna(X.mean())
# ------------------------------------------- #

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

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
cv_scores = cross_val_score(reg_all, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# ------------------------------------------- #
num_columns_important_submission = ['Id','OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt'    
,'YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces']  
df_test = df_test.fillna(df_test.mean())
df_test = df_test[num_columns_important_submission]
# ------------------------------------------- #

# Outputting the result
ids = df_test['Id']
predictions = reg_all.predict(df_test)
# predictions = reg_all.predict(data_test.drop('Id', axis=1))

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('house-predictions_youngseok.csv', index = False)
output.head()

# replace negatives with min

# ------------------------------------------- #
# ------------------------------------------- #
# ------------------------------------------- #

category = ['HeatingQC','Foundation','GarageType','MasVnrType',
	'GarageFinish','BsmtQual','KitchenQual','ExterQual','BsmtFinType1',
	'Neighborhood','SaleType','SaleCondition','FireplaceQu','BsmtExposure','Exterior2nd','Exterior1st']

df_category = df[category]	