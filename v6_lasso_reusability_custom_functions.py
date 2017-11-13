# Visit http://arcanumysl.wordpress.com for recent works.
# I am new to Machine Learning. This is my first solution without referring or looking any other kernels.
# Comment me for suggestions.

# What to do next
# 1. Include categorical(string) data whose corr is high enough
# 2. Perform Feature Engineering to make better representative feature data by removing/combing the current dataset.
# 3. Fine-tune the model: Grid Search, HyperParameters
# 4. More Feature Engineering using grid_search.best_estimator_.feature_importances_
# 5. Do these analysis with other models such as SVM
# 6. Pick the best combination and submit.

# ------------------------------------------- #

# Removes certain rows for outlier data # Example: df = remove_outlier(df,[34,127,359])
def remove_outlier(df,id_array):
    for i in range(len(id_array)):
        df = df.drop(df[df['Id'] == id_array[i]].index)
    return df
	
# Returns df whose types are numbers and correlation with target is more than corr	
def filter_num(df, target, corr):
	numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
	df_num = df.select_dtypes(include=numerics)
	important = abs(np.array(df_num.corr()[target])) > corr	
	important_index = df_num.keys()[important]
	df_num_important = df_num.loc[:,important_index]
	return df_num_important.drop(target, axis=1), df_num_important.loc[:,target], important_index.drop(['SalePrice'])	
	
def filter_cat(df, attributes):
	df_cat = df[cat_attribs]
	df_cat_dummy = pd.get_dummies(df_cat)
	return df_cat_dummy

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
%matplotlib inline


df = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/train.csv')
df_test = pd.read_csv('D:/DataScience/Kaggle/HousePrice/Data/test.csv')


#applying log transformation	
df['SalePrice'] = np.log1p(df['SalePrice'])
df_num, df_label, important_index = filter_num(df, 'SalePrice', 0.5)

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

num_pipeline = Pipeline([
		('imputer', Imputer(strategy="median")),
		('std_scaler', StandardScaler()),
		])
	
df_num = num_pipeline.fit_transform(df_num) # pipeline returns np.array
df_num = pd.DataFrame(df_num)
			

		
cat_attribs = ['Id', 'HeatingQC','Foundation','GarageType','MasVnrType',
	'GarageFinish','BsmtQual','KitchenQual','ExterQual','BsmtFinType1',
	'Neighborhood','SaleType','SaleCondition','FireplaceQu','BsmtExposure','Exterior2nd','Exterior1st']

df_cat_dummy = filter_cat(df, cat_attribs)
df_combined = pd.concat([df_num, df_cat_dummy], axis=1, copy=False)

# Data Pre-processing done by this point
X = df_combined
y = df_label

# Start Feature Engineering
df_combined = remove_outlier(df_combined,[1299, 524])

# ------------------------------------------- #

# Import necessary modules
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def CV_test_score(model, X_test, y_test):
	# Predict on the test data: y_pred
	y_pred_lasso = model.predict(X_test)
	
	# [model] - Compute and print R^2 and RMSE
	print("R^2: {}".format(model.score(X_test, y_test)))
	rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
	print("Root Mean Squared Error: {}".format(rmse))

def CV_Kfold(model, X, y, cv):
	cv_scores = cross_val_score(model, X, y, cv=5)
	print(cv_scores)
	print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor:
forest_reg = RandomForestRegressor()
lasso_reg = Lasso()

# Fit the regressor to the training data
forest_reg.fit(X_train, y_train)
lasso_reg.fit(X_train, y_train)


CV_test_score(lasso_reg, X_test, y_test)
CV_Kfold(lasso_reg, X, y, 5)


# ------------------------------------------- #
# [Data preparation for test data]

df_test_num = df_test.loc[:,important_index]
df_test_num = num_pipeline.fit_transform(df_test_num)
df_test_num = pd.DataFrame(df_test_num)

df_cat_dummy_submission = filter_cat(df_test,cat_attribs)

# https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data
# Get missing columns in the training test
missing_cols = set( df_cat_dummy.columns ) - set( df_cat_dummy_submission.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    df_cat_dummy_submission[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
df_cat_dummy_submission = df_cat_dummy_submission[df_cat_dummy.columns]

df_combined_submission = pd.concat([df_test_num, df_cat_dummy_submission], axis=1)
# ------------------------------------------- #

from sklearn.model_selection import GridSearchCV

param_grid = [{'alpha': [1e-5,4e-4,7e-4,1e-3, 1e-2]},]

grid_search = GridSearchCV(lasso_reg, param_grid, cv=5, scoring='neg_mean_squared_error')		
grid_search.fit(X_train, y_train)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)		

print(grid_search.best_params_)

regressor = grid_search.best_estimator_

CV_test_score(regressor, X_test, y_test)
	
ids = df_test['Id']		
predictions = regressor.predict(df_combined_submission) # Lasso performed better and I modify here.
predictions = np.expm1(predictions)

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('houseprice_youngseok.csv', index = False)
output.head()

