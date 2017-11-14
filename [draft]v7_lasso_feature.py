# Visit http://arcanumysl.wordpress.com for recent works.
# I am new to Machine Learning. This is my first solution without referring or looking any other kernels.
# Comment me for suggestions.

# Version 7: Trying to make helper functions for automation. 
#            Feature Engineering to remove, combine and create new variable-in-training-and-testing-data
#            version 6 was lasso regression.
# ------------------------------------------- #

def load_data(location):
	if(location == 1): # Desktop
		path_train = "D:/DataScience/Kaggle/HousePrice/Data/train.csv"
		path_test = "D:/DataScience/Kaggle/HousePrice/Data/test.csv"
	elif(location == 2): # Laptop
		path_train = "C:/Users/User/Documents/GitHub/HousePrice/Data/train.csv"
		path_test = "C:/Users/User/Documents/GitHub/HousePrice/Data/test.csv"
	elif(location == 3): # Kaggle Kernel
		path_train = "../input/train.csv"
		path_test =  "../input/test.csv"
	else:
		print("No location found")
	return pd.read_csv(path_train), pd.read_csv(path_test)
	
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
	
def cat_analysis(df, cat, label):
	cat_df_dummy = pd.get_dummies(df[cat])
	catdf_analysis = pd.concat([df['SalePrice'], cat_df_dummy], axis=1)
	print(catdf_analysis.corr()['SalePrice'].sort_values(ascending=False))
	print(df[cat].value_counts())
	plot_categories(df,cat,label)

# https://www.kaggle.com/helgejo/an-interactive-data-science-tutorial	
def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()	
	

	
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import warnings
%matplotlib inline
warnings.filterwarnings('ignore')

# Load data set	
df, df_test = load_data(2)
# Applying log transformation	
df['SalePrice'] = np.log1p(df['SalePrice'])


#cat_analysis(df,"HeatingQC","SalePrice")




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

