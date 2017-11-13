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

# Here I am only dealing with numeric data as one-step at a time approach.

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/train.csv")

df = df.drop(df[df['Id'] == 1299].index)
df = df.drop(df[df['Id'] == 524].index)

#applying log transformation
df['SalePrice'] = np.log1p(df['SalePrice'])

# [Selecting numeric values] : whose corr > 0.4
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_num = df.select_dtypes(include=numerics)
# print(df_num.corr()['SalePrice'].sort_values(ascending=False))

important = abs(np.array(df_num.corr()['SalePrice'])) > 0.4
important_index = df_num.keys()[important]
df_num_important = df_num.loc[:,important_index]


from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer.fit(df_num_important)
df_num_important = df_num_important.fillna(df_num_important.mean()) # imputer, somehow, had no impact on NaN values.
# df_num_important.isnull().values.any() => False

scaler = StandardScaler()
scaler.fit(df_num_important)

cat_attribs = ['HeatingQC','Foundation','GarageType','MasVnrType',
	'GarageFinish','BsmtQual','KitchenQual','ExterQual','BsmtFinType1',
	'Neighborhood','SaleType','SaleCondition','FireplaceQu','BsmtExposure','Exterior2nd','Exterior1st']

df_cat = df[cat_attribs]
df_cat_dummy = pd.get_dummies(df_cat)

df_combined = pd.concat([df_num_important, df_cat_dummy], axis=1)

# [Data preparation to feed the model]
X = df_combined.drop('SalePrice', axis=1)
y = df_combined['SalePrice']

X.isnull().values.any()
# X = X.fillna(X.mean())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X)

# ------------------------------------------- #

# Import necessary modules
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor:
forest_reg = RandomForestRegressor()

# Fit the regressor to the training data
forest_reg.fit(X_train, y_train)


# Predict on the test data: y_pred
y_pred_forest = forest_reg.predict(X_test)


# [Random Forest] - Compute and print R^2 and RMSE
print("R^2: {}".format(forest_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_forest))
print("Root Mean Squared Error: {}".format(rmse))

# ------------------------------------------- #
# Import the necessary modules
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(forest_reg, X, y, cv=5)
print(cv_scores)
print("Average Forest_reg 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# Look like forest model is more powerful as expected

# ------------------------------------------- #
# [Data preparation for test data]
df_test = pd.read_csv("../input/test.csv")
important_index_wo_price = important_index.drop(['SalePrice'])

df_test_num = df_test.loc[:,important_index_wo_price]
df_test_num = df_test_num.fillna(df_test_num.mean())
# print(df_test_num.isnull().values.any())

df_cat_submission = df_test[cat_attribs]
df_cat_dummy_submission = pd.get_dummies(df_cat_submission)

# https://stackoverflow.com/questions/41335718/keep-same-dummy-variable-in-training-and-testing-data
# Get missing columns in the training test
missing_cols = set( df_cat_dummy.columns ) - set( df_cat_dummy_submission.columns )
# Add a missing column in test set with default value equal to 0
for c in missing_cols:
    df_cat_dummy_submission[c] = 0
# Ensure the order of column in the test set is in the same order than in train set
df_cat_dummy_submission = df_cat_dummy_submission[df_cat_dummy.columns]

df_combined_submission = pd.concat([df_test_num, df_cat_dummy_submission], axis=1) # df_combined.shape -> (1459,135)
scaler.fit(df_combined_submission) # scaling made slight difference # score from 0.17423 -> 0.17194
# ------------------------------------------- #

from sklearn.model_selection import GridSearchCV

param_grid = [
				{'n_estimators': [10, 100, 200], 'max_features': [ 10, 20]},
				{'bootstrap': [False], 'n_estimators': [100, 200], 'max_features': [15, 25]},
			]
			
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')		
grid_search.fit(X_train, y_train)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)		

print(grid_search.best_params_)

from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
import xgboost as xgb

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

model_ridge = Ridge()

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")

model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y)
coef = pd.Series(model_lasso.coef_, index = X_train.columns)

imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])
                     
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb":xgb_preds, "lasso":lasso_preds})
predictions.plot(x = "xgb", y = "lasso", kind = "scatter")
preds = 0.7*lasso_preds + 0.3*xgb_preds



# I don't want to fit for 'Id' features.		
ids = df_test['Id']		
predictions = grid_search.predict(df_combined_submission)
# predictions = np.expm1(predictions)
output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('houseprice_youngseok.csv', index = False)
output.head()

