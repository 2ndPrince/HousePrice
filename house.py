# ------------------------------------------------------------------------------------- #
# [Loading the data]

# Import numpy and pandas
import numpy as np
import pandas as pd

# [Labtop file location]
# data_train = pd.read_csv('C:/Users/User/Desktop/DataScience/Contest/House_Prices/train.csv')
# data_test = pd.read_csv('C:/Users/User/Desktop/DataScience/Contest/House_Prices/test.csv')

# [PC file location]
# I have only included numeric variables of the dataset by deleting non-numerics

df = pd.read_csv('D:/DataScience/Kaggle/HousePrice/train_num_only_noNA.csv')
df2 = pd.read_csv('D:/DataScience/Kaggle/HousePrice/test_num_only_noNA.csv')

# NA values are converted to the mean value by the below.
df_keys = df.keys()
for i in range(df_keys.size):
	df[df_keys[i]] = df[df_keys[i]].fillna(df[df_keys[i]].mean())
	
df2_keys = df2.keys()	
for i in range(df2_keys.size):
	df2[df2_keys[i]] = df2[df2_keys[i]].fillna(df2[df2_keys[i]].mean())	
	
#### Exploring the data ####
# Create arrays for features and target variable
y = df['SalePrice']
X = df.drop(['SalePrice'], axis=1) # all other variables considered
X_test_submit = df2 # For submission.


%pylab inline
%matplotlib inline
import seaborn as sns
sns.heatmap(df.corr(), square=True, cmap='RdYlGn')

# ------------------------------------------------------------------------------------- #
# [Linear Regression] - Train/test split

# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Create the regressor: reg_all
reg_linear = LinearRegression()

# Fit the regressor to the training data
reg_linear.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_linear.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_linear.score(X_test, y_test)))
# => R^2: 0.8216645356897889

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))
# => Root Mean Squared Error: 35276.64370170862
# => RMS Error is too large

# ------------------------------------------------------------------------------------- #
# [Bayesian Ridge]
# http://scikit-learn.org/stable/modules/linear_model.html
from sklearn import linear_model
reg_bayes = linear_model.BayesianRidge()
reg_bayes.fit(X_train, y_train)

y_pred_Bayesian = reg_bayes.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_bayes.score(X_test, y_test)))
# => R^2: 0.787867155954712

rmse = np.sqrt(mean_squared_error(y_test, y_pred_Bayesian))
print("Root Mean Squared Error: {}".format(rmse))
# => Root Mean Squared Error: 38474.44444722671
# => RMS Error is too large

# ------------------------------------------------------------------------------------- #
# [Lasso Reg]

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
clf.fit(X_train, y_train)
print(clf.coef_)
print(clf.intercept_)

y_pred_lasso = clf.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(clf.score(X_test, y_test)))
# => R^2: 0.8475662728328496

rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print("Root Mean Squared Error: {}".format(rmse))
# => Root Mean Squared Error: 32614.36555418206
# => RMS Error is too large

# ------------------------------------------------------------------------------------- #
#### [5-fold cross validation]

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg_linear = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg_linear, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)
# => [ 0.84503429  0.81338856  0.82233746  0.81762712  0.62919185]

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))
# => Average 5-Fold CV Score: 0.7855158564081826
# ------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------- #
#### Outputting the result

# Create the regressor: reg_all
# reg_all = LinearRegression()

# Fit the regressor to the training data
# reg_all.fit(X_train, y_train)
y_test_submit = reg_linear.predict(X_test_submit)
y_test_submit2 = reg_bayes.predict(X_test_submit)

ids = df2['Id']
# predictions = reg_all.predict(df2.drop('Id', axis=1))

output = pd.DataFrame({ 'Id' : ids, 'SalePrice': y_test_submit })
output.to_csv('house_youngseok.csv', index = False)
output.head()

#  	Id	SalePrice
# 0	1461	113324.100113
# 1	1462	124421.942282
# 2	1463	171843.120405
# 3	1464	197641.067813
# 4	1465	195574.027440