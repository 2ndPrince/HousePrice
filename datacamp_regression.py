# boston housing data
# boston = pd.read_csv('boston.csv')
# boston.head()
# X = boston.drop('MEDV', axis=1).values
# y = boston['MEDV'].values # values return numpy arrays

# Predicting house value from a single feature
# X_rooms = X[:,5] # fifth column

# type(X_rooms), type(y)
# => (numpy.ndarray, numpy.ndarray) # both numpy arrays
# y = y.reshape(-1,1)
# X_rooms = X_rooms.reshape(-1,1)

# plt.scatter(X_rooms, y)
# plt.ylabel('Value of house /1000 ($)')
# plt.xlabel('Number of rooms')
# plt.show()

# import numpy as np
# from sklearn import linear_model
# reg = linear_model.LinearRegression()
# reg.fit(X_rooms, y)
# prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)
# plt.scatter(X_rooms, y, color='blue')
# plt.plot(prediction_space, reg.predict(prediction_space),
# ... color='black', linewidth=3)
# plt.show()

# ------------------------------------------------------------------------------------- #
# [Loading the data]

# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life']
X = df['fertility']

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape)) # (139,)
print("Dimensions of X before reshaping: {}".format(X.shape)) # (139,)

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape)) # (139,1)
print("Dimensions of X after reshaping: {}".format(X.shape)) # (139,1)

# [Exploring the data]
# sns.heatmap(df.corr(), square=True, cmap='RdYlGn') # this was given for correlation.
# life and fertility are negatively correlated.
df.shape()
df.life.mean()

# Regression mechanics: y=ax+b, y: target, x:single feature, a&b:parameters of model
# how do we choose a and b?
# => Define an error function for any given line
# => Choose the line that minimizes the error function
# => For a line, calcaulte the vertical distance(Residual) between the point and line
# (Ordinary least squares (OLS): Minimize sum of squares of residuals
# this is not same as mean square erorr. see statistics curriculum for more details)
# Scikit-learn API fit function performs this OLS.

# Linear regression in higher dimensions: y=a1x1+a2x2+a3x3+....+b
# Must specify coefficient for each feature and the variable b
# Scikit-learn API works exactly the same way: pass two arrays: feature, and target

# Linear regression on all features
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# reg_all = linear_model.LinearRegression()
# reg_all.fit(X_train, y_train)
# y_pred = reg_all.predict(X_test)
# reg_all.score(X_test, y_test)
# => In real world, we never do like this. We need regularization which will be covered soon.

# [Fit & predict for regression]
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

#### [Train/test split for regression]

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

# ------------------------------------------------------------------------------------- #
# Cross-validation's motivation
# : Model performance is dependent on way the data is split
# : Not representative of the model's ability to generalize
# => Split the dataset by 5 Folds. Hold the first fold as test data and the rest as training
# => 5 folds = 5-fold CV(cross validation) ==> k-fold CV: the more fold, the more computationally expensive CV becomes.

# from sklearn.model_selection import cross_val_score
# reg = linear_model.LinearRegression()
# cv_results = cross_val_score(reg, X, y, cv=5)
# print(cv_results)
# np.mean(cv_results)

# [5-fold cross validation]

# Import the necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

# [K-Fold CV comparison]
# in ipython shell, use %timeit to measure computation time
# %timeit cross_val_score(reg, X, y, cv = ____)
# ------------------------------------------------------------------------------------- #

# Regularized regression
# Why regularize?
# Recall: Linear regression minimizes a loss function
# It chooses a coefficient for each feature variable
# Large coefficients can lead to overfitting
# Penalizing large coefficients: Regularization

# 1. Rdige regression
# Loss function = OLS loss function + alpha*sum(all (a_n coefficients **2))
# a_n variable(a1, a2, a3 ...) with large positive or negative will be penalized
# Alpha(or lambda) is the parameter we need to choose: similar to picking k in k-NN : Hyperparameter tuning (more in chapter3)
# Alpha controls model complexity
# Alpha 0 : OLS (can lead to overfitting)
# Alpha very high : Can lead to underfitting

# from sklearn.linear_model import Ridge
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
# ridge = Ridge(alpha=0.1, normalize=True)
# ridge.fit(X_train, y_train)
# ridge_pred = ridge.predict(X_test)
# ridge.score(X_test, y_test)

# 2. Lasso regression
# Loss function = OLS loss function + alpha*sum(all abs(a_n coefficients))
# from sklearn.linear_model import Lasso
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
# lasso = Lasso(alpha=0.1, normalize=True)
# lasso.fit(X_train, y_train)
# lasso_pred = lasso.predict(X_test)
# lasso.score(X_test, y_test)

##
# Lasso regression for feature selection
# => Can be used to select import features(variables) of a dataset
# => lasso shrinks the coefficients of less important features to exactly 0

# from sklearn.linear_model import Lasso
# names = boston.drop('MEDV', axis=1).columns
# lasso = Lasso(alpha=0.1)
# lasso_coeff = lasso.fit(X,y).coef_
# _ = plt.plot(range(len(names)), lasso_coef)
# _ = plt.xticks(range(len(names)), names, rotation=60)
# - = plt.ylabel('Coefficients')
# plt.show()
# which features are important predictor
##

# [Regularization 1: Lasso]

# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso.fit(X,y)

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

# [Regularization 2: Ridge]

def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
	
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)
# => Note that CV score change with different alphas. Which alpha to choose? 
# => fine-tune model. next chapter.









