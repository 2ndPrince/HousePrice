# Preprocessing
# Need to encode categorical feature numerically
# Convert to 'dummy variables'
# 0: Observation was not that categorym, 1 was

# Dealing with categorical features in Python
# scikit-learn: OneHotEncoder()
# pandas: get_dummies()

# import pandas as pd
# df = pd.read_csv('auto.csv')
# df_origin = pd.get_dummies(df)
# print(df_origin.head())

# df_origin = df_origin.drop('origin_Asia', axis=1)
# print(df_origin.head())
# With this processing, we could proceed.

# [Exploring categorical features]
# Region data has good information
# Import pandas
import pandas as pd 

# Read 'gapminder.csv' into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create a boxplot of life expectancy per region
df.boxplot('life', 'Region', rot=60)

# Show the plot
plt.show()

# [Creating dummy variables]
# Create dummy variables: df_region
df_region = pd.get_dummies(df)

# Print the columns of df_region
print(df_region.columns)

# Create dummy variables with drop_first=True: df_region
df_region = pd.get_dummies(df, drop_first=True)

# Print the new columns of df_region
print(df_region.columns)
# Now we could use Region data into our model

# [Regression with categorical features]

# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Instantiate a ridge regressor: ridge
ridge = Ridge(alpha=0.5, normalize=True)

# Perform 5-fold cross-validation: ridge_cv
ridge_cv = cross_val_score(ridge, X, y, cv=5)

# Print the cross-validated scores
print(ridge_cv)

# ------------------------------------------------------------------------------------- #
# Handling missing data

# Dropping missing data
# df.insulin.replace(0, np.nan, inplace=True)
# df.triceps.replace(0, np.nan, inplace=True)
# df.bmi.replace(0, np.nan, inplace=True)
# df.info()

# 1. Drop the rows with missing data
# df = df.dropna()
# df.shape -> many data is missing and this is unacceptable

# 2. Imputing missing data
# Making an educated guess about the missing values
# Using the mean of the non-missing entries

# from sklaern.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0) # axis=0 : column # axis=1 : row.
# imp.fit(X)
# X = imp.transform(X) # imputing is known as transformers

# Imputing within a pipeline
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import Imputer
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
# logreg = LogisticRegression()
# steps = [('imputation', imp), ('logistic_regression', logreg)] # tuples
# pipeline = Pipeline(steps)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
# pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)
# pipeline.score(X_test, y_test)

# [Dropping missing data]
# Convert '?' to NaN
df[df == '?'] = np.nan # 'NaN should be np.nan

# Print the number of NaNs
print(df.isnull().sum())

# Print shape of original DataFrame
print("Shape of Original DataFrame: {}".format(df.shape))

# Drop missing values and print shape of new DataFrame
df = df.dropna()

# Print shape of new DataFrame
print("Shape of DataFrame After Dropping All Rows with Missing Values: {}".format(df.shape))

# [Imputing missing data in a ML Pipeline 1]
# Import the Imputer module
from sklearn.preprocessing import Imputer
from sklearn.svm import SVC

# Setup the Imputation transformer: imp
imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)

# Instantiate the SVC classifier: clf
clf = SVC()

# Setup the pipeline with the required steps: steps
steps = [('imputation', imp),
        ('SVM', clf)]

# [Imputing missing data in a ML Pipeline 2]
# Import necessary modules
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='most_frequent', axis=0)),
        ('SVM', SVC())]

# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the pipeline to the train set
pipeline.fit(X_train, y_train)

# Predict the labels of the test set
y_pred = pipeline.predict(X_test)

# Compute metrics
print(classification_report(y_test, y_pred))
# ------------------------------------------------------------------------------------- #
# Centering and scaling
# Motivation for data scaling:
# Many models use some form of distance to inform them
# Features on larger scales can unduly influence the model
# We want features to be on a similar scales
# => Normalizing (or scaling and centering)

# Ways to normalize the data
# Standardization: subtract the mean and divide by variance
# -> all features are centered around zero and have variance one
# Can also subtract the minimum and divide by the range
# -> Minimum zero and maximum one

# from sklearn.preprocessing import scale
# X_scaled = scale(X)
# np.mean(X), np.std(X)
# => (8.13, 16.72)
# np.mean(X_scaled), np.std(X_scaled)
# => (2.54e-15, 1.0)

# from sklearn.preprocessing import StandardScaler
# steps = [('scaler', StandardScaler()), 
#			('knn', KNeighborsClassifier())]
# pipeline = Pipeline(steps)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=21)		
# knn_scaled = pipeline.fit(X_train, y_train)
# y_pred = pipeline.predict(X_test)

# accuracy_score(y_test, y_pred)
# -> 0.956

# knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)
# knn_unscaled.score(X_test, y_test)
# -> 0.928

# CV(Cross validation) and scaling in a pipeline

# stpes =[('scaler', StandardScaler()), (('knn', KNeighborsClassifier())]
# pipeline = Pipeline(steps)
# parameters = {knn__n_neighbors=np.arange(1,50)}
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=21)	
# cv = GridSearchCV(pipeline, param_grid=parameters)
# cv.fit(X_train, y_train)
# y_pred = cv.predict(X_test)

# print(cv.best_params_)
# -> {'knn_n_neighbors': 41}

# print(cv.score(X_test, y_test))
# -> 0.956

# print(classification_report(y_test, y_pred))
# -> precision, recall f1-score, support 


# [Centering and scaling your data]
# Import scale
from sklearn.preprocessing import scale

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))


# [ Centering and scaling in a pipeline]
# Import the necessary modules
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Setup the pipeline steps: steps
steps = [('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())]
        
# Create the pipeline: pipeline
pipeline = Pipeline(steps)

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

# Fit the pipeline to the training set: knn_scaled
knn_scaled = pipeline.fit(X_train, y_train)

# Instantiate and fit a k-NN classifier to the unscaled data
knn_unscaled = KNeighborsClassifier().fit(X_train, y_train)

# Compute and print metrics
print('Accuracy with Scaling: {}'.format(knn_scaled.score(X_test, y_test)))
print('Accuracy without Scaling: {}'.format(knn_unscaled.score(X_test, y_test)))


# [Bringing it all together 1: Pipeline for classification]

# Setup the pipeline
steps = [('scaler', StandardScaler()),
         ('SVM', SVC())]

pipeline = Pipeline(steps)

# Specify the hyperparameter space
# hyperparameter space using the following notation: 'step_name__parameter_name'
parameters = {'SVM__C':[1, 10, 100],
              'SVM__gamma':[0.1, 0.01]}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=21)	

# Instantiate the GridSearchCV object: cv
cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
cv.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred = cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))
print("Tuned Model Parameters: {}".format(cv.best_params_))


# [Bringing it all together 2: Pipeline for regression]

# Setup the pipeline steps: steps
steps = [('imputation', Imputer(missing_values='NaN', strategy='mean', axis=0)),
         ('scaler', StandardScaler()),
         ('elasticnet', ElasticNet())]

# Create the pipeline: pipeline 
pipeline = Pipeline(steps)

# Specify the hyperparameter space
# Specify the hyperparameter space for the l1l1 ratio using the following notation: 
# 'step_name__parameter_name'. Here, the step_name is elasticnet, and the parameter_name is l1_ratio
parameters = {'elasticnet__l1_ratio':np.linspace(0,1,30)}

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)	

# Create the GridSearchCV object: gm_cv
gm_cv = GridSearchCV(pipeline, param_grid=parameters)

# Fit to the training set
gm_cv.fit(X_train, y_train)

# Compute and print the metrics
r2 = gm_cv.score(X_test, y_test)
print("Tuned ElasticNet Alpha: {}".format(gm_cv.best_params_))
print("Tuned ElasticNet R squared: {}".format(r2))

# ------------------------------------------------------------------------------------- #
# What you've learned
# ML techniques to build predictive models for regression and classification problems
# Underfitting and overfitting
# Test-train split
# Cross-validation(CV)
# Grid Search
# Regularization, lasso and ridge regression
# Data preprofessing
# For more: Check out the scikit-learn documentation
















