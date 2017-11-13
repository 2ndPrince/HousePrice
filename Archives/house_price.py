# House Prices
%pylab inline
%matplotlib inline

# [CSV to DataFrame]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data_train = pd.read_csv('C:/Users/User/Desktop/DataScience/Contest/House_Prices/train.csv')
data_test = pd.read_csv('C:/Users/User/Desktop/DataScience/Contest/House_Prices/test.csv')

data_train.sample(3)

# [Visualizing Data]

# data_train.describe() # This works only for numeric data.


# Visualizing data is crucial for recognizing 
# underlying patterns to exploit in the model.

# sns.barplot(x="YearBuilt", y="SalePrice", data=data_train);
# sns.barplot(x="Pclass", y="Survived", hue="Sex", data=data_train);


# sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=data_train,
#              palette={"male": "blue", "female": "pink"},
#              markers=["*", "o"], linestyles=["-", "--"]);

	
# Transforming features into categories.
def simplify_years(df):
    df.YearBuilt = df.YearBuilt.fillna(-0.5)
    bins = (1850, 1900, 1950, 1960, 1980, 1990, 2000, 2005, 2010)
    group_names = ['Oldest', 'grp1', 'grp2', 'grp3', 'grp4', 'grp5', 'grp6', 'grp7']
    categories = pd.cut(df.YearBuilt, bins, labels=group_names)
    df.YearBuilt = categories
    return df
data_train = simplify_years(data_train)
sns.barplot(x="YearBuilt", y="SalePrice", data=data_train);	
sns.barplot(x="YearRemodAdd", y="SalePrice", data=data_train);	

sns.barplot(x="MSSubClass", y="SalePrice", data=data_train);	
sns.barplot(x="MSZoning", y="SalePrice", data=data_train);	

from sklearn import datasets, linear_model
regr = linear_model.LinearRegression()

# What I want at this point is
# 1. Declare a linear model object: obj = linear.model()
# 2. Fit x and y variables based on train data
# 3. Predict the test data
# 4. Output the result

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



from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# display coefficients
print(regressor.coef_)

X = df["YearBuilt"]
y = 

iris.data.shape
iris.target_names
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns = iris.feature_names)
	
def simplify_cabins(df):
    df.Cabin = df.Cabin.fillna('N')
    df.Cabin = df.Cabin.apply(lambda x: x[0])
    return df

def simplify_fares(df):
    df.Fare = df.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(df.Fare, bins, labels=group_names)
    df.Fare = categories
    return df

def format_name(df):
    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])
    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])
    return df    
    
def drop_features(df):
    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)

def transform_features(df):
    df = simplify_ages(df)
    df = simplify_cabins(df)
    df = simplify_fares(df)
    df = format_name(df)
    df = drop_features(df)
    return df

data_train = transform_features(data_train)
data_test = transform_features(data_test)
# data_train.head()

# sns.barplot(x="Age", y="Survived", hue="Sex", data=data_train);
# sns.barplot(x="Cabin", y="Survived", hue="Sex", data=data_train);
# sns.barplot(x="Fare", y="Survived", hue="Sex", data=data_train);


# Final Encoding to normalize labels. LabelEncoder in Scikit-learn convert each unique 
# string value into a number, making out data more flexible for various algorithms.

from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
data_train, data_test = encode_features(data_train, data_test)
# data_train.head()	


# Split up the dataset for train & test

from sklearn.model_selection import train_test_split

X_all = data_train.drop(['Survived', 'PassengerId'], axis=1)
y_all = data_train['Survived']

num_test = 0.20
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)

# Tuning and fitting an Algorithm

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, y_train)

predictions = clf.predict(X_test)
print(accuracy_score(y_test, predictions))


# Validate the effectiveness of the model using KFold
from sklearn.cross_validation import KFold

def run_kfold(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Fold {0} accuracy: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Mean Accuracy: {0}".format(mean_outcome)) 

run_kfold(clf)

# Outputting the result
ids = data_test['Id']
predictions = clf.predict(data_test.drop('Id', axis=1))


output = pd.DataFrame({ 'Id' : ids, 'SalePrice': predictions })
output.to_csv('house-predictions_youngseok.csv', index = False)
output.head()

# [Editor's comment]
# @Cielo. Sounds like overfitting. The joy of machine learning is finding 
# the optimal bias-variance tradeoff. Here's a few ways the model could be improved.

# Better preprocessing or feature engineering. Do extensive visualization on the data, 
# then try to isolate important features or create new ones.

# Better model tuning. Try experimenting with parameters in the RandomForestClassifier docs, 
# or try a different algorithm all together.

# BTW. Users with very high scores, say high 80's and up, are training against 
# the public test set by trial and error, just overfitting to the test data. In ranked competitions, 
# Kaggle uses two test sets (public and private) to prevent these types of models from succeeding, 
# as they are not valuable in the real world.
