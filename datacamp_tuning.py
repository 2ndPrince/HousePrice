# Fine-tuning your model
# In classification, accuracy is not always useful metric
# Class imbalance example: Email spam
# 99% of emails are real; 1% is spem. 
# Build a classifier to predict all emails are real => 99% accuracy
# => Need more nuanced metrics

# Confusion matrix
#				Predicted Spam		Predicted Email
# Actual Spam	True Positive		False Negative
# Actual Real	False Positive		True Negative
# => Positive class is typically of interest because spam makes positive class
# Why do we care? We can calculate metrics such as accuacy, precision, recall and F1 score.
# High precision: Not many real emails predicted as spam
# High recall: predicted most spam emails correctly

# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# knn = KNeighborsClassifier(n_neighbors=8)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test)
# print(confusion_matrix(y_test, y_pred)) # always first argument : true value(actual label)
# print(classification_report(y_test, y_pred)) # second argument: prediction

# [Metrics for classification]
# Import necessary modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Create training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Instantiate a k-NN classifier: knn
knn = KNeighborsClassifier(n_neighbors=6)

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#    [[176  30]
#     [ 52  50]]
#                 precision    recall  f1-score   support
#    
#              0       0.77      0.85      0.81       206
#              1       0.62      0.49      0.55       102
#    
#    avg / total       0.72      0.73      0.72       308

# [Logistic regression and the ROC curve]
# Despite its name, it is used to classification problem not regression.
# Logistic regression(Log Reg) outputs probabilties
# If p is greater than 0.5, the data is labeled 1. Below then 0.

# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split
# logreg = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)

# Probability thresholds. By default, log reg threshold = 0.5
# But it is not specific to logistic regression. k-NN classifiers also have thresholds
# What happens if we vary the threshold to the false and true positive rate?
# p=0. Model predicts 1 for all data. tp=fp=1
# p=1. Model predicts 0 for all data. tp=fp=0
# the ROC curve: the curve from varying all possible thresholds.
# Receiver Operating Characteristic Curve

# from sklearn.metrics import roc_curve
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr, tpr, label='Logistic Regression')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Logistic Regression ROC Curve')
# plt.show();

# We used predicted probability of this model.
# assigning a value of 1 to the observationing question.
# this is because to compute the ROC
# we do not merely want the prediction of the dataset
# but the probability that our log reg model outputs 
# before using a threshold to predict the label.
# to do this, we apply the method prdict_proba to the model
# and the test data
# logreg.predict_proba(X_test)[:,1]
# prddict_proba returns an array of two columns 
# each column contains the probability for the respect target value.
# we choose the second column, prob of predictive label being 1.
# ------------------------------------------------------------------------------------- #
# [Building a logistic regression model]

# Import the necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)

# Create the classifier: logreg
logreg = LogisticRegression()

# Fit the classifier to the training data
logreg.fit(X_train, y_train)

# Predict the labels of the test set: y_pred
y_pred =logreg.predict(X_test)

# Compute and print the confusion matrix and classification report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# [Plotting an ROC curve]
# ROC curve provides a nice visual way to access classfier's performance

# Import necessary modules
from sklearn.metrics import roc_curve

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# ------------------------------------------------------------------------------------- #
# [Precision-recall Curve]
# 1. A recall of 1 corresponds to a classifier with a low threshold in which all females 
# who contract diabetes were correctly classified as such, at the expense of many 
# misclassifications of those who did not have diabetes.
# 2. Precision is undefined for a classifier which makes no positive predictions, that is, 
# classifies everyone as not having diabetes.
# 3. When the threshold is very close to 1, precision is also 1, 
# because the classifier is absolutely certain about its predictions.

# Area under the ROC Curve = (AUC)
# Larger area under the ROC curve = better model
# Largest area when TF = 1 and FP = 0 (Upper left corner)
# To compute AUC, 

# 1. AUC in scikit-learn
# from sklearn.metrics import roc_auc_score
# logreg = LogisticRegression()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
# logreg.fit(X_train, y_train)
# y_pred_prob = logreg.predict_proba(X_test)[:,1]
# roc_auc_score(y_test, y_pred_prob)
# => 0.997466

# 2. AUC using cross-validation
# from sklearn.model_selection import cross_val_score
# cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
# print(cv_scores)
# => [ 0.9967 0.99183 0.99 ~~~ ]

# [AUC computation]
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]

# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))

# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')

# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))
# => AUC: 0.8254806777079764
# => AUC scores computed using 5-fold cross-validation: 
#    [ 0.80148148  0.8062963   0.81481481  0.86245283  0.8554717 ]

# ------------------------------------------------------------------------------------- #
# Hyperparameter tuning
# Linear Regression: Chooing parameters to fit the data the best
# Ridge/Lasso Regression: Choosing alpha
# k-Nearest Neighbors: Chooing n_neighbors
# Such parameters, need to be specified before fitting the model, are called hyperparameters
# => Hyperparameter cannot be learned by fitting the model

# Choosing the correct hyperparameters is the key for successful model
# Try a bunch of different hyperparameters values
# Fit all of them separately
# See how well each performs
# Choose the best performing one
# => It is essential to use cross-validation to avoid overfitting

# Grid search cross-validation
# Given, for example, two parameters to choose,
# We try all cases C[0.1~0.5] and Alpha[0.1~0.4] with kFold cross validation on each grid
# (combination of hyperparameters) and then choose the best performing ones.

# from sklearn.model_selection import GridSearchCV
# param_grid = {'n_neighbors': np.arange(1,50)}
# knn = KNeighborsClassifier()
# knn_cv = GridSearchCV(knn, param_grid, cv=5) #cv is number of folds that we wish to use
# knn_cv.fit(X, y) # grid search performed
# knn_cv.best_params_
# =>{'n_neighbors': 12}
# knn_cv.best_score_
# => 0.933216

# [Hyperparameter tuning with GridSearchCV]
# Import necessary modules
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Setup the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Instantiate a logistic regression classifier: logreg
logreg = LogisticRegression()

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the data
logreg_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_)) 
print("Best score is {}".format(logreg_cv.best_score_))

# [Hyperparameter tuning with RandomizedSearchCV]
# GridSearch can be computationally expensive.
# Alternatives are RandomizedSearchCV
# Decision Tree(Ideal use for RandomizedSearchCV) has parameters:
# => max_features, max_depth, min_samples_leaf 

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))

# This never outperforms GridSearchCV, but it is useful saving the computation time.
# ------------------------------------------------------------------------------------- #
# Hold-out set reasoning
# => We want to be absolutely certain about model's ability to genernalize to unseen data.















