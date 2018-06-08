# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 13:04:16 2017

@author: naveenkumar
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# Importing the dataset
dataset = pd.read_csv("C:/Users/naveenkumar/Desktop/dataset_1.csv")

#pre processing --- last column is NAN so ignoring
dataset = dataset.drop(dataset.index[len(dataset)-1])
dataset['Employee Annual Salary'] = dataset['Employee Annual Salary'].str.replace('\$','')
dataset['Employee Annual Salary'] = dataset['Employee Annual Salary'].convert_objects(convert_numeric=True)

#bins = [0.00,60000.00,85000.00,100000.00,200000.00]
#bins = [0.00, 20000.00, 55000.00, 95000.00, 200000.00]
bins = [0.00, 20000.00, 55000.00, 105000.00, 3500000.00]
labels = [ "low","medium","high","very high"]
Salary_Labels = pd.cut(dataset['Employee Annual Salary'], bins, labels=labels)
dataset["Salary_Labels"] = pd.cut(dataset["Employee Annual Salary"], bins, labels=labels)
dataset["Salary_bins"] = pd.cut(dataset["Employee Annual Salary"], bins, labels)

bin_values=pd.value_counts(dataset['Salary_Labels'])

# applying Label Encoder for all columns 
dataset =dataset.apply(LabelEncoder().fit_transform)
X = dataset[dataset.columns[1:3]]
y = dataset.iloc[:, 4].values

# Logistic Regression without balancing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


## Fitting Decision Tree Classification to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print( "CM", cm);
# Decision tree Classifier without balancing
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

## Without cross validation
# Cross Validation for Logistic
from sklearn.linear_model import LogisticRegression
def do_cross_val_LR(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    precision = []
    recall = []
    f_measure = []
    for train_idx, test_idx in cv:
        clf = LogisticRegression()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        #accuracies.append(acc)
        prec = precision_score(y[test_idx], predicted, average='weighted')
        f1 = f1_score(y[test_idx], predicted, average='weighted')
        recall1 = recall_score(y[test_idx], predicted, average='weighted') 
        accuracies.append(acc)
        precision.append(prec)
        recall.append(recall1)
        f_measure.append(f1)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute)
    print("Mean Squared Error", Mean_squared)
    print("LR", accuracies)
    avg = np.mean(accuracies)
    avg_prec = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f_measure)    
    return avg

# Cross Validation
def do_cross_val_Decision(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    for train_idx, test_idx in cv:
        clf = DecisionTreeClassifier()
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute)
    print("Mean Squared Error", Mean_squared)
    print("DT MSE", squared_error)
    print("DT MAE", mse_error)
    avg = np.mean(accuracies)
    return avg

## Cross validation for random forest    
from sklearn.ensemble import RandomForestClassifier
def do_cross_val_RForest(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    for train_idx, test_idx in cv:
        clf = RandomForestClassifier( )
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute)
    print("Mean Squared Error", Mean_squared)
    print("RF MSE", squared_error)
    print("RF MAE", mse_error)
    avg = np.mean(accuracies)
    return avg

    
##Solving the imbalanced property
# Cross Validation Deciision Tree with balanced data
from sklearn.tree import DecisionTreeClassifier
def do_cross_val_Decision_Balanced(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    for train_idx, test_idx in cv:
        clf = DecisionTreeClassifier(class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])       
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute)
    print("Mean Squared Error", Mean_squared)
    print("DT", accuracies)
    avg = np.mean(accuracies)
    return avg   
acc = do_cross_val_Decision_Balanced(np.array(X),y,10)
print("Accuracy", acc)

from sklearn.ensemble import RandomForestClassifier
# Decision tree Classifier without balancing
def do_cross_val_RF_Balanced(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    for train_idx, test_idx in cv:       
        clf = RandomForestClassifier(class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
        Mean_absolute = np.mean(mse_error)
        Mean_squared  = np.mean(squared_error)
        print("Mean Absolute Error", Mean_absolute);
        print("Mean Squared Error", Mean_squared);
    print("RF", accuracies)
    avg = np.mean(accuracies)
    return avg   
acc =do_cross_val_RF_Balanced(np.array(X),y,10)
print("Accuracy", acc)

# Logistic Regression for balanced
def do_cross_val_LR_Balanced(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    for train_idx, test_idx in cv:
        clf = LogisticRegression(class_weight='balanced')
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute);
    print("Mean Squared Error", Mean_squared);
    print("LR MSE", squared_error)
    print("LR MAE", mse_error)
    avg = np.mean(accuracies)
    return avg
acc =do_cross_val_LR_Balanced(np.array(X),y,10)
print("Accuracy", acc);


##Balancing by undersampling
from imblearn.under_sampling import RandomUnderSampler
# Applx.sizey the random under-sampling
rus = RandomUnderSampler(return_indices=True)
X_resampled, y_resampled, idx_resampled = rus.fit_sample(X, y)
print('Original dataset shape {}'.format(Counter(y)))
print('Under Sampled dataset shape {}'.format(Counter(y_resampled)))
## Decision Tree
acc =do_cross_val_Decision(np.array(X_resampled),y_resampled,10)
print("Accuracy", acc);
## Logistic Regression 
acc =do_cross_val_LR(np.array(X_resampled),y_resampled,10)
print("Accuracy", acc);
##Random Forest
acc =do_cross_val_RForest(np.array(X_resampled),y_resampled,10)
print("Accuracy", acc);

##Balancing by SMOTE
from imblearn.over_sampling import SMOTE 
print('Original dataset shape {}'.format(Counter(y)))
sm = SMOTE(random_state=0)
X_res, y_res = sm.fit_sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))
from imblearn.over_sampling import SMOTE 
print('Original dataset shape {}'.format(Counter(y)))
sm = SMOTE(random_state=42)
sm.fit(X, y)
X_res, y_res = sm.sample(X, y)
print('Resampled dataset shape {}'.format(Counter(y_res)))
X_res1, y_res1 = sm.fit_sample(X_res, y_res)
print('Resampled dataset shape {}'.format(Counter(y_res1)))
X_res2, y_res2 = sm.fit_sample(X_res1, y_res1)
print('Resampled dataset shape {}'.format(Counter(y_res2)))

## Decision Tree
acc =do_cross_val_Decision(X_res2,y_res2,10)
print("Accuracy", acc);
## Logistic Regression
acc =do_cross_val_LR(X_res2,y_res2,10)
print("Accuracy", acc);
## Random Forest
acc =do_cross_val_RForest(X_res2,y_res2,10)
print("Accuracy", acc);



from pylab import *

# make a square figure and axes
figure(1, figsize=(6,6))
ax = axes([0.1, 0.1, 0.8, 0.8])

# The slices will be ordered and plotted counter-clockwise.
labels = 'low', 'medium', 'high', 'very high'
fracs = [bin_values[0], bin_values[1], bin_values[2], bin_values[3]]
explode=(0, 0.05, 0, 0)

pie(fracs, explode=explode, labels=labels,
                autopct='%1.1f%%', shadow=True, startangle=90)
                # The default startangle is 0, which would start
                # the Frogs slice on the x-axis.  With startangle=90,
                # everything is rotated counter-clockwise by 90 degrees,
                # so the plotting starts on the positive y-axis.

title('distribution of salary', bbox={'facecolor':'0.8', 'pad':5})

show()
## plotting other graphs
import matplotlib.pyplot as plt

accs2 = [0.30974442521128193, 0.3275633845840546, 0.30842073108644741, 0.091233071988595871, 0.012219959266802444, 0.1459266802443992, 0.4053971486761711, 0.33380855397148679, 0.064969450101832998, 0.063951120162932792]
accs = [0.31626107321046737, 0.33082170858364729, 0.31290092658588736, 0.093167701863354033, 0.012627291242362525, 0.14572301425661915, 0.40386965376782075, 0.34175152749490834, 0.052443991853360489, 0.055804480651731159]
accs1 = [0.58808855628313061, 0.57748674773932018, 0.58172177167810357, 0.62507797878976923, 0.58265751715533376, 0.59451029320024951, 0.58858390517779169, 0.57579538365564564, 0.59232688708671244, 0.62414223331253904]


cs = [0,1,2,3,4,5,6,7,8,9]
plt.figure()
plt.plot(cs, accs, 'bo-', label='Random Forest Classifier',color="red")
plt.plot(cs, accs2, 'bo-', label='Decision Tree Classifier',color="green")
plt.plot(cs, accs1, 'bo-', label='Logistic Regression')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=5)
plt.xlabel('Folds')
plt.ylabel('MAE')
plt.savefig('classifier.png')

plt.show()

#Parameter Tuning
# Cross Validation
def do_cross_val(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    precision = []
    recall = []
    f_measure = []
    mse_error = []
    squared_error = []
    for train_idx, test_idx in cv:
        #clf = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=20,
        #                             random_state=99)
        #clf = DecisionTreeClassifier(min_samples_split=10, max_leaf_nodes=5,criterion='gini',
        #                             max_depth = None, min_samples_leaf= 1, random_state=99)
        clf = DecisionTreeClassifier(min_samples_split=20, criterion='gini',
                                     random_state=99)
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        prec = precision_score(y[test_idx], predicted, average='weighted')
        f1 = f1_score(y[test_idx], predicted, average='weighted')
        recall1 = recall_score(y[test_idx], predicted, average='weighted') 
        accuracies.append(acc)
        precision.append(prec)
        recall.append(recall1)
        f_measure.append(f1)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute)
    print("Mean Squared Error", Mean_squared)
    avg = np.mean(accuracies)    
    avg_prec = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f_measure)
    return avg,avg_prec,avg_recall,avg_f1
    
avg,avg_prec,avg_recall,avg_f1 =do_cross_val(np.array(X),y,10)
print("Accuracy", avg)
print("Precision", avg_prec)
print("Recall", avg_recall)
print("Accuracy", avg_f1)
    
# Cross Validation
def do_cross_val(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    precision = []
    recall = []
    f_measure = []
    for train_idx, test_idx in cv:
        #clf = LogisticRegression(random_state=99, max_iter=100, multi_class='ovr')
        #clf = LogisticRegression(random_state=99, max_iter=100, class_weight = 'balanced')
        #clf = LogisticRegression(random_state=99, max_iter=10000, class_weight = 'balanced')
        clf = LogisticRegression(random_state=99, max_iter=10000, multi_class='ovr')
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        #accuracies.append(acc)
        prec = precision_score(y[test_idx], predicted, average='weighted')
        f1 = f1_score(y[test_idx], predicted, average='weighted')
        recall1 = recall_score(y[test_idx], predicted, average='weighted') 
        accuracies.append(acc)
        precision.append(prec)
        recall.append(recall1)
        f_measure.append(f1)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute)
    print("Mean Squared Error", Mean_squared)
    avg = np.mean(accuracies)
    avg_prec = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f_measure)
    
    return avg,avg_prec,avg_recall,avg_f1
avg,avg_prec,avg_recall,avg_f1 =do_cross_val(np.array(X),y,10)
print("Accuracy", avg)
print("Precision", avg_prec)
print("Recall", avg_recall)
print("Accuracy", avg_f1)

from sklearn.ensemble import RandomForestClassifier
# Cross Validation
def do_cross_val_RForest(X, y, nfolds):
    """ Compute average cross-validation acccuracy."""
    cv = KFold(len(y), nfolds)
    accuracies = []
    mse_error = []
    squared_error = []
    for train_idx, test_idx in cv:
        #clf = DecisionTreeClassifier( min_samples_split=20,
        #                             random_state=99)
        #clf = DecisionTreeClassifier(min_samples_split=10, max_leaf_nodes=5,
        #                             max_depth = None, min_samples_leaf= 1, random_state=99)
        clf = DecisionTreeClassifier(min_samples_split=20, random_state=99)
                                     
        clf.fit(X[train_idx], y[train_idx])
        predicted = clf.predict(X[test_idx])
        acc = accuracy_score(y[test_idx], predicted)
        accuracies.append(acc)
        err1 = mean_absolute_error(y[test_idx], predicted)
        mse_error.append(err1)
        err2 = mean_squared_error(y[test_idx], predicted)
        squared_error.append(err2)
    Mean_absolute = np.mean(mse_error)
    Mean_squared  = np.mean(squared_error)
    print("Mean Absolute Error", Mean_absolute)
    print("Mean Squared Error", Mean_squared)
    avg = np.mean(accuracies)
    return avg
    return avg,avg_prec,avg_recall,avg_f1

acc =do_cross_val_RForest(np.array(X),y,10)
print("Accuracy", acc)   


from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
def box_plot(X,Y):
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('RF',RandomForestClassifier()))
    results=[]
    names = []
    for name,model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=42)
        cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')
        names.append(name)
        results.append(cv_results)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

box_plot(X,y)