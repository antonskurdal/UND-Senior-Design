#!/usr/bin/env python
# coding: utf-8

# In[410]:


#Load random forest classifier
from sklearn.ensemble import RandomForestClassifier


# In[525]:


#Pandas
import pandas as pd


# In[526]:


#Numpy
import numpy as np


# In[527]:


import sys
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', 100000)


# In[528]:


#Seed
seed = 1
print("Seed: ", seed)
np.random.seed(seed)


# In[529]:


#Create dataset object
fname = "FDIinject3.csv"
dataset = pd.read_csv(fname)


# In[530]:


#Drop bad columns
del dataset["Date"]
del dataset["Time"]


# In[558]:


#Create Test and Train data
dataset['is_train'] = np.random.uniform(0, 1, len(dataset)) <= .65
pd.set_option('display.max_columns', None)


# In[559]:


#Create train and test datasets
train = dataset[dataset['is_train'] == True]
test = dataset[dataset['is_train'] == False]


# In[560]:


#Show lengths of each
print("Training length: ", len(train))
print("Testing length: ", len(test))


# In[561]:


#Create list of feature column's names
features = dataset.columns[:1]
#print("Features",features)


# In[562]:


#Converting each output possibility into digits
y_train = pd.factorize(train['Output'])[0]
#print(y_train)


# In[563]:


#Creating random forest classifier
clf = RandomForestClassifier(n_jobs = 2, random_state = 0)


# In[564]:


#Training the classifier
clf.fit(train[features], y_train)
#print(clf)


# In[565]:


#Apply the trained classifier to the test
#print(test[features])
#print(clf.predict(test[features]))


# In[566]:


#View the predicted probabilities of the first 10 observations
#print(clf.predict_proba(test[features])[0:10])

#from 10 to 20
#print(clf.predict_proba(test[features])[10:20])

#Failed attempt to map names the 'easy' way
#preds = dataset['Output'][clf.predict(test[features])]
preds = clf.predict(test[features])


# In[567]:


#Map names the actual easy way
preds_read = []

for x in range(len(preds)):
	#print(preds[x])
	if preds[x] == 0:
		preds_read.append("Attack")
	else:
		preds_read.append("Normal")

#print (preds_read)


# In[568]:


#Calculate prediction accuracy
accuracy = []
correct = 0
incorrect = 0


# In[569]:


test2 = test['Output'].to_numpy()


# In[570]:


for x in range(len(preds_read)):
	if preds_read[x] == test2[x]:
		#print("Correct!")
		correct = correct + 1
	else:
		#print("Incorrect.")
		incorrect = incorrect + 1


#print("TEST IDX:",test.iloc[9]['Output'])


# In[571]:


test_actual_labels = test['Output'].to_numpy()
test_actual = []
for label in test_actual_labels:
    if label == 'Attack':
        test_actual.append(0)
    else:
        test_actual.append(1)
y_test = np.array(test_actual)


# In[572]:


#Print accuracy numbers
print("\nPREDICTIONS:")
print("Correct:", correct)
print("Incorrect: ", incorrect)

total = correct + incorrect
percentRight = correct/total

print("Accuracy:% 0.3f%%" %(percentRight * 100))

#View the PREDICTIONS for the first 5 observations
#print(preds[0:5])

#View the ACTUAL first 5 observations
#print(test['Output'].head())


# In[573]:


#Random Forest hyperparameters

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100,stop = 1000, num = 10)] # 10 was best out of [10, 120, 230, 340, 450, 560, 670, 780, 890, 1000]
# Number of features to consider at every split
max_features = ['auto'] # auto > sqrt and log2
# Measures quality of the split
criterion= ['entropy','gini'] # 'entropy' better than 'gini index'
# Maximum number of levels in tree
max_depth = [10] #10 best out of [2,5,10,15,20,50]
# Minimum number of samples required to split a node
min_samples_split = [10] # 10 best out of [2,5,10,20]
# minimum number of samples required at each leaf node
min_samples_leaf = [15] # 15 best out of [5,10,15,20,25,30]
# Method of selecting samples for training each tree
bootstrap = [True] # False was best out [True, False]



# In[574]:


#Creating the Param Grid

param_grid = {'n_estimators' : n_estimators,
              'criterion' : criterion,
             'max_features' : max_features,
             'max_depth' : max_depth,
             'min_samples_split' : min_samples_split,
             'min_samples_leaf': min_samples_leaf,
             'bootstrap' : bootstrap}
print(param_grid)


# In[575]:


from sklearn.model_selection import GridSearchCV
clf_Grid = GridSearchCV(clf,param_grid = param_grid, cv=3, verbose=10, n_jobs = -1)


# In[576]:


clf_Grid.fit(train[features], y_train)
clf_Grid.best_params_


# In[577]:


#print(f'Train Accuracy - : {clf_Grid.score(train[features], y_train)}')
print(f'Test Accuracy - : {clf_Grid.score(test[features], y_test)}')


# In[578]:


# # Make a numpy array of test labels

# test_predictions = clf_Grid.predict(test[features])

# def accuracy(predicted, actual):
#     if predicted.shape == actual.shape:
#         correct = 0
#         for x,y in zip(predicted, actual):
#             if x == y:
#                 correct +=1
#         return correct/len(predicted)
    


# In[579]:


# acc


# In[580]:


import seaborn as sns

table = pd.pivot_table(pd.DataFrame(clf_Grid.cv_results_),
    values='mean_test_score', index='param_n_estimators', 
                       columns='param_criterion')
     
sns.heatmap(table)


# In[581]:


from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score


# In[582]:


space = {'criterion': hp.choice('criterion', ['entropy', 'gini']),
        'max_depth': hp.quniform('max_depth', 1, 880, 10),
        'max_features': hp.choice('max_features', ['auto', 'sqrt','log2', None]),
        'min_samples_leaf': hp.uniform ('min_samples_leaf', 0, 0.5),
        'min_samples_split' : hp.uniform ('min_samples_split', 0, 1),
        'n_estimators' : hp.choice('n_estimators', [10, 50, 300, 750, 1200])
    }


# In[583]:


def objective(space):
    model = RandomForestClassifier(criterion = space['criterion'], 
                                   max_depth = space['max_depth'],
                                 max_features = space['max_features'],
                                 min_samples_leaf = space['min_samples_leaf'],
                                 min_samples_split = space['min_samples_split'],
                                 n_estimators = space['n_estimators'], 
                                 )
    accuracy = cross_val_score(model, test[features], y_test, cv = 4).mean()
    
    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -accuracy, 'status': STATUS_OK }
    
trials = Trials()
best = fmin(fn= objective,
            space= space,
            algo= tpe.suggest,
            max_evals = 60,
            trials= trials)
best   


# In[584]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'auto', 1: 'sqrt', 2: 'log2', 3: None}
est = {0: 10, 1: 50, 2: 300, 3: 750, 4: 1200}

trainedforest = RandomForestClassifier(criterion = crit[best['criterion']], 
                                       max_depth = best['max_depth'], 
                                       max_features = feat[best['max_features']], 
                                       min_samples_leaf = best['min_samples_leaf'], 
                                       min_samples_split = best['min_samples_split'], 
                                       n_estimators = est[best['n_estimators']]
                                      ).fit(train[features], y_train)
predictionforest = trainedforest.predict(test[features])
print(confusion_matrix(y_test,predictionforest))
print(classification_report(y_test,predictionforest))
acc5 = accuracy_score(y_test,predictionforest)


# In[ ]:





# In[ ]:




