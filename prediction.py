import numpy as np
import pandas as pd 
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix ,classification_report,precision_score, recall_score 
,f1_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC 
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('dataset.csv')
data.head()
data.info()
data.describe()
data.describe().columns
data.rename(columns={'chest pain type': 'chest_pain_type', 'resting bp s': 'resting_bp_s',
 'fasting blood sugar': 'fasting_blood_sugar',
 'resting ecg': 'resting_ecg',
 'max heart rate': 'max_heart_rate',
 'exercise angina': 'exercise_angina',
 'ST slope':'ST_slope'}, inplace=True)
data_num = data[['age','resting_bp_s','cholesterol','max_heart_rate','oldpeak']]
data_cat =data[['sex','chest_pain_type','fasting_blood_sugar','resting_ecg','exercise_angina']]
data.describe().columns
for i in data_num.columns:
 plt.hist(data_num[i])
 plt.title(i)
 plt.show()
pd.pivot_table(data, index='target', values=['age','resting_bp_s','cholesterol','max_heart_rate','oldpeak'])
for i in data_cat.columns:
 sns.barplot(data_cat[i].value_counts().index,data_cat[i].value_counts()).set_title(i)
 plt.show()
print(pd.pivot_table(data,index='target',columns='sex', values='age'))
print("="*100)
print(pd.pivot_table(data,index='target',columns='cholesterol', values='age'))
print("="*100)
print(pd.pivot_table(data,index='target',columns='fasting_blood_sugar', values='age'))
print("="*100)
print(pd.pivot_table(data,index='target',columns='resting_ecg', values='age'))
print("="*100)
print(pd.pivot_table(data,index='target',columns='exercise_angina', values='age'))
print("="*100)
print(pd.pivot_table(data,index='target',columns='sex', values='cholesterol'))
print("="*100)
#print(pd.pivot_table(data,index='target',columns='cholesterol', values='cholesterol'))
#print("="*100)
print(pd.pivot_table(data,index='target',columns='fasting_blood_sugar', values='cholesterol'))
print("="*100)
print(pd.pivot_table(data,index='target',columns='resting_ecg', values='cholesterol'))
print("="*100)
print(pd.pivot_table(data,index='target',columns='exercise_angina', values='cholesterol'))
for i in data_num.columns:
 sns.boxplot(data_num[i])
 plt.title(i)
 plt.show()
def outlinefree(dataCol):
 sorted(dataCol)
 Q1,Q3 = np.percentile(dataCol,[25,75])
 IQR = Q3-Q1
 LowerRange = Q1-(1.5 * IQR)
 UpperRange = Q3+(1.5 * IQR)
 return LowerRange,UpperRange
lwtrtbps,uptrtbps = outlinefree(data['resting_bp_s'])
lwchol,upchol = outlinefree(data['cholesterol'])
lwoldpeak,upoldpeak = outlinefree(data['oldpeak'])
data['resting_bp_s'].replace(list(data[data['resting_bp_s'] > uptrtbps].resting_bp_s) ,uptrtbps,inplace=True)
data['cholesterol'].replace(list(data[data['cholesterol'] > upchol].cholesterol) ,upchol,inplace=True)
data['oldpeak'].replace(list(data[data['oldpeak'] > upoldpeak].oldpeak) ,upoldpeak,inplace=True)
features = data.iloc[:,:-1].values
label = data.iloc[:,-1].values
#### Logistic_Regression ####
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=102)
classimodel= LogisticRegression() 
classimodel.fit(X_train, y_train)
trainscore = classimodel.score(X_train,y_train)
testscore = classimodel.score(X_test,y_test) 
print("test score: {} train score: {}".format(testscore,trainscore),'\n')
y_pred = classimodel.predict(X_test)
#from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)
print(' f1 score: ',f1_score(y_test, y_pred),'\n')
print(' precision score: ',precision_score(y_test, y_pred),'\n')
print(' recall score: ',recall_score(y_test, y_pred),'\n')
print(classification_report(y_test, y_pred))
#### Support_Vector_Machine ####
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=8) 
svcmodel = SVC(probability=True) 
svcmodel.fit(X_train, y_train) 
trainscore = svcmodel.score(X_train,y_train)
testscore = svcmodel.score(X_test,y_test) 
print("test score: {} train score: {}".format(testscore,trainscore),'\n')
y_predsvc = svcmodel.predict(X_test)
print(confusion_matrix(y_test, y_predsvc))
print("f1_score: ",f1_score(y_test, y_predsvc),'\n')
print("precision_score: ",precision_score(y_test, y_predsvc),'\n')
print("recall_score: ",recall_score(y_test, y_predsvc),'\n')
print(classification_report(y_test, y_predsvc),'\n')
#### KNN ####
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=193) 
classifier= KNeighborsClassifier() 
knnmodel = classifier.fit(X_train, y_train) 
trainscore = knnmodel.score(X_train,y_train)
testscore = knnmodel.score(X_test,y_test) 
print("test score: {} train score: {}".format(testscore,trainscore),'\n')
y_predknn = knnmodel.predict(X_test)
print(confusion_matrix(y_test, y_predknn))
print("f1_score: ",f1_score(y_test, y_predknn),'\n')
print("precision_score: ",precision_score(y_test, y_predknn),'\n')
print("recall_score: ",recall_score(y_test, y_predknn),'\n')
print(classification_report(y_test, y_predknn))
#### Navie Baye's ####
X_train, X_test, y_train, y_test= train_test_split(features,label, test_size= 0.25, random_state=34) 
NBmodel = GaussianNB() 
NBmodel.fit(X_train, y_train) 
trainscore = NBmodel.score(X_train,y_train)
testscore = NBmodel.score(X_test,y_test) 
#print("test score: {} train score: {}".format(testscore,trainscore),'\n')
y_predNB = NBmodel.predict(X_test)
print(confusion_matrix(y_test, y_predNB))
print("f1_score: ",f1_score(y_test, y_predNB),'\n')
print("precision_score: ",precision_score(y_test, y_predNB),'\n')
print("recall_score: ",recall_score(y_test, y_predNB),'\n')
print(classification_report(y_test, y_predNB))
prediction=NBmodel.predict([[40,1,2,140,289,0,0,172,0,0,1]])
prediction
output=int((prediction).astype(int))
output
import pickle
# open a file, where you ant to store the data
#file = open('capstone_project.pkl', 'wb')
# dump information to that file
#pickle.dump(NBmodel, file)
filename = 'capstone_project.pkl'
with open(filename, 'rb') as f:
 classification_dict = pickle.load(f)