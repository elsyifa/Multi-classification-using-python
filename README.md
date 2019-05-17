**Name: Syifa Silfiyana S (syifa.silfiyana@gmail.com)**



## Langkah-langkah pemecahan masalah
   Untuk memprediksi target pada data test ini bahasa pemprograman yang digunakan adalah python dan tool yang digunakan adalah jupyter notebook. Adapun langkah-langkah nya adalah sebagai berikut:
   
   a. Load data ke dalam pandas dataframe
   
   Dalam hal ini, data train dan data test diupload ke dalam bentuk pandas dataframe.
      
   b. Check data train dan data test
   
   Dalam step ini, baik data train maupun data test dicek seperti apa datanya mulai dari dimensi (jumlah kolom dan baris), type data di setiap kolom sampai summary statistic dari kedua data tersebut.
	  
   c. Check & Handle Missing values
   
   Mengecek apakah di dalam kedua data tersebut ada values yang kosong atau tidak. Seperti diketahui bahwa data-data yang kosong 
akan berpengaruh pada model prediksi dan bisa menyebabkan error pada saat modelling, sehingga data-data kosong tersebut perlu diisi atau dibuang namun membuang data sangat tidak dianjurkan. Ada banyak method yang dapat digunakan dalam mengisi data kosong, dalam hal ini data kosong pada variable-variable berjenis kategory akan diisi oleh modus  dari kategory tersebut sedangkan untuk variable-variable numeric makan data yang kosong akan diisi oleh nilai rata-ratanya. Dalam case ini karena semua variable adalah variable numeric maka data yang hilang diisi oleh nilai rata-ratanya.

   d. EDA (Exploratory Data Analyst)
   
   Pada langkah ini dibuat visualisasi dari setiap variable dan bagaimana visualisasinya terhadap target. Langkah ini dilakukan untuk memperoleh insight dari data tersebut, dalam hal ini data train.
      
   e. Check & Handle Outlier
   
   Outlier bisa juga mempengaruhi performa dari model yang dibuat. Oleh karena itu, dalam step ini semua variable numeric yang memiliki outlier diganti dengan nilai upper side atau lower sidenya. 
      
   f. Split data train into train and test
   
   Pada step ini, data train displit menjadi data train dan data test dengan proporsi 0.7 untuk data train dan 0.3 untuk data test. Data train yang sebanyak 0.7 tersebut akan digunakan untuk membuat model sedangkan 0.3 data test tadi akan digunakan sebagai validasi saat mengevaluasi performa model.
      
   g. Modelling, Prediksi dan Evaluation
   
   Algoritma machine learning yang digunakan untuk modelling dalam case ini ada 4 yaitu:
	  - Decision Tree
	  - Random Forest
	  - SVM
	  - Gradient Boosting
    Sedang metric yang digunakan untuk mengevaluasi performa model adalah ROC dan confusion matrix (accuracy, precision dan recall)
      
   h. Implemantasi model pada data test
   
   Ini merupakan tahap terakhir, pada tahap ini dilakukan prediksi target pada data test menggunakan model yang memiliki performa paling baik. Hasil prediksi tersebut disimpan ke dalam file .csv

## Jawaban Pertanyaan.
   1. Algoritma machine learning apa yang anda gunakan?
   
   Jawab: Gradient Boosting
      
   2. Apa alasan anda menggunakan algoritma tersebut?
   
   Jawab: Gradient Boosting memiliki model evaluasi yang paling bagus dari semua model yang digunakan.
   
   Accuracy: 0.788, 
   Precision Average : 0.79, 
   Recall Average : 0.79,
   ROC curve of class 0: 0.94,
   ROC curve of class 1: 0.90,
   ROC curve of class 2: 0.88
	     
   3. Variabel-variabel mana saja yang paling penting diantara (x_0, x_1, ..., x_19)?
   
   Jawab: top 10 variable-variable yang paling penting adalah x_7, x_3, x_17, x_4, x_2, x_10, x_0, x_8, x_5, x_1
      
   4. Apakah anda menemukan variabel yang collinear/redundant? Variabel mana sajakah itu?
   
   Jawab: ada, yaitu x_2, x_3, x_4, x_7, x_17
   
   5. Metric apa sajakah yang anda gunakan untuk mengevaluasi performansi model? Berapa nilai dari tiap metric tersebut?
   
   Jawab: Matrix yang digunakan adalah ROC dan confusion matrix.
   
   Gradient Boosting --> Accuracy nya: 0.78,
   Precision Average : 0.79,
   Recall Average : 0.79,
   ROC curve of class 0: 0.94,
   ROC curve of class 1: 0.90,
   ROC curve of class 2: 0.88,
   
   Random Forest --> Accuracy nya: 0.766,
   Precision Average : 0.76,
   Recall Average : 0.77,
   ROC curve of class 0: 0.95,
   ROC curve of class 1: 0.90,
   ROC curve of class 2: 0.86,
   
   Decison Tree --> Accuracy nya: 0.701,
   Precision Average : 0.71,
   Recall Average : 0.70,
   ROC curve of class 0: 0.89,
   ROC curve of class 1: 0.84,
   ROC curve of class 2: 0.70
   
   Support Vector Machine --> Accuracy nya: 0.33,
   Precision Average : 0.11,
   Recall Average : 0.33,
   ROC curve of class 0: 0.50,
   ROC curve of class 1: 0.50,
   ROC curve of class 2: 0.50

### Penejelasan Souce code
#### Load libraries
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
```
#### Load data
``` python
#load train dataset ke dalam pandas dan memberi nama columns
df_train = pd.read_csv('training_set.csv', names = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9','x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19','target'])

#menampilkan 5 baris pertama data train
df_train.head(5)

#load test dataset ke dalam pandas dan memberi nama columns
df_test = pd.read_csv('test_set.csv', names = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18', 'x_19'])

#menampilkan 5 baris pertama data test
df_test.head(5)
```
#### Check Dataset
``` python
#check data train information
df_train.info()

#define numerical variables
numcolumns = ['x_0', 'x_1', 'x_2', 'x_3', 'x_4', 'x_5', 'x_6', 'x_7', 'x_8', 'x_9', 'x_10', 'x_11', 'x_12', 'x_13', 'x_14', 'x_15', 'x_16', 'x_17', 'x_18','x_19']

#check data train dimension
df_train.shape

#change target data type to integer
df_train['target'] = df_train['target'].astype('int64')
df_train.dtypes

#check summary statistics of data train
df_train.describe()

#check target proportion in data train
df_train.target.value_counts()

#check data test dimension
df_test.shape

#check data test information
df_test.info()

#check summary statistics of data test
df_test.describe()
```
#### Check Missing Values
``` python
def missing_values_table(df_pd):
    """Input pandas dataframe and Return columns with missing value&percentage and stored as pandas dataframe"""
    mis_val = df_pd.isnull().sum() #count total of null in each columns in dataframe
    mis_val_percent = 100 * df_pd.isnull().sum() / len(df_pd) #count percentage of null in each columns
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)  #join to left (as column) between mis_val and mis_val_percent and create it as dataframe
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'}) #rename columns in table, mis_val to Missing Values and mis_val_percent to % of Total Values
    mis_val_table_ren_columns = mis_val_table_ren_columns[
    mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1) #sort column % of Total Values descending and round 1 after point(coma)
    print ("Your selected dataframe has " + str(df_pd.shape[1]) + " columns.\n"    #.shape[1] : just view total columns in dataframe  
    "There are " + str(mis_val_table_ren_columns.shape[0]) +              
    " columns that have missing values.") #.shape[0] : just view total rows in dataframe
    return mis_val_table_ren_columns

#check missing values in data train
a=missing_values_table(df_train)
a

#check missing values in data test
b=missing_values_table(df_test)
b
```
#### Handle Missing Values
``` python
#fill missing values in data train with average value
df_train=df_train.fillna(df_train.mean())

#Check missing values in data train after filling with average
df_train.isnull().sum()

#fill missing values in data test with average value
df_test=df_test.fillna(df_test.mean())

#Check missing values in data test after filling with average
df_test.isnull().sum()
```
#### EDA
Check distibusi pada semua data numeric di data train
``` python
#density plot x_0, x_1, x_2
plt.figure(figsize=(24,5))
plt.subplot(131)
sns.distplot(df_train['x_0'])
plt.subplot(132)
sns.distplot(df_train['x_1'])
plt.subplot(133)
sns.distplot(df_train['x_2'])
plt.show()

# Numerical variable VS Target
#show distribution x_0, x_1, x_2 in type of target
plt.figure(figsize=(20,8))
plt.subplot(131)
sns.kdeplot(df_train[df_train["target"]==0]["x_0"], label="0", color="green")
sns.kdeplot(df_train[df_train["target"]==1]["x_0"], label="1", color="red")
sns.kdeplot(df_train[df_train["target"]==2]["x_0"], label="2", color="blue")
plt.title("x_0 VS target")
plt.subplot(132)
sns.kdeplot(df_train[df_train["target"]==0]["x_1"], label="0", color="green")
sns.kdeplot(df_train[df_train["target"]==1]["x_1"], label="1", color="red")
sns.kdeplot(df_train[df_train["target"]==2]["x_1"], label="2", color="blue")
plt.title("x_1 VS target")
plt.subplot(133)
sns.kdeplot(df_train[df_train["target"]==0]["x_2"], label="0", color="green")
sns.kdeplot(df_train[df_train["target"]==1]["x_2"], label="1", color="red")
sns.kdeplot(df_train[df_train["target"]==2]["x_2"], label="2", color="blue")
plt.title("x_2 VS target")
plt.show()

# drop target from dataframe
df_train_cor=df_train.drop(['target'],1)

# check correlation each variables using heatmap
plt.figure(figsize=(20,8))
sns.heatmap(df_train_cor.corr(), annot=True)
plt.show()
```
#### Handle Outlier
``` python
#define dataframe for calculate quantile from data train
df_train_describe=df_train.describe()
df_train_describe

#function to calculate uppler side
def upper_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun upper value from the column"""
    q1 = b[c].iloc[4] #select value of q1 from the column
    q2 = b[c].iloc[5] #select value of q2 from the column
    q3 = b[c].iloc[6] #select value of q3 from the column
    IQR=q3-q1  #calculate the value of IQR
    upper= q3 + (IQR*1.5)   #calculate the value of upper side
    return upper

#function to calculate lower side
def lower_value(b,c):
    """Input is quantile dataframe and name of numerical column and Retrun upper value from the column"""
    q1 = b[c].iloc[4] #select value of q1 from the column
    q2 = b[c].iloc[5] #select value of q2 from the column
    q3 = b[c].iloc[6] #select value of q3 from the column
    IQR=q3-q1  #calculate the value of IQR
    lower= q1-(IQR*1.5)   #calculate the value of lower side
    return lower

#replace outlier with upper side in data train
for i in numcolumns:
    upper=upper_value(df_train_describe,i)
    df_train.loc[(df_train[i] > upper), i] = upper
    
#replace outlier with lower side in data train
for i in numcolumns:
    lower=lower_value(df_train_describe,i)
    df_train.loc[(df_train[i] < lower), i] = lower
    
#define dataframe for calculate quantile from data test
df_test_describe=df_test.describe()
df_test_describe

#replace outlier with upper side in data test
for i in numcolumns:
    upper=upper_value(df_test_describe,i)
    df_test.loc[(df_test[i] > upper),i] = upper
    
#replace outlier with lower side in data test
for i in numcolumns:
    lower=lower_value(df_test_describe,i)
    df_test.loc[(df_test[i] < lower),i] = lower
```

#### Features Importance
``` python
# use heatmap to see feature importance
corrmat= df_train.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
sns.heatmap(df_train[top_corr_features].corr(),annot=True)
plt.show()
```
#### Split data train to train and test data
``` python
#define X and Y
X = df_train.drop(['target'],1)
Y = df_train['target']

#split X, Y to 0.7 train and 0.3 test
train_X,test_X,train_y,test_y= train_test_split(X,Y,test_size=0.3,random_state=123)
```
#### Modelling & Evaluation
There are 4 algorithma machine learning that used for modelling in this case, they are:
 - Desicion Tree
 - Random Forest
 - SVM
 - Gradient Boosting
 
Model evaluation that used are confusion matrix and ROC, there is different on treating Y (target) multi-class at those both model evaluation. If want evaluate model using ROC so Y (target) have to be changed to binary firstly but does not if model evaluation that used is confusion matrix.

#### Decision Tree
``` python
#create model decision tree
decisiontree = DecisionTreeClassifier(random_state = 123, max_depth = 2)
decisiontree.fit(train_X, train_y)

#create prediction
dtpred = decisiontree.predict(test_X)

#evaluation model using confusion matrix
cm = confusion_matrix(test_y,dtpred)
print(cm)
acc= accuracy_score(test_y, dtpred)
print('acc:',acc)
print(classification_report(test_y, dtpred))

#evaluate model using ROC
# Binarize the output for ROC evaluation
train_y2 = label_binarize(train_y, classes=[0, 1, 2])
test_y2 = label_binarize(test_y, classes=[0, 1, 2])
n_classes= test_y2.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(decisiontree)
y_score = classifier.fit(train_X, train_y2).predict_proba(test_X)

# Compute ROC curve and ROC area for each class
lw=2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y2[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
```
#### Random Forest
``` python
#Create model using random forest
rf = RandomForestClassifier(n_estimators = 10, random_state = 123)
rf.fit(train_X, train_y)

#create prediction
rfpred = rf.predict(test_X)

#evaluation model using confusion matrix
cm = confusion_matrix(test_y,rfpred)
print(cm)
acc= accuracy_score(test_y, rfpred)
print('acc:',acc)
print(classification_report(test_y, rfpred))

# Evaluate model using ROC
# Binarize the output for ROC evaluation
train_y2 = label_binarize(train_y, classes=[0, 1, 2])
test_y2 = label_binarize(test_y, classes=[0, 1, 2])
n_classes= test_y2.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(rf)
y_score = classifier.fit(train_X, train_y2).predict_proba(test_X)

# Compute ROC curve and ROC area for each class
lw=2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y2[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
```
#### SVM Model
``` python
#Create SVM Model
svm_model= SVC(C = 1, random_state = 123)
svm_model.fit(train_X, train_y)

#create prediction
svm_pred = svm_model.predict(test_X) 

#evaluation model using confusion matrix
cm = confusion_matrix(test_y,svm_pred)
print(cm)
acc= accuracy_score(test_y, svm_pred)
print('acc:',acc)
print(classification_report(test_y, svm_pred))

# Evaluate model using ROC
# Binarize the output for ROC evaluation
train_y2 = label_binarize(train_y, classes=[0, 1, 2])
test_y2 = label_binarize(test_y, classes=[0, 1, 2])
n_classes= test_y2.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(svm_model)
y_score = classifier.fit(train_X, train_y2).decision_function(test_X)

# Compute ROC curve and ROC area for each class
lw=2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y2[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
```
#### Gradient Boosting
``` python
#create GradientBoosting Model
GBT = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=7, random_state=0)
GBT.fit(train_X, train_y)

#Create prediction
GBT_pred= GBT.predict(test_X)

#Evaluate model with metric score from GBT
Accuracy = GBT.score(test_X, test_y)
print('Accuracy:', Accuracy)

#evaluation model using confusion matrix
cm = confusion_matrix(test_y,GBT_pred)
print(cm)
acc= accuracy_score(test_y, GBT_pred)
print('acc:',acc)
print(classification_report(test_y, GBT_pred))

# Evaluate model using ROC
# Binarize the output for ROC evaluation
train_y2 = label_binarize(train_y, classes=[0, 1, 2])
test_y2 = label_binarize(test_y, classes=[0, 1, 2])
n_classes= test_y2.shape[1]

# Learn to predict each class against the other
classifier = OneVsRestClassifier(GBT)
y_score = classifier.fit(train_X, train_y2).decision_function(test_X)

# Compute ROC curve and ROC area for each class
lw=2
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(test_y2[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
```
#### Implementasion on Data Test
It can be seen from those models above that gradient boosting has best model evaluation. Accuracy from gradient boosting is 0.79 dan average of ROC curve is around 0.91. So, gradient boosting will be used to predict data test.
``` python
# Make prediction on data test using gradient Boosting model
my_prediction= GBT.predict(df_test)

#convert my_prediction to pandas dataframe
my_prediction=pd.DataFrame(my_prediction)

#show 4 first of prediction
my_prediction.head(4)

#add column name to my_prediction
my_prediction.columns=['target']

#concat my_prediction with data test
my_prediction_data=pd.concat([df_test,my_prediction],axis=1 )

#Save my_prediction to csv and named it output.csv
my_prediction_data.to_csv('E:/Me/output.csv', index=False, header=True)

#Save my_prediction_data to csv and named it output2.csv
my_prediction_data.to_csv('E:/Me/output2.csv', index=False, header=True)
