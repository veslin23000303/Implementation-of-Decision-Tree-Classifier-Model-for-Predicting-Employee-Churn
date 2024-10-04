# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sushiendar M
RegisterNumber: 212223040217
*/
import pandas as pd
df=pd.read_csv("Employee.csv")
df

df.head()

df.info()

df.isnull().sum()

df["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["salary"]=le.fit_transform(df["salary"])
df.head()

x=df[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=df["left"]


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy


dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### Data head:
![headmlexp4](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/2d80e64d-2af6-4896-9fb1-c528cebc4c7e)

### Data info:
![infomlexp6](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/766d43af-4734-4928-9e8d-0eaaccec3d42)

### Null values:
![null values](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/e478eda7-e031-4d65-9c93-bb94dcadbe75)

### Values Count in Left Column:
![valexp6](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/a286317e-2ec9-4897-8fdb-47a9e3c1be89)

### Dataset transformed head:
![shsk](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/26be2a72-d417-4878-9d85-fd97ab8b40c4)

### x.head():
![xheadexp6](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/ffd3a395-b744-417a-a69a-dc16f4f0e61d)

### Accuracy:
![accuracyexp6](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/5d62da04-811f-4df6-8454-d19f29267430)

### Data Prediction:
![;lkj](https://github.com/JananiSoundararajan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119477549/396d9606-b88c-4d39-954a-46ab380ad9d8)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
