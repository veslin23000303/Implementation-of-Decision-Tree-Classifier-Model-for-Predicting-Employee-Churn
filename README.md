# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import the required libraries.

2.Upload and read the dataset.

3.Check for any null values using the isnull() function.

4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.

5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

Developed by:   Adhithya Perumal.D

RegisterNumber: 212222230007 

```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
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

## Initial data set:

![272784342-4fdab09e-a67b-45ba-b6ea-1430530c1f44](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/6ce30cd4-040a-47ca-9afa-47f31b6dc14b)

### Data info:

![272784393-40bbf9f1-0b4b-4554-9398-67eb9f3e3a6c](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/b8fa412e-8435-4d57-9e07-c607d029c228)

## Optimization of null values:

![272784411-130e597d-f5e9-42fa-96e7-5bcfb522f28a](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/e06d5ab1-35ff-42bb-af4c-d52940cf8ec4)

## Assignment of x value:

![272784425-b638f42f-9423-4dcd-81ce-d6af81aa7bc4](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/45417fa7-6a1d-4b82-896e-93376b8184b9)

## Assignment of y value:

![272784454-48b19d12-f586-4f21-8f4f-44a9e7562409](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/84e13da4-51ac-405f-9b19-52a5b2704cbc)

## Converting string literals to numerical values using label encoder:

![272784485-efd59b97-4b8e-4855-8f54-6194f2ec6d44](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/216a9fe1-874e-4f44-82ea-df2c51d28f59)

## Accuracy:

![273142795-126faecd-11de-4cdb-9637-e98c7bba520e](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/22cf25d5-6367-4d2d-829d-86ff17c08cfa)

## Prediction:

![272784517-655f10f8-8fff-4d48-9115-89736a07c6fb](https://github.com/Adhithya4116/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/118707079/3d7089ea-1be6-403e-bf29-850c75fe1ba6)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
