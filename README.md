# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset and display basic information, including null values and class distribution of the `left` column.
2. Encode the categorical `salary` column using Label Encoding.
3. Define the features (`X`) and target (`y`) by selecting relevant columns.
4. Split the data into training and testing sets (80-20 split).
5. Initialize a Decision Tree Classifier with the entropy criterion and train it on the training data.
6. Predict the target values for the test set.
7. Calculate and display the model's accuracy.
8. Compute and display the confusion matrix for the predictions.
9. Predict the `left` status for a new employee sample.

## Program:
```Python
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
*/

import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
data=pd.read_csv('Employee.csv')
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[['satisfaction_level','last_evaluation','number_project','average_montly_hours',
'time_spend_company','Work_accident','left','promotion_last_5years']]
x.head()
y=data[['left']]
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
cm=metrics.confusion_matrix(y_test,y_pred)
cm
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
![image](https://github.com/user-attachments/assets/8c02a398-2420-4538-839c-f5dfcc6c2837)
![image](https://github.com/user-attachments/assets/8d5b1162-7251-418b-9023-f33c1dbc88ed)
![image](https://github.com/user-attachments/assets/4b25b691-a2d0-4fb3-982f-e0bb8273ab4c)
![image](https://github.com/user-attachments/assets/6ed089ac-ed71-4b5c-bb90-a515b8670dbc)
![image](https://github.com/user-attachments/assets/c345d378-5ebd-4269-8561-fb1ab8b63e64)
![image](https://github.com/user-attachments/assets/b601c4b3-6a3d-4252-b2f9-438dbab975f8)
![image](https://github.com/user-attachments/assets/5ad6d70d-fdeb-4622-9784-434595259c2e)

![image](https://github.com/user-attachments/assets/692d81de-59ba-4931-90d3-f323af85d416)

![image](https://github.com/user-attachments/assets/1b59ab37-712e-43b7-b2dc-eeb122c3104c)
![image](https://github.com/user-attachments/assets/d54f5cf7-14f4-4938-9441-900c808b5902)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
