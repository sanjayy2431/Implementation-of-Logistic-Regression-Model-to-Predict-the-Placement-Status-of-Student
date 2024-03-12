# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: v.sanjay
RegisterNumber:  212223230188
*/
```

## Output:
```
import pandas as pd
data=pd.read_csv("C:/Users/admin/Desktop/Placement_Data.csv")
data.head()
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/ace4da3d-4eb1-4c52-8fbc-486544a0b5eb)
```
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/057f505a-48ae-4053-8956-51a3fe4d0347)
```
data1.isnull()
```

![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/eac5ec7e-db34-47b7-9a87-da94c972a4ef)

```
data1.duplicated().sum()
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/034da63e-13af-483a-ae46-8768948cbd10)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/e87135d0-7f1e-47ea-8fcc-886eee8b380f)
```
x=data1.iloc[:,:-1]
x
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/fd2ede0e-176a-45c6-9158-c18ba9b2d930)
```
y=data1["status"]
y
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/0de8823d-ffb5-4405-80a7-d686975db8b7)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/43773e61-93af-4366-89b2-d9eea9c8a87a)
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/92f5972f-910d-4519-87cb-9538e1253489)
```
from sklearn.metrics import  classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/2b4ee5d4-25a1-42ab-8ead-6bc378546d82)
```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/sanjayy2431/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149365143/4014a5dc-2496-479c-92f9-33faabc1540f)












## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
