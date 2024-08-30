# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

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

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Vikaash P
RegisterNumber:212223240180

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)

```

## Output:
Dataset:
![image](https://github.com/user-attachments/assets/c91d8d24-6180-4023-a0f9-c936c03ff3f8)


Head value:
![image](https://github.com/user-attachments/assets/1d03c82a-69c5-4f40-985b-f1ac6dc74c99)

Tail value:
![image](https://github.com/user-attachments/assets/794aecf6-c253-46af-914d-872ac61caeaf)

X,Y Values:
![image](https://github.com/user-attachments/assets/d5e68a84-cd5c-47a5-97c8-03fa2f4ad3e7)

Prediction Values :
![image](https://github.com/user-attachments/assets/96c45a37-cac9-4163-ae62-729e80cde115)

Training Set:
![image](https://github.com/user-attachments/assets/03d3f761-b950-450a-8ed5-2e0affc42560)

Testing Set :
![image](https://github.com/user-attachments/assets/55a3db79-75b5-4cc0-b3b8-f0ef09c9166b)

MSE , MAE AND RMSE;
![image](https://github.com/user-attachments/assets/51d51b70-534c-4af1-9ce6-a0c0754554b3)










## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
