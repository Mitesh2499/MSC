#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



df = pd.read_csv("Admission_Predict.csv")

df=df.rename(columns={'Chance of Admit ':'chance'})


var=df.columns.values.tolist()


x=pd.DataFrame(df['CGPA'])
y=df['chance']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)



# Fitting logistic regression model
lr = LinearRegression()
lr.fit(x_train, y_train)


pred_y=lr.predict(x_test)
print('Linear regression accuracy: {:.3f}'.format(r2_score(y_test,pred_y )))


plt.xlabel("CGPA")
plt.ylabel("Chance")
plt.scatter(x=df['CGPA'],y=df['chance'])
plt.plot(df['CGPA'],lr.predict(x),color="red")
plt.show()


