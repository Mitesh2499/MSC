import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df=pd.read_csv("Admission_Predict.csv")

#Rename column chance of Admit to chance
df=df.rename(columns={'Chance of Admit ':'chance'})

#Drop Serial Number as it is not depenedent feature.
df.drop(['Serial No.'],axis=1,inplace=True)

#List of All columns
columns_list=df.columns.values.tolist()

y=df["chance"]
x = df.drop('chance', 1)


#Split data into Traning data => 80% and Testing Data => 20%
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

'''set Value to 1 if chance greater than 0.82 
Meaning If student chance is greater than 0.82 then student is consider as selected '''
# cy_train=[]
# for chance in y_train:
#         if chance > 0.82:
#                 cy_train.append(1) 
#         else:
#                  cy_train.append(0) 
cy_train=[1 if chance > 0.82 else 0 for chance in y_train]
cy_train=np.array(cy_train)

cy_test=[1 if chance > 0.82 else 0 for chance in y_test]
cy_test=np.array(cy_test)

#xs=MinMaxScaler()
#x_train[x_train.columns] = xs.fit_transform(x_train[x_train.columns])
#x_test[x_test.columns] = xs.transform(x_test[x_test.columns])

# Fitting logistic regression model
lr = LogisticRegression()
lr.fit(x_train, cy_train)

# Printing accuracy score & confusion matrix
pred_y=lr.predict(x_test)
print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(cy_test,pred_y )))

cm = confusion_matrix(cy_test, pred_y)
print(cm)





