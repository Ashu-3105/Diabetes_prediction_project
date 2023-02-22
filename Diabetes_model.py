import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection  import train_test_split
df=pd.read_csv('diabetes.csv')
scaler = StandardScaler()
scaler.fit(df)
df1 = scaler.transform(df)
df1=pd.DataFrame(df)
Q1=df1.quantile(0.25)
Q3=df1.quantile(0.75)
IQR=Q3-Q1
df_out = df1[~((df1 < (Q1 - 1.5 * IQR)) |(df1 > (Q3 + 1.5 * IQR))).any(axis=1)]
X=df_out.drop(columns=['Outcome'])
y=df_out['Outcome']
train_X,test_X,train_y,test_y=train_test_split(X,y,test_size=0.2)



def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(train_X,train_y)
    return trained_model




