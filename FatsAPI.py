from fastapi import FastAPI
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

df=pd.read_json(r"dataset.json")
X=df.iloc[:,:-1]
y=df.iloc[:,-1]
label=LabelEncoder()
X1=label.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3)
X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
y_train=np.ravel(y_train)
y_test=np.ravel(y_test)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)

app=FastAPI()
status_list=set(list(X['externalStatus']))

status_dict = {index: value for index, value in enumerate(status_list)}

@app.get("/Internal_Status")
def home():
    return status_dict
@app.get("/External_Status")
def status(status: str):
    if status in status_list:
        inv_test = label.inverse_transform(X_test)
        inv_test = list(inv_test)
        pos = inv_test.index(status)
        X_test1 = X_test[pos].reshape(1, -1)
        y_pred = model.predict(X_test1)
        return y_pred[0]
    return {"Data":"Not found"}
