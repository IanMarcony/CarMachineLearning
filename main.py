from threading import Thread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

path_dataset = './autompg.csv'


def training(lr, X_train, y_train):
  lr.fit(X_train, y_train) 
  
def predict(lr, X_test, y_test):
  y_prev = lr.predict(X_test)
  error = mean_squared_error(y_test, y_prev)**0.5
  score = r2_score(y_test, y_prev)

  print(f'Mean Squared Error: {error}')
  print(f'R2 Score: {score}')
  
  return y_prev

def plotMachineLearning(y_test, y_prev):
  residuos = []
  for (x,y) in zip(y_test,y_prev):
      residuos.append((x-y)**2)


  x = [0,int(max(y_test))]
  y = [0,0]
  plt.plot(x,y,linewidth=3)
  plt.plot(y_test,residuos,'ro')
  plt.ylabel('Residuos')
  plt.xlabel('kml')
  plt.show()
  

df = pd.read_csv(path_dataset, sep = ';') 

df.dropna(inplace = True)
df["kml"] = round(df["mpg"]*0.425,2)
df.drop(["mpg","name"], axis=1, inplace=True)

y = df["kml"]
X = df.drop(["kml"], axis = 1)

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.3)

lr = LinearRegression()
y_prev = []


training(lr, X_train, y_train)

y_prev = predict(lr, X_test, y_test)

plotMachineLearning(y_test, y_prev)