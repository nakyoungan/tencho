import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")

data_used = data.iloc[:, 1:4]   #개장가, 최고가, 최저가 추가
data_used["Close"] = data["Close"]  #종가 추가
hist = data_used.hist()
plt.show()