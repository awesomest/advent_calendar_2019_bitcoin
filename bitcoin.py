import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

csv = pd.read_csv("history_20191218.csv")

class BitcoinClassification:
    def __init__(self, csv):
        self.csv = csv.rename(columns={"日付け": "date", "終値": "close", "始値": "start", "高値": "high", "安値": "low",  "出来高": "volume",  "前日比%": "diff"}).sort_values(["date"])
        self.data = self.csv[["date", "start", "close", "low", "high", "volume", "diff"]]

    def format(self):
        self.data["date"] = pd.to_datetime(self.csv["date"], format="%Y年%m月%d日").astype(int) /  10**9
        self.data["start"] = self.csv["start"].str.replace(",", "").astype(int)
        self.data["close"] = self.csv["close"].str.replace(",", "").astype(int)
        self.data["low"] = self.csv["low"].str.replace(",", "").astype(int)
        self.data["high"] = self.csv["high"].str.replace(",", "").astype(int)
        self.data["volume"] = self.csv["volume"].str.replace("K", "").astype(float)
        self.data["diff"] = self.csv["diff"].str.replace("%", "").astype(float)
        self.data["year"] = pd.to_datetime(self.csv["date"], format="%Y年%m月%d日").dt.strftime("%Y").astype(int)
        self.data["month"] = pd.to_datetime(self.csv["date"], format="%Y年%m月%d日").dt.strftime("%m").astype(int)
        self.data["day"] = pd.to_datetime(self.csv["date"], format="%Y年%m月%d日").dt.strftime("%d").astype(int)
        self.data["weekday"] = pd.to_datetime(self.csv["date"], format="%Y年%m月%d日").dt.strftime("%w").astype(int)

    def out(self):
        return self.data

    def removeOutlier(self):
        self.data = self.data[self.data["start"] > 0]
        self.data = self.data[self.data["close"] > 0]
        self.data = self.data[self.data["low"] > 0]
        self.data = self.data[self.data["volume"] > 0]

    def addColumnResult(self):
        self.data["result"] = self.data["diff"].shift(1)
        self.data["result"] = [1 if l >= 0.0 else -1 for l in self.data["result"]]

    def train_and_test(self):
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(self.data_train, self.label_train)
        predict = clf.predict(self.data_test)
        return metrics.accuracy_score(self.label_test, predict)

    def calcAvgPred(self):
        sum_score = 0
        for i in range(100):
            ac_score = self.train_and_test()
            sum_score += ac_score

        return sum_score / 100

    def setTrainTestDataset(self, n):
        #label = "diff" + str(n)
        labels = ["date", "year", "month", "day", "weekday", "start", "close", "low", "high", "volume", "diff"]
        for i in list(range(2, n+1)):
            labels.append("diff" + str(i))
        self.data_train, self.data_test, self.label_train, self.label_test = train_test_split(self.data[labels][n:], self.data["result"][n:], random_state=1)

    def setColumnDiffN(self, n):
        label = "diff" + str(n)
        self.data[label] = self.data["start"].shift(n-1)
        self.data[label] = self.data["close"] / self.data[label] - 1

b = BitcoinClassification(csv)
b.format()
b.removeOutlier()
b.addColumnResult()
d = 10
for i in list(range(2, d + 1)):
    b.setColumnDiffN(i)

x = list(range(10))
y = []
for i in list(range(1, d + 1)):
    b.setTrainTestDataset(i)
    pred = b.calcAvgPred()
    y.append(pred)
    print("平均[%d]: %f" % (i, pred))

plt.plot(x, y)
plt.show()
