import numpy as np
import pandas as pd

path = "Concrete_Data.xls"
learningRate = 0.000001
numEpochs = 1000000
standardized = False

def gradDescUnivariate(Xb, y, learningRate, numEpochs):
    n = Xb.shape[0]
    w = np.zeros(2)
    for _ in range(numEpochs):
        yHat = Xb @ w
        grad = (2.0 / n) * (Xb.T @ (yHat - y))
        w -= learningRate * grad
    return w

df = pd.read_excel(path)

targetCol = df.columns[-1]
featureCols = [c for c in df.columns if c != targetCol]

testIdx = np.arange(500, 630)
allIdx = np.arange(len(df))
trainIdx = np.setdiff1d(allIdx, testIdx)

if standardized:
    dfFinal = (df - df.mean()) / df.std()
else:
    dfFinal = df

y = df[targetCol].to_numpy(dtype=float)
yTrain, yTest = y[trainIdx], y[testIdx]

results = []
for feat in featureCols:
    x = dfFinal[feat].to_numpy(dtype=float)
    xTrain, xTest = x[trainIdx], x[testIdx]
    XbTrain = np.c_[np.ones((xTrain.shape[0],)), xTrain]
    XbTest = np.c_[np.ones((xTest.shape[0],)), xTest]

    w = gradDescUnivariate(XbTrain, yTrain, learningRate, numEpochs)

    yHatTrain = XbTrain @ w
    yHatTest = XbTest @ w

    results.append({
        "feature": feat,
        "trainMSE": round(np.mean((yHatTrain - yTrain) ** 2), 4),
        "trainR2": round(1 - np.mean((yHatTrain - yTrain) ** 2) / np.var(yTrain), 4),
        "testMSE": round(np.mean((yHatTest - yTest) ** 2), 4),
        "testR2": round(1 - np.mean((yHatTest - yTest) ** 2) / np.var(yTest), 4),
        "bias": round(w[0], 4),
        "coef": round(w[1], 4),
    })

resultsDf = pd.DataFrame(results)
print(resultsDf.to_string(index=False))


