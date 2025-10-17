import numpy as np
import pandas as pd

path = "Concrete_Data.xls"
learningRate = 0.0000001
numEpochs = 20000
standardized = False

def gradDescMultivariate(Xb, y, learningRate, numEpochs):
    n, d1 = Xb.shape
    w = np.zeros(d1, dtype=float)
    for _ in range(numEpochs):
        yHat = Xb @ w
        grad = (2.0 / n) * (Xb.T @ (yHat - y))
        w -= learningRate * grad
    return w

df = pd.read_excel(path)

targetCol = df.columns[-1]
featureCols = [c for c in df.columns if c != targetCol]

if standardized:
    dfFinal = (df - df.mean()) / df.std()
else:
    dfFinal = df

x = dfFinal[featureCols].to_numpy(dtype=float)
y = df[targetCol].to_numpy(dtype=float)

n = len(df)
testIdx = np.arange(500, 630)
allIdx = np.arange(n)
trainIdx = np.setdiff1d(allIdx, testIdx)

Xtr, Xte = x[trainIdx], x[testIdx]
yTr, yTe = y[trainIdx], y[testIdx]

XbTr = np.c_[np.ones((Xtr.shape[0], 1)), Xtr]
XbTe = np.c_[np.ones((Xte.shape[0], 1)), Xte]

w = gradDescMultivariate(XbTr, yTr, learningRate, numEpochs)

yHatTr = XbTr @ w
yHatTe = XbTe @ w

trainMSE = float(np.mean((yHatTr - yTr) ** 2))
testMSE  = float(np.mean((yHatTe - yTe) ** 2))
trainR2  = 1.0 - trainMSE / float(np.var(yTr))
testR2   = 1.0 - testMSE / float(np.var(yTe))

print(f"Train MSE: {trainMSE:.4f}   |   Train R^2 (var explained): {trainR2:.4f}")
print(f"Test  MSE: {testMSE:.4f}    |   Test  R^2 (var explained): {testR2:.4f}")

coefDf = pd.DataFrame({
    "term": ["bias"] + featureCols,
    "weight": w
})

print(coefDf.to_string(index=False))

    
    


