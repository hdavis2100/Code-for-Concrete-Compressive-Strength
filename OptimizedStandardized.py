import pandas as pd
import numpy as np
import statsmodels.api as sm

path = "Concrete_Data.xls"

df = pd.read_excel(path)

targetCol = df.columns[-1]
featureCols = [c for c in df.columns if c != targetCol]

XallRaw = df[featureCols].astype(float)
yAll = df[targetCol].astype(float)
XallStd = (XallRaw - XallRaw.mean()) / XallRaw.std()

testIdx = np.arange(500, 630)
allIdx = np.arange(len(df))
trainIdx = np.setdiff1d(allIdx, testIdx)

Xtrain = XallStd.iloc[trainIdx]
yTrain = yAll.iloc[trainIdx]
Xtest  = XallStd.iloc[testIdx]
yTest  = yAll.iloc[testIdx]

XtrainConst = sm.add_constant(Xtrain)
XtestConst  = sm.add_constant(Xtest)

model = sm.OLS(yTrain, XtrainConst).fit()

print(model.summary())

yHatTrain = model.predict(XtrainConst)
yHatTest  = model.predict(XtestConst)

trainMSE = float(np.mean((yTrain - yHatTrain)**2))
testMSE  = float(np.mean((yTest  - yHatTest )**2))
trainR2  = 1.0 - trainMSE / float(np.var(yTrain))
testR2   = 1.0 - testMSE  / float(np.var(yTest))

print(f"Train MSE: {trainMSE:.4f}")
print(f"Train R^2: {trainR2:.4f}")
print(f"Test MSE:  {testMSE:.4f}")
print(f"Test R^2:  {testR2:.4f}")

