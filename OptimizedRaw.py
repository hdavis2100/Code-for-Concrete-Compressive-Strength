import pandas as pd
import numpy as np
import statsmodels.api as sm

path = "Concrete_Data.xls"

df = pd.read_excel(path)

targetCol = df.columns[-1]
featureCols = [c for c in df.columns if c != targetCol]

testIdx = np.arange(500, 630)
allIdx = np.arange(len(df))
trainIdx = np.setdiff1d(allIdx, testIdx)

Xtrain = df.loc[trainIdx, featureCols]
yTrain = df.loc[trainIdx, targetCol]

Xtest = df.loc[testIdx, featureCols]
yTest = df.loc[testIdx, targetCol]

XtrainConst = sm.add_constant(Xtrain)
XtestConst = sm.add_constant(Xtest)

model = sm.OLS(yTrain, XtrainConst).fit()

print(model.summary())

yHatTrain = model.predict(XtrainConst)
yHatTest  = model.predict(XtestConst)

trainMSE = np.mean((yTrain - yHatTrain) ** 2)
testMSE  = np.mean((yTest - yHatTest) ** 2)

trainR2 = 1 - trainMSE / np.var(yHatTrain)
testR2  = 1 - testMSE / np.var(yHatTest)

print(f"Train MSE: {trainMSE:.4f}")
print(f"Train R^2: {trainR2:.4f}")
print(f"Test  MSE: {testMSE:.4f}")
print(f"Test  R^2: {testR2:.4f}")

