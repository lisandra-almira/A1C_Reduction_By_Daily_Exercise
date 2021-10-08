import pandas as pd
import numpy as np
from sklearn import linear_model

df= pd.read_cvs("A1ClevelReductionExerciseDuration")
df

reg = linear_model.LinearRegression()
reg.fit(df[['Excersise']], df.A1CRegresion)

reg.coef_

reg.intercept_

reg.predict([[0.9]])

val = input("Enter daily hours of exercise: ")

y= reg.predict([[val]])
print(y)


