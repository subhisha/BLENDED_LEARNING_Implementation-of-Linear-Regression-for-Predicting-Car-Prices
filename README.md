# BLENDED_LEARNING
# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import Libraries & Load Dataset
2. Divide the dataset into training and testing sets.
3. Select a suitable ML model, train it on the training data, and make predictions.
4. Assess model performance using metrics and interpret the results.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

df = pd.read_csv('CarPrice_Assignment.csv')
df.head()
X=df[['enginesize','horsepower','citympg','highwaympg']]
y=df['price']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

print('Name:SUBHISHA P ')
print('Reg. No: 212225040143')
print("MODEL COEFFICIENTS:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature:>12}: {coef:>10.2f}")
print(f"{'Intercept':>12}: {model.intercept_:>10.2f}")

print("\nMODEL PERFORMANCE:")
print(f"{'MSE':>12}: {mean_squared_error(y_test, y_pred):>10.2f}")
print(f"{'RMSE':>12}: {np.sqrt(mean_squared_error(y_test, y_pred)):>10.2f}")
print(f"{'R-squared':>12}: {r2_score(y_test, y_pred):>10.2f}")
print(f"{'MAE':>12}: {mean_absolute_error(y_test, y_pred):>10.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(), y.max()],'r--')
plt.title("Linearity Check: Actual vs Predicted Prices")
plt.xlabel("Actual Price ($)")
plt.ylabel("Predicted Price($)")
plt.grid(True)
plt.show()
```

## Output:
<img width="1248" height="440" alt="image" src="https://github.com/user-attachments/assets/353f819e-abf8-4e7e-8c21-9b5f46299e00" />

<img width="590" height="345" alt="image" src="https://github.com/user-attachments/assets/f1bb6405-e7a6-4d1e-b034-b84549b74cea" />

<img width="821" height="332" alt="image" src="https://github.com/user-attachments/assets/d864e367-4774-4e86-86ae-bef0b269b67c" />

<img width="1186" height="777" alt="image" src="https://github.com/user-attachments/assets/08b38ef2-1925-44b2-a201-a7880ad22a83" />

<img width="1195" height="766" alt="image" src="https://github.com/user-attachments/assets/b5071757-5886-49c3-a3e2-305facb1462a" />

<img width="1272" height="702" alt="image" src="https://github.com/user-attachments/assets/ae719960-4648-4028-b6a6-6e8d2fadc339" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
