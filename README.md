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
<img width="877" height="175" alt="image" src="https://github.com/user-attachments/assets/3611bdac-85a8-4adf-85ba-2fd51b8c80b2" />
<img width="1269" height="287" alt="Screenshot 2026-02-23 134227" src="https://github.com/user-attachments/assets/88be92f0-4511-4078-91f9-3a89cde9af08" />
<img width="660" height="151" alt="image" src="https://github.com/user-attachments/assets/7754867f-1858-4e2f-b6de-33f6beda0f79" />
<img width="294" height="203" alt="Screenshot 2026-02-23 134347" src="https://github.com/user-attachments/assets/a8076ef9-c034-4eb7-b24c-ac852c732f0a" />
<img width="827" height="385" alt="image" src="https://github.com/user-attachments/assets/348b928c-b89d-4443-a496-bbe2ce806ced" />
<img width="1182" height="777" alt="image" src="https://github.com/user-attachments/assets/359dacfd-f6dd-431b-a0a6-367c04d08508" />
<img width="1920" height="1080" alt="Screenshot (28)" src="https://github.com/user-attachments/assets/25e8a334-f136-413c-9938-32cbf4a0077c" />
<img width="1920" height="1080" alt="Screenshot (29)" src="https://github.com/user-attachments/assets/a29ee081-1937-4ff7-837c-592ea00b6f90" />


## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
