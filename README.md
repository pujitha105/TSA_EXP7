## Devloped by: k.pujitha
## Register Number: 212223240074
## Date: 10-5-2025

# Ex.No: 07-AUTO-REGRESSIVE MODEL

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
### ALGORITHM :

### Step 1 :

Import necessary libraries.

### Step 2 :

Read the CSV file into a DataFrame.

### Step 3 :

Perform Augmented Dickey-Fuller test.

### Step 4 :

Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags.

### Step 5 :

Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF).

### Step 6 :

Make predictions using the AR model.Compare the predictions with the test data.

### Step 7 :

Calculate Mean Squared Error (MSE).Plot the test data and predictions.

### PROGRAM

#### Import necessary libraries :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error
```

#### Read the CSV file into a DataFrame :

```python
data = pd.read_csv('smooth_data.csv',parse_dates=['Date'],index_col='Date')
```

#### Perform Augmented Dickey-Fuller test :

```python
result = adfuller(data['yahoo_price']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

#### Split the data into training and testing sets :

```python
x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]
```

#### Fit an AutoRegressive (AR) model with 13 lags :

```python
lag_order = 13
model = AutoReg(train_data['yahoo_price'], lags=lag_order)
model_fit = model.fit()
```

#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :

```python
plt.figure(figsize=(10, 6))
plot_acf(data['yahoo_price'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['yahoo_price'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()
```

#### Make predictions using the AR model :

```python
predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
```

#### Compare the predictions with the test data :

```python
mse = mean_squared_error(test_data['yahoo_price'], predictions)
print('Mean Squared Error (MSE):', mse)
```

#### Plot the test data and predictions :

```python
plt.figure(figsize=(12, 6))
plt.plot(test_data['yahoo_price'], label='Test Data - Yahoo_stock_price')
plt.plot(predictions, label='Predictions - Stock Price of Yahoo',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()

```

### OUTPUT:

Dataset:

![image](https://github.com/user-attachments/assets/c263186a-cfbf-483d-b2b5-60fc22c10fef)

ADF test result:

![image](https://github.com/user-attachments/assets/316cf05e-e7d8-4268-a89a-4598bff51edf)

PACF plot:

!![image](https://github.com/user-attachments/assets/e4407c79-8c4b-48c5-a9d2-4815e9b6b080)


ACF plot:

![image](https://github.com/user-attachments/assets/722cc4be-913b-4f16-8518-5d5d5460172c)


Accuracy:

![image](https://github.com/user-attachments/assets/579c3397-449f-4b6c-a7aa-053946f4d92d)


Prediction vs test data:

![image](https://github.com/user-attachments/assets/18e1b80a-a234-4c63-8365-2691926c5b60)

### RESULT:
Thus we have successfully implemented the auto regression function using python.
