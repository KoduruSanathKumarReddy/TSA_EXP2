<img width="903" alt="image" src="https://github.com/user-attachments/assets/19a86f69-d1f1-48bc-8edc-0b2cc503e3c6"># Ex.No: 02 LINEAR AND POLYNOMIAL TREND ESTIMATION
Date:
### AIM:
To Implement Linear and Polynomial Trend Estiamtion Using Python.

### ALGORITHM:
Import necessary libraries (NumPy, Matplotlib)

Load the dataset

Calculate the linear trend values using least square method

Calculate the polynomial trend values using least square method

End the program
### PROGRAM:
A - LINEAR TREND ESTIMATION
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the dataset
data = pd.read_csv('electric_production.csv')

# Convert the Date column to datetime format and set it as the index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Prepare the data
X = np.arange(len(data)).reshape(-1, 1)  # Time as a numeric value
y = data['Production'].values  # Electric production values

# Linear trend estimation
linear_model = LinearRegression()
linear_model.fit(X, y)
linear_trend = linear_model.predict(X)


~~~

B- POLYNOMIAL TREND ESTIMATION
~~~
# Polynomial trend estimation (degree 2 for quadratic)
degree = 2
polynomial_features = PolynomialFeatures(degree=degree)
X_poly = polynomial_features.fit_transform(X)
polynomial_model = LinearRegression()
polynomial_model.fit(X_poly, y)
polynomial_trend = polynomial_model.predict(X_poly)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label='Original Data', color='blue')
plt.plot(data.index, linear_trend, label='Linear Trend', color='red')
plt.plot(data.index, polynomial_trend, label=f'Polynomial Trend (Degree {degree})', color='green')
plt.xlabel('Date')
plt.ylabel('Electric Production')
plt.title('Linear and Polynomial Trend Estimation')
plt.legend()
plt.show()

~~~

### OUTPUT
A - LINEAR TREND ESTIMATION
<img width="905" alt="image" src="https://github.com/user-attachments/assets/62e19fba-dccc-4d1a-bd52-109c6ac5d5d0">

B- POLYNOMIAL TREND ESTIMATION
<img width="511" alt="image" src="https://github.com/user-attachments/assets/b73c64dc-e720-4491-b8a1-ad1c9f1859a0">


### RESULT:
Thus the python program for linear and Polynomial Trend Estiamtion has been executed successfully.
