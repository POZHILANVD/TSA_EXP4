# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
## Date: 26.09.25

### AIM:

To implement ARMA model in python.
### ALGORITHM:

1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

data = pd.read_csv('/content/FINAL_USO.csv')

# Assuming 'Close' column represents the price and needs cleaning
data['Close'] = data['Close'].astype(str).str.replace(',', '').astype(float)

X = data['Close'] # Use the 'Close' column
plt.rcParams['figure.figsize'] = [12, 6]
plt.plot(X)
plt.title('Original Gold Price Data')
plt.show()

plt.subplot(2, 1, 1)
plot_acf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data ACF')

plt.subplot(2, 1, 2)
plot_pacf(X, lags=int(len(X)/4), ax=plt.gca())
plt.title('Original Data PACF')

plt.tight_layout()
plt.show()

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])

N = 1000
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)

plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Gold Prices')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])

ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)

plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Gold Prices')
plt.xlim([0, 500])
plt.show()

plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()

```

### OUTPUT:

<img width="1084" height="539" alt="image" src="https://github.com/user-attachments/assets/ef9a02b7-6ee1-436e-931d-5f207011a9d6" />
<img width="1289" height="306" alt="image" src="https://github.com/user-attachments/assets/1f4e2f0f-35df-4093-a9d8-df0c8b72e180" />
<img width="1266" height="294" alt="image" src="https://github.com/user-attachments/assets/44aa272d-37a9-477d-9351-9c0824992d00" />
<img width="1085" height="534" alt="image" src="https://github.com/user-attachments/assets/eeec7bab-c984-441f-bcaa-36a7e01a9103" />
<img width="1075" height="541" alt="image" src="https://github.com/user-attachments/assets/279de96d-3c1b-4cb1-b447-e38f08f6503b" />
<img width="1078" height="532" alt="image" src="https://github.com/user-attachments/assets/69d0dbd5-2afa-4f57-82cf-17c7ab62f39c" />
<img width="1080" height="533" alt="image" src="https://github.com/user-attachments/assets/8ddbf0ac-370d-408d-8cab-9dd8d6b60a9b" />
<img width="912" height="443" alt="image" src="https://github.com/user-attachments/assets/707dc7a4-45b8-4858-8c22-0075f00dd818" />
<img width="878" height="445" alt="image" src="https://github.com/user-attachments/assets/b58d4741-2fd6-4d62-b74e-d3c158d0820a" />


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
