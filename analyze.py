def FMPHist(ticker):
    return f'https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey='

import numpy as np
import pandas as pd
import requests 
import json
import matplotlib.pyplot as plt

'''
    Bayesian Update for Normal Returns
'''

def Rho(x, y):
    cov = np.cov(x, y)
    sd1 = np.sqrt(cov[0, 0])
    sd2 = np.sqrt(cov[1, 1])
    rho = cov[0, 1]/(sd1*sd2)
    return rho

ticker = 'MSFT'
data = requests.get(FMPHist(ticker)).json()
data = pd.DataFrame(data['historical'])[::-1]

lookback = 300
window = 50

close = data['adjClose'].values[-lookback:]
ror = close[1:]/close[:-1] - 1.0

ror_pre = ror[:window]
ror_post = ror[window:]

mu = np.mean(ror_pre)
tau = np.std(ror_pre)**2

returns = []
breturns = []

for i in range(window, len(ror_post)):
    hold = ror_post[i-window:i]
    r = hold[-1]
    variance = np.std(hold)**2
    tau = 1.0 / ((1.0/tau) + (1.0/variance))
    mu = tau*((mu/tau)+(r/variance))
    returns.append(r)
    breturns.append(mu)
    tau = variance
    mu = np.mean(hold)

xx = list(range(len(returns)))

plt.plot(xx, returns, color='blue', label='Regular Returns')
plt.plot(xx, breturns, color='red', label='Bayesian Returns')
plt.title(f'{ticker} | Correlation: {Rho(returns, breturns)}')
plt.legend()
plt.show()