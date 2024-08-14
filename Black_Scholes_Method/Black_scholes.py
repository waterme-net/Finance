import math
from scipy.stats import norm

S = 42
K = 40
T = 0.5
r = 0.1
vol = 0.2

d1 = (math.log(S/K) + (r + 0.5 * vol**2)*T / (vol * math.sqrt(T)))
d2 = d1 - (vol * math.sqrt(T))

C = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)

P = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

print('The value of d1 is:', round(d1,4))
print('The value of d2 is:', round(d2,4))
print('The price of the call option is: $', round(C,2))
print('The price of the put option is: $', round(P,2))

