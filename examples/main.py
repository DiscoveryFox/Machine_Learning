import runpy

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
speed = [86, 87, 88, 86, 87, 85, 86]
print("mean: ")
print(np.mean(speed))
print("Standart Deviation: ")
print(np.std(speed))
print("Variance:")
print(np.var(speed))
print("Percentile: ")
print(np.percentile(speed, 100))
"""

"""
x = np.random.normal(5.0, 1.0, 100000)
plt.hist(x, 100)
plt.show( )
print(x)
"""

"""
x = np.random.normal(5.0, 1.0, 1000)
y = np.random.normal(10.0, 2.0, 1000)

plt.scatter(x, y)
plt.show()
"""

# Lineare Regression
x = [5, 7, 8, 7, 2, 17, 2, 9, 4, 11, 12, 9, 6]  # Age OF THE CARS
y = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]  # Speed   OF THE CARS

slope, intercept, r, p, std_err = stats.linregress(x, y)

print(r)


def myfunc(x):
    return slope * x + intercept


print(myfunc(10))

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show( )
