import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

numpy.random.seed(2)

x = numpy.random.normal(3, 1, 1000)
y = numpy.random.normal(150, 40, 1000) / x

train_x = x[:800]
train_y = y[:800]

test_x = x[800:]
test_y = y[800:]

mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 1000)
plt.scatter(train_x, train_y)
# plt.scatter(test_x, test_y)

plt.plot(myline, mymodel(myline), 'g')
plt.show( )
r = r2_score(train_x, mymodel(train_y))
print(r)
print( )
rtest = r2_score(test_x, mymodel(test_y))
print(rtest)
print( )
print(mymodel(5))
