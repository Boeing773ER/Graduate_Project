from numpy.random import binomial

for i in range(10):
    result = binomial(2, 0.5)
    print(result, type(result))

