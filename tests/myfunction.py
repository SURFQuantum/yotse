import sys
import numpy as np


def costfunction(x, y, var):
    f = x**2 + y**2 + var*np.sin(x)
    return f


if __name__ == "__main__":
    var = float(sys.argv[2])
    print("# f x y")
    xvalues = np.linspace(0, 1, 10)
    yvalues = np.linspace(0, 1, 10)
    for x in xvalues:
        for y in yvalues:
            print(costfunction(x, y, var), x, y)
