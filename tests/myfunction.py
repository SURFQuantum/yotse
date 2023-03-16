import sys
import csv
import numpy as np
from datetime import datetime


# def costfunction(x, y, var):
#     f = x**2 + y**2 + var*np.sin(x)
#     return f

if __name__ == "__main__":
    var = float(sys.argv[2])
    filename = f'myoutput_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.csv'
    print("f x y")
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter=' ')
        writer.writerow(["f", "x", "y"])
    xvalues = np.linspace(0, 1, 10)
    yvalues = np.linspace(0, 1, 10)

    for x in xvalues:
        for y in yvalues:
            with open(filename, 'a') as file:
                writer = csv.writer(file, delimiter=' ')
                writer.writerow([x, y])
