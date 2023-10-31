import csv
import sys
from datetime import datetime

import numpy as np


if __name__ == "__main__":
    print("Calling __main__ of `myfunction.py`")
    if len(sys.argv) > 1:
        # used as run script with params as input
        var = float(sys.argv[2])
        filename = f'myoutput_{datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.csv'
        print("f x y  -> .csv")
        with open(filename, "w") as file:
            writer = csv.writer(file, delimiter=" ")
            writer.writerow(["f", "x", "y"])
        xvalues = np.linspace(0, 1, 10)
        yvalues = np.linspace(0, 1, 10)

        for x in xvalues:
            for y in yvalues:
                with open(filename, "a") as file:
                    writer = csv.writer(file, delimiter=" ")
                    writer.writerow([x, y])
        print("'myfunction.py' done writing values.")
    else:
        print("No input params. 'myfunction.py' used as analysis script.")
        pass
    print("End of `myfunction.py`.")
