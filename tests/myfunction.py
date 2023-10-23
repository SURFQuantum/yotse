import csv
import sys
from datetime import datetime

import numpy as np


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # used as run script with params as input
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
    else:
        # used as analysis script with no output
        pass
