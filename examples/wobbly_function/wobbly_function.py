"""
Usage:
    python3 function.py -filebasename <folder-to-store-output>/<somefilebasename> -x 3 -y 5.5
"""

from argparse import ArgumentParser
import csv
import numpy as np


def function(x, y):
    """Returns wobbly function value"""

    radius_squared = x ** 2 + y ** 2
    return radius_squared + np.sin(radius_squared)


if __name__ == "__main__":
    # Parse the input argument
    parser = ArgumentParser()
    parser.add_argument('--filebasename', type=str, default="test", required=False)
    parser.add_argument('--x', type=float, required=False, default=1.0)
    parser.add_argument('--y', type=float, required=False, default=1.0)
    parser.add_argument('--test', type=float, required=False, default=1.0)
    args = parser.parse_args()
    parameter_values = [args.x, args.y]

    # Run the "simulation"
    output_value = function(x=args.x, y=args.y)

    # Store the output in a file
    csv_filename = args.filebasename + ".csv"
    with open(csv_filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ')
        csv_writer.writerow(['x', 'y'])
        csv_writer.writerow(parameter_values)
        csv_file.close()
