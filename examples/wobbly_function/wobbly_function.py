"""
Usage:
    python3 function.py -filebasename <folder-to-store-output>/<somefilebasename> -x 3 -y 5.5
"""

import csv
import numpy as np
from argparse import ArgumentParser


def function(x, y):
    """Returns wobbly function value f(x,y) = (x^2 + y^2) * sin(x^2 + y^2)."""

    radius_squared = x ** 2 + y ** 2
    return radius_squared + np.sin(radius_squared)


if __name__ == "__main__":
    # Parse the input argument
    parser = ArgumentParser()
    parser.add_argument('--filebasename', type=str)
    parser.add_argument('--x', type=float)
    parser.add_argument('--y', type=float)
    parser.add_argument('--test', type=float)
    args = parser.parse_args()
    parameter_values = [args.x, args.y]

    # Run the "simulation"
    output_value = function(x=args.x, y=args.y)
    print(f"Output of wobbly_function with input {args.x},{args.y} is {output_value}.")

    # Store the output value of the wobbly function in a file
    csv_filename = args.filebasename + ".csv"
    with open(csv_filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',')
        csv_writer.writerow(['f(x,y)'])
        csv_writer.writerow([output_value])
