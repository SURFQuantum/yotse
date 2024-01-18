"""
Usage:
    python3 function.py -filebasename <folder-to-store-output>/<somefilebasename> -x 3 -y 5.5
"""
import csv
from argparse import ArgumentParser
from typing import Any

import numpy as np


def function(x: float, y: float) -> Any:
    """Returns wobbly function value f(x,y) = (x^2 + y^2) + sin(x^2 + y^2)."""

    radius_squared = x**2 + y**2
    return radius_squared + np.sin(radius_squared)


def ackley_function_2d(x: float, y: float) -> Any:
    """Returns function value of the 2d ackley function."""
    f = (
        -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2)))
        - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
        + np.e
        + 20
    )
    return f


if __name__ == "__main__":
    # Parse the input argument
    parser = ArgumentParser()
    parser.add_argument("--filebasename", type=str)
    parser.add_argument("--x", type=float)
    parser.add_argument("--y", type=float)
    parser.add_argument("--z", type=float)
    parser.add_argument("--test", type=float)
    args = parser.parse_args()
    if args.z is None:
        # default behavior

        # Run the "simulation"
        output_value = ackley_function_2d(x=args.x, y=args.y)
        print(
            f"Output of wobbly_function with input {args.x},{args.y} is {output_value}."
        )

        # Store the output value of the wobbly function in a file together with respective input values
        csv_filename = args.filebasename + ".csv"
        with open(csv_filename, mode="w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=" ")
            csv_writer.writerow(["f(x,y)", "x", "y"])
            csv_writer.writerow([output_value, args.x, args.y])
    else:
        # z parameter is given, implementation for parameter switching
        if args.x is not None:
            raise ValueError(
                f"When inputting z parameter, x has to be None not {args.x}"
            )

        # Run the "simulation"
        output_value = ackley_function_2d(x=args.y, y=args.z)
        print(
            f"Output of wobbly_function with input {args.y},{args.z} is {output_value}."
        )

        # Store the output value of the wobbly function in a file together with respective input values
        csv_filename = args.filebasename + ".csv"
        with open(csv_filename, mode="w") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=" ")
            csv_writer.writerow(["f(y,z)", "y", "z"])
            csv_writer.writerow([output_value, args.y, args.z])
