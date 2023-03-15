# ! /usr/bin/env python3
'''concatenate last rows of .csv files under output/ into one file: csv_output.csv'''

import os
import csv
from argparse import ArgumentParser


if __name__ == "__main__":
    # Create source_directory
    parser = ArgumentParser()
    parser.add_argument('--step', required=False, type=str,
                        help='Optimization step')
    args = parser.parse_args()

    current_path = os.getcwd()
    sorted_dir = sorted(os.listdir(current_path))
    for d in sorted_dir:
        if os.path.isdir(d):
            print(os.path.join(d, 'wobbly_example.csv'))
            with open(os.path.join(d, 'wobbly_example.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                for row in reader:
                    new_row = row
                    with open('output.csv', 'a', newline='') as output_file:
                        writer = csv.writer(output_file, delimiter=' ')
                        writer.writerow(new_row)
                        output_file.close()

