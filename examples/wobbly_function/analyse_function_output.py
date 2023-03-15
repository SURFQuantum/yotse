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
    for d, directory in enumerate(sorted_dir):
        if os.path.isdir(directory):
            print(os.path.join(directory, 'wobbly_example.csv'))
            with open(os.path.join(directory, 'wobbly_example.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                if d == 0:
                    with open('output.csv', 'w', newline='') as output_file:
                        writer = csv.writer(output_file, delimiter=' ')
                        for row in reader:
                            writer.writerow(row)
                        output_file.close()
                else:
                    with open('output.csv', 'a', newline='') as output_file:
                        writer = csv.writer(output_file, delimiter=' ')
                        writer.writerow(list(reader)[1])
                        output_file.close()
                csvfile.close()

