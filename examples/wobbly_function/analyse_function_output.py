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
    # ensure that output.csv is ordered by job_id
    for job_id in range(len(os.listdir(current_path))):
        directory = f'job{job_id}'
        if os.path.isdir(directory):
            print(os.path.join(directory, 'wobbly_example.csv'))
            with open(os.path.join(directory, 'wobbly_example.csv'), newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=' ')
                # the function we are interested in optimizing is the output of the wobbly_function, in other words:
                # the cost C = f(x,y) and we can just copy the values of the jobs without applying any function
                if job_id == 0:
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
