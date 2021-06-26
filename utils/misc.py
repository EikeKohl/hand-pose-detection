import os
import csv
import yaml

"""
This script contains miscellaneous helper functions.
"""


def create_folder(folderpath):

    if not os.path.exists(folderpath):
        os.mkdir(folderpath)

def read_multiple_csv(csv_data):
    data = []
    for file in csv_data:
        with open(file) as f:

            reader = csv.reader(f)
            header = next(reader)

            for line in reader:
                data.append(line)

    return data

def read_yaml_config(filepath):

    with open(filepath) as f:
        config = yaml.safe_load(f)
        return config
