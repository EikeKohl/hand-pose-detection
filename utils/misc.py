import os
import csv
import yaml

"""
This script contains miscellaneous helper functions.
"""


def create_folder(folderpath):
    """
    This method creates a folder if it does not exist yet.

    Parameters
    ----------
    folderpath: the path of the folder to be created (str)

    Returns
    -------
    None
    """

    if not os.path.exists(folderpath):
        os.mkdir(folderpath)


def read_multiple_csv(csv_data):
    """
    This function reads multiple csv files in a folder and concatenates their content in a list.

    Parameters
    ----------
    csv_data: List of csv files in a directory (list)

    Returns
    -------
    The concatenated content of all csv files.
    """

    data = []
    for file in csv_data:
        with open(file) as f:

            reader = csv.reader(f)
            header = next(reader)

            for line in reader:
                data.append(line)

    return data


def read_yaml_config(filepath):
    """
    This method reads a yaml config.

    Parameters
    ----------
    filepath: The filepath to the yaml config (str)

    Returns
    -------
    The config content
    """

    with open(filepath) as f:
        config = yaml.safe_load(f)
        return config
