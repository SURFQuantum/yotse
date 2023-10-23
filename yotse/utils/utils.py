import os

import pandas


def get_files_by_extension(directory: str, extension: str) -> list:
    """
    Returns a list of files in the given directory with the specified extension.

    Parameters:
    -----------
    directory: str
        The directory to search for files in.
    extension: str
        The file extension to search for.

    Returns:
    --------
    list
        A list of files (and their actual location) in the given directory with the specified extension.
    """
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(extension)]


def file_list_to_single_df(files: list) -> pandas.DataFrame:
    """
    Reads CSV files from a list and combines their content in a single dataframe.

    Parameters:
    -----------
    files: list
        A list of CSV files to read.

    Returns:
    --------
    df : pandas.Dataframe
        Pandas dataframe containing the combined contents of all the CSV files.
    """
    dfs = [pandas.read_csv(file, delimiter=' ') for file in files]
    return pandas.concat(dfs, ignore_index=True)
