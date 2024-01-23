"""utils.py.

This module provides functions for file processing, including searching for files with a specific extension
in a directory, combining the content of multiple CSV, JSON, or pickle files into a single pandas DataFrame,
and converting between NumPy arrays and Python lists.

Functions
---------
get_files_by_extension(directory: str, extension: str) -> List[str]:
    Returns a list of files in the given directory with the specified extension.

file_list_to_single_df(files: List[str], extension: str) -> pandas.DataFrame:
    Reads CSV, json, or pickle files from a list and combines their content in a single pandas DataFrame.

ndarray_to_list(numpy_array: np.ndarray) -> List[Any]:
    Convert a NumPy array to a Python list.

list_to_numpy_array(list_data: List[Any]) -> np.ndarray:
    Convert a Python list to a NumPy array.
"""
import os
from typing import Any
from typing import List

import numpy as np
import pandas


def get_files_by_extension(directory: str, extension: str) -> List[str]:
    """Returns a list of files in the given directory with the specified extension.

    Parameters
    ----------
    directory : str
        The directory to search for files in.
    extension : str
        The file extension to search for.

    Returns
    -------
    List[str]
        A list of files (and their actual location) in the given directory with the specified extension.
    """
    return [
        os.path.join(directory, file)
        for file in os.listdir(directory)
        if file.endswith(extension)
    ]


def file_list_to_single_df(files: List[str], extension: str) -> pandas.DataFrame:
    """Reads CSV, json or pickle files from a list and combines their content in a
    single pandas dataframe.

    Parameters
    ----------
    files : List[str]
        A list of files to read.
    extension : str
        File extension of the files in the list.

    Returns
    -------
    pandas.Dataframe
        Pandas dataframe containing the combined contents of all the files.
    """
    if extension == "csv":
        dfs = [pandas.read_csv(file, delimiter=" ") for file in files]
    elif extension == "json":
        dfs = [pandas.read_json(file) for file in files]
    elif extension == "pickle":
        dfs = [pandas.read_pickle(file) for file in files]
    else:
        raise NotImplementedError(
            f"Reading file extension {extension} not implemented yet."
        )
        # Note: See https://pandas.pydata.org/docs/reference/io.html for more IO functions for e.g. XML files.
    return pandas.concat(dfs, ignore_index=True)


def ndarray_to_list(numpy_array: np.ndarray) -> List[Any]:
    """Convert a NumPy array to a Python list.

    Parameters
    ----------
    numpy_array : np.ndarray
        The NumPy array to be converted.

    Returns
    -------
    List[Any]
        A Python list containing the elements of the input NumPy array.
    """
    return numpy_array.tolist()  # type: ignore[no-any-return]


def list_to_numpy_array(list_data: List[Any]) -> np.ndarray:
    """Convert a Python list to a NumPy array.

    Parameters
    ----------
    list_data : List[Any]
        The Python list to be converted.

    Returns
    -------
    np.ndarray
        A NumPy array containing the elements of the input Python list.
    """
    return np.array(list_data)
