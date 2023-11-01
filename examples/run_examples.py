import importlib
import inspect
import os
import sys
import time
from typing import Any
from typing import Callable

# This file is used to run all the examples in this folder and subfolders
# Each file should be of the form example*.py and contain a `main`-method.
# If the `main`-method takes an argument `no_output` this will be passed as `True`
# to avoid creating plots etc.


def main() -> None:
    path_to_here = os.path.dirname(os.path.abspath(__file__))
    total_execution_time = 0.0
    for root, folders, files in os.walk(path_to_here):
        for filename in files:
            if filename.startswith("example") and filename.endswith(".py"):
                if not filename.startswith("example_blueprint_main"):
                    filepath = os.path.join(root, filename)
                    example_execution_time = _run_example(filepath)
                    total_execution_time += example_execution_time
    print(f"Total execution time for all examples: {total_execution_time:.2f} seconds")


def run_blueprint_example() -> None:
    print(" --- Executing NlBlueprint Example --- ")
    path_to_here = os.path.dirname(os.path.abspath(__file__))
    total_execution_time = 0.0
    for root, folders, files in os.walk(path_to_here):
        for filename in files:
            if filename.startswith("example") and filename.endswith(".py"):
                if filename.startswith("example_blueprint_main"):
                    filepath = os.path.join(root, filename)
                    example_execution_time = _run_example(filepath)
                    total_execution_time += example_execution_time
    print(f"Execution time for NlBlueprint example: {total_execution_time:.2f} seconds")


def _run_example(filepath: str) -> float:  # Modified return type to float
    cwd = os.getcwd()
    sys.path.append(os.path.dirname(filepath))
    example_module_name = os.path.basename(filepath)[: -len(".py")]
    example_module = importlib.import_module(example_module_name)
    if hasattr(example_module, "main"):
        main = getattr(example_module, "main")
    else:
        return 0.0  # Return 0.0 if there is no main function
    os.chdir(os.path.dirname(filepath))
    start_time = time.time()
    if _has_no_output_arg(main):
        main(no_output=True)
    else:
        main()
    end_time = time.time()
    os.chdir(cwd)
    sys.path.pop()
    execution_time = end_time - start_time
    print(f"{filepath} took {execution_time:.2f} seconds to execute")
    return execution_time


def _has_no_output_arg(func: Callable[..., Any]) -> bool:
    return "no_output" in inspect.getfullargspec(func).args


if __name__ == "__main__":
    main()
