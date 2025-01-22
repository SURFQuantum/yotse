"""Module to provide poetry with functions to run all examples.

This file is used to run all the examples in this folder and subfolders
Each file should be of the form example*.py and contain a `main`-method.
If the `main`-method takes an argument `no_output` this will be passed as `True`
to avoid creating plots etc.
"""

import importlib
import inspect
import os
import sys
import time
from typing import Any
from typing import Callable


def run_examples() -> None:
    """Run all examples except the blueprint example, which takes long."""
    path_to_here = os.path.dirname(os.path.abspath(__file__))
    total_execution_time = 0.0
    passed_examples = 0
    for root, folders, files in os.walk(path_to_here):
        for filename in files:
            if filename.startswith("example") and filename.endswith(".py"):
                if not filename.startswith("example_blueprint_main"):
                    filepath = os.path.join(root, filename)
                    example_execution_time = _run_example(filepath)
                    total_execution_time += example_execution_time
                    passed_examples += 1
    print(f"Total execution time for all examples: {total_execution_time:.2f} seconds")
    print(f"\033[92mAll {passed_examples} examples passed!\033[0m")


def run_blueprint_example() -> None:
    """Run only the blueprint example."""
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
    print("\033[92mExamples passed ok\033[0m")


def _run_example(filepath: str) -> float:
    """Run the example specified in `filepath`."""
    cwd = os.getcwd()
    sys.path.append(os.path.dirname(filepath))
    example_module_name = os.path.basename(filepath)[: -len(".py")]
    example_module = importlib.import_module(example_module_name)
    if hasattr(example_module, "main"):
        example_main = getattr(example_module, "main")
    else:
        return 0.0
    os.chdir(os.path.dirname(filepath))
    start_time = time.time()
    if _has_no_output_arg(example_main):
        example_main(no_output=True)
    else:
        example_main()
    end_time = time.time()
    os.chdir(cwd)
    sys.path.pop()
    execution_time = end_time - start_time
    print(f"{filepath} took {execution_time:.2f} seconds to execute")
    return execution_time


def _has_no_output_arg(func: Callable[..., Any]) -> bool:
    """Check if `func` has no output."""
    return "no_output" in inspect.getfullargspec(func).args


if __name__ == "__main__":
    run_examples()
