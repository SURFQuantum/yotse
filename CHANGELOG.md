CHANGELOG
=========

[//]: # (For more details refer to the [documentation]&#40;&#41;.)

2024-01-31 (1.0.0):
---
* separated optimizations in blackbox and analytical (where the function is known).
* added `SciPyOpt` and `BayesOpt` to available optimization algorithms
* expanded unittest to cover missing functions
* full docs coverage
* refactored optimization
* updated `qcg-pilotjob`

2023-11-08 (0.3.1):
---
* fixed SLURM workflow in `yotse.pre.SystemSetup`
* added documentation about usage with SLURM to `README.md`
* added GitHub pages documentation at https://surfquantum.github.io/yotse/


2023-11-02 (0.3.0)
---
* added `json` and `pickle` output file support in `execution.py`
* added `examples`, `tests` and `show-cov` commands for `poetry`
* added strict type checking with `mypy` to help catch type issues
* added commandline function for analysis script in `execution.py`
* improved CI/CD and added integration tests
* added check for `cost_function` property in `pre.py`


2023-10-23 (0.2.0)
---
* refactor to new, less convoluted code structure
* made experiments resumable by saving/loading state
* switched to dependency management using poetry
* updated installation instructions and documentation in `README.md`
* added `qcq_cfg` parameter to `pre.SystemSetup` to enable direct passing of config parameters to the QCG `LocalManager`


2023-07-13 (0.0.1)
---
* minimal viable product
