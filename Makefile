PYTHON3         = python3
TEST_DIR        = tests
SOURCE_DIR     	= yotse
REQUIREMENTS    = requirements.txt
MIN_COV       	= 60
EXAMPLES     	= examples
RUN_EXAMPLES  	= ${EXAMPLES}/run_examples.py

help:
	@echo "install           Installs the package."
	@echo "lint				 Checking if package conforms with coding style."
	@echo "test              Runs the tests."
	@echo "test-cov          Runs the tests with coverage."
	@echo "open-cov-report   Generates coverage report and opens it."
	@echo "example           Runs the examples."
	@echo "verify            Verifies the installation."
	@echo "clean             Cleans .pyc and built files."
#	@echo "test-deps         Installs dependencies for running tests."
	@echo "python-deps       Installs dependencies for using this package."
	@echo "build             Builds the wheel of the package."

install: python-deps
	@echo -e "\n*** Installing $(SOURCE_DIR) package locally"
	@$(PYTHON3) -m pip install -e .

lint:
	@echo -e "\n*** Checking that $(SOURCE_DIR) conforms to PEP8 coding style"
	@$(PYTHON3) -m flake8 ${SOURCE_DIR} ${TEST_DIR} ${EXAMPLES}

test:
	@echo -e "\n*** Running unit tests for $(SOURCE_DIR)"
	@$(PYTHON3) -m pytest --verbose $(TEST_DIR)

test-cov:
	@echo -e "\n*** Running unit tests for $(SOURCE_DIR) including coverage"
	@$(PYTHON3) -m pytest --verbose --cov-fail-under=${MIN_COV} --cov=${SOURCE_DIR} $(TEST_DIR)

open-cov-report:
	@${PYTHON} -m pytest --cov=${SOURCE_DIR} --cov-report=html tests && open htmlcov/index.html

example:
	@echo -e "\n*** Running examples of $(SOURCE_DIR)"
	@$(PYTHON3) $(RUN_EXAMPLES) > /dev/null && echo "Examples OK!"

verify: clean python-deps lint test-cov example _verified

_verified:
	@echo "$(SOURCE_DIR) is verified."

clean: _delete_pyc _clear_build

_delete_pyc:
	@find . -name '*.pyc' -delete

python-deps:
	@${PYTHON3} -m pip install -r ${REQUIREMENTS}

#test-deps:
#	@${PIP} install -r test_requirements.txt

_remove_build:
	@rm -f -r build

_remove_dist:
	@rm -f -r dist

_remove_egg_info:
	@rm -f -r *.egg-info

_clear_build: _remove_build _remove_dist _remove_egg_info

_build:
	@${PYTHON3} -m build

build: _clear_build _build

.PHONY: install lint test test-cov open-cov-report example verify clean python-deps build