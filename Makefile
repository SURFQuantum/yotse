PYTHON3         = python3
TEST_DIR        = tests
SOURCE_DIR     	= qiaopt
REQUIREMENTS    = requirements.txt
MINCOV       	= 90
EXAMPLES     	= examples
RUN_EXAMPLES  	= ${EXAMPLES}/run_examples.py

all: install test

install:
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
	@$(PYTHON3) -m pytest --verbose --cov=${SOURCE_DIR} $(TEST_DIR)

example:
	@echo -e "\n*** Running examples of $(SOURCE_DIR)"
	@$(PYTHON3) $(RUN_EXAMPLES) > /dev/null && echo "Examples OK!"

clean:
	@find . -name "*.pyc" -delete

verify: clean install lint test-cov example _verified

_verified:
	@echo "$(SOURCE_DIR) is verified :)"
