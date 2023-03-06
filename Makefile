PYTHON3         = python3
TEST_DIR        = tests
SOURCE_DIR       = qiaopt
REQUIREMENTS    = requirements.txt

all: install test

install:
	@echo -e "\n*** Installing $(SOURCE_DIR) package locally"
	@$(PYTHON3) -m pip install -e .

lint:
	@echo -e "\n*** Checking that $(SOURCE_DIR) conforms to PEP8 coding style"
	@$(PYTHON3) -m flake8 ${SOURCE_DIR} ${TEST_DIR}

test:
	@echo -e "\n*** Running unit tests for $(SOURCE_DIR)"
	@$(PYTHON3) -m pytest --verbose $(TEST_DIR)

clean:
	@find . -name "*.pyc" -delete