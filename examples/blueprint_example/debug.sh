#!/bin/bash

# Define the base directory
BASE_DIR="examples/blueprint_example/output"

# Check if the base directory exists
if [[ ! -d "$BASE_DIR" ]]; then
  echo "Base directory $BASE_DIR does not exist."
  exit 1
fi

# Find the first folder in the base directory
FIRST_FOLDER=$(ls -d "$BASE_DIR"/*/ | head -n 1)

# Check if a folder was found
if [[ -z "$FIRST_FOLDER" ]]; then
  echo "No folders found in $BASE_DIR."
  exit 1
fi

# Define the file path
FILE_PATH_ERR="${FIRST_FOLDER}step0/job0/stdout0.err"
FILE_PATH_OUT="${FIRST_FOLDER}step0/job0/stdout0.txt"

# Check if the file exists
if [[ ! -f "$FILE_PATH_ERR" ]]; then
  echo "File $FILE_PATH_ERR does not exist."
  exit 1
fi

# Output the contents of the file
echo "Opening File $FILE_PATH_ERR"
cat "$FILE_PATH_ERR"
echo "Opening File $FILE_PATH_OUT"
cat "$FILE_PATH_OUT"
