#!/bin/bash

# Check if a directory was provided as an argument
if [ $# -eq 0 ]
  then
    echo "Please provide a directory path"
    exit 1
fi

# Recursively find all .md files within the directory and call format_myst.jl on them
find "$1" -name "*.md" -type f -print0 | while read -d $'\0' file
do
  julia format_myst.jl "$file"
  if [ $? -ne 0 ]
  then
    echo "Error processing file: $file.  Stopping formatting."
    exit 1
  fi
done