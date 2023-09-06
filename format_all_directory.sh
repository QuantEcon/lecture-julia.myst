#!/bin/bash

# Check if at least one argument is given
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 path/to/md/files/ [use_replacements]"
  exit 1
fi

# Directory containing markdown files
dir_path=$1

# Optional use_replacements flag, default to "false" if not provided
use_replacements=${2:-false}

# Loop over all .md files in the given directory
for file_path in "$dir_path"*.md; do
  # Call the Julia script with the current .md file and the use_replacements flag
  julia format_myst.jl "$file_path" $use_replacements
done
